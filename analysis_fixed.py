
import os
import re
from datetime import datetime

# NOTE:
# - This script is expected to run in your existing environment where `start_date_time`, `end_date_time`,
#   `pd` and `z` are already defined.
# - It writes markdown reports into: .\output\reports\{start_date_time}_to_{end_date_time}\moneyness_type_rank\by_log\


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0


def _build_summary_metrics(d, key_col='cf_volume_over_open_interest'):
    """
    Build 2x6 summary matrix for one stock_owner:
      rows: CALL, PUT
      cols: DEEP_ITM(key), DEEP_OTM(key), ATM(key), DEEP_ITM(OI), DEEP_OTM(OI), ATM(OI)
    Also computes:
      S_call, S_put based on key_col (3 buckets)
      diff = abs(S_call - S_put)
      denom = max(S_call, S_put)
      ratio = diff/denom if denom>0 else None
    """
    # split buckets
    im = d[d['moneyness_bucket'] == 'DEEP_ITM']
    om = d[d['moneyness_bucket'] == 'DEEP_OTM']
    atm = d[d['moneyness_bucket'] == 'ATM']

    # key sums
    put_itm = _safe_float(im[im["option_type"] == 'PUT'][key_col].sum())
    put_otm = _safe_float(om[om["option_type"] == 'PUT'][key_col].sum())
    put_atm = _safe_float(atm[atm["option_type"] == 'PUT'][key_col].sum())
    call_itm = _safe_float(im[im["option_type"] == 'CALL'][key_col].sum())
    call_otm = _safe_float(om[om["option_type"] == 'CALL'][key_col].sum())
    call_atm = _safe_float(atm[atm["option_type"] == 'CALL'][key_col].sum())

    # OI sums
    put_itm_oi = int(im[im["option_type"] == 'PUT']["option_open_interest"].sum())
    put_otm_oi = int(om[om["option_type"] == 'PUT']["option_open_interest"].sum())
    put_atm_oi = int(atm[atm["option_type"] == 'PUT']["option_open_interest"].sum())
    call_itm_oi = int(im[im["option_type"] == 'CALL']["option_open_interest"].sum())
    call_otm_oi = int(om[om["option_type"] == 'CALL']["option_open_interest"].sum())
    call_atm_oi = int(atm[atm["option_type"] == 'CALL']["option_open_interest"].sum())

    s_call = call_itm + call_otm + call_atm
    s_put = put_itm + put_otm + put_atm
    diff = abs(s_call - s_put)
    denom = max(s_call, s_put)
    ratio = (diff / denom) if denom > 0 else None

    matrix = {
        "CALL": {
            "DEEP_ITM_key": call_itm, "DEEP_OTM_key": call_otm, "ATM_key": call_atm,
            "DEEP_ITM_oi": call_itm_oi, "DEEP_OTM_oi": call_otm_oi, "ATM_oi": call_atm_oi,
        },
        "PUT": {
            "DEEP_ITM_key": put_itm, "DEEP_OTM_key": put_otm, "ATM_key": put_atm,
            "DEEP_ITM_oi": put_itm_oi, "DEEP_OTM_oi": put_otm_oi, "ATM_oi": put_atm_oi,
        }
    }
    return matrix, s_call, s_put, diff, denom, ratio


def _format_summary_table(code_, matrix, key_label='cf_volume_over_open_interest'):
    lines = []
    lines.append(f"| 方向 | DEEP_ITM({key_label}) | DEEP_OTM({key_label}) | ATM({key_label}) | DEEP_ITM(OI) | DEEP_OTM(OI) | ATM(OI) |")
    lines.append(f"|------|---------------:|---------------:|---------:|------------:|------------:|--------:|")
    lines.append(
        f"| CALL | {matrix['CALL']['DEEP_ITM_key']:.4f} | {matrix['CALL']['DEEP_OTM_key']:.4f} | {matrix['CALL']['ATM_key']:.4f} | "
        f"{matrix['CALL']['DEEP_ITM_oi']} | {matrix['CALL']['DEEP_OTM_oi']} | {matrix['CALL']['ATM_oi']} |"
    )
    lines.append(
        f"| PUT  | {matrix['PUT']['DEEP_ITM_key']:.4f} | {matrix['PUT']['DEEP_OTM_key']:.4f} | {matrix['PUT']['ATM_key']:.4f} | "
        f"{matrix['PUT']['DEEP_ITM_oi']} | {matrix['PUT']['DEEP_OTM_oi']} | {matrix['PUT']['ATM_oi']} |"
    )
    return lines


def find_common_codes_in_top_diff_md(md_paths):
    """
    From multiple report_top_diff.md paths, find common stock codes.
    Code lines are like: '### 1. US.TSLA'
    """
    per_file = {}
    for p in md_paths:
        try:
            with open(p, 'r', encoding='utf-8') as f:
                txt = f.read()
        except Exception:
            per_file[p] = set()
            continue
        codes = set(re.findall(r'^###\s+\d+\.\s+([A-Za-z]{1,5}\.[A-Za-z0-9\-_]+)\s*$', txt, flags=re.M))
        per_file[p] = codes

    sets = [s for s in per_file.values() if s is not None]
    if not sets:
        return set(), per_file
    common = set.intersection(*sets) if sets else set()
    return common, per_file


# ---------------- Main report generation ----------------

file_path = f".\\output\\reports\\{start_date_time}_to_{end_date_time}\\moneyness_type_rank\\by_log\\"
df = z['factors_contract'].copy()

key_ = 'cf_volume_over_open_interest'
top_n = 20          # per-category listing length in report_summary.md
rank_n = 20         # top N for diff/ration ranking reports

# Only these 4 categories currently used
category_map = {
    'DEEP_OTM_CALL': ('DEEP_OTM', 'CALL'),
    'DEEP_OTM_PUT':  ('DEEP_OTM', 'PUT'),
    'DEEP_ITM_CALL': ('DEEP_ITM', 'CALL'),
    'DEEP_ITM_PUT':  ('DEEP_ITM', 'PUT'),
}

lines = [
    "# 期权 Moneyness 分类报告\n",
    f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n",
    f"> 指标: `{key_}`  |  Top {top_n}\n\n---\n"
]

# cache unique stock summary for ranking reports
summary_cache = {}  # code_ -> dict(metrics)

for cat_name, (bucket, otype) in category_map.items():
    csv_path = f'{file_path}{cat_name}.csv'
    try:
        a = pd.read_csv(csv_path)
    except Exception:
        a = None

    lines.append(f"\n## {cat_name}\n")

    if a is None or a.shape[0] == 0:
        lines.append("*该类别无数据*\n")
        lines.append("---\n")
        continue

    a_sorted = a.sort_values('sum_volume_open_interest', ascending=False).reset_index(drop=True)
    cat_top_n = min(top_n, a_sorted.shape[0])

    for rank in range(cat_top_n):
        row = a_sorted.iloc[rank]
        code_ = row['stock_owner']
        d = df[df['stock_owner'] == code_]

        matrix, s_call, s_put, diff, denom, ratio = _build_summary_metrics(d, key_col=key_)

        # update cache (dedupe by code_)
        if code_ not in summary_cache:
            summary_cache[code_] = {
                "code": code_,
                "matrix": matrix,
                "s_call": s_call,
                "s_put": s_put,
                "diff": diff,
                "denom": denom,
                "ratio": ratio,
            }

        lines.append(f"### {rank+1}. {code_}\n")
        lines.extend(_format_summary_table(code_, matrix, key_label=key_))
        lines.append("")
        lines.append(f"- diff: {diff:.6f}")
        lines.append(f"- ratio: {ratio:.6f}" if ratio is not None else f"- ratio: NA")
        lines.append("")

        # Drill-down details for the specific bucket+type in this category
        detail = d[(d['moneyness_bucket'] == bucket) & (d['option_type'] == otype)]
        if detail.empty:
            lines.append(f"*该股票在 {bucket} {otype} 无合约数据*\n")
            continue

        lines.append(f"<details><summary>📋 {bucket} {otype} 逐日合约明细 (点击展开)</summary>\n")
        for trade_date, grp in detail.groupby('trade_date'):
            lines.append(f"**{trade_date}**\n")
            lines.append(f"| code | volume | turnover | open_interest |")
            lines.append(f"|------|--------|----------|---------------|")
            for _, r in grp.iterrows():
                lines.append(f"| {r['code']} | {r['volume']} | {r['turnover']:.2f} | {r['option_open_interest']} |")
            lines.append("")
        lines.append(f"</details>\n")

    lines.append("---\n")

# write summary report
out_path = f"{file_path}report_summary.md"
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"报告已生成: {out_path}")


# ---------------- Ranking reports ----------------

def _write_ranking_report(out_md_path, title, subtitle_lines, ranked_items, key_label):
    lines = [f"# {title}\n\n"]
    lines.extend([ln + "\n" if not ln.endswith("\n") else ln for ln in subtitle_lines])
    lines.append("\n")
    for idx, item in enumerate(ranked_items, start=1):
        code_ = item["code"]
        matrix = item["matrix"]
        diff = item["diff"]
        ratio = item["ratio"]

        lines.append(f"### {idx}. {code_}\n\n")
        lines.extend(_format_summary_table(code_, matrix, key_label=key_label))
        lines.append("\n")
        lines.append(f"- diff: {diff:.6f}")
        lines.append(f"- ratio: {ratio:.6f}" if ratio is not None else f"- ratio: NA")
        lines.append("\n")
    with open(out_md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"已生成: {out_md_path}")


all_items = list(summary_cache.values())

# Top diff
items_by_diff = sorted(all_items, key=lambda x: x["diff"], reverse=True)
top_diff_items = items_by_diff[:min(rank_n, len(items_by_diff))]
top_diff_path = f"{file_path}report_top_diff.md"
_write_ranking_report(
    top_diff_path,
    "Top Diff 标的（按 |S_call - S_put|）",
    [
        f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"> 指标: `{key_}`  |  Top {rank_n}",
    ],
    top_diff_items,
    key_label=key_,
)

# Top ratio with your rule:
# - valid items (diff>0 and denom>0) compete for rank_n
# - zero items (diff==0 or denom==0) are appended at the end and do NOT consume rank_n quota
valid_ratio_items = [it for it in all_items if (it["diff"] != 0 and it["denom"] != 0)]
zero_ratio_items = [it for it in all_items if not (it["diff"] != 0 and it["denom"] != 0)]

items_by_ratio_valid = sorted(valid_ratio_items, key=lambda x: x["ratio"], reverse=True)
top_ratio_valid = items_by_ratio_valid[:min(rank_n, len(items_by_ratio_valid))]

# Append all zero items at the end (keeping deterministic order)
zero_ratio_items_sorted = sorted(zero_ratio_items, key=lambda x: (x["denom"] == 0, x["diff"], x["code"]))
top_ratio_items = top_ratio_valid + zero_ratio_items_sorted

top_ratio_path = f"{file_path}report_top_ratio.md"
_write_ranking_report(
    top_ratio_path,
    "Top Ratio 标的（按 |S_call - S_put| / max(S_call,S_put)）",
    [
        f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"> 指标: `{key_}`  |  Top {rank_n}",
    ],
    top_ratio_items,
    key_label=key_,
)

# Print intersection codes between top-diff and top-ratio(valid) in terminal
common_codes = set([x["code"] for x in top_diff_items]).intersection(set([x["code"] for x in top_ratio_valid]))
print("Intersection of Top-Diff and Top-Ratio (valid) codes:")
for c in sorted(common_codes):
    print(c)
