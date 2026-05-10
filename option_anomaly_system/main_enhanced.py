"""
期权异动多维分析系统 v1.0 - 循环版本

PyCharm Console 用法:
    import sys, os
    sys.path.insert(0, r"V:\pythonproject20240816\scanner\option_anomaly_system")
    exec(open("main_optimized.py").read())

命令行用法:
    python main_optimized.py
"""
from __future__ import annotations
import logging, sys, os, re
import pandas as pd
from datetime import datetime
from pathlib import Path
from option_anomaly_system. analyze_repeated_stocks import analysis_repeat_func#analyze_repeated_stocks,get_available_end_dates

# ── 路径自适应 ──
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from option_anomaly_system.config import Config
from option_anomaly_system.loader import load_multi_day, preprocess
from option_anomaly_system.factor import compute_all_contract_factors
from option_anomaly_system.scorer import compute_factor_scores, merge_persistence_scores, build_symbol_rank
from option_anomaly_system.aggregator import (
    aggregate_underlying_daily, aggregate_by_moneyness, deep_itm_analysis,
    compute_persistence, detect_stop_events, compute_moneyness_type_rank,
)
from option_anomaly_system.archiver import (
    archive_daily, archive_persistence, archive_report, archive_moneyness_type_rank,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# 配置参数（在此修改）
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    "data_dir": r".\filter_data",
    "output_dir": r".\output",
    "start_date": "auto",  # 自动检测（倒数第3个文件日期）；或指定具体日期如 "2026-05-07"
    "end_date": "auto",    # 自动检测（最后一个文件日期）；或指定具体日期如 "2026-05-09"
    "top_n": 50,
    "moneyness_rank_agg": "sum",
    "report_top_n": 70,
    "report_rank_n": 70,
    "key_metric": "cf_volume_over_open_interest",
}




# ═══════════════════════════════════════════════════════════════════════════════
# 核心分析流程
# ═══════════════════════════════════════════════════════════════════════════════

def run_analysis(cfg_dict: dict) -> dict:
    """
    完整分析流程：ALL / CALL / PUT 三版并行。

    Args:
        cfg_dict: 配置字典，包含 data_dir, output_dir, start_date, end_date, top_n, moneyness_rank_agg

    Returns:
        dict，包含所有中间结果和最终结果
    """
    cfg = Config(
        data_dir=cfg_dict["data_dir"],
        output_dir=cfg_dict["output_dir"],
        top_n_report=cfg_dict["top_n"],
        moneyness_rank_agg=cfg_dict["moneyness_rank_agg"],
    )
    _banner("期权异动多维分析系统 v1.0")

    # STEP 1: 加载 & 预处理
    logger.info("STEP 1 ── 加载数据")
    raw = load_multi_day(cfg, start_date=cfg_dict["start_date"], end_date=cfg_dict["end_date"])
    df = preprocess(raw)

    # 统计实际交易日数
    total_trade_days = _count_trade_days(cfg, cfg_dict["start_date"], cfg_dict["end_date"])
    logger.info(f"  区间内交易日数（m）= {total_trade_days}")

    # STEP 2: 合约级因子
    logger.info("STEP 2 ── 计算合约级因子")
    df = compute_all_contract_factors(df, cfg)

    # STEP 3: 按日聚合 & 每日存档
    logger.info("STEP 3 ── 标的级聚合 & moneyness 分段（ALL / CALL / PUT）")
    lists = {
        k: [] for k in [
            "und_all", "und_call", "und_put",
            "mon_all", "mon_call", "mon_put",
            "deep_all", "deep_call", "deep_put",
        ]
    }

    for trade_date, day_df in df.groupby("trade_date"):
        logger.info(f"  {trade_date}: {len(day_df)} 行")
        und_trio = aggregate_underlying_daily(day_df, cfg)
        mon_trio = aggregate_by_moneyness(day_df, cfg)
        deep_trio = deep_itm_analysis(day_df, cfg)

        lists["und_all"].append(und_trio[0])
        lists["und_call"].append(und_trio[1])
        lists["und_put"].append(und_trio[2])
        lists["mon_all"].append(mon_trio[0])
        lists["mon_call"].append(mon_trio[1])
        lists["mon_put"].append(mon_trio[2])
        for key, part in zip(["deep_all", "deep_call", "deep_put"], deep_trio):
            if not part.empty:
                lists[key].append(part)

        archive_daily(
            trade_date=trade_date, factors_contract=day_df,
            underlying_trio=und_trio, moneyness_trio=mon_trio,
            deep_itm_trio=deep_trio, cfg=cfg,
        )

    def _concat(lst):
        frames = [f for f in lst if f is not None and not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    underlying_all, underlying_call, underlying_put = _concat(lists["und_all"]), _concat(lists["und_call"]), _concat(
        lists["und_put"])
    moneyness_all, moneyness_call, moneyness_put = _concat(lists["mon_all"]), _concat(lists["mon_call"]), _concat(
        lists["mon_put"])
    deep_itm_all, deep_itm_call, deep_itm_put = _concat(lists["deep_all"]), _concat(lists["deep_call"]), _concat(
        lists["deep_put"])

    logger.info(
        f"  标的日级  ALL={len(underlying_all)}行 | CALL={len(underlying_call)}行 | PUT={len(underlying_put)}行\n"
        f"  moneyness ALL={len(moneyness_all)}行 | CALL={len(moneyness_call)}行 | PUT={len(moneyness_put)}行\n"
        f"  深实值    ALL={len(deep_itm_all)}行 | CALL={len(deep_itm_call)}行 | PUT={len(deep_itm_put)}行"
    )

    # STEP 3.5: Moneyness × Call/Put 跨日排名
    logger.info("STEP 3.5 ── Moneyness × Call/Put 跨日排名")
    date_range = f"{df['trade_date'].min()}_to_{df['trade_date'].max()}"
    moneyness_type_rank = compute_moneyness_type_rank(df, cfg, total_trade_days)
    archive_moneyness_type_rank(moneyness_type_rank, date_range, cfg)
    _print_moneyness_type_rank(moneyness_type_rank, cfg)

    # STEP 4: 持续性分析 + Stop Event
    logger.info("STEP 4 ── 持续性分析 & Stop Event")
    persistence = compute_persistence(underlying_all, moneyness_all, cfg)
    stop_events = detect_stop_events(underlying_all, persistence, cfg)
    archive_persistence(persistence, stop_events, cfg)
    _print_persistence_summary(persistence, stop_events, cfg)

    # STEP 5: 多因子打分
    logger.info("STEP 5 ── 多因子打分（ALL / CALL / PUT）")

    def _inject_deep(und, deep):
        if deep.empty:
            und["deep_turnover_share"] = 0.0
            return und
        pivot = deep.groupby(["stock_owner", "trade_date"]).agg(
            deep_turnover_share=("deep_turnover_share", "max"),
            deep_churning_count=("is_churning", "sum"),
        ).reset_index()
        merged = und.merge(pivot, on=["stock_owner", "trade_date"], how="left")
        merged["deep_turnover_share"] = merged["deep_turnover_share"].fillna(0)
        return merged

    und_all_e = _inject_deep(underlying_all, deep_itm_all)
    und_call_e = _inject_deep(underlying_call, deep_itm_call)
    und_put_e = _inject_deep(underlying_put, deep_itm_put)

    scored = compute_factor_scores(und_all_e, cfg)
    scored = merge_persistence_scores(scored, persistence, stop_events, cfg)
    scored_call = compute_factor_scores(und_call_e, cfg)
    scored_put = compute_factor_scores(und_put_e, cfg)

    # STEP 6: 标的排名 & 报告落盘
    logger.info("STEP 6 ── 生成报告")
    symbol_rank_all = build_symbol_rank(scored)
    symbol_rank_call = build_symbol_rank(scored_call)
    symbol_rank_put = build_symbol_rank(scored_put)

    archive_report(
        scored_daily=scored, symbol_rank_all=symbol_rank_all,
        symbol_rank_call=symbol_rank_call, symbol_rank_put=symbol_rank_put,
        date_range=date_range, cfg=cfg,
    )

    _print_final_report(symbol_rank_all, symbol_rank_call, symbol_rank_put, deep_itm_all, cfg)

    return {
        "factors_contract": df,
        "underlying_all": underlying_all, "underlying_call": underlying_call, "underlying_put": underlying_put,
        "moneyness_all": moneyness_all, "moneyness_call": moneyness_call, "moneyness_put": moneyness_put,
        "deep_itm_all": deep_itm_all, "deep_itm_call": deep_itm_call, "deep_itm_put": deep_itm_put,
        "persistence": persistence, "stop_events": stop_events,
        "scored": scored, "scored_call": scored_call, "scored_put": scored_put,
        "symbol_rank_all": symbol_rank_all, "symbol_rank_call": symbol_rank_call, "symbol_rank_put": symbol_rank_put,
        "moneyness_type_rank": moneyness_type_rank,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 报告生成模块
# ═══════════════════════════════════════════════════════════════════════════════

def generate_reports(results: dict, cfg_dict: dict) -> None:
    """
    生成所有Markdown报告
    """
    df = results["factors_contract"].copy()
    start_date = cfg_dict["start_date"]
    end_date = cfg_dict["end_date"]
    key_metric = cfg_dict["key_metric"]
    top_n = cfg_dict["report_top_n"]
    rank_n = cfg_dict["report_rank_n"]

    file_path = Path(
        cfg_dict["output_dir"]) / "reports" / f"{start_date}_to_{end_date}" / "moneyness_type_rank" / "by_log"
    file_path.mkdir(parents=True, exist_ok=True)

    category_map = {
        'DEEP_OTM_CALL': ('DEEP_OTM', 'CALL'),
        'DEEP_OTM_PUT': ('DEEP_OTM', 'PUT'),
        'DEEP_ITM_CALL': ('DEEP_ITM', 'CALL'),
        'DEEP_ITM_PUT': ('DEEP_ITM', 'PUT'),
    }

    summary_cache = {}

    # 主报告：摘要 + 详情
    lines = [
        "# 期权 Moneyness 分类报告\n",
        f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n",
        f"> 指标: `{key_metric}`  |  Top {top_n}\n\n---\n"
    ]

    for cat_name, (bucket, otype) in category_map.items():
        csv_path = file_path / f"{cat_name}.csv"
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

            matrix, s_call, s_put, diff, denom, ratio = _build_summary_metrics(d, key_metric)

            if code_ not in summary_cache:
                summary_cache[code_] = {
                    "code": code_, "matrix": matrix,
                    "s_call": s_call, "s_put": s_put, "diff": diff, "denom": denom, "ratio": ratio,
                }

            lines.append(f"### {rank + 1}. {code_}\n")
            lines.extend(_format_summary_table(code_, matrix, key_metric))
            lines.append("")
            lines.append(f"- diff: {diff:.6f}")
            lines.append(f"- ratio: {ratio:.6f}" if ratio is not None else f"- ratio: NA")
            lines.append("")

            detail = d[(d['moneyness_bucket'] == bucket) & (d['option_type'] == otype)]
            if not detail.empty:
                lines.append(f"<details><summary>📋 {bucket} {otype} 逐日合约明细</summary>\n")
                for trade_date, grp in detail.groupby('trade_date'):
                    lines.append(f"**{trade_date}**\n")
                    lines.append(f"| code | volume | turnover | open_interest |")
                    lines.append(f"|------|--------|----------|---------------|")
                    for _, r in grp.iterrows():
                        lines.append(
                            f"| {r['code']} | {r['volume']} | {r['turnover']:.2f} | {r['option_open_interest']} |")
                    lines.append("")
                lines.append(f"</details>\n")

        lines.append("---\n")

    # 写入主报告
    summary_path = file_path / "report_summary.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"✓ 报告已生成: {summary_path}")

    # 排名报告：Top Diff 和 Top Ratio
    all_items = list(summary_cache.values())

    items_by_diff = sorted(all_items, key=lambda x: x["diff"], reverse=True)
    top_diff_items = items_by_diff[:min(rank_n, len(items_by_diff))]
    _write_ranking_report(
        file_path / f"report_top_diff_{start_date}_{end_date}.md",
        "Top Diff 标的（按 |S_call - S_put|）",
        [
            f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
            f"> 指标: `{key_metric}`  |  Top {rank_n}",
        ],
        top_diff_items,
        key_metric,
    )
    file_path2 = Path(cfg_dict["output_dir"]) / "important"
    file_path2.mkdir(parents=True, exist_ok=True)

    _write_ranking_report(
        file_path2 / f"Report_top_diff_{start_date}_{end_date}.md",
        "Top Diff 标的（按 |S_call - S_put|）",
        [
            f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
            f"> 指标: `{key_metric}`  |  Top {rank_n}",
        ],
        top_diff_items,
        key_metric,
    )

    valid_ratio_items = [it for it in all_items if (it["diff"] != 0 and it["denom"] != 0)]
    zero_ratio_items = [it for it in all_items if not (it["diff"] != 0 and it["denom"] != 0)]
    items_by_ratio_valid = sorted(valid_ratio_items, key=lambda x: x["ratio"], reverse=True)
    top_ratio_valid = items_by_ratio_valid[:min(rank_n, len(items_by_ratio_valid))]
    zero_ratio_items_sorted = sorted(zero_ratio_items, key=lambda x: (x["denom"] == 0, x["diff"], x["code"]))
    top_ratio_items = top_ratio_valid + zero_ratio_items_sorted

    _write_ranking_report(
        file_path / "report_top_ratio.md",
        "Top Ratio 标的（按 |S_call - S_put| / max(S_call,S_put)）",
        [
            f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
            f"> 指标: `{key_metric}`  |  Top {rank_n}",
        ],
        top_ratio_items,
        key_metric,
    )

    # 打印交集
    common_codes = set([x["code"] for x in top_diff_items]).intersection(set([x["code"] for x in top_ratio_valid]))
    if common_codes:
        print("\n✓ Top-Diff 和 Top-Ratio 交集:")
        for c in sorted(common_codes):
            print(f"  {c}")


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def _count_trade_days(cfg: "Config", start_date: str, end_date: str) -> int:
    """统计区间内的交易日数"""
    pattern = re.compile(r"anomaly_results(\d{4}-\d{2}-\d{2})\.csv$")
    sd = pd.Timestamp(start_date) if start_date else pd.Timestamp("1900-01-01")
    ed = pd.Timestamp(end_date) if end_date else pd.Timestamp("2999-12-31")
    return sum(
        1 for f in Path(cfg.data_dir).glob("anomaly_results*.csv")
        if (m := pattern.match(f.name)) and sd <= pd.Timestamp(m.group(1)) <= ed
    )


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _build_summary_metrics(d: pd.DataFrame, key_col: str = "cf_volume_over_open_interest") -> tuple:
    """构建 2x6 汇总矩阵"""
    im = d[d['moneyness_bucket'] == 'DEEP_ITM']
    om = d[d['moneyness_bucket'] == 'DEEP_OTM']
    atm = d[d['moneyness_bucket'] == 'ATM']

    put_itm = _safe_float(im[im["option_type"] == 'PUT'][key_col].sum())
    put_otm = _safe_float(om[om["option_type"] == 'PUT'][key_col].sum())
    put_atm = _safe_float(atm[atm["option_type"] == 'PUT'][key_col].sum())
    call_itm = _safe_float(im[im["option_type"] == 'CALL'][key_col].sum())
    call_otm = _safe_float(om[om["option_type"] == 'CALL'][key_col].sum())
    call_atm = _safe_float(atm[atm["option_type"] == 'CALL'][key_col].sum())

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


def _format_summary_table(code_: str, matrix: dict, key_label: str = "cf_volume_over_open_interest") -> list:
    """格式化汇总表"""
    lines = [
        f"| 方向 | DEEP_ITM({key_label}) | DEEP_OTM({key_label}) | ATM({key_label}) | DEEP_ITM(OI) | DEEP_OTM(OI) | ATM(OI) |",
        f"|------|---------------:|---------------:|---------:|------------:|------------:|--------:|",
        f"| CALL | {matrix['CALL']['DEEP_ITM_key']:.4f} | {matrix['CALL']['DEEP_OTM_key']:.4f} | {matrix['CALL']['ATM_key']:.4f} | "
        f"{matrix['CALL']['DEEP_ITM_oi']} | {matrix['CALL']['DEEP_OTM_oi']} | {matrix['CALL']['ATM_oi']} |",
        f"| PUT  | {matrix['PUT']['DEEP_ITM_key']:.4f} | {matrix['PUT']['DEEP_OTM_key']:.4f} | {matrix['PUT']['ATM_key']:.4f} | "
        f"{matrix['PUT']['DEEP_ITM_oi']} | {matrix['PUT']['DEEP_OTM_oi']} | {matrix['PUT']['ATM_oi']} |",
    ]
    return lines


def _write_ranking_report(out_path: Path, title: str, subtitles: list, items: list, key_label: str) -> None:
    """写入排名报告"""
    lines = [f"# {title}\n\n"]
    lines.extend([ln + "\n" if not ln.endswith("\n") else ln for ln in subtitles])
    lines.append("\n")
    for idx, item in enumerate(items, start=1):
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
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"✓ 已生成: {out_path}")


def _banner(title: str) -> None:
    """打印横幅"""
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


def _print_moneyness_type_rank(rank_dict: dict, cfg: "Config") -> None:
    """打印 moneyness_type_rank 摘要"""
    if not rank_dict:
        return
    agg = cfg.moneyness_rank_agg
    col_log = f"{agg}_log_turnover"
    col_mktval = f"{agg}_turnover_mktval"
    print("\n" + "═" * 70)
    print(f"  Moneyness × Type 排名（跨日聚合: {agg.upper()}）")
    print("═" * 70)
    for key, (by_log, by_mktval) in rank_dict.items():
        print(f"\n  ── {key} ──")
        if by_log.empty:
            print("    （无数据）")
            continue
        preview_n = min(5, len(by_log))
        print(f"    [排序依据: {col_log}]")
        show_cols = ["stock_owner", col_log, "rank_by_log", col_mktval, "rank_by_mktval", "appear_days",
                     "total_trade_days"]
        show = [c for c in show_cols if c in by_log.columns]
        print(by_log.head(preview_n)[show].to_string())
        print(f"\n    [排序依据: {col_mktval}]")
        show = [c for c in show_cols if c in by_mktval.columns]
        print(by_mktval.head(preview_n)[show].to_string())


def _print_persistence_summary(persistence: pd.DataFrame, stop_events: pd.DataFrame, cfg: "Config") -> None:
    """打印持续性分析摘要"""
    if persistence.empty:
        return
    multi = persistence[persistence["active_days"] >= cfg.min_persistence_days]
    print(f"\n  持续 ≥{cfg.min_persistence_days} 天的标的: {len(multi)}")
    if not multi.empty:
        show_cols = ["stock_owner", "active_days", "max_consecutive_days", "persistence_ratio", "dominant_bucket",
                     "total_turnover"]
        show = [c for c in show_cols if c in multi.columns]
        print(multi.sort_values("active_days", ascending=False).head(10)[show].to_string(index=False))
    if not stop_events.empty:
        stopped = stop_events[stop_events["stop_status"] == "STOPPED"]
        if not stopped.empty:
            print(f"\n  ⚠  停止事件 ({len(stopped)} 个):")
            print(stopped.to_string(index=False))


def _print_rank_table(rank: pd.DataFrame, label: str, top_n: int) -> None:
    """打印标的排名表"""
    print(f"\n{'─' * 70}")
    print(f"  📊 标的排名 [{label}]  Top {min(top_n, len(rank))}")
    print(f"{'─' * 70}")
    if rank.empty:
        print("  （无数据）")
        return
    display_cols = [
        "stock_owner", "max_score", "avg_score", "signal_days",
        "total_turnover", "max_daily_turnover",
        "active_days", "max_consecutive_days",
        "dominant_bucket", "stop_status",
    ]
    show = [c for c in display_cols if c in rank.columns]
    print(rank.head(top_n)[show].to_string())


def _print_final_report(rank_all: pd.DataFrame, rank_call: pd.DataFrame, rank_put: pd.DataFrame,
                        deep_itm_all: pd.DataFrame, cfg: "Config") -> None:
    """打印最终报告"""
    _print_rank_table(rank_all, "ALL", cfg.top_n_report)
    _print_rank_table(rank_call, "CALL", cfg.top_n_report)
    _print_rank_table(rank_put, "PUT", cfg.top_n_report)

    if not deep_itm_all.empty and "is_churning" in deep_itm_all.columns:
        churning = deep_itm_all[deep_itm_all["is_churning"]]
        if not churning.empty:
            print(f"\n{'─' * 70}")
            print(f"  🔴 深实值高周转（过手模式）Top 20 [ALL]")
            print(f"{'─' * 70}")
            churn_cols = [
                "stock_owner", "trade_date",
                "deep_turnover", "deep_avg_churn", "deep_max_churn",
                "deep_avg_notional_churn", "deep_turnover_share", "deep_contracts",
            ]
            show = [c for c in churn_cols if c in churning.columns]
            print(churning.sort_values("deep_turnover", ascending=False).head(20)[show].to_string(index=False))

    print(f"\n  输出目录: {cfg.output_dir}")
    print("═" * 70)


def get_all_trade_dates(data_dir: str) -> list:
    """从 data_dir 中提取所有交易日期，按升序返回"""
    pattern = re.compile(r"anomaly_results(\d{4}-\d{2}-\d{2})\.csv$")
    dates = []

    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning(f"数据目录不存在: {data_dir}")
        return []

    for f in data_path.glob("anomaly_results*.csv"):
        if match := pattern.match(f.name):
            dates.append(match.group(1))

    if not dates:
        logger.warning(f"未在 {data_dir} 中找到 anomaly_results_*.csv 文件")
        return []

    return sorted(dates)


def quick_lookup_option_info(stock_code, df):
    """快速查询指定股票的期权信息"""
    stock_data = df[df['stock_owner'] == stock_code].copy()

    for i, j in stock_data.groupby('moneyness_bucket'):
        for c_p in ["CALL", "PUT"]:
            print(f'{i}_{c_p}')
            temp = j[j['option_type'] == c_p]
            print(temp[['name', 'turnover', 'option_open_interest', "option_delta"]])
            print(f'sum of turnover = {temp["turnover"].sum()/1e6}, sum of open interest = {temp["option_open_interest"].sum()/1e4}')
    return stock_data

# ═══════════════════════════════════════════════════════════════════════════════
# 主入口 - 循环版本
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ──────────────────────────────────────────────────────────────────
    # STEP 0: 自动检测日期范围
    # ──────────────────────────────────────────────────────────────────

    # 获取所有可用的交易日期
    all_dates = get_all_trade_dates(CONFIG["data_dir"])
    if not all_dates:
        logger.error(f"无法从 {CONFIG['data_dir']} 中检测到任何交易日期")
        sys.exit(1)

    logger.info(f"检测到交易日期: {all_dates[0]} 至 {all_dates[-1]} (共 {len(all_dates)} 天)")

    # 自动设置 end_date（最后一个文件的日期）
    end_date = CONFIG.get("end_date", "auto")
    if not end_date or end_date == "auto":
        end_date = all_dates[-1]
        logger.info(f"✓ end_date 自动设置为: {end_date}")
    else:
        logger.info(f"✓ end_date 使用指定值: {end_date}")

    # 自动设置 start_date（倒数第3个文件的日期）
    start_date = CONFIG.get("start_date", "auto")
    if not start_date or start_date == "auto":
        if len(all_dates) >= 3:
            start_date = all_dates[-3]
            logger.info(f"✓ start_date 自动设置为（倒数第3个日期）: {start_date}")
        else:
            start_date = all_dates[0]
            logger.warning(f"⚠  文件少于3个，start_date 设置为最早日期: {start_date}")
    else:
        logger.info(f"✓ start_date 使用指定值: {start_date}")

    # ──────────────────────────────────────────────────────────────────
    # STEP 1: 生成循环日期对（保持 end_date 固定，start_date 倒序移动）
    # ──────────────────────────────────────────────────────────────────

    # 筛选出在 [start_date, end_date] 范围内的日期
    sd_ts = pd.Timestamp(start_date)
    ed_ts = pd.Timestamp(end_date)
    available_dates = [d for d in all_dates if sd_ts <= pd.Timestamp(d) <= ed_ts]

    if len(available_dates) < 2:
        logger.error(f"有效日期数不足2个，无法执行循环")
        sys.exit(1)

    logger.info(f"\n有效交易日期: {available_dates[0]} 至 {available_dates[-1]} (共 {len(available_dates)} 天)")

    # 生成日期对：end_date 固定，start_date 从 available_dates[0] 倒序至 available_dates[-1]
    # 即：[available_dates[0], end_date], [available_dates[1], end_date], ..., [available_dates[-1], end_date]
    date_pairs = []
    for i in range(len(available_dates)):
        pair_start = available_dates[i]
        pair_end = available_dates[-1]  # 固定为最后一个日期
        date_pairs.append((pair_start, pair_end))

    logger.info(f"\n生成 {len(date_pairs)} 个日期对进行循环分析:")
    for idx, (s, e) in enumerate(date_pairs, 1):
        print(f"  [{idx:2d}] {s} -> {e}")

    # ──────────────────────────────────────────────────────────────────
    # STEP 2: 循环执行分析
    # ──────────────────────────────────────────────────────────────────

    all_results = []  # 存储所有循环的结果

    for loop_idx, (loop_start, loop_end) in enumerate(date_pairs, 1):
        _banner(f"循环 [{loop_idx}/{len(date_pairs)}]: {loop_start} ──> {loop_end}")

        # 更新 CONFIG
        CONFIG["start_date"] = loop_start
        CONFIG["end_date"] = loop_end

        try:
            # 运行分析
            results = run_analysis(CONFIG)

            # 生成报告
            generate_reports(results, CONFIG)

            # 保存结果
            all_results.append({
                "loop_idx": loop_idx,
                "start_date": loop_start,
                "end_date": loop_end,
                "results": results,
                "status": "SUCCESS"
            })

            print(f"\n✅ 循环 [{loop_idx}] 分析完成！\n")

        except Exception as e:
            logger.error(f"❌ 循环 [{loop_idx}] 执行失败: {e}", exc_info=True)
            all_results.append({
                "loop_idx": loop_idx,
                "start_date": loop_start,
                "end_date": loop_end,
                "results": None,
                "status": f"FAILED: {str(e)}"
            })

    # ──────────────────────────────────────────────────────────────────
    # STEP 3: 最终总结
    # ──────────────────────────────────────────────────────────────────

    _banner("全部循环分析完成")

    success_count = sum(1 for r in all_results if r["status"] == "SUCCESS")
    logger.info(f"\n总计: {len(all_results)} 个循环，成功: {success_count} 个，失败: {len(all_results) - success_count} 个")

    # 最后一个成功的结果用于展示样本
    for result in reversed(all_results):
        if result["status"] == "SUCCESS":
            df = result["results"]['factors_contract'].copy()
            print(f"\n【最后成功的循环样本 - {result['start_date']} to {result['end_date']}】")
            print(f"快速查询: US.TSLA\n")
            stock_data = quick_lookup_option_info('US.QQQ', df)
            break

    print(f"\n📊 所有输出已保存至: {CONFIG['output_dir']}")
    print("═" * 70)
    summary = analysis_repeat_func(end_date,5)
    summary[summary['标的代码']=='US.QQQ']
    stock_data = quick_lookup_option_info('US.QQQ', df)
