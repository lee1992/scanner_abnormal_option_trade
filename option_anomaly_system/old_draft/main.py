"""
主入口 — PyCharm Console & 命令行均可运行。

PyCharm Console 用法:
    import sys, os
    sys.path.insert(0, r"V:\pythonproject20240816\scanner\option_anomaly_system")
    from main import run
    results = run(
        data_dir   = r"V:\pythonproject20240816\scanner\filter_data",
        output_dir = r"V:\pythonproject20240816\scanner\output",
        start_date = "2026-05-01",
        end_date   = "2026-05-05",
    )

命令行用法:
    python main.py --data-dir ./filter_data --start-date 2026-05-01 --end-date 2026-05-05
"""
from __future__ import annotations
import logging
import sys
import os
import pandas as pd

# ── 路径自适应（PyCharm Console / 命令行 均可）──────────────────────────────
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)



from option_anomaly_system.config    import Config
from option_anomaly_system.loader    import load_multi_day, preprocess
from option_anomaly_system.factor    import compute_all_contract_factors
from option_anomaly_system.scorer  import compute_factor_scores, merge_persistence_scores, build_symbol_rank
from option_anomaly_system.aggregator import (
    aggregate_underlying_daily,
    aggregate_by_moneyness,
    deep_itm_analysis,
    compute_persistence,
    detect_stop_events,
    compute_moneyness_type_rank,          # ← 新增
)
from option_anomaly_system.archiver import (
    archive_daily,
    archive_persistence,
    archive_report,
    archive_moneyness_type_rank,          # ← 新增
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)




# ═══════════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════════

def run(
    data_dir:   str        = r".\filter_data",
    output_dir: str        = r".\output",
    start_date: str | None = None,
    end_date:   str | None = None,
    top_n:      int        = 50,
    moneyness_rank_agg: str = "mean",
) -> dict:
    """
    完整分析流程：ALL / CALL / PUT 三版并行。

    Returns
    -------
    dict，键包含所有中间结果和最终结果：
      factors_contract,
      underlying_all/call/put,
      moneyness_all/call/put,
      deep_itm_all/call/put,
      persistence, stop_events,
      scored, scored_call, scored_put,
      symbol_rank_all, symbol_rank_call, symbol_rank_put
    """
    #cfg = Config(data_dir=data_dir, output_dir=output_dir, top_n_report=top_n)
    cfg = Config(
        data_dir=data_dir,
        output_dir=output_dir,
        top_n_report=top_n,
        moneyness_rank_agg=moneyness_rank_agg,  # ← 新增
    )
    _banner("期权异动多维分析系统 v1.0")

    # ══════════════════════════════════════════════════════════════════
    # STEP 1  加载 & 预处理
    # ══════════════════════════════════════════════════════════════════
    logger.info("STEP 1 ── 加载数据")
    raw = load_multi_day(cfg, start_date=start_date, end_date=end_date)
    df  = preprocess(raw)
    # ── 统计区间内实际交易日数（m = filter_data 中匹配文件数）───────────────
    import re as _re
    from pathlib import Path as _Path
    _data_path = _Path(cfg.data_dir)
    _pat       = _re.compile(r"anomaly_results(\d{4}-\d{2}-\d{2})\.csv$")
    _sd = pd.Timestamp(start_date) if start_date else pd.Timestamp("1900-01-01")
    _ed = pd.Timestamp(end_date)   if end_date   else pd.Timestamp("2999-12-31")
    total_trade_days = sum(
        1 for _f in _data_path.glob("anomaly_results*.csv")
        if (_m := _pat.match(_f.name)) and _sd <= pd.Timestamp(_m.group(1)) <= _ed
    )
    logger.info(f"  区间内交易日数（m）= {total_trade_days}")
    # ══════════════════════════════════════════════════════════════════
    # STEP 2  合约级因子
    # ══════════════════════════════════════════════════════════════════
    logger.info("STEP 2 ── 计算合约级因子")
    df = compute_all_contract_factors(df, cfg)

    # ══════════════════════════════════════════════════════════════════
    # STEP 3  按日聚合（ALL / CALL / PUT）+ 每日存档
    # ══════════════════════════════════════════════════════════════════
    logger.info("STEP 3 ── 标的级聚合 & moneyness 分段（ALL / CALL / PUT）")

    lists: dict[str, list[pd.DataFrame]] = {
        k: [] for k in [
            "und_all", "und_call", "und_put",
            "mon_all", "mon_call", "mon_put",
            "deep_all", "deep_call", "deep_put",
        ]
    }

    for trade_date, day_df in df.groupby("trade_date"):
        logger.info(f"  {trade_date}: {len(day_df)} 行")

        und_trio  = aggregate_underlying_daily(day_df, cfg)
        mon_trio  = aggregate_by_moneyness(day_df, cfg)
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
            trade_date       = trade_date,
            factors_contract = day_df,
            underlying_trio  = und_trio,
            moneyness_trio   = mon_trio,
            deep_itm_trio    = deep_trio,
            cfg              = cfg,
        )

    def _concat(lst: list[pd.DataFrame]) -> pd.DataFrame:
        frames = [f for f in lst if f is not None and not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    underlying_all  = _concat(lists["und_all"])
    underlying_call = _concat(lists["und_call"])
    underlying_put  = _concat(lists["und_put"])
    moneyness_all   = _concat(lists["mon_all"])
    moneyness_call  = _concat(lists["mon_call"])
    moneyness_put   = _concat(lists["mon_put"])
    deep_itm_all    = _concat(lists["deep_all"])
    deep_itm_call   = _concat(lists["deep_call"])
    deep_itm_put    = _concat(lists["deep_put"])

    logger.info(
        f"  标的日级  ALL={len(underlying_all)}行 | "
        f"CALL={len(underlying_call)}行 | PUT={len(underlying_put)}行\n"
        f"  moneyness ALL={len(moneyness_all)}行 | "
        f"CALL={len(moneyness_call)}行 | PUT={len(moneyness_put)}行\n"
        f"  深实值    ALL={len(deep_itm_all)}行 | "
        f"CALL={len(deep_itm_call)}行 | PUT={len(deep_itm_put)}行"
    )
    # ══════════════════════════════════════════════════════════════════
    # STEP 3.5  Moneyness × Call/Put 跨日排名
    # ══════════════════════════════════════════════════════════════════
    logger.info("STEP 3.5 ── Moneyness × Call/Put 跨日排名")
    date_range = f"{df['trade_date'].min()}_to_{df['trade_date'].max()}"
    moneyness_type_rank = compute_moneyness_type_rank(df, cfg, total_trade_days)
    archive_moneyness_type_rank(moneyness_type_rank, date_range, cfg)
    _print_moneyness_type_rank(moneyness_type_rank,cfg)
    # ══════════════════════════════════════════════════════════════════
    # STEP 4  持续性分析 + Stop Event（基于 ALL）
    # ══════════════════════════════════════════════════════════════════
    logger.info("STEP 4 ── 持续性分析 & Stop Event")
    persistence = compute_persistence(underlying_all, moneyness_all, cfg)
    stop_events = detect_stop_events(underlying_all, persistence, cfg)
    archive_persistence(persistence, stop_events, cfg)
    _print_persistence_summary(persistence, stop_events, cfg)

    # ══════════════════════════════════════════════════════════════════
    # STEP 5  多因子打分（三路并行）
    # ══════════════════════════════════════════════════════════════════
    logger.info("STEP 5 ── 多因子打分（ALL / CALL / PUT）")

    def _inject_deep(und: pd.DataFrame, deep: pd.DataFrame) -> pd.DataFrame:
        """将深实值专项的 deep_turnover_share 合并进 underlying 表"""
        if deep.empty:
            und["deep_turnover_share"] = 0.0
            return und
        pivot = deep.groupby(["stock_owner", "trade_date"]).agg(
            deep_turnover_share = ("deep_turnover_share", "max"),
            deep_churning_count = ("is_churning",         "sum"),
        ).reset_index()
        merged = und.merge(pivot, on=["stock_owner", "trade_date"], how="left")
        merged["deep_turnover_share"] = merged["deep_turnover_share"].fillna(0)
        return merged

    und_all_e  = _inject_deep(underlying_all,  deep_itm_all)
    und_call_e = _inject_deep(underlying_call, deep_itm_call)
    und_put_e  = _inject_deep(underlying_put,  deep_itm_put)

    # ALL：因子打分 + 持续性注入 → final_score
    scored      = compute_factor_scores(und_all_e,  cfg)
    scored      = merge_persistence_scores(scored, persistence, stop_events, cfg)

    # CALL / PUT：只做因子打分，daily_score 即为最终分数
    scored_call = compute_factor_scores(und_call_e, cfg)
    scored_put  = compute_factor_scores(und_put_e,  cfg)

    # ══════════════════════════════════════════════════════════════════
    # STEP 6  标的排名 & 报告落盘
    # ══════════════════════════════════════════════════════════════════
    logger.info("STEP 6 ── 生成报告")
    symbol_rank_all  = build_symbol_rank(scored)
    symbol_rank_call = build_symbol_rank(scored_call)
    symbol_rank_put  = build_symbol_rank(scored_put)

    date_range = f"{df['trade_date'].min()}_to_{df['trade_date'].max()}"
    archive_report(
        scored_daily     = scored,
        symbol_rank_all  = symbol_rank_all,
        symbol_rank_call = symbol_rank_call,
        symbol_rank_put  = symbol_rank_put,
        date_range       = date_range,
        cfg              = cfg,
    )

    _print_final_report(symbol_rank_all, symbol_rank_call, symbol_rank_put,   deep_itm_all, cfg)

    return {
        "factors_contract":  df,
        "underlying_all":    underlying_all,
        "underlying_call":   underlying_call,
        "underlying_put":    underlying_put,
        "moneyness_all":     moneyness_all,
        "moneyness_call":    moneyness_call,
        "moneyness_put":     moneyness_put,
        "deep_itm_all":      deep_itm_all,
        "deep_itm_call":     deep_itm_call,
        "deep_itm_put":      deep_itm_put,
        "persistence":       persistence,
        "stop_events":       stop_events,
        "scored":            scored,
        "scored_call":       scored_call,
        "scored_put":        scored_put,
        "symbol_rank_all":   symbol_rank_all,
        "symbol_rank_call":  symbol_rank_call,
        "symbol_rank_put":   symbol_rank_put,
        "moneyness_type_rank": moneyness_type_rank,
    }

def _print_moneyness_type_rank(
    rank_dict: dict,
    cfg: Config,
) -> None:
    """终端打印 moneyness_type_rank 双排序摘要（各组 Top 5 预览）"""
    if not rank_dict:
        return

    agg = cfg.moneyness_rank_agg
    col_log    = f"{agg}_log_turnover"
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

        # 按 log_turnover 排序版
        print(f"    [排序依据: {col_log}]")
        show_cols = ["stock_owner", col_log, "rank_by_log",
                     col_mktval, "rank_by_mktval",
                     "appear_days", "total_trade_days"]
        show = [c for c in show_cols if c in by_log.columns]
        print(by_log.head(preview_n)[show].to_string())

        # 按 turnover_mktval 排序版
        print(f"\n    [排序依据: {col_mktval}]")
        show = [c for c in show_cols if c in by_mktval.columns]
        print(by_mktval.head(preview_n)[show].to_string())
# ═══════════════════════════════════════════════════════════════════════════════
# 打印辅助
# ═══════════════════════════════════════════════════════════════════════════════

def _banner(title: str) -> None:
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


def _print_persistence_summary(
    persistence: pd.DataFrame,
    stop_events: pd.DataFrame,
    cfg: Config,
) -> None:
    if persistence.empty:
        return
    multi = persistence[persistence["active_days"] >= cfg.min_persistence_days]
    print(f"\n  持续 ≥{cfg.min_persistence_days} 天的标的: {len(multi)}")
    if not multi.empty:
        show_cols = ["stock_owner", "active_days", "max_consecutive_days",
                     "persistence_ratio", "dominant_bucket", "total_turnover"]
        show = [c for c in show_cols if c in multi.columns]
        print(
            multi.sort_values("active_days", ascending=False)
            .head(10)[show].to_string(index=False)
        )
    if not stop_events.empty:
        stopped = stop_events[stop_events["stop_status"] == "STOPPED"]
        if not stopped.empty:
            print(f"\n  ⚠  停止事件 ({len(stopped)} 个):")
            print(stopped.to_string(index=False))


def _print_rank_table(rank: pd.DataFrame, label: str, top_n: int) -> None:
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


def _print_final_report(
    rank_all:     pd.DataFrame,
    rank_call:    pd.DataFrame,
    rank_put:     pd.DataFrame,
    deep_itm_all: pd.DataFrame,
    cfg:          Config,
) -> None:
    _print_rank_table(rank_all,  "ALL",  cfg.top_n_report)
    _print_rank_table(rank_call, "CALL", cfg.top_n_report)
    _print_rank_table(rank_put,  "PUT",  cfg.top_n_report)

    # 深实值过手摘要（ALL 口径）
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
            print(
                churning.sort_values("deep_turnover", ascending=False)
                .head(20)[show].to_string(index=False)
            )

    print(f"\n  输出目录: {cfg.output_dir}")
    print("═" * 70)

def option_quick_check(df,code):
    #df = z['factors_contract'].copy()
    df1 = df[df['stock_owner'] == code ].copy()
    df2 = df1[df1['moneyness_bucket']=='DEEP_ITM'].copy()
    df3 = df1[df1['moneyness_bucket'] == 'DEEP_OTM'].copy()
    print("ITM-put \n ",df2[df2['option_type']=='PUT'][['code','trade_date','turnover']],"\n")
    print("OTM-put\n",df3[df3['option_type'] == 'PUT'][['code', 'trade_date', 'turnover']],"\n")
    print("ITM-call\n",df2[df2['option_type']=='CALL'][['code','trade_date','turnover']],"\n")
    print("OTM-call\n",df3[df3['option_type'] == 'CALL'][['code', 'trade_date', 'turnover']],"\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 命令行入口（parse_known_args 兼容 PyCharm Console 注入的额外参数）
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="期权异动多维分析系统")
    parser.add_argument("--data-dir",   default=r".\filter_data")
    parser.add_argument("--output-dir", default=r".\output")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date",   default=None)
    parser.add_argument("--top-n",      default=50, type=int)
    parser.add_argument(
        "--moneyness-rank-agg",
        default="sum",
        choices=["mean", "sum"],
        help="moneyness_type_rank 跨日聚合方式: mean（均值）或 sum（求和）",
    )
    args, _ = parser.parse_known_args()   # _ 忽略 PyCharm 注入的 --mode/--host/--port
    start_date_time = '2026-05-07'
    end_date_time = '2026-05-09'
    z=run(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        start_date=start_date_time,#args.start_date,
        end_date=end_date_time,#args.end_date,
        top_n=args.top_n,
        moneyness_rank_agg=args.moneyness_rank_agg,  # ← 新增
    )


