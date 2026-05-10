"""
存档模块：每个分析维度同时落盘 ALL / CALL / PUT 三张表。

目录结构：
  output/
  ├── archive/
  │   └── 2026-05-05/
  │       ├── factors_contract.csv
  │       ├── underlying_all.csv
  │       ├── underlying_call.csv
  │       ├── underlying_put.csv
  │       ├── moneyness_all.csv
  │       ├── moneyness_call.csv
  │       ├── moneyness_put.csv
  │       ├── deep_itm_all.csv
  │       ├── deep_itm_call.csv
  │       └── deep_itm_put.csv
  ├── reports/
  │   └── {date_range}/
  │       ├── scored_daily.csv
  │       ├── symbol_rank_all.csv
  │       ├── symbol_rank_call.csv
  │       └── symbol_rank_put.csv
  └── stop_events/
      ├── persistence_{ts}.csv
      └── stop_events_{ts}.csv
"""
from __future__ import annotations
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from option_anomaly_system.config import Config

logger = logging.getLogger(__name__)


# ── 底层保存 ──────────────────────────────────────────────────────────────────

def _save(df: pd.DataFrame | None, path: Path, label: str = "") -> None:
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        logger.debug(f"  跳过空表: {label}")
        return
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"  ✓ {label}: {path.name}  ({len(df)} 行)")


def _save_trio(
    all_df:   pd.DataFrame,
    call_df:  pd.DataFrame,
    put_df:   pd.DataFrame,
    base_dir: Path,
    prefix:   str,
    label:    str,
) -> None:
    """一次调用，同时保存 all / call / put 三个文件"""
    _save(all_df,  base_dir / f"{prefix}_all.csv",  f"{label} [ALL]")
    _save(call_df, base_dir / f"{prefix}_call.csv", f"{label} [CALL]")
    _save(put_df,  base_dir / f"{prefix}_put.csv",  f"{label} [PUT]")


# ═══════════════════════════════════════════════════════════════════════════════
# 每日存档
# ═══════════════════════════════════════════════════════════════════════════════

def archive_daily(
    trade_date:      str,
    factors_contract: pd.DataFrame,
    underlying_trio:  tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    moneyness_trio:   tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    deep_itm_trio:    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    cfg: Config,
) -> None:
    """
    每日全量结果落盘（ALL / CALL / PUT 各一文件）。

    Parameters
    ----------
    underlying_trio : (all, call, put)，来自 aggregate_underlying_daily
    moneyness_trio  : (all, call, put)，来自 aggregate_by_moneyness
    deep_itm_trio   : (all, call, put)，来自 deep_itm_analysis
    """
    d = cfg.archive_dir(trade_date)

    # 合约级因子：精简保留核心列
    try:
        from factor import CONTRACT_FACTORS
        base_cols = [
            "code", "name", "trade_date", "stock_owner",
            "option_type", "option_strike_price",
            "option_expiry_date_distance", "option_open_interest",
            "option_implied_volatility",
            "option_delta", "option_gamma", "option_theta", "option_vega",
            "last_price", "open_price", "high_price", "low_price", "prev_close_price",
            "volume", "turnover", "market_val",
            "moneyness_bucket", "dte_bucket", "directional_moneyness",
        ] + list(CONTRACT_FACTORS.keys())
        contract_cols = [c for c in base_cols if c in factors_contract.columns]
    except Exception:
        contract_cols = list(factors_contract.columns)

    _save(factors_contract[contract_cols], d / "factors_contract.csv", "合约因子")
    _save_trio(*underlying_trio, d, "underlying", "标的日级")
    _save_trio(*moneyness_trio,  d, "moneyness",  "moneyness分段")
    _save_trio(*deep_itm_trio,   d, "deep_itm",   "深实值专项")


# ═══════════════════════════════════════════════════════════════════════════════
# 持续性存档
# ═══════════════════════════════════════════════════════════════════════════════

def archive_persistence(
    persistence: pd.DataFrame,
    stop_events: pd.DataFrame,
    cfg: Config,
) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d  = cfg.stop_events_dir()
    _save(persistence, d / f"persistence_{ts}.csv", "持续性统计")
    _save(stop_events, d / f"stop_events_{ts}.csv", "停止事件")


# ═══════════════════════════════════════════════════════════════════════════════
# 报告存档
# ═══════════════════════════════════════════════════════════════════════════════

def archive_report(
    scored_daily:     pd.DataFrame,
    symbol_rank_all:  pd.DataFrame,
    symbol_rank_call: pd.DataFrame,
    symbol_rank_put:  pd.DataFrame,
    date_range:       str,
    cfg: Config,
) -> None:
    d = cfg.reports_dir() / date_range
    d.mkdir(parents=True, exist_ok=True)

    _save(scored_daily,     d / "scored_daily.csv",      "每日评分 [ALL]")
    _save(symbol_rank_all,  d / "symbol_rank_all.csv",   "标的排名 [ALL]")
    _save(symbol_rank_call, d / "symbol_rank_call.csv",  "标的排名 [CALL]")
    _save(symbol_rank_put,  d / "symbol_rank_put.csv",   "标的排名 [PUT]")

def archive_moneyness_type_rank(
    rank_dict: dict,
    date_range: str,
    cfg: "Config",
) -> None:
    """
    存档 compute_moneyness_type_rank 的结果。

    目录结构：
        output/reports/{date_range}/moneyness_type_rank/
            by_log/          ← 按 cf_log_turnover 排序的6个CSV
                ATM_CALL.csv
                ATM_PUT.csv
                ...
            by_mktval/       ← 按 cf_turnover_over_market_val 排序的6个CSV
                ATM_CALL.csv
                ...
    """
    if not rank_dict:
        logger.warning("archive_moneyness_type_rank: 无数据，跳过")
        return

    base_dir  = cfg.reports_dir() / date_range / "moneyness_type_rank"
    dir_log    = base_dir / "by_log"
    dir_mktval = base_dir / "by_mktval"
    dir_log.mkdir(parents=True, exist_ok=True)
    dir_mktval.mkdir(parents=True, exist_ok=True)

    for key, (by_log, by_mktval) in rank_dict.items():
        _save(by_log,    dir_log    / f"{key}.csv", f"Moneyness排名[by_log][{key}]")
        _save(by_mktval, dir_mktval / f"{key}.csv", f"Moneyness排名[by_mktval][{key}]")

# def archive_moneyness_type_rank(
#     rank_dict: dict,
#     date_range: str,
#     cfg: "Config",
# ) -> None:
#     """
#     将 compute_moneyness_type_rank 的结果落盘到
#     output/reports/{date_range}/moneyness_type_rank/
#     """
#     if not rank_dict:
#         logger.warning("archive_moneyness_type_rank: 无数据，跳过")
#         return
#
#     out_dir = cfg.reports_dir() / date_range / "moneyness_type_rank"
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     for key, df in rank_dict.items():
#         _save(df, out_dir / f"{key}.csv", f"Moneyness排名 [{key}]")