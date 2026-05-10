"""
多因子打分 & 报告生成。

流程:
  1. 对标的日级各因子聚合列做百分位归一化（0-100）
  2. 加权求和 → daily_score
  3. 注入持续性分数 & stop_bonus → final_score（仅 ALL 版）
  4. 标的级汇总 → symbol_rank（ALL / CALL / PUT 各一张）
"""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd

from option_anomaly_system.config import Config

logger = logging.getLogger(__name__)


# ── 归一化工具 ────────────────────────────────────────────────────────────────

def _rank_normalize(series: pd.Series) -> pd.Series:
    """百分位归一化到 [0, 100]，NaN 按最低值处理"""
    s = series.fillna(0)
    return (s.rank(pct=True, na_option="bottom") * 100).clip(0, 100)


# ── 因子列映射表 ──────────────────────────────────────────────────────────────
# key   = config.score_weights 中的权重 key
# value = underlying_daily 中对应的聚合列名
# None  = 由外部注入（如 deep_itm_share），不在此映射
FACTOR_COL_MAP: dict[str, str | None] = {
    "cf_log_turnover":          "cf_log_turnover_sum",
    "cf_churn":                 "cf_churn_max",
    "cf_notional_churn":        "cf_notional_churn_max",
    "cf_delta_flow":            "cf_delta_flow_sum",
    "cf_vega_flow":             "cf_vega_flow_sum",
    "cf_gamma_flow":            "cf_gamma_flow_sum",
    "cf_iv_level":              "cf_iv_level_mean",
    "cf_volume_oi_ratio":       "cf_volume_oi_ratio_max",
    "cf_turnover_per_contract": "cf_turnover_per_contract_max",
    "cf_intraday_range":        "cf_intraday_range_max",
    "uf_pcr_turnover":          "uf_pcr_turnover",    # 只在 ALL 版有值
    "uf_deep_itm_share":        "deep_turnover_share", # 由 main 注入后存在
    # 以下两个由 merge_persistence_scores 注入，不在此计算
    "uf_persistence_ratio":     None,
    "uf_stop_bonus":            None,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 因子打分
# ═══════════════════════════════════════════════════════════════════════════════

def compute_factor_scores(underlying_daily: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    对标的日级表逐因子归一化后加权合计 daily_score。

    Parameters
    ----------
    underlying_daily : aggregate_underlying_daily 返回的 all/call/put 任一张表，
                       可额外含 deep_turnover_share 列（由 main 注入）
    """
    df = underlying_daily.copy()
    total_score  = pd.Series(0.0, index=df.index)
    total_weight = 0.0

    for factor_key, col_name in FACTOR_COL_MAP.items():
        weight = cfg.score_weights.get(factor_key, 0.0)
        if weight == 0.0 or col_name is None:
            continue
        if col_name not in df.columns:
            continue

        normalized = _rank_normalize(df[col_name])
        df[f"fscore_{factor_key}"] = normalized
        total_score  += normalized * weight
        total_weight += weight

    df["daily_score"] = (
        (total_score / total_weight).clip(0, 100).round(2)
        if total_weight > 0 else 0.0
    )

    n_factors = sum(
        1 for k, v in FACTOR_COL_MAP.items()
        if v and v in df.columns and cfg.score_weights.get(k, 0) > 0
    )
    logger.info(f"因子打分完成，参与因子: {n_factors} 个")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 注入持续性分数（仅 ALL 版调用）
# ═══════════════════════════════════════════════════════════════════════════════

def merge_persistence_scores(
    scored:      pd.DataFrame,
    persistence: pd.DataFrame,
    stop_events: pd.DataFrame,
    cfg:         Config,
) -> pd.DataFrame:
    """
    将持续性分数 & stop_bonus 注入 scored，重新计算 final_score。
    通常只对 ALL 版调用；CALL/PUT 版保持 daily_score 作为最终分数。
    """
    df = scored.copy()

    # ── 持续性 ───────────────────────────────────────────────────────
    if persistence is not None and not persistence.empty:
        pers = persistence[["stock_owner", "active_days", "persistence_ratio"]].copy()
        pers["persist_score"] = _rank_normalize(pers["persistence_ratio"])
        df = df.merge(pers[["stock_owner", "persist_score", "active_days",
                             "persistence_ratio"]],
                      on="stock_owner", how="left")
        df["persist_score"] = df["persist_score"].fillna(0)
    else:
        df["persist_score"]    = 0.0
        df["active_days"]      = 0
        df["persistence_ratio"] = 0.0

    # ── Stop bonus ───────────────────────────────────────────────────
    if stop_events is not None and not stop_events.empty:
        stop_map = stop_events[["stock_owner", "stop_status",
                                 "stop_date", "peak_date",
                                 "dominant_bucket"]].copy()
        stop_map["stop_bonus"] = stop_map["stop_status"].map(
            {"STOPPED": 100.0, "ONGOING": 60.0, "ACTIVE": 30.0}
        ).fillna(0)
        df = df.merge(stop_map, on="stock_owner", how="left")
        df["stop_bonus"]  = df["stop_bonus"].fillna(0)
        df["stop_status"] = df["stop_status"].fillna("N/A")
    else:
        df["stop_bonus"]      = 0.0
        df["stop_status"]     = "N/A"
        df["dominant_bucket"] = "N/A"
        df["peak_date"]       = None
        df["stop_date"]       = None

    # ── final_score（三项加权）───────────────────────────────────────
    w_daily   = 1.0
    w_persist = cfg.score_weights.get("uf_persistence_ratio", 3.0)
    w_stop    = cfg.score_weights.get("uf_stop_bonus",         2.0)
    total_w   = w_daily + w_persist + w_stop

    df["final_score"] = (
        (df["daily_score"]   * w_daily
         + df["persist_score"] * w_persist
         + df["stop_bonus"]    * w_stop)
        / total_w
    ).clip(0, 100).round(2)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 标的排名
# ═══════════════════════════════════════════════════════════════════════════════

def build_symbol_rank(scored: pd.DataFrame) -> pd.DataFrame:
    """
    按标的聚合，输出最终排名表。
    兼容 ALL 版（含 final_score）和 CALL/PUT 版（含 daily_score）。
    """
    # 选用最终分数列：ALL 版有 final_score，CALL/PUT 版只有 daily_score
    score_col = "final_score" if "final_score" in scored.columns else "daily_score"

    # 找 turnover 汇总列
    t_sum_col = next(
        (c for c in ["turnover_sum", "turnover"] if c in scored.columns), None
    )

    agg_args: dict[str, tuple] = {
        "max_score":   (score_col,    "max"),
        "avg_score":   (score_col,    "mean"),
        "signal_days": ("trade_date", "nunique"),
    }
    if t_sum_col:
        agg_args["total_turnover"]     = (t_sum_col, "sum")
        agg_args["max_daily_turnover"] = (t_sum_col, "max")

    rank = scored.groupby("stock_owner").agg(**agg_args).reset_index()

    # 附加持续性/停止字段（如果已 merge 进 scored）
    extra_cols = [
        "stock_owner", "active_days", "max_consecutive_days",
        "persistence_ratio", "dominant_bucket",
        "stop_status", "stop_date", "peak_date",
    ]
    available = [c for c in extra_cols if c in scored.columns]
    if len(available) > 1:
        extra = scored.drop_duplicates("stock_owner")[available]
        rank  = rank.merge(extra, on="stock_owner", how="left")

    rank = rank.sort_values("max_score", ascending=False).reset_index(drop=True)
    rank.index = rank.index + 1
    rank.index.name = "rank"
    return rank
