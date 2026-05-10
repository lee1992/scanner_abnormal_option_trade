"""
聚合模块：
  所有函数均返回 (all_df, call_df, put_df) 三元组。
  1. aggregate_underlying_daily  → (stock_owner, trade_date) 级汇总
  2. aggregate_by_moneyness      → 分段统计
  3. deep_itm_analysis           → 深实值专项
  4. compute_persistence         → 跨日持续性（基于 ALL 数据）
  5. detect_stop_events          → 停止事件检测（基于 ALL 数据）
"""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd

from option_anomaly_system.config import Config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 内部辅助
# ═══════════════════════════════════════════════════════════════════════════════

def _underlying_agg_for(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    对传入 df（全量 / 仅CALL / 仅PUT）做标的日级聚合。
    key = (stock_owner, trade_date)
    """
    KEY = ["stock_owner", "trade_date"]
    if df.empty:
        return pd.DataFrame(columns=KEY)

    numeric_agg: dict[str, list] = {
        "turnover":                  ["sum", "max"],
        "volume":                    ["sum", "max"],
        "option_open_interest":      ["sum", "mean"],
        "option_implied_volatility": ["mean"],
        "option_delta":              ["mean"],
        "market_val":                ["last"],   # 标的市值（每日取最后一条即可）
        "cf_log_turnover":           ["sum", "mean"],
        "cf_churn":                  ["mean", "max"],
        "cf_notional_churn":         ["mean", "max"],
        "cf_delta_flow":             ["sum"],
        "cf_vega_flow":              ["sum"],
        "cf_gamma_flow":             ["sum"],
        "cf_iv_level":               ["mean"],
        "cf_volume_oi_ratio":        ["mean", "max"],
        "cf_turnover_per_contract":  ["mean", "max"],
        "cf_intraday_range":         ["mean", "max"],
        "cf_theta_vega_ratio":       ["mean"],
        "cf_dte_weight":             ["mean"],
        "cf_market_val_log":         ["last"],
    }

    # 只聚合实际存在且非全空的列
    existing_agg = {
        k: v for k, v in numeric_agg.items()
        if k in df.columns and not df[k].isna().all()
    }

    g = df.groupby(KEY).agg(existing_agg)
    g.columns = ["_".join(col) for col in g.columns]
    g = g.reset_index()

    # 合约数量
    cnt = df.groupby(KEY)["code"].nunique().reset_index()
    cnt.columns = KEY + ["contracts"]
    g = g.merge(cnt, on=KEY, how="left")

    return g


def _moneyness_agg_for(df: pd.DataFrame) -> pd.DataFrame:
    """
    对传入 df 做 (stock_owner, trade_date, moneyness_bucket) 聚合。
    """
    KEY = ["stock_owner", "trade_date", "moneyness_bucket"]
    if df.empty:
        return pd.DataFrame()

    g = df.groupby(KEY, observed=True).agg(
        bucket_turnover              = ("turnover",                  "sum"),
        bucket_volume                = ("volume",                    "sum"),
        bucket_contracts             = ("code",                      "nunique"),
        bucket_oi_sum                = ("option_open_interest",      "sum"),
        bucket_avg_iv                = ("option_implied_volatility", "mean"),
        bucket_avg_delta             = ("option_delta",              "mean"),
        bucket_avg_churn             = ("cf_churn",                  "mean"),
        bucket_max_churn             = ("cf_churn",                  "max"),
        bucket_avg_notional_churn    = ("cf_notional_churn",         "mean"),
        bucket_max_notional_churn    = ("cf_notional_churn",         "max"),
        bucket_delta_flow            = ("cf_delta_flow",             "sum"),
        bucket_vega_flow             = ("cf_vega_flow",              "sum"),
    ).reset_index()

    # 占标的当日总成交额的比例
    total = (
        df.groupby(["stock_owner", "trade_date"])["turnover"]
        .sum().reset_index()
        .rename(columns={"turnover": "day_total_turnover"})
    )
    g = g.merge(total, on=["stock_owner", "trade_date"], how="left")
    g["bucket_turnover_share"] = (
        g["bucket_turnover"] / g["day_total_turnover"].replace(0, np.nan)
    )
    return g


def _deep_itm_for(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    对传入 df 做 DEEP_ITM 专项统计，key = (stock_owner, trade_date)。
    """
    KEY = ["stock_owner", "trade_date"]

    deep = df[df["moneyness_bucket"] == "DEEP_ITM"].copy()
    if deep.empty:
        return pd.DataFrame()

    g = deep.groupby(KEY).agg(
        deep_turnover            = ("turnover",                  "sum"),
        deep_volume              = ("volume",                    "sum"),
        deep_contracts           = ("code",                      "nunique"),
        deep_oi_sum              = ("option_open_interest",      "sum"),
        deep_avg_churn           = ("cf_churn",                  "mean"),
        deep_max_churn           = ("cf_churn",                  "max"),
        deep_avg_notional_churn  = ("cf_notional_churn",         "mean"),
        deep_avg_iv              = ("option_implied_volatility", "mean"),
        deep_avg_delta           = ("option_delta",              "mean"),
    ).reset_index()

    # 占当日总成交额比例（分母取 df 全量，不只看 DEEP_ITM）
    total = (
        df.groupby(KEY)["turnover"]
        .sum().reset_index()
        .rename(columns={"turnover": "day_total_turnover"})
    )
    g = g.merge(total, on=KEY, how="left")
    g["deep_turnover_share"] = (
        g["deep_turnover"] / g["day_total_turnover"].replace(0, np.nan)
    )

    # "过手" flag：成交额 >= 门槛 且 名义周转率 >= 阈值
    g["is_churning"] = (
        (g["deep_turnover"]           >= cfg.min_turnover_usd)
        & (g["deep_avg_notional_churn"] >= cfg.deep_itm_churn_threshold)
    )
    return g


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 标的-日级聚合（ALL / CALL / PUT）
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_underlying_daily(
    df: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    (all_df, call_df, put_df)
    all_df 附带 PCR 列；call_df / put_df 无 PCR。
    """
    KEY = ["stock_owner", "trade_date"]

    all_g  = _underlying_agg_for(df,                               cfg)
    call_g = _underlying_agg_for(df[df["option_type"] == "CALL"], cfg)
    put_g  = _underlying_agg_for(df[df["option_type"] == "PUT"],  cfg)

    # ── PCR（仅 all_g）────────────────────────────────────────────────
    if not call_g.empty and not put_g.empty and "turnover_sum" in call_g.columns:
        pcr_call = call_g[KEY + ["turnover_sum", "volume_sum"]].rename(
            columns={"turnover_sum": "_c_to", "volume_sum": "_c_vo"})
        pcr_put  = put_g[KEY + ["turnover_sum", "volume_sum"]].rename(
            columns={"turnover_sum": "_p_to", "volume_sum": "_p_vo"})
        all_g = all_g.merge(pcr_call, on=KEY, how="left")
        all_g = all_g.merge(pcr_put,  on=KEY, how="left")
        all_g["uf_pcr_turnover"] = (
            all_g["_p_to"].fillna(0) / all_g["_c_to"].fillna(0).replace(0, np.nan)
        ).fillna(0)
        all_g["uf_pcr_volume"] = (
            all_g["_p_vo"].fillna(0) / all_g["_c_vo"].fillna(0).replace(0, np.nan)
        ).fillna(0)
        all_g.drop(columns=["_c_to", "_c_vo", "_p_to", "_p_vo"], inplace=True)
    else:
        all_g["uf_pcr_turnover"] = 0.0
        all_g["uf_pcr_volume"]   = 0.0

    return all_g, call_g, put_g


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Moneyness 分段统计（ALL / CALL / PUT）
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_by_moneyness(
    df: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    (all_df, call_df, put_df)
    每张表：(stock_owner, trade_date, moneyness_bucket)
    """
    all_g  = _moneyness_agg_for(df)
    call_g = _moneyness_agg_for(df[df["option_type"] == "CALL"])
    put_g  = _moneyness_agg_for(df[df["option_type"] == "PUT"])
    return all_g, call_g, put_g


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 深实值专项（ALL / CALL / PUT）
# ═══════════════════════════════════════════════════════════════════════════════

def deep_itm_analysis(
    df: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    (all_df, call_df, put_df)
    每张表：(stock_owner, trade_date) 深实值统计
    """
    all_g  = _deep_itm_for(df,                               cfg)
    call_g = _deep_itm_for(df[df["option_type"] == "CALL"],  cfg)
    put_g  = _deep_itm_for(df[df["option_type"] == "PUT"],   cfg)
    return all_g, call_g, put_g


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 持续性计算（基于 ALL 数据）
# ═══════════════════════════════════════════════════════════════════════════════

def _max_consecutive(active_dates: list[str], all_dates: list[str]) -> int:
    """在 all_dates 时间轴上，计算 active_dates 中最长连续出现天数"""
    active_set = set(active_dates)
    max_run = cur_run = 0
    for d in all_dates:
        if d in active_set:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    return max_run


def compute_persistence(
    underlying_all: pd.DataFrame,
    moneyness_all:  pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """
    对每个 stock_owner，在 lookback_days 窗口内计算：
    - active_days / persistence_ratio / max_consecutive_days
    - dominant_bucket（全量口径）
    - 每个 moneyness bucket 的持续天数和比例（扁平化列）

    Parameters
    ----------
    underlying_all : ALL 版标的日级聚合（aggregate_underlying_daily 返回的第0个）
    moneyness_all  : ALL 版 moneyness 分段（aggregate_by_moneyness 返回的第0个）
    """
    all_dates  = sorted(underlying_all["trade_date"].unique())
    lookback   = cfg.lookback_days
    window     = all_dates[-lookback:] if len(all_dates) >= lookback else all_dates
    total_days = len(window)

    # 找 turnover 汇总列
    t_col = next(
        (c for c in ["turnover_sum", "turnover"] if c in underlying_all.columns),
        None,
    )

    records = []
    for stock, grp in underlying_all.groupby("stock_owner"):
        grp_w        = grp[grp["trade_date"].isin(window)].sort_values("trade_date")
        active_dates = sorted(grp_w["trade_date"].unique())
        active_days  = len(active_dates)
        persist_ratio = active_days / total_days if total_days > 0 else 0.0
        max_consec    = _max_consecutive(active_dates, window)

        # 成交额统计
        if t_col and t_col in grp_w.columns:
            total_turnover   = grp_w[t_col].sum()
            max_daily_t      = grp_w[t_col].max()
            peak_idx         = grp_w[t_col].idxmax()
            peak_date        = grp_w.loc[peak_idx, "trade_date"]
        else:
            total_turnover = max_daily_t = 0.0
            peak_date = active_dates[-1] if active_dates else None

        # 主导 moneyness bucket（成交额最大的分段）
        dominant_bucket = "N/A"
        mn_w = pd.DataFrame()
        if moneyness_all is not None and not moneyness_all.empty:
            mn_w = moneyness_all[
                (moneyness_all["stock_owner"] == stock)
                & (moneyness_all["trade_date"].isin(window))
            ]
            if not mn_w.empty and "bucket_turnover" in mn_w.columns:
                bt = mn_w.groupby("moneyness_bucket", observed=True)["bucket_turnover"].sum()
                if not bt.empty:
                    dominant_bucket = str(bt.idxmax())

        # 分段级持续性（扁平化为列：bp_{bucket}_active_days / bp_{bucket}_ratio）
        bucket_stats: dict = {}
        if not mn_w.empty:
            for bn, bg in mn_w.groupby("moneyness_bucket", observed=True):
                b_days = bg["trade_date"].nunique()
                key    = str(bn)
                bucket_stats[f"bp_{key}_active_days"] = b_days
                bucket_stats[f"bp_{key}_ratio"]       = (
                    b_days / total_days if total_days > 0 else 0.0
                )

        row = {
            "stock_owner":          stock,
            "active_days":          active_days,
            "persistence_ratio":    round(persist_ratio, 4),
            "max_consecutive_days": max_consec,
            "total_turnover":       total_turnover,
            "max_daily_turnover":   max_daily_t,
            "peak_date":            peak_date,
            "dominant_bucket":      dominant_bucket,
            "window_days":          total_days,
        }
        row.update(bucket_stats)
        records.append(row)

    result = pd.DataFrame(records)
    logger.info(f"持续性分析: {len(result)} 个标的 | 窗口 {total_days} 天")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Stop Event 检测（基于 ALL 数据）
# ═══════════════════════════════════════════════════════════════════════════════

def detect_stop_events(
    underlying_all: pd.DataFrame,
    persistence:    pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """
    对有持续性信号的标的，检测"停止事件"。

    stop_status:
      ONGOING : 最后一天仍是活跃日
      ACTIVE  : 峰值后未出现连续衰减
      STOPPED : 峰值后连续 stop_window_days 天 < 峰值 × stop_drop_ratio
    """
    t_col = next(
        (c for c in ["turnover_sum", "turnover"] if c in underlying_all.columns),
        None,
    )
    all_dates = sorted(underlying_all["trade_date"].unique())
    last_date = all_dates[-1] if all_dates else None
    records   = []

    for _, prow in persistence.iterrows():
        stock = prow["stock_owner"]
        if prow["active_days"] < cfg.min_persistence_days:
            continue

        grp = (
            underlying_all[underlying_all["stock_owner"] == stock]
            .sort_values("trade_date")
        )
        if grp.empty or t_col is None or t_col not in grp.columns:
            continue

        peak_idx  = grp[t_col].idxmax()
        peak_val  = grp.loc[peak_idx, t_col]
        peak_date = grp.loc[peak_idx, "trade_date"]

        after = grp[grp["trade_date"] > peak_date].head(cfg.stop_window_days)

        if len(after) == 0 or peak_date == last_date:
            status    = "ONGOING"
            stop_date = None
        elif (
            len(after) == cfg.stop_window_days
            and (after[t_col] < peak_val * cfg.stop_drop_ratio).all()
        ):
            status    = "STOPPED"
            stop_date = after["trade_date"].iloc[0]
        else:
            status    = "ACTIVE"
            stop_date = None

        records.append({
            "stock_owner":       stock,
            "stop_status":       status,
            "peak_date":         peak_date,
            "peak_turnover":     peak_val,
            "stop_date":         stop_date,
            "active_days":       prow["active_days"],
            "persistence_ratio": prow["persistence_ratio"],
            "dominant_bucket":   prow.get("dominant_bucket", "N/A"),
        })

    result = pd.DataFrame(records)
    if not result.empty:
        n_stopped = (result["stop_status"] == "STOPPED").sum()
        n_ongoing = (result["stop_status"] == "ONGOING").sum()
        logger.info(f"Stop Event: {n_stopped} 个已停止 | {n_ongoing} 个持续中")
    return result

### 新增
# ═══════════════════════════════════════════════════════════════════════════════
# 新增：Moneyness × Call/Put 跨日排名
# ═══════════════════════════════════════════════════════════════════════════════

def compute_moneyness_type_rank(
    df: pd.DataFrame,
    cfg: "Config",
    total_trade_days: int,
) -> dict:
    """
    6 组（ATM/DEEP_ITM/DEEP_OTM × CALL/PUT）排名。

    每组输出两个 DataFrame（存在 tuple 里）：
        (rank_by_log, rank_by_mktval)

    聚合方式由 cfg.moneyness_rank_agg 控制：
        "mean" → 跨日取平均（日均活跃度）
        "sum"  → 跨日求和（区间总量）

    列说明：
        {agg}_log_turnover      : 跨日 mean/sum 的 cf_log_turnover
        {agg}_turnover_mktval   : 跨日 mean/sum 的 cf_turnover_over_market_val
        rank_by_log             : 按 log_turnover 的全量排名（全量计算，head 前）
        rank_by_mktval          : 按 turnover_mktval 的全量排名（全量计算，head 前）
        appear_days             : 该股在该组出现的交易日数
        total_trade_days        : 区间内总交易日数（= filter_data 文件数）
    """
    TARGET_BUCKETS = {"ATM", "DEEP_ITM", "DEEP_OTM"}
    agg_method = cfg.moneyness_rank_agg  # "mean" or "sum"

    if agg_method not in ("mean", "sum"):
        logger.warning(
            f"moneyness_rank_agg='{agg_method}' 不合法，回退为 'mean'"
        )
        agg_method = "mean"

    prefix = agg_method  # 列名前缀，方便区分

    # ── Step 1: 每股每天聚合（同股票多合约直接求和）─────────────────────────
    sub = df[df["moneyness_bucket"].isin(TARGET_BUCKETS)].copy()
    if sub.empty:
        logger.warning("compute_moneyness_type_rank: 过滤后无数据")
        return {}

    daily = (
        sub.groupby(
            ["stock_owner", "trade_date", "moneyness_bucket", "option_type"],
            observed=True,
        )
        .agg(
            day_log_turnover    = ("cf_log_turnover",             "sum"),
            day_turnover_mktval = ("cf_turnover_over_market_val", "sum"),
            day_volume_over_open_interest= ("cf_volume_over_open_interest", "sum"),
            day_open_interest = ("option_open_interest", "sum"),
        )
        .reset_index()
    )

    # ── Step 2: 跨日聚合（mean 或 sum）+ 出现天数 ─────────────────────────
    cross_day = (
        daily.groupby(
            ["stock_owner", "moneyness_bucket", "option_type"],
            observed=True,
        )
        .agg(
            **{
                f"{prefix}_log_turnover":    ("day_log_turnover",    agg_method),
                f"{prefix}_turnover_mktval": ("day_turnover_mktval", agg_method),
                f"{prefix}_volume_open_interest": ("day_volume_over_open_interest", agg_method),
                f"{prefix}_open_interest": ("day_open_interest", agg_method),
            },
            appear_days = ("trade_date", "nunique"),
        )
        .reset_index()
    )
    cross_day["total_trade_days"] = total_trade_days

    col_log    = f"{prefix}_log_turnover"
    col_mktval = f"{prefix}_turnover_mktval"
    col_volume_open_interest = f"{prefix}_volume_open_interest"
    col_open_interest = f"{prefix}_open_interest"

    # ── Step 3: 拆成 6 组，每组生成两份排序表 ────────────────────────────────
    result = {}
    top_n  = cfg.top_n_report

    for bucket in ["ATM", "DEEP_ITM", "DEEP_OTM"]:
        for otype in ["CALL", "PUT"]:
            key  = f"{bucket}_{otype}"
            mask = (
                (cross_day["moneyness_bucket"] == bucket) &
                (cross_day["option_type"]      == otype)
            )
            grp = cross_day[mask].copy()
            grp = grp.drop(columns=["moneyness_bucket", "option_type"])

            # 全量双排名（head 之前计算，保证 rank 是全局排名）
            grp["rank_by_volume"]    = grp[col_volume_open_interest].rank(
                ascending=False, method="min").astype(int)
            grp["rank_by_mktval"] = grp[col_mktval].rank(
                ascending=False, method="min").astype(int)
            if grp.empty:
                result[key] = (pd.DataFrame(), pd.DataFrame())
                continue



            base_cols = [
                "stock_owner",
                col_log,  col_volume_open_interest,  "rank_by_volume",
                col_mktval, "rank_by_mktval",
                "appear_days", "total_trade_days",col_open_interest
            ]
            # 版本1：按 log_turnover 降序排列，取 top_n
            by_volume = (
                grp.sort_values(col_volume_open_interest, ascending=False)
                   .head(top_n)
                   .reset_index(drop=True)
            )[base_cols].copy()
            by_volume.index = by_volume.index + 1
            by_volume.index.name = "rank"

            # 版本2：按 turnover_mktval 降序排列，取 top_n
            by_mktval = (
                grp.sort_values(col_mktval, ascending=False)
                   .head(top_n)
                   .reset_index(drop=True)
            )[base_cols].copy()
            by_mktval.index = by_mktval.index + 1
            by_mktval.index.name = "rank"

            result[key] = (by_volume, by_mktval)
            logger.info(f"  [{key}] volume排名:{len(by_volume)}行 mktval排名:{len(by_mktval)}行")

    return result

# def compute_moneyness_type_rank(
#     df: pd.DataFrame,
#     cfg: "Config",
#     total_trade_days: int,
# ) -> dict:
#     """
#     对合约级 df（STEP 2 之后）按 (moneyness_bucket, option_type) 分成 6 组，
#     每组内对每个 stock_owner 跨日取均值并排序。
#
#     Parameters
#     ----------
#     df : pd.DataFrame
#         compute_all_contract_factors 之后的合约级 DataFrame，
#         必须含列：stock_owner, trade_date, moneyness_bucket, option_type,
#                   cf_log_turnover, cf_turnover_over_market_val
#     cfg : Config
#     total_trade_days : int
#         start_date ~ end_date 区间内 filter_data 目录中
#         anomaly_results*.csv 的文件数量（即交易日总数 m）
#
#     Returns
#     -------
#     dict，key 为 "ATM_CALL" / "ATM_PUT" / "DEEP_ITM_CALL" /
#               "DEEP_ITM_PUT" / "DEEP_OTM_CALL" / "DEEP_OTM_PUT"，
#     value 为排名 DataFrame。
#     """
#     TARGET_BUCKETS = {"ATM", "DEEP_ITM", "DEEP_OTM"}
#     REQUIRED_COLS  = {
#         "stock_owner", "trade_date", "moneyness_bucket", "option_type",
#         "cf_log_turnover", "cf_turnover_over_market_val",
#     }
#
#     missing = REQUIRED_COLS - set(df.columns)
#     if missing:
#         logger.warning(f"compute_moneyness_type_rank: 缺少列 {missing}，跳过")
#         return {}
#
#     # ── Step 1: 每股每天聚合（同股票多合约直接求和）─────────────────────────
#     sub = df[df["moneyness_bucket"].isin(TARGET_BUCKETS)].copy()
#     if sub.empty:
#         logger.warning("compute_moneyness_type_rank: 过滤后无数据")
#         return {}
#
#     daily = (
#         sub.groupby(
#             ["stock_owner", "trade_date", "moneyness_bucket", "option_type"],
#             observed=True,
#         )
#         .agg(
#             day_log_turnover    = ("cf_log_turnover",             "sum"),
#             day_turnover_mktval = ("cf_turnover_over_market_val", "sum"),
#         )
#         .reset_index()
#     )
#
#     # ── Step 2: 跨日均值 + 出现天数 ─────────────────────────────────────────
#     ranked = (
#         daily.groupby(
#             ["stock_owner", "moneyness_bucket", "option_type"],
#             observed=True,
#         )
#         .agg(
#             avg_log_turnover    = ("day_log_turnover",    "mean"),
#             avg_turnover_mktval = ("day_turnover_mktval", "mean"),
#             appear_days         = ("trade_date",          "nunique"),
#         )
#         .reset_index()
#     )
#     ranked["total_trade_days"] = total_trade_days
#
#     # ── Step 3: 拆成 6 组并排序 ──────────────────────────────────────────────
#     result = {}
#     top_n  = cfg.top_n_report
#
#     for bucket in ["ATM", "DEEP_ITM", "DEEP_OTM"]:
#         for otype in ["CALL", "PUT"]:
#             key  = f"{bucket}_{otype}"
#             mask = (
#                 (ranked["moneyness_bucket"] == bucket) &
#                 (ranked["option_type"]      == otype)
#             )
#             grp = ranked[mask].copy()
#             if grp.empty:
#                 result[key] = pd.DataFrame()
#                 continue
#
#             grp = grp.drop(columns=["moneyness_bucket", "option_type"])
#
#             # 双排名列（全量计算，head 之前）
#             grp["rank_by_log"]    = grp["avg_log_turnover"].rank(
#                 ascending=False, method="min").astype(int)
#             grp["rank_by_mktval"] = grp["avg_turnover_mktval"].rank(
#                 ascending=False, method="min").astype(int)
#
#             # 主排序：按 avg_turnover_mktval 降序，取 top_n
#             grp = (
#                 grp.sort_values("avg_turnover_mktval", ascending=False)
#                    .head(top_n)
#                    .reset_index(drop=True)
#             )
#             grp.index      = grp.index + 1
#             grp.index.name = "rank"
#
#             # 列顺序
#             grp = grp[[
#                 "stock_owner",
#                 "avg_log_turnover",    "rank_by_log",
#                 "avg_turnover_mktval", "rank_by_mktval",
#                 "appear_days",         "total_trade_days",
#             ]]
#             result[key] = grp
#             logger.info(f"  [{key}] {len(grp)} 行")
#
#     return result