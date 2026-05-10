"""
合约级单因子计算。

设计规范：
  - 每个因子函数签名: func(df: pd.DataFrame, cfg: Config) -> pd.Series
  - CONTRACT_FACTORS 字典：注册所有数值因子
  - CATEGORY_FACTORS 字典：注册分类/分桶因子（返回 str Series）
  - 增删因子：只改注册字典 + config.score_weights，其他代码无需改动
"""
from __future__ import annotations
import logging
from typing import Callable

import numpy as np
import pandas as pd

from option_anomaly_system.config import Config, MoneynessBucket

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════════════════════════

def _assign_bucket(series: pd.Series, buckets: list[MoneynessBucket]) -> pd.Series:
    """将连续值映射到 MoneynessBucket.name"""
    result = pd.Series("UNKNOWN", index=series.index, dtype=str)
    for b in buckets:
        mask = (series >= b.lower) & (series < b.upper)
        result[mask] = b.name
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 合约级数值因子
# ═══════════════════════════════════════════════════════════════════════════════

def cf_log_turnover(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """对数成交额，消除量纲偏斜"""
    return np.log1p(df["turnover"].fillna(0))


def cf_churn(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """
    合约周转率 = volume / OI(T-1)
    高周转 + OI 不增加 = 典型"过手"信号
    """
    oi = df["option_open_interest"].fillna(0).clip(lower=cfg.churn_oi_floor)
    return df["volume"].fillna(0) / oi


def cf_notional_churn(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """
    名义周转率 = turnover / (OI × contract_size × stock_price)
    不同价位合约可比，更稳健
    """
    notional_oi = (
        df["option_open_interest"].fillna(0)
        * df["option_contract_size"].fillna(100)
        * df["last_price"].fillna(0)
    ).clip(lower=cfg.notional_oi_floor)
    return df["turnover"].fillna(0) / notional_oi


def cf_delta_flow(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Delta 加权成交额：turnover × |delta|，衡量方向性资金力度"""
    return df["turnover"].fillna(0) * df["option_delta"].abs().fillna(0)


def cf_vega_flow(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Vega 加权成交额：turnover × |vega|，衡量波动率押注力度"""
    return df["turnover"].fillna(0) * df["option_vega"].abs().fillna(0)


def cf_gamma_flow(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Gamma 加权成交额，衡量短线 Gamma 暴露"""
    return df["turnover"].fillna(0) * df["option_gamma"].abs().fillna(0)


def cf_iv_level(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """隐含波动率绝对水平"""
    return df["option_implied_volatility"].fillna(0)


def cf_volume_oi_ratio(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """volume / OI（与 cf_churn 相同，独立命名便于权重单独调节）"""
    return cf_churn(df, cfg)


def cf_turnover_per_contract(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """
    每手平均成交额 = turnover / volume
    大单推动时此值高（深实值尤为明显）
    """
    vol = df["volume"].replace(0, np.nan)
    return df["turnover"].fillna(0) / vol


def cf_intraday_range(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """
    期权日内振幅（高-低）/ 昨收
    缺失时返回 0，不影响其他因子
    """
    return df["intraday_range"].fillna(0)


def cf_theta_vega_ratio(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """
    |theta| / |vega|：时间衰减 vs 波动率敏感性之比
    比值极小 = 纯波动率押注；比值大 = 时间价值为主
    """
    vega = df["option_vega"].abs().replace(0, np.nan)
    return (df["option_theta"].abs().fillna(0) / vega).fillna(0)


def cf_dte_weight(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """
    到期日紧迫度权重 = 1 / (DTE + 1)
    DTE 越小，信号越紧迫
    """
    return 1.0 / (df["option_expiry_date_distance"].fillna(30) + 1)


def cf_market_val_log(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """
    对数市值（归一化标的规模）
    在标的间横向对比时作为规模控制因子
    """
    return np.log1p(df["market_val"].fillna(0))

def cf_turnover_over_market_val(df: pd.DataFrame, cfg: Config) -> pd.Series:
    temp = np.where(
        # 规则1：分子无效 → 0
        (df["turnover"].isna()) | (df["turnover"] == 0),
        0,
        # 分子有效，判断分母
        np.where(
            df["market_val"].isna(),  # 分母无值 → 极小值
            df["turnover"] / 1e18,
            np.where(
                df["market_val"] == 0,  # 分母为0 → 1
                1,
                # 分母有效：正常除法 + 最大值封顶2
                np.minimum(df["turnover"] / df["market_val"], 2)
            )
        )
    )
    return df["turnover"].fillna(1)/df["market_val"].fillna(1e18)
# ═══════════════════════════════════════════════════════════════════════════════
# 注册表
# 增删因子：只改这两个字典
# ═══════════════════════════════════════════════════════════════════════════════
def cf_volume_over_open_interest(df: pd.DataFrame, cfg: Config) -> pd.Series:
    # df["volume"].replace(0, 1).fillna(1)    /   df["option_open_interest"].replace(0, 1e18).fillna(1e18)  )# df["volume"].fillna(1)/df["option_open_interest"].fillna(1e18)
    temp = np.where(
        # 规则1：分子无效 → 0
        (df["volume"].isna()) | (df["volume"] == 0),
        0,
        # 分子有效，判断分母
        np.where(
            df["option_open_interest"].isna(),  # 分母无值 → 极小值
            df["volume"] / 1e18,
            np.where(
                df["option_open_interest"] == 0,  # 分母为0 → 1
                1,
                # 分母有效：正常除法 + 最大值封顶2
                np.minimum(df["volume"] / df["option_open_interest"], 2)
            )
        )
    )
    return   temp #df["volume"].replace(0, 1).fillna(1)    /   df["option_open_interest"].replace(0, 1e18).fillna(1e18)      #np.where(    (df["volume"] == 0) | (df["option_open_interest"] == 0),    0,    df["volume"] / df["option_open_interest"])


CONTRACT_FACTORS: dict[str, Callable[[pd.DataFrame, Config], pd.Series]] = {
    "cf_log_turnover":           cf_log_turnover,
    "cf_churn":                  cf_churn,
    "cf_notional_churn":         cf_notional_churn,
    "cf_delta_flow":             cf_delta_flow,
    "cf_vega_flow":              cf_vega_flow,
    "cf_gamma_flow":             cf_gamma_flow,
    "cf_iv_level":               cf_iv_level,
    "cf_volume_oi_ratio":        cf_volume_oi_ratio,
    "cf_turnover_per_contract":  cf_turnover_per_contract,
    "cf_intraday_range":         cf_intraday_range,
    "cf_theta_vega_ratio":       cf_theta_vega_ratio,
    "cf_dte_weight":             cf_dte_weight,
    "cf_market_val_log":         cf_market_val_log,
    "cf_turnover_over_market_val": cf_turnover_over_market_val,
    "cf_volume_over_open_interest": cf_volume_over_open_interest,
}

CATEGORY_FACTORS: dict[str, Callable[[pd.DataFrame, Config], pd.Series]] = {
    "moneyness_bucket": lambda df, cfg: _assign_bucket(
        df["directional_moneyness"], cfg.moneyness_buckets
    ),
    "dte_bucket": lambda df, cfg: pd.cut(
        df["option_expiry_date_distance"].fillna(0),
        bins=[0, 7, 30, 90, 180, 9999],
        labels=["0-7d", "8-30d", "31-90d", "91-180d", "180d+"],
        right=True,
        include_lowest=True,
    ).astype(str),
}


def compute_all_contract_factors(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    计算所有合约级因子，结果列附加到 df 上返回。
    任何单个因子报错不中断整体流程（仅记录 ERROR 日志）。
    """
    df = df.copy()

    for name, func in CONTRACT_FACTORS.items():
        try:
            df[name] = func(df, cfg)
        except Exception as e:
            logger.error(f"  因子 {name} 计算失败: {e}")
            df[name] = np.nan

    for name, func in CATEGORY_FACTORS.items():
        try:
            df[name] = func(df, cfg)
        except Exception as e:
            logger.error(f"  分类因子 {name} 计算失败: {e}")
            df[name] = "UNKNOWN"

    logger.info(
        f"合约级因子计算完成: {len(CONTRACT_FACTORS)} 个数值因子 "
        f"+ {len(CATEGORY_FACTORS)} 个分类因子"
    )
    return df
