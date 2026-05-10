"""
数据加载 & 预处理。
从文件名 anomaly_resultsYYYY-MM-DD.csv 提取 trade_date，不依赖 scan_timestamp。
"""
from __future__ import annotations
import re
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from option_anomaly_system.config import Config

logger = logging.getLogger(__name__)

# 实际 CSV 字段（以 2026-05-05 文件为准）
NUMERIC_COLS = [
    "last_price", "volume", "turnover",
    "option_strike_price", "option_open_interest",
    "option_implied_volatility",
    "option_delta", "option_gamma", "option_theta", "option_vega",
    "option_expiry_date_distance", "option_contract_size",
    "atm_degree_pct",
    "open_price", "high_price", "low_price", "prev_close_price",
    "market_val",
]


def _extract_date(filename: str) -> Optional[str]:
    """从文件名提取 YYYY-MM-DD"""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    return m.group(1) if m else None


def load_multi_day(
    cfg: Config,
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
) -> pd.DataFrame:
    """
    扫描 cfg.data_dir 下所有 anomaly_results*.csv，
    按文件名日期过滤后合并返回。
    """
    files = sorted(Path(cfg.data_dir).glob("anomaly_results*.csv"))
    if not files:
        raise FileNotFoundError(f"在 {cfg.data_dir} 下未找到 anomaly_results*.csv")

    frames = []
    for f in files:
        trade_date = _extract_date(f.name)
        if trade_date is None:
            logger.warning(f"无法提取日期，跳过: {f.name}")
            continue
        if start_date and trade_date < start_date:
            continue
        if end_date and trade_date > end_date:
            continue
        try:
            df = pd.read_csv(f, low_memory=False)
            df["trade_date"] = trade_date
            frames.append(df)
            logger.info(f"  ✓ {f.name}: {len(df)} 行, trade_date={trade_date}")
        except Exception as e:
            logger.error(f"  ✗ {f.name}: {e}")

    if not frames:
        raise ValueError(f"日期范围 [{start_date}, {end_date}] 内无可用数据")

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        f"合并完成: {len(combined)} 行, "
        f"{combined['trade_date'].nunique()} 天, "
        f"范围 {combined['trade_date'].min()} ~ {combined['trade_date'].max()}"
    )
    return combined


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    类型转换、缺失值处理、基础派生字段。
    """
    df = df.copy()

    # 数值列强制转换
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    # option_type 统一大写
    df["option_type"] = df["option_type"].str.upper().fillna("UNKNOWN")

    # 保留合法方向，过滤异常值
    df = df[df["option_type"].isin(["CALL", "PUT"])].copy()

    # contract_size 默认 100（美股标准）
    df["option_contract_size"] = df["option_contract_size"].fillna(100)

    # stock_owner 兜底（CSV 里已有）
    if "stock_owner" not in df.columns:
        df["stock_owner"] = df.get("scan_symbol", "UNKNOWN")

    # ── 方向化 moneyness ──────────────────────────────────────────────
    # atm_degree_pct 定义：(stock_price - strike) / stock_price
    #   Call: 正 = 标的 > 行权价 = ITM ✓
    #   Put:  正 = 标的 > 行权价 = OTM（需翻转，使正=ITM）
    df["directional_moneyness"] = df["atm_degree_pct"].copy()
    is_put = df["option_type"] == "PUT"
    df.loc[is_put, "directional_moneyness"] = df.loc[is_put, "atm_degree_pct"]

    # ── 日内振幅（期权）────────────────────────────────────────────────
    # 若 high/low/prev_close 存在则计算，否则 NaN
    prev = df["prev_close_price"].replace(0, np.nan)
    has_range = df["high_price"].notna() & df["low_price"].notna() & prev.notna()
    df["intraday_range"] = np.where(
        has_range,
        (df["high_price"] - df["low_price"]) / prev,
        np.nan,
    )

    return df
