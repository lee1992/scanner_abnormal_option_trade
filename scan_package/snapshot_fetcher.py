# =============================================================================
# snapshot_fetcher.py —— 批量快照获取（自适应批次 + 限频）
# =============================================================================
"""
富途 get_market_snapshot 限制：
  - 每次最多 400 个代码
  - 每 30 秒最多 60 次调用

本模块：
  - 默认 SNAPSHOT_BATCH_SIZE=100 个一批（远低于上限，留余量）
  - 批次失败自动缩小：100 → 50 → 25
  - 全程使用 snapshot_limiter 控制频率
"""
import logging

import pandas as pd
from futu import RET_OK

from scan_package import config
from scan_package.connection import ensure_connection
from scan_package.rate_limiter import snapshot_limiter, api_retry

logger = logging.getLogger(__name__)


def _fetch_snapshot_batch(codes: list[str]) -> pd.DataFrame:
    """
    对一批代码调用快照，带限频 + 重试。
    返回 DataFrame，失败返回空 DataFrame。
    """
    ctx = ensure_connection()
    snapshot_limiter.acquire()

    @api_retry
    def _call():
        return ctx.get_market_snapshot(codes)

    try:
        ret, df = _call()
        return df if not df.empty else pd.DataFrame()
    except RuntimeError as e:
        logger.error(f"快照批次失败: {e}")
        return pd.DataFrame()


def fetch_snapshots(option_codes: list[str],
                    batch_size: int = None) -> pd.DataFrame:
    """
    批量获取期权快照，自动分批 + 失败缩小批次重试。

    参数:
        option_codes: 期权代码列表
        batch_size:   每批数量（默认 SNAPSHOT_BATCH_SIZE）

    返回:
        包含所有成功获取合约快照的 DataFrame
    """
    if not option_codes:
        return pd.DataFrame()

    batch_size = batch_size or config.SNAPSHOT_BATCH_SIZE
    results: list[pd.DataFrame] = []
    i = 0
    total = len(option_codes)

    while i < total:
        chunk = option_codes[i: i + batch_size]
        logger.debug(f"快照批次 [{i + 1}~{i + len(chunk)}/{total}]，批次大小={len(chunk)}")

        df = _fetch_snapshot_batch(chunk)

        if df.empty and batch_size > config.SNAPSHOT_BATCH_MIN:
            # 缩小批次重试
            new_batch = max(batch_size // 2, config.SNAPSHOT_BATCH_MIN)
            logger.warning(
                f"批次失败，缩小批次 {batch_size} → {new_batch} 重试…"
            )
            # 递归处理这段
            sub_df = fetch_snapshots(chunk, batch_size=new_batch)
            if not sub_df.empty:
                results.append(sub_df)
        elif not df.empty:
            results.append(df)
        else:
            logger.warning(f"批次 [{i + 1}~{i + len(chunk)}] 获取失败，跳过")

        i += batch_size

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    logger.info(f"快照获取完成：{len(combined)}/{total} 个合约")
    return combined


def extract_option_snapshot_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    从快照 DataFrame 中提取期权相关关键字段。
    返回精简后的 DataFrame。
    """
    if df.empty:
        return df

    keep_cols = [
        'code', 'name', 'last_price', 'volume', 'turnover',
        'option_type', 'strike_time', 'option_strike_price',
        'option_open_interest', 'option_implied_volatility',
        'option_delta', 'option_gamma', 'option_theta', 'option_vega',
        'option_expiry_date_distance', 'option_contract_size',
        'stock_owner',    'open_price',
    'high_price',                  # 最高价
    'low_price',                   # 最低价
    'prev_close_price',            # 昨收价
    ]
    existing = [c for c in keep_cols if c in df.columns]
    return df[existing].copy()


# ── 测试入口 ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    # 用已知期权代码测试（替换为实际存在的代码）
    # 格式示例：US.AAPL  250117C00185000
    test_codes = [
        # 'US.AAPL250117C00150000',
        # 'US.AAPL250117P00250000',
    ]

    if not test_codes:
        print("⚠️  请在 test_codes 中填入实际期权代码后运行本模块测试")
        print("   可先运行 option_chain_fetcher.py 获取候选合约列表")
    else:
        df = fetch_snapshots(test_codes)
        df_slim = extract_option_snapshot_fields(df)
        print(df_slim.to_string())

    from connection import close_connection

    close_connection()
