# =============================================================================
# option_chain_fetcher.py —— 期权链获取（成交量过滤 + 批次优化）
# =============================================================================
"""
流程：
  1. get_option_expiration_date  → 拿到所有到期日
  2. 过滤掉超过 MAX_EXPIRY_DAYS 的到期日，并按 ALLOWED_CYCLES 白名单过滤
  3. 将到期日按 28 天分批，每批一次 get_option_chain 调用
     附带 OptionDataFilter(vol_min=200)
  4. 返回符合条件的期权代码列表
"""
import logging
import time
from datetime import datetime, timedelta

import pandas as pd
from futu import RET_OK, OptionDataFilter, OptionType, ExpirationCycle

from scan_package import config
from scan_package.connection import ensure_connection
from scan_package.rate_limiter import option_chain_limiter, api_retry

logger = logging.getLogger(__name__)


# 只保留这些交割周期，其余（日度散碎合约）跳过
# 可按需调整，例如去掉 QUARTERLY 或加入 END_OF_MONTH
ALLOWED_CYCLES = {
    ExpirationCycle.WEEK,
    ExpirationCycle.MONTH,
    ExpirationCycle.ENDOFMONTH,
    ExpirationCycle.QUARTERLY,
}


# ── 内部：带限频 + 重试的单次期权链调用 ───────────────────────────────────────
def _fetch_chain_once(code: str, start: str, end: str) -> list[str]:
    """
    调用一次 get_option_chain，返回 [start, end] 范围内符合条件的期权代码列表。
    已内置限频等待 + 重试。
    """
    ctx = ensure_connection()

    data_filter = OptionDataFilter()
    data_filter.vol_min = 200  # 服务端过滤成交量 < 200 的合约

    option_chain_limiter.acquire()

    @api_retry
    def _call():
        return ctx.get_option_chain(
            code=code,
            start=start,
            end=end,
            data_filter=data_filter,
        )

    try:
        ret, df = _call()
    except RuntimeError as e:
        logger.error(f"[{code}] get_option_chain 失败: {e}")
        return []

    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return []

    return df['code'].tolist()


# ── 内部：将到期日列表按 batch_days 天分批 ────────────────────────────────────
def _batch_expiry_dates(expiry_dates: list[str], batch_days: int = 29) -> list[tuple[str, str]]:
    """
    将排序后的到期日按不超过 batch_days 天的窗口分组，
    返回 [(start_date, end_date), ...] 列表。
    保守用 28 天，留出余量避免触碰 API 30 天上限。
    """
    if not expiry_dates:
        return []

    dates = sorted(expiry_dates)
    batches: list[tuple[str, str]] = []
    batch_start = dates[0]
    anchor = datetime.strptime(dates[0], '%Y-%m-%d')
    prev = dates[0]

    for d in dates[1:]:
        if (datetime.strptime(d, '%Y-%m-%d') - anchor).days > batch_days:
            batches.append((batch_start, prev))
            batch_start = d
            anchor = datetime.strptime(d, '%Y-%m-%d')
        prev = d

    batches.append((batch_start, prev))
    return batches


# ── 获取到期日列表（过滤超期 + 交割周期白名单） ───────────────────────────────
def get_valid_expiry_dates(code: str) -> list[str]:
    """
    返回 [today, today + MAX_EXPIRY_DAYS] 范围内、且属于 ALLOWED_CYCLES 的到期日列表。
    """
    ctx = ensure_connection()

    @api_retry
    def _call():
        return ctx.get_option_expiration_date(code=code)

    try:
        ret, df = _call()
    except RuntimeError as e:
        logger.error(f"[{code}] get_option_expiration_date 失败: {e}")
        return []

    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return []

    cutoff = datetime.today() + timedelta(days=config.MAX_EXPIRY_DAYS)
    today = datetime.today().date()

    valid = []
    for _, row in df.iterrows():
        distance = row.get('option_expiry_date_distance', None)
        date_str = row['strike_time']
        try:
            exp_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            exp_date = datetime.strptime(date_str[:10], '%Y-%m-%d').date()

        # 跳过已过期（distance < 0 或日期已过）
        if distance is not None and distance < 0:
            continue
        if exp_date < today:
            continue
        if exp_date > cutoff.date():
            continue

        # 跳过不在白名单内的交割周期（如日度合约）
        cycle = row.get('expiration_cycle')
        if cycle not in ALLOWED_CYCLES:
            logger.debug(f"  [{code}] 跳过 {date_str}，交割周期={cycle}")
            continue

        valid.append(date_str)

    logger.debug(f"[{code}] 有效到期日 {len(valid)} 个（截止 {cutoff.date()}）")
    return valid


# ── 主函数：获取单个标的的所有候选期权代码 ────────────────────────────────────
def get_candidate_options(code: str) -> list[str]:
    """
    对指定标的：
    - 拉取有效到期日（已按 ALLOWED_CYCLES + MAX_EXPIRY_DAYS 过滤）
    - 将到期日按 28 天分批，每批一次 get_option_chain 调用
    - 返回去重后的期权代码列表
    """
    expiry_dates = get_valid_expiry_dates(code)
    if not expiry_dates:
        logger.info(f"[{code}] 无有效到期日，跳过")
        return []

    batches = _batch_expiry_dates(expiry_dates, batch_days=28)
    logger.info(f"[{code}] {len(expiry_dates)} 个到期日 → {len(batches)} 批次调用")

    all_codes: set[str] = set()

    for start, end in batches:
        codes = _fetch_chain_once(code, start, end)
        if codes:
            logger.debug(f"  [{code}] {start}~{end}: {len(codes)} 个合约")
            all_codes.update(codes)
        time.sleep(0.3)

    result = list(all_codes)

    if len(result) > config.SINGLE_STOCK_MAX_OPTIONS:
        logger.warning(
            f"[{code}] ⚠️ 过滤后合约数 {len(result)} 超过阈值 "
            f"{config.SINGLE_STOCK_MAX_OPTIONS}，后续快照将自动分批处理"
        )

    logger.info(f"[{code}] 候选合约共 {len(result)} 个")
    return result


# ── 测试入口 ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    test_symbol = 'US.META'
    print(f"\n===== 测试 {test_symbol} 期权链获取 =====")

    dates = get_valid_expiry_dates(test_symbol)
    print(f"有效到期日（前5个）: {dates[:5]}")

    candidates = get_candidate_options(test_symbol)
    print(f"候选合约数: {len(candidates)}")
    if candidates:
        print(f"前5个合约: {candidates[:5]}")

    from connection import close_connection
    close_connection()