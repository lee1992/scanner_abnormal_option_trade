# =============================================================================
# option_chain_fetcher.py —— 期权链获取（成交量过滤）
# =============================================================================
"""
流程：
  1. get_option_expiration_date  → 拿到所有到期日
  2. 过滤掉超过 MAX_EXPIRY_DAYS 的到期日
  3. 对每个到期日调用 get_option_chain，附带 OptionDataFilter(vol_min=200)
  4. 返回符合条件的期权代码列表
"""
import logging
import time
from datetime import datetime, timedelta

import pandas as pd
from futu import RET_OK, OptionDataFilter, OptionType, ExpirationCycle

import config
from connection import ensure_connection
from rate_limiter import option_chain_limiter, api_retry

logger = logging.getLogger(__name__)


# ── 内部：带限频 + 重试的单次期权链调用 ───────────────────────────────────────
def _fetch_chain_once(code: str, start: str, end: str) -> list[str]:
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


# 只保留这些交割周期，其余（日度散碎合约）跳过
# 可按需调整，例如去掉 QUARTERLY 或加入 END_OF_MONTH
ALLOWED_CYCLES = {
    ExpirationCycle.WEEK,
    ExpirationCycle.MONTH,
    ExpirationCycle.ENDOFMONTH,
    ExpirationCycle.QUARTERLY,
}


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
    expiry_dates = get_valid_expiry_dates(code)
    if not expiry_dates:
        logger.info(f"[{code}] 无有效到期日，跳过")
        return []

    all_codes: set[str] = set()

    for expiry in expiry_dates:
        codes = _fetch_chain_once(code, expiry, expiry)
        if codes:
            logger.debug(f"  [{code}] {expiry}: {len(codes)} 个合约")
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
    test_symbol = 'US.AAPL'
    print(f"\n===== 测试 {test_symbol} 期权链获取 =====")

    dates = get_valid_expiry_dates(test_symbol)
    print(f"有效到期日（前5个）: {dates[:5]}")

    candidates = get_candidate_options(test_symbol)
    print(f"候选合约数: {len(candidates)}")
    if candidates:
        print(f"前5个合约: {candidates[:5]}")

    from connection import close_connection
    close_connection()
