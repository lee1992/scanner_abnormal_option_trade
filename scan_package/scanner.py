# =============================================================================
# scanner.py —— 主扫描器（编排各模块，支持逐步调试 / 全量运行）
# =============================================================================
"""
PyCharm Console 逐步运行示例：

  # Step 1: 初始化
  from scanner import Scanner
  s = Scanner()

  # Step 2: 测试单个标的（看期权链获取是否正常）
  codes = s.step_fetch_option_codes('US.AAPL')

  # Step 3: 测试单个标的快照
  snapshots = s.step_fetch_snapshots(codes)

  # Step 4: 测试异常检测
  anomalies = s.step_detect('US.AAPL', snapshots)

  # Step 5: 全量扫描
  s.run_all()
"""
import logging
import sys
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from scan_package import config
from scan_package.connection import ensure_connection, close_connection
from scan_package.option_chain_fetcher import get_candidate_options
from scan_package.snapshot_fetcher import fetch_snapshots, extract_option_snapshot_fields
from scan_package.anomaly_detector import detect_anomalies, format_anomaly_report, save_anomalies_to_csv
from scan_package.checkpoint import get_pending_symbols, mark_done, clear_checkpoint


def init_config():
    """在真正开始运行时，用当前日期覆盖配置"""
    import datetime
    today = datetime.date.today().strftime("%Y-%m-%d")
    config.CHECKPOINT_FILE = f'.\\log\\checkpoint{today}.json'
    config.RESULTS_FILE    = f'.\\filter_data\\anomaly_results{today}.csv'
    config.ERROR_LOG_FILE  = f'.\\log\\process_errors{today}.log'

# ── 日志配置 ──────────────────────────────────────────────────────────────────
def setup_logging(level=logging.INFO):
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.ERROR_LOG_FILE, encoding='utf-8'),
    ]
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        handlers=handlers,
        force=True,
    )



# =============================================================================
class Scanner:
    """
    期权异常交易扫描器主类。
    支持两种使用方式：
    1. 逐步调试（PyCharm Console）：调用 step_* 方法
    2. 全量运行：调用 run_all()
    """

    def __init__(self, symbols: list[str] = None, fresh_start: bool = False):
        """
        参数:
            symbols:     自定义标的列表，默认使用 config.SYMBOLS
            fresh_start: True 则清除断点缓存从头扫描
        """
        self.symbols = symbols or config.SYMBOLS
        if fresh_start:
            clear_checkpoint()
            logger.info("已清除断点，全新开始")

        self.all_anomalies: list[pd.DataFrame] = []
        self._start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        logger.info(f"Scanner 初始化完成，共 {len(self.symbols)} 个标的")
        logger.info(f"配置: MAX_EXPIRY_DAYS={config.MAX_EXPIRY_DAYS}, "
                    f"MIN_VOLUME={config.MIN_VOLUME}, "
                    f"TURNOVER_ABS_THRESHOLD=${config.TURNOVER_ABS_THRESHOLD:,}")

    # ── 逐步调试接口 ──────────────────────────────────────────────────────────

    def step_fetch_option_codes(self, symbol: str) -> list[str]:
        """
        Step 1: 获取单个标的的候选期权代码列表。

        示例：

            codes = scanner.step_fetch_option_codes('US.AAPL')
            print(f"获取到 {len(codes)} 个候选合约")
            print(codes[:5])
        """
        ctx = ensure_connection()
        for i in range(2):
            print( f'fetch {symbol} last price and market val ' )
            temp = ctx.get_market_snapshot(symbol)
            if temp[0]==0:
                market_value,last_price = float(temp[1]['total_market_val'].values[0]),float(temp[1]['last_price'].values[0])
                print( f'{symbol} market val = {market_value}, last price={last_price}')
                break
            else:
                print('get_market_snapshot rate limit, sleep 30sec')
                time.sleep(30)
        if market_value!=0:
            logger.info(f"[STEP1] 获取 {symbol} 候选期权链…")
            #ensure_connection()
            codes = get_candidate_options(symbol)
            logger.info(f"[STEP1] {symbol}: 候选合约 {len(codes)} 个")
        else:
            logger.info(f"{symbol} 市值过大，pass")
            codes=[]
        return codes,last_price,market_value

    def step_fetch_snapshots(self, option_codes: list[str]) -> pd.DataFrame:
        """
        Step 2: 批量获取期权快照并提取关键字段。

        示例：
            snapshots = scanner.step_fetch_snapshots(codes)
            print(snapshots[['code','volume','turnover','option_delta']].head(10))
        """
        logger.info(f"[STEP2] 获取 {len(option_codes)} 个合约快照…")
        if not option_codes:
            logger.warning("[STEP2] 合约列表为空，跳过")
            return pd.DataFrame()

        raw_df = fetch_snapshots(option_codes)
        slim_df = extract_option_snapshot_fields(raw_df)
        logger.info(f"[STEP2] 成功获取 {len(slim_df)} 条快照")
        return slim_df

    def step_detect(self, symbol: str, snapshot_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 3: 对快照数据执行异常检测并打印报告。

        示例：
            anomalies = scanner.step_detect('US.AAPL', snapshots)
            print(anomalies[['code','turnover','option_delta','anomaly_reason']])
        """
        logger.info(f"[STEP3] 对 {symbol} 执行异常检测…")
        anomalies = detect_anomalies(snapshot_df)
        print(format_anomaly_report(anomalies, symbol))

        if not anomalies.empty:
            save_anomalies_to_csv(anomalies, symbol)

        return anomalies

    def step_run_single(self, symbol: str) -> pd.DataFrame:
        """
        完整处理单个标的（Step1 + Step2 + Step3 合并）。
        适合先用少量标的调试整个流程。

        示例：
            result = scanner.step_run_single('US.NVDA')
        """
        logger.info(f"{'─' * 50}")
        logger.info(f"处理标的: {symbol}")

        try:
            #获取期权链 和  当天标的价格的收盘价，这个要测试一下
            codes,prev_close_price,market_val = self.step_fetch_option_codes(symbol)
            snapshots = self.step_fetch_snapshots(codes)
            snapshots['last_price'] =  prev_close_price
            denominator = snapshots['last_price'].replace(0, np.nan)
            conditions = [snapshots['option_type'] == 'CALL',snapshots['option_type'] == 'PUT' ]
            choices = [ # 看涨：(标的价-执行价)/标的价 *100 → 实值为正，虚值为负
                (snapshots['last_price'] - snapshots['option_strike_price']) / denominator,
                # 看跌：(执行价-标的价)/标的价 *100 → 实值为正，虚值为负
                (snapshots['option_strike_price'] - snapshots['last_price']) / denominator ]
            snapshots['atm_degree_pct'] = np.select(conditions, choices, default=np.nan)
            snapshots['market_val'] = market_val
            anomalies = self.step_detect(symbol, snapshots)
            return anomalies
        except Exception as e:
            logger.error(f"[{symbol}] 处理异常: {e}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    # ── 全量扫描 ──────────────────────────────────────────────────────────────
    def run_all(self, symbols: list[str] = None) -> pd.DataFrame:
        """
        全量扫描所有标的，支持断点续跑。

        示例：
            scanner.run_all()
            # 或指定部分标的
            scanner.run_all(['US.SPY', 'US.QQQ'])
        """
        target_symbols = symbols or self.symbols
        pending, already_done = get_pending_symbols(target_symbols)
        completed_list = list(already_done)

        logger.info(f"开始全量扫描: 待处理 {len(pending)} 个标的")
        logger.info(f"已跳过（断点续跑）: {len(already_done)} 个")

        total_anomalies: list[pd.DataFrame] = []
        start_ts = datetime.now()

        for idx, symbol in enumerate(pending, 1):
            eta = self._estimate_eta(idx - 1, len(pending), start_ts)
            logger.info(
                f"[{idx}/{len(pending)}] {symbol}  "
                f"完成率 {idx / (len(pending)):.0%}  预计剩余 {eta}"
            )

            try:
                anomalies = self.step_run_single(symbol)
                if not anomalies.empty:
                    total_anomalies.append(anomalies)

                mark_done(symbol, completed_list)

            except Exception as e:
                logger.error(f"[{symbol}] 未捕获异常: {e}")
                logger.error(traceback.format_exc())
                # 记录错误但继续下一个标的

        # 汇总报告
        final_df = self._build_summary(total_anomalies)
        self._print_final_summary(final_df, len(target_symbols), len(already_done))

        return final_df

    # ── 内部工具 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _estimate_eta(done: int, total: int, start_ts: datetime) -> str:
        if done == 0:
            return "计算中…"
        elapsed = (datetime.now() - start_ts).total_seconds()
        per_item = elapsed / done
        remaining = per_item * (total - done)
        m, s = divmod(int(remaining), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h{m:02d}m"
        return f"{m}m{s:02d}s"

    @staticmethod
    def _build_summary(anomaly_list: list[pd.DataFrame]) -> pd.DataFrame:
        if not anomaly_list:
            return pd.DataFrame()
        return pd.concat(anomaly_list, ignore_index=True)

    @staticmethod
    def _print_final_summary(df: pd.DataFrame, total: int, skipped: int):
        print(f"\n{'═' * 60}")
        print(f"  扫描完成汇总")
        print(f"{'═' * 60}")
        print(f"  标的总数:       {total}")
        print(f"  断点跳过:       {skipped}")
        print(f"  实际处理:       {total - skipped}")
        print(f"  异常合约数:     {len(df)}")
        if not df.empty and 'stock_owner' in df.columns:
            print(f"  涉及标的数:     {df['stock_owner'].nunique()}")
        if not df.empty and 'turnover' in df.columns:
            top5 = df.nlargest(5, 'turnover')[['code', 'turnover', 'option_delta']]
            print(f"\n  成交额 TOP 5:")
            for _, row in top5.iterrows():
                print(f"    {row['code']:40s}  ${row['turnover']:>15,.0f}  δ={row['option_delta']:.3f}")
        print(f"{'═' * 60}")
        print(f"  结果文件: {config.RESULTS_FILE}")
        print(f"  错误日志: {config.ERROR_LOG_FILE}")
        print(f"{'═' * 60}\n")


# ── 程序入口 ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import time
    import pytz  # pip install pytz

    from datetime import datetime
    def wait_until_market_close():
        """
        如果当前美东时间在 08:00–16:15 之间，则等待到 16:16 再运行。
        否则直接运行。
        """
        et_tz = pytz.timezone('America/New_York')
        now_et = datetime.now(et_tz)

        market_open = now_et.replace(hour=8, minute=0, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=15, second=0, microsecond=0)
        run_at = now_et.replace(hour=16, minute=16, second=0, microsecond=0)

        if market_open <= now_et <= market_close:
            wait_seconds = (run_at - now_et).total_seconds()
            print(f"当前美东时间: {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print(f"处于交易时段，将在 16:16 ET 开始运行（等待 {wait_seconds / 3600:.1f} 小时）...")
            time.sleep(wait_seconds)
        else:
            print(f"当前美东时间: {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print("不在交易时段，直接开始运行。")
    wait_until_market_close()
    init_config()
    setup_logging()
    logger = logging.getLogger('scanner')

    from datetime import datetime
    scanner = Scanner()
    try:
        scanner.run_all()
    finally:
        close_connection()


    #from datetime import datetime, timedelta
    # # --- 1. 计算需要 Sleep 的时间 ---
    # now = datetime.now()
    # # 设定目标时间：今天的 04:15:00
    # target_time = datetime(now.year, now.month, now.day, 4, 15, 0)
    #
    # # 如果现在已经过了 04:15，就把目标设为明天的 04:15
    # if now > target_time:
    #     target_time += timedelta(days=1)
    #
    # # 计算时间差（秒）
    # wait_seconds = (target_time - now).total_seconds()
    #
    # print(f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    # print(f"将在 {target_time.strftime('%Y-%m-%d %H:%M:%S')} 运行任务 (等待 {wait_seconds / 3600:.1f} 小时)...")
    #
    # # --- 2. 开始等待 ---
    # time.sleep(wait_seconds)
    """
    直接运行：python scanner.py
    或在 PyCharm Console 中逐步执行：

        from scanner import Scanner
        s = Scanner()

        # 单标的测试
        codes,prev = s.step_fetch_option_codes('US.BILI')
        snaps = s.step_fetch_snapshots(codes)
        anom  = s.step_detect('US.AAPL', snaps)

        # 全量
        s.run_all()
    """
