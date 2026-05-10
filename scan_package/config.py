# =============================================================================
# config.py —— 全局配置（所有参数集中在此修改）
# =============================================================================
import pandas as pd
# ── OpenD 连接 ────────────────────────────────────────────────────────────────
OPEND_HOST = '127.0.0.1'
OPEND_PORT = 11111
SYMBOLS = pd.read_csv(".\\layer1_stock_pool_latest.csv")
SYMBOLS = [f'US.{ticker}' for ticker in SYMBOLS['ticker']]
# ── 扫描标的列表 ──────────────────────────────────────────────────────────────
# 格式：富途代码，如 'US.AAPL'、'US.SPY'
# 可替换为从文件读取：SYMBOLS = [f"US.{s}" for s in open('symbols.txt').read().split()]
# SYMBOLS = [
#     'US.AAPL', 'US.MSFT', 'US.NVDA', 'US.TSLA', 'US.AMZN',
#     'US.GOOGL', 'US.META', 'US.SPY',  'US.QQQ',  'US.IWM',
#     # ↑ 测试阶段用少量标的，正式运行替换为完整 500 个
# ]
# ── 期权链过滤参数 ────────────────────────────────────────────────────────────
MAX_EXPIRY_DAYS = 210          # 只扫描 N 天内到期的合约（修改这一个数字即可）
MIN_VOLUME     = 1            # 服务端成交量过滤起点（>=1 过滤掉 0 成交僵尸合约）

# Delta 区间定义（三段不连续，每段独立调用一次期权链）
# 深实值 CALL:  delta ∈ [0.80, 1.00]
# 深实值 PUT:   delta ∈ [-1.00, -0.80]  → 富途用负数表示 put delta
# 深虚值 CALL:  delta ∈ [0.00, 0.20]
# 深虚值 PUT:   delta ∈ [-0.20, 0.00]
# 注意：富途 get_option_chain 的 OptionDataFilter.delta 字段对 call/put 统一用实际值
DELTA_FILTERS = [
    # (delta_min, delta_max, label)
    (0.9,  1.00,  'deep_itm_call'),   # 深实值 CALL
    (-1.00, -0.9, 'deep_itm_put'),    # 深实值 PUT
    (0.00,  0.1,  'deep_otm_call'),   # 深虚值 CALL
    (-0.1, 0.00,  'deep_otm_put'),    # 深虚值 PUT
]
# 实际上 delta_min=-1.00 ~ -0.80 和 0.00 ~ 0.20 可各合并一次调用：
# 合并方案：(-1.00, -0.80) 一次，(0.80, 1.00) 一次，(0.00, 0.20) / (-0.20, 0.00) 两次
# 为清晰可读，以上保留 4 段（也可合并为 3 次，见 README）

# ── 异常检测参数 ──────────────────────────────────────────────────────────────
# 使用快照当日成交额与 N 日均值比较（当前版本基于单日快照 turnover 绝对值判断）
TURNOVER_ABS_THRESHOLD  = 300_000     # 单合约当日成交额 >= 500 万美元视为候选
TURNOVER_RATIO_THRESHOLD = 5.0        # 成交额倍数阈值（相对于历史均值，预留）
ANOMALY_LOOKBACK_DAYS    = 20         # 历史对比窗口天数（预留，当前版本暂不使用历史K线）

# ── 批量快照参数 ──────────────────────────────────────────────────────────────
SNAPSHOT_BATCH_SIZE = 100             # 默认每批快照数量
SNAPSHOT_BATCH_MIN  = 25              # 失败时最小退避批次
SNAPSHOT_RATE_LIMIT = 60             # 每 30 秒最多 60 次快照（富途限制），取整留余量
SNAPSHOT_RATE_WINDOW = 30            # 限频窗口（秒）

# ── 大标的阈值 ────────────────────────────────────────────────────────────────
SINGLE_STOCK_MAX_OPTIONS = 200       # 单标的过滤后合约数超过此值时打印警告

# ── 重试参数 ──────────────────────────────────────────────────────────────────
MAX_RETRIES        = 5               # API 调用最多重试次数
RETRY_BASE_DELAY   = 5.0             # 指数退避基础延迟（秒）
RATE_LIMIT_WAIT    = 65              # 触发限频后等待秒数（富途限频窗口 + 缓冲）


# import datetime
# # ── 断点续跑缓存 ──────────────────────────────────────────────────────────────
# dt_used_in_save = datetime.date.today().strftime("%Y-%m-%d")
# CHECKPOINT_FILE    = f'.\\log\\checkpoint{dt_used_in_save}.json'
# RESULTS_FILE       = f'.\\filter_data\\anomaly_results{dt_used_in_save}.csv'
# ERROR_LOG_FILE     = f'.\\log\\process_errors{dt_used_in_save}.log'
