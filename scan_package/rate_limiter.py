# =============================================================================
# rate_limiter.py —— 限频控制 + 指数退避重试装饰器
# =============================================================================
import time
import logging
import functools
from collections import deque
from typing import Callable

from futu import RET_OK

from scan_package import config

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    滑动窗口限频器。
    默认：30 秒内最多 60 次（快照接口富途限制）。
    """

    def __init__(self, max_calls: int = None, window_sec: float = None):
        self.max_calls = max_calls or config.SNAPSHOT_RATE_LIMIT
        self.window_sec = window_sec or config.SNAPSHOT_RATE_WINDOW
        self._timestamps: deque = deque()

    def acquire(self):
        """阻塞直到可以发起下一次请求"""
        now = time.monotonic()
        # 移除窗口外的旧时间戳
        while self._timestamps and now - self._timestamps[0] >= self.window_sec:
            self._timestamps.popleft()

        if len(self._timestamps) >= self.max_calls:
            wait = self.window_sec - (now - self._timestamps[0]) + 0.05
            logger.debug(f"限频等待 {wait:.2f}s …")
            time.sleep(wait)
            # 递归直到通过
            self.acquire()
        else:
            self._timestamps.append(time.monotonic())

    def reset(self):
        """触发限频错误后强制重置"""
        self._timestamps.clear()


# 全局限频器实例（供快照调用使用）
snapshot_limiter = RateLimiter()

# 期权链调用频率更宽松，单独一个慢一点的限频器
# 富途期权链接口限制约 10次/30秒（文档未明确，保守估计）
option_chain_limiter = RateLimiter(max_calls=10, window_sec=30)


def api_retry(func: Callable = None,
              max_retries: int = None,
              base_delay: float = None,
              rate_limit_keywords: tuple = ('rate limit', 'frequency', '限频', 'RET_ERROR_API_TIMEOUT')):
    """
    装饰器：对富途 API 调用自动进行指数退避重试。

    - 遇到限频关键字：等待 RATE_LIMIT_WAIT 秒后重试，并重置限频器
    - 遇到其他错误：指数退避重试
    - 超过 max_retries：抛出最后一个异常

    用法：
        @api_retry
        def my_call(): ...

        或直接调用：api_retry(my_func)()
    """
    _max = max_retries or config.MAX_RETRIES
    _delay = base_delay or config.RETRY_BASE_DELAY

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(1, _max + 1):
                try:
                    ret, data = fn(*args, **kwargs)
                    if ret == RET_OK:
                        return ret, data

                    err_str = str(data).lower()
                    # 判断是否限频
                    if any(kw.lower() in err_str for kw in rate_limit_keywords):
                        logger.warning(
                            f"[{fn.__name__}] 触发限频，等待 {config.RATE_LIMIT_WAIT}s 后重试…"
                        )
                        snapshot_limiter.reset()
                        option_chain_limiter.reset()
                        time.sleep(config.RATE_LIMIT_WAIT)
                    else:
                        wait = _delay * (2 ** (attempt - 1))
                        logger.warning(
                            f"[{fn.__name__}] 第 {attempt}/{_max} 次失败: {data}，"
                            f"{wait:.1f}s 后重试…"
                        )
                        time.sleep(wait)
                    last_err = data

                except Exception as e:
                    wait = _delay * (2 ** (attempt - 1))
                    logger.error(
                        f"[{fn.__name__}] 第 {attempt}/{_max} 次异常: {e}，"
                        f"{wait:.1f}s 后重试…"
                    )
                    last_err = e
                    time.sleep(wait)

            raise RuntimeError(f"{fn.__name__} 超过最大重试次数，最后错误: {last_err}")

        return wrapper

    # 支持 @api_retry 和 @api_retry() 两种写法
    if func is not None:
        return decorator(func)
    return decorator


# ── 测试入口 ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    print("测试限频器（模拟 65 次快速请求，窗口 30s/60次）…")
    limiter = RateLimiter(max_calls=5, window_sec=3)
    for i in range(7):
        limiter.acquire()
        print(f"  请求 #{i + 1} 通过，time={time.strftime('%H:%M:%S')}")
    print("限频器测试完成 ✅")
