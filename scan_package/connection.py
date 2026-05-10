# =============================================================================
# connection.py —— OpenD 连接管理（单例，自动重连）
# =============================================================================
import logging
from futu import OpenQuoteContext, RET_OK

from scan_package import config

logger = logging.getLogger(__name__)

_quote_ctx: OpenQuoteContext | None = None


def get_quote_ctx(force_reconnect: bool = False) -> OpenQuoteContext:
    """
    获取全局 QuoteContext 单例。
    - 首次调用自动连接
    - force_reconnect=True 或连接断开时自动重连
    """
    global _quote_ctx

    if force_reconnect and _quote_ctx is not None:
        try:
            _quote_ctx.close()
        except Exception:
            pass
        _quote_ctx = None

    if _quote_ctx is None:
        logger.info(f"连接 OpenD  {config.OPEND_HOST}:{config.OPEND_PORT} ...")
        _quote_ctx = OpenQuoteContext(host=config.OPEND_HOST, port=config.OPEND_PORT)
        logger.info("连接成功")

    return _quote_ctx


def check_connection() -> bool:
    """探测连接是否存活（发一次无害请求）"""
    try:
        ctx = get_quote_ctx()
        ret, _ = ctx.get_global_state()
        return ret == RET_OK
    except Exception as e:
        logger.warning(f"连接探测失败: {e}")
        return False


def ensure_connection() -> OpenQuoteContext:
    """确保连接存活，断开则重连，返回可用的 ctx"""
    if not check_connection():
        logger.warning("连接断开，正在重连…")
        return get_quote_ctx(force_reconnect=True)
    return get_quote_ctx()


def close_connection():
    """程序结束时调用，释放连接"""
    global _quote_ctx
    if _quote_ctx is not None:
        try:
            _quote_ctx.close()
            logger.info("连接已关闭")
        except Exception:
            pass
        _quote_ctx = None


# ── 测试入口（PyCharm Console 直接运行此文件） ────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    ctx = get_quote_ctx()
    alive = check_connection()
    print(f"连接状态: {'✅ 正常' if alive else '❌ 异常'}")
    close_connection()
