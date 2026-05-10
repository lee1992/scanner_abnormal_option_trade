# =============================================================================
# checkpoint.py —— 断点续跑缓存管理
# =============================================================================
import json
import logging
import os
from datetime import datetime

from scan_package import config

logger = logging.getLogger(__name__)


def load_checkpoint() -> dict:
    """
    读取断点缓存。
    返回格式：
    {
        "completed": ["US.AAPL", "US.MSFT", ...],
        "start_time": "2025-01-01 10:00:00",
        "last_update": "2025-01-01 12:30:00"
    }
    """
    if not os.path.exists(config.CHECKPOINT_FILE):
        return {"completed": [], "start_time": None, "last_update": None}

    try:
        with open(config.CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(
            f"读取断点缓存：已完成 {len(data.get('completed', []))} 个标的"
        )
        return data
    except Exception as e:
        logger.warning(f"读取断点缓存失败: {e}，将从头开始")
        return {"completed": [], "start_time": None, "last_update": None}


def save_checkpoint(completed_symbols: list[str], start_time: str = None):
    """将已完成的标的列表写入缓存文件"""
    checkpoint = load_checkpoint()

    if start_time and not checkpoint.get("start_time"):
        checkpoint["start_time"] = start_time

    checkpoint["completed"]   = list(set(completed_symbols))
    checkpoint["last_update"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
        with open(config.CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"写入断点缓存失败: {e}")


def mark_done(symbol: str, completed_list: list[str]):
    """标记单个标的为已完成并实时保存"""
    if symbol not in completed_list:
        completed_list.append(symbol)
    save_checkpoint(completed_list)


def get_pending_symbols(all_symbols: list[str]) -> tuple[list[str], list[str]]:
    """
    返回 (待处理列表, 已完成列表)。
    已完成的从 all_symbols 中剔除。
    """
    checkpoint = load_checkpoint()
    done_set   = set(checkpoint.get("completed", []))
    pending    = [s for s in all_symbols if s not in done_set]
    done       = [s for s in all_symbols if s in done_set]

    logger.info(
        f"任务状态：共 {len(all_symbols)} 个标的，"
        f"已完成 {len(done)} 个，待处理 {len(pending)} 个"
    )
    return pending, done


def clear_checkpoint():
    """清除断点缓存（从头开始运行时调用）"""
    if os.path.exists(config.CHECKPOINT_FILE):
        os.remove(config.CHECKPOINT_FILE)
        logger.info("断点缓存已清除")
    if os.path.exists(config.RESULTS_FILE):
        os.remove(config.RESULTS_FILE)
        logger.info("结果文件已清除")


# ── 测试入口 ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    # 模拟断点续跑
    all_syms = ['US.AAPL', 'US.MSFT', 'US.NVDA', 'US.TSLA', 'US.SPY']

    print("=== 第一次运行（模拟处理前3个后中断）===")
    done_list = []
    for sym in all_syms[:3]:
        mark_done(sym, done_list)
    print(f"已保存完成标的: {done_list}")

    print("\n=== 第二次运行（续跑）===")
    pending, done = get_pending_symbols(all_syms)
    print(f"待处理: {pending}")
    print(f"已完成: {done}")

    print("\n=== 清除断点 ===")
    clear_checkpoint()
    pending2, _ = get_pending_symbols(all_syms)
    print(f"清除后待处理: {pending2} ✅")
