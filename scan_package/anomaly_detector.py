# =============================================================================
# anomaly_detector.py —— 异常交易检测逻辑
# =============================================================================
"""
当前版本：基于单日快照绝对成交额阈值检测（不依赖历史K线，无额度消耗）。
预留接口：基于历史均值倍数检测（需历史K线，暂不启用）。
"""
import logging
from datetime import datetime

import pandas as pd

from scan_package import config

logger = logging.getLogger(__name__)


def detect_anomalies(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    """
    对快照 DataFrame 进行异常检测。

    检测规则（满足其一即标记）：
    1. 当日成交额 turnover >= TURNOVER_ABS_THRESHOLD（绝对值）
    2. 成交量 volume > 0（已由服务端前置过滤，此处为双重确认）

    返回:
        异常合约 DataFrame，附加 anomaly_reason 列
    """
    if snapshot_df.empty:
        return pd.DataFrame()

    df = snapshot_df.copy()

    # 确保数值类型
    for col in ['turnover', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 规则 1：成交额绝对值
    mask_turnover = df['turnover'] >= config.TURNOVER_ABS_THRESHOLD

    # 规则 2：成交量（冗余保险）
    mask_volume = df['volume'] > 0

    # 综合：两个条件都满足（成交额达标且有成交）
    mask_anomaly = mask_turnover & mask_volume

    anomalies = df[mask_anomaly].copy()

    if anomalies.empty:
        logger.info("本批次无异常合约")
        return pd.DataFrame()

    # 标注异常原因
    anomalies['anomaly_reason'] = anomalies.apply(
        lambda row: _build_reason(row), axis=1
    )

    # 按成交额降序排列
    if 'turnover' in anomalies.columns:
        anomalies = anomalies.sort_values('turnover', ascending=False)

    logger.info(f"发现异常合约 {len(anomalies)} 个")
    return anomalies.reset_index(drop=True)


def _build_reason(row: pd.Series) -> str:
    """构建可读的异常原因描述"""
    parts = []
    turnover = row.get('turnover', 0)
    volume = row.get('volume', 0)

    if turnover >= config.TURNOVER_ABS_THRESHOLD:
        parts.append(f"成交额 ${turnover:,.0f} ≥ 阈值 ${config.TURNOVER_ABS_THRESHOLD:,}")
    if volume > 0:
        parts.append(f"成交量 {int(volume):,} 手")

    return " | ".join(parts) if parts else "未知"


def format_anomaly_report(anomalies: pd.DataFrame,
                          symbol: str = '') -> str:
    """
    格式化异常报告为可读字符串（用于控制台打印）。
    """
    if anomalies.empty:
        return f"[{symbol}] 无异常合约\n"

    lines = [f"\n{'=' * 60}"]
    lines.append(f"  [{symbol}] 发现 {len(anomalies)} 个异常期权合约")
    lines.append(f"{'=' * 60}")

    display_cols = [
        'code', 'option_type', 'option_strike_price', 'strike_time',
        'last_price', 'volume', 'turnover',
        'option_delta', 'option_implied_volatility',
        'option_open_interest', 'anomaly_reason'
    ]
    show_cols = [c for c in display_cols if c in anomalies.columns]

    for _, row in anomalies[show_cols].iterrows():
        lines.append("")
        for col in show_cols:
            val = row[col]
            if col == 'turnover' and isinstance(val, (int, float)):
                val = f"${val:,.0f}"
            elif col == 'volume' and isinstance(val, (int, float)):
                val = f"{int(val):,}"
            lines.append(f"  {col:<35} {val}")

    lines.append(f"{'=' * 60}\n")
    return "\n".join(lines)


def save_anomalies_to_csv(anomalies: pd.DataFrame,
                          symbol: str,
                          filepath: str = None):
    """
    将异常结果追加写入 CSV 文件。
    """
    if anomalies.empty:
        return

    filepath = filepath or config.RESULTS_FILE
    df = anomalies.copy()
    df['scan_symbol'] = symbol
    df['scan_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    mode = 'a'
    header = False

    import os
    if not os.path.exists(filepath):
        mode = 'w'
        header = True

    df.to_csv(filepath, mode=mode, header=header, index=False)
    logger.info(f"已追加写入 {len(df)} 条异常记录 → {filepath}")


# ── 测试入口 ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    # 构造模拟快照数据测试异常检测逻辑
    mock_data = pd.DataFrame([
        {
            'code': 'US.AAPL250117C00100000',
            'option_type': 'CALL',
            'option_strike_price': 100.0,
            'strike_time': '2025-01-17',
            'last_price': 88.5,
            'volume': 5000,
            'turnover': 4_400_000,  # ✅ 超阈值
            'option_delta': 0.95,
            'option_implied_volatility': 45.2,
            'option_open_interest': 1200,
            'stock_owner': 'US.AAPL',
        },
        {
            'code': 'US.AAPL250117P00300000',
            'option_type': 'PUT',
            'option_strike_price': 300.0,
            'strike_time': '2025-01-17',
            'last_price': 0.05,
            'volume': 10,
            'turnover': 50,  # ❌ 低于阈值
            'option_delta': -0.02,
            'option_implied_volatility': 120.0,
            'option_open_interest': 50,
            'stock_owner': 'US.AAPL',
        },
    ])

    anomalies = detect_anomalies(mock_data)
    print(format_anomaly_report(anomalies, 'US.AAPL'))
    print(f"期望检出 1 条，实际: {len(anomalies)} 条 {'✅' if len(anomalies) == 1 else '❌'}")
