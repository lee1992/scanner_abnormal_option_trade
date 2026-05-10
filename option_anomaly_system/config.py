"""
全局配置：所有参数集中管理。
修改任何阈值/路径/权重：只改这一个文件。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MoneynessBucket:
    name:  str
    lower: float   # 含下界
    upper: float   # 不含上界（右开区间）


@dataclass
class Config:
    # ── 路径 ──────────────────────────────────────────────────────────────────
    data_dir:   str = r".\filter_data"
    output_dir: str = r".\output"

    # ── Moneyness 分段（正=ITM，负=OTM，方向化后）────────────────────────────
    # 增删分段：只改这个列表，其他代码无需改动
    moneyness_buckets: list = field(default_factory=lambda: [
        MoneynessBucket("DEEP_OTM",   -9999.0, -0.07),   # >20% ITM
        #MoneynessBucket("ITM",        -0.20,  -0.10),   # 10–20% ITM
        #MoneynessBucket("SLIGHT_ITM", -0.10,  -0.05),   # 5–10% ITM
        MoneynessBucket("ATM",        -0.07,   0.07),   # ±5%
        #MoneynessBucket("SLIGHT_OTM",  0.05,   0.10),   # 5–10% OTM
        #MoneynessBucket("OTM",         0.10,   0.20),   # 10–20% OTM
        MoneynessBucket("DEEP_ITM",    0.07,  9999.0),   # >20% OTM
    ])

    # ── 因子计算参数 ──────────────────────────────────────────────────────────
    churn_oi_floor:    float = 1.0          # OI 为 0 时的地板（防除零）
    notional_oi_floor: float = 1.0          # 名义 OI 地板（USD）
    min_turnover_usd:  float = 500_000.0    # 初筛成交额门槛（与扫描脚本一致）
    deep_itm_churn_threshold: float = 1.0   # 深实值"过手"判定：名义周转率 >= 此值

    # ── 持续性参数 ────────────────────────────────────────────────────────────
    lookback_days:         int   = 10        # 持续性回看窗口（交易日）
    min_persistence_days:  int   = 2         # 最低活跃天数（2天即触发，3天以上才进报告）

    # ── Stop Event 参数 ───────────────────────────────────────────────────────
    stop_drop_ratio:   float = 0.30          # 峰值后跌至此比例以下 = 停止
    stop_window_days:  int   = 3             # 连续 N 天低于阈值才确认停止

    # ── 多因子打分权重 ────────────────────────────────────────────────────────
    # 增删因子：在 factor.py 注册后，在此加权重；权重=0 表示不参与总分
    score_weights: dict = field(default_factory=lambda: {
        # 合约级因子（聚合后用 max/sum）
        "cf_log_turnover":           1.5,
        "cf_churn":                  2.0,
        "cf_notional_churn":         2.5,
        "cf_delta_flow":             1.5,
        "cf_vega_flow":              1.0,
        "cf_gamma_flow":             0.8,
        "cf_iv_level":               0.8,
        "cf_volume_oi_ratio":        1.5,
        "cf_turnover_per_contract":  1.0,
        "cf_intraday_range":         0.5,
        # 标的级专项因子
        "uf_pcr_turnover":           1.0,    # 只在 ALL 版有效
        "uf_deep_itm_share":         2.5,    # 深实值成交额占比
        # 跨日因子（merge_persistence_scores 注入）
        "uf_persistence_ratio":      3.0,
        "uf_stop_bonus":             2.0,
    })

    # ── 报告参数 ──────────────────────────────────────────────────────────────
    top_n_report: int = 100
    # moneyness_type_rank 跨日聚合方式: "mean"（均值）或 "sum"（求和）
    # "mean": 每股每天先 sum 合约，再对所有交易日取平均 → 反映日均活跃度
    # "sum":  每股每天先 sum 合约，再对所有交易日求和   → 反映区间总量
    moneyness_rank_agg: str = "sum"

    # ── 路径工具方法 ──────────────────────────────────────────────────────────
    def archive_dir(self, trade_date: str) -> Path:
        p = Path(self.output_dir) / "archive" / trade_date
        p.mkdir(parents=True, exist_ok=True)
        return p

    def reports_dir(self) -> Path:
        p = Path(self.output_dir) / "reports"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def stop_events_dir(self) -> Path:
        p = Path(self.output_dir) / "stop_events"
        p.mkdir(parents=True, exist_ok=True)
        return p
