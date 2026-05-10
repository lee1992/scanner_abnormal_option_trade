# tech_analysis 功能解析与波形结构读取指南

## 一、整体架构

`tech_analysis` 是一个"**三理论串联约束**"的量化分析包，三个理论层层过滤：

```
原始K线数据
    ↓
道氏理论 (DowEngine)      ← 确定主次趋势方向，输出"约束"
    ↓
江恩网格 (GannEngine)     ← 计算关键价格位，输出"吸附评分"
    ↓
艾略特波浪 (ElliottEngine) ← 在上述约束内枚举波浪计数候选
    ↓
多周期共振 (MultiTimeframeSync) ← 跨周期对齐，输出共振信号
    ↓
信号生成 (SignalGenerator)  ← 输出具体交易信号（入场/止损/止盈）
```

---

## 二、各模块功能

### 1. DowEngine → `dow_constraint`
判断趋势结构，输出：
- `primary_trend`：主趋势方向（UP / DOWN / NEUTRAL）
- `secondary_trend`：次级趋势方向
- `allowed_wave_types`：当前趋势下允许存在的浪型
- `forbidden_wave_types`：**被禁止的浪型**（波浪引擎会过滤掉这些）

### 2. GannEngine → `gann_grid`
计算江恩价格网格，输出：
- `price_levels`：一组 `GannLevel`，每个含 `value`（价格）和 `strength`（强度）
- `time_levels`：时间轴上的比例位置
- `resonance`：价格位与时间位的**共振点**，带 `score` 评分

### 3. ElliottEngine → `wave_counts`
这是最核心的波形结构部分，输出一个 **候选列表**，每个 `WaveCount` 包含：

| 字段 | 含义 |
|------|------|
| `wave_type` | 整体结构类型（见下表） |
| `labels` | 每一个子浪的详细标注 |
| `score` | 综合评分 0-100，**越高越优先** |
| `hard_rule_pass` | 是否通过艾略特硬规则 |
| `dow_consistent` | 是否与道氏趋势一致 |
| `gann_snap_score` | 浪的端点是否吸附在江恩位上 |
| `guideline_score` | 符合波浪"指南"的程度（如3浪最长等）|
| `description` | 文字描述 |

**wave_type 枚举值说明：**

| 值 | 中文含义 |
|----|---------|
| `impulse_up` | 向上推动浪（5浪上涨） |
| `impulse_down` | 向下推动浪（5浪下跌） |
| `corrective_zigzag` | 锯齿形调整（A-B-C，B浪不超A浪起点）|
| `corrective_flat` | 平坦形调整（A-B-C，B浪接近A浪起点）|
| `corrective_triangle` | 三角形调整（收敛/扩散）|
| `corrective_complex` | 复杂调整（双三、三三等）|
| `diagonal_up` | 向上斜角浪（楔形推动）|
| `diagonal_down` | 向下斜角浪（楔形下跌）|

**每个 `WaveLabel`（子浪标注）包含：**

| 字段 | 含义 |
|------|------|
| `wave_id` | 浪号，如 "1","2","3","4","5" 或 "A","B","C" |
| `start_price` | 该浪起点价格 |
| `end_price` | 该浪终点价格 |
| `start_time` | 起点时间 |
| `end_time` | 终点时间 |
| `degree` | 层级（0=当前分析层，1=上一级，-1=次级）|

### 4. TimeframeAnalysis（单周期综合结论）
每个时间周期的分析结论，关键字段：
- `current_position`：当前所处浪位，如 `"推动3浪中"` / `"调整B浪中"`
- `bias`：方向偏向，`'bullish'` / `'bearish'` / `'neutral'`
- `primary_wave`：**评分最高的波浪计数**（就是 `wave_counts[0]`）

### 5. ResonanceSignal（多周期共振信号）
- `direction`：方向（`'long'` / `'short'`）
- `strength`：共振强度 0-1
- `large_tf` / `small_tf`：大小周期名称
- `entry_condition`：入场条件文字描述
- `invalidation`：失效条件描述
- `key_levels`：含止损/目标等关键价格

---

## 三、如何读取波形结构（解决"看不懂"的问题）

### 方法1：直接打印关键字段（推荐先这样做）

```python
# 运行分析后，result 已经得到
analyses = result['analyses']

for freq, ta in analyses.items():
    print(f"\n{'='*50}")
    print(f"周期: {freq}")
    print(f"  趋势偏向: {ta.bias}")
    print(f"  当前位置: {ta.current_position}")
    
    if ta.primary_wave:
        pw = ta.primary_wave
        print(f"  最优波形: {pw.wave_type.value}  评分={pw.score:.1f}")
        print(f"  通过硬规则: {pw.hard_rule_pass}  道氏一致: {pw.dow_consistent}")
        print(f"  各子浪:")
        for lbl in pw.labels:
            print(f"    浪{lbl.wave_id}: {lbl.start_price:.2f} → {lbl.end_price:.2f}"
                  f"  ({lbl.start_time} ~ {lbl.end_time})")
    
    # 所有候选（按评分排序）
    print(f"  候选数量: {len(ta.wave_counts)}")
    for i, wc in enumerate(ta.wave_counts[:3]):  # 只看前3
        print(f"  候选{i+1}: {wc.wave_type.value} 评分={wc.score:.1f} | {wc.description}")
```

### 方法2：打印共振信号（最有用的交易信息）

```python
print("\n多周期共振信号:")
for sig in result['resonance_signals']:
    print(f"  方向={sig.direction}  强度={sig.strength:.2f}")
    print(f"  {sig.large_tf}({sig.large_bias}) + {sig.small_tf}({sig.small_bias})")
    print(f"  大周期位置: {sig.large_position}")
    print(f"  小周期位置: {sig.small_position}")
    print(f"  入场条件: {sig.entry_condition}")
    print(f"  失效条件: {sig.invalidation}")
    print(f"  关键价位: {sig.key_levels}")
```

### 方法3：生成可读性报告（写入文件）

```python
import json

def wave_summary(result):
    lines = []
    for freq, ta in result['analyses'].items():
        lines.append(f"\n## {freq} 周期")
        lines.append(f"- 偏向: {ta.bias}  |  位置: {ta.current_position}")
        if ta.primary_wave:
            pw = ta.primary_wave
            lines.append(f"- 波形结构: **{pw.wave_type.value}** (评分 {pw.score:.0f}/100)")
            lines.append(f"- {pw.description}")
            for lbl in pw.labels:
                amp = lbl.end_price - lbl.start_price
                lines.append(f"  - 浪{lbl.wave_id}: {lbl.start_price:.2f}→{lbl.end_price:.2f}"
                             f"  幅度={amp:+.2f}")
    
    lines.append("\n## 共振信号")
    for sig in result['resonance_signals']:
        lines.append(f"- [{sig.direction.upper()}] 强度={sig.strength:.0%}"
                    f" | {sig.large_tf}+{sig.small_tf}")
        lines.append(f"  入场: {sig.entry_condition}")
    
    return "\n".join(lines)

# 保存报告
with open("wave_report.md", "w", encoding="utf-8") as f:
    f.write(wave_summary(result))
print("报告已保存到 wave_report.md")
```

---

## 四、multi_timeframe_result.json 结构说明

这个 JSON 文件（21MB）存储的是 `result['analyses']` 的序列化内容，结构为：

```json
{
  "1D": {
    "freq": "1D",
    "bias": "bullish",
    "current_position": "推动3浪中",
    "wave_counts": [
      {
        "wave_type": "impulse_up",
        "score": 82.5,
        "hard_rule_pass": true,
        "dow_consistent": true,
        "gann_snap_score": 0.74,
        "guideline_score": 0.68,
        "description": "...",
        "labels": [
          {"wave_id": "1", "start_price": 100.0, "end_price": 120.0, ...},
          {"wave_id": "2", "start_price": 120.0, "end_price": 108.0, ...},
          ...
        ]
      }
    ],
    "dow_constraint": { "primary_trend": "UP", ... },
    "gann_grid": { "price_levels": [...], ... }
  },
  "60min": { ... },
  "5min": { ... }
}
```

**score 评分的构成参考：**
- `hard_rule_pass = False` → 直接排除（违反波浪基本规则）
- `gann_snap_score`：浪的端点有多贴近江恩价格位（0-1）
- `guideline_score`：3浪最长、4浪不入1浪区等指南符合度（0-1）
- 最终 `score` = 综合加权，越高说明该计数越"干净"

---

## 五、常见问题

**Q: wave_counts 里有好几个候选，看哪个？**
> 看 `primary_wave`，就是 `wave_counts[0]`，评分最高的那个。但若第1和第2分数差距小于10分，说明结构不确定，建议两种路径都参考。

**Q: current_position 怎么理解？**
> 直接告诉你"现在在哪个浪里"，如 `"推动3浪中"` 意味着正处于第3上涨浪内部，尚未完成。

**Q: 文件太大（21MB）打不开？**
> 别直接打开，用代码按周期读取：
> ```python
> with open("multi_timeframe_result.json", "r", encoding="utf-8") as f:
>     data = json.load(f)
> print(data["1D"]["current_position"])  # 只看日线当前位置
> ```
