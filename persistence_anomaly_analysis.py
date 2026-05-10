"""
持续性异常行为检测（逐日审核版）
输出4个CSV + 1个Markdown报告
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
CONFIG = {
    'archive_dir': Path(r'.\output\archive'),
    'output_dir': Path(r'.\output\reports\persistence_anomaly'),
    'window_days': 5,                    # 窗口天数（n）
    'min_pct_active': 0.7,              # 需求1、4的活跃比例阈值（70%）
    'min_pct_atm_otm': 0.8,            # 需求2、3的窗口内比例阈值（80%），注意k_days可设小，这里直接用window_days
    'deep_turnover_threshold': 2_000_000,  # 深度实值大额阈值（美元）
    'deep_volume_oi_ratio': 2.0,        # 成交量/持仓量 > 2 视为“远大于”，用于排序
    'atm_ratio_threshold': 5.0,         # 成交额比值 > 5 或 < 0.2 异常
    'otm_ratio_threshold': 5.0,
    'stop_consecutive_days': 2,         # 停止事件：连续无DEEP_ITM天数（基于每日是否有deep_itm大额交易）
    'iv_diff_pct': 0.3,                 # IV比值偏离30%算显著
    'match_strike_tolerance': 0.02,     # 执行价相近容忍度2%
}

def load_all_archive_dates(base_dir):
    """返回字典 {日期: {'deep_itm_all': df, 'moneyness_all': df, 'underlying_all': df}}"""
    dates = sorted([d.name for d in base_dir.glob('*') if d.is_dir() and d.name[:4].isdigit()])
    data = {}
    for d in dates:
        daily = {}
        for fname in ['deep_itm_all.csv', 'moneyness_all.csv', 'underlying_all.csv']:
            fpath = base_dir / d / fname
            if fpath.exists():
                daily[fname.replace('.csv', '')] = pd.read_csv(fpath)
        if daily:
            data[d] = daily
    return data

def load_moneyness_by_type(archive_dir, date):
    """加载当日 moneyness_call 和 moneyness_put"""
    call_path = archive_dir / date / 'moneyness_call.csv'
    put_path  = archive_dir / date / 'moneyness_put.csv'
    call_df = pd.read_csv(call_path) if call_path.exists() else None
    put_df  = pd.read_csv(put_path)  if put_path.exists()  else None
    return call_df, put_df

def load_factors_contract(archive_dir, date):
    """加载合约级因子（用于IV对比）"""
    fpath = archive_dir / date / 'factors_contract.csv'
    if fpath.exists():
        # 只读必要列以节省内存
        usecols = ['stock_owner', 'moneyness_bucket', 'option_type', 'option_implied_volatility',
                   'option_strike_price', 'option_expiry_date_distance']
        return pd.read_csv(fpath, usecols=usecols)
    return None

def check_deep_itm_daily(deep_df, stock, threshold):
    """检查当日该股票是否有深度实值大额交易，返回 (has_deep, deep_turnover_sum, volume_oi_ratio)"""
    if deep_df is None or deep_df.empty:
        return False, 0.0, 0.0
    sub = deep_df[deep_df['stock_owner'] == stock]
    if sub.empty:
        return False, 0.0, 0.0
    # 大额过滤
    sub = sub[sub['deep_turnover'] >= threshold]
    if sub.empty:
        return False, 0.0, 0.0
    total_turnover = sub['deep_turnover'].sum()
    # 计算成交量/持仓量（如果有 volume 和 open_interest，deep_itm_all 里有 deep_volume 和 deep_oi_sum）
    # 平均每份合约成交手数/持仓
    volume = sub['deep_volume'].sum()
    oi = sub['deep_oi_sum'].sum()
    vol_oi_ratio = volume / oi if oi > 0 else np.inf
    return True, total_turnover, vol_oi_ratio

def check_atm_ratio_daily(call_df, put_df, stock, threshold):
    """检查当日ATM Call/Put成交额比值是否异常，返回 (is_anomaly, ratio, call_turnover, put_turnover, iv_ratio, iv_match)"""
    if call_df is None or put_df is None:
        return False, 0.0, 0.0, 0.0, 0.0, False
    call_atm = call_df[(call_df['stock_owner'] == stock) & (call_df['moneyness_bucket'] == 'ATM')]
    put_atm  = put_df[(put_df['stock_owner'] == stock) & (put_df['moneyness_bucket'] == 'ATM')]
    if call_atm.empty or put_atm.empty:
        return False, 0.0, 0.0, 0.0, 0.0, False
    call_turn = call_atm['bucket_turnover'].sum()
    put_turn  = put_atm['bucket_turnover'].sum()
    if call_turn + put_turn == 0:
        return False, 0.0, 0.0, 0.0, 0.0, False
    ratio = call_turn / (put_turn + 1e-6)
    anomaly = (ratio > threshold) or (ratio < 1/threshold)
    # IV 对比需要合约级数据，这里暂返回占位，后续在外部调用更精确的IV对比
    return anomaly, ratio, call_turn, put_turn, 0.0, False

def check_otm_ratio_daily(call_df, put_df, stock, threshold):
    """类似，DEEP_OTM"""
    if call_df is None or put_df is None:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, False
    call_otm = call_df[(call_df['stock_owner'] == stock) & (call_df['moneyness_bucket'] == 'DEEP_OTM')]
    put_otm  = put_df[(put_df['stock_owner'] == stock) & (put_df['moneyness_bucket'] == 'DEEP_OTM')]
    if call_otm.empty or put_otm.empty:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, False
    call_turn = call_otm['bucket_turnover'].sum()
    put_turn  = put_otm['bucket_turnover'].sum()
    call_oi   = call_otm['bucket_oi_sum'].sum()
    put_oi    = put_otm['bucket_oi_sum'].sum()
    ratio_turn = call_turn / (put_turn + 1e-6)
    ratio_oi   = call_oi / (put_oi + 1e-6)
    anomaly_turn = (ratio_turn > threshold) or (ratio_turn < 1/threshold)
    anomaly_oi   = (ratio_oi > threshold) or (ratio_oi < 1/threshold)
    anomaly = anomaly_turn or anomaly_oi
    return anomaly, ratio_turn, ratio_oi, call_turn, put_turn, call_oi, put_oi

def compute_iv_ratio_for_bucket(archive_dir, date, stock, bucket, use_strike_match=True):
    """
    对于某个 bucket (ATM/DEEP_OTM) 和股票，计算该日所有 call 和 put 合约的平均 IV 比值。
    如果 use_strike_match，则尝试配对相近执行价和相近到期日（距离差<=7天）的合约对，计算平均比值。
    返回 (iv_ratio, match_count, is_reliable)
    """
    factors = load_factors_contract(archive_dir, date)
    if factors is None:
        return 0.0, 0, False
    sub = factors[(factors['stock_owner'] == stock) & (factors['moneyness_bucket'] == bucket)]
    if sub.empty:
        return 0.0, 0, False
    calls = sub[sub['option_type'] == 'CALL']
    puts  = sub[sub['option_type'] == 'PUT']
    if calls.empty or puts.empty:
        return 0.0, 0, False
    if not use_strike_match:
        # 简单平均
        iv_call = calls['option_implied_volatility'].mean()
        iv_put  = puts['option_implied_volatility'].mean()
        if iv_put == 0:
            return 0.0, 0, False
        return iv_call / iv_put, len(calls) + len(puts), False
    else:
        # 相近执行价配对
        ratios = []
        # 对每个call，寻找执行价相近且到期日相近（距离差<=7天）的put
        for _, call_row in calls.iterrows():
            strike_c = call_row['option_strike_price']
            dte_c = call_row['option_expiry_date_distance']
            # 找put中执行价相差<2%且到期日差≤7天的
            for _, put_row in puts.iterrows():
                strike_p = put_row['option_strike_price']
                dte_p = put_row['option_expiry_date_distance']
                if abs(strike_c - strike_p) / max(strike_c, 1e-6) < CONFIG['match_strike_tolerance']:
                    if abs(dte_c - dte_p) <= 7:
                        iv_ratio = call_row['option_implied_volatility'] / (put_row['option_implied_volatility'] + 1e-6)
                        ratios.append(iv_ratio)
                        break  # 每个call只配一个最近的put
        if not ratios:
            return 0.0, 0, False
        return np.mean(ratios), len(ratios), True

def get_daily_summaries(archive_dir, date, all_stocks_set):
    """返回当日所有股票的深度实值、ATM比值、OTM比值汇总，以及IV补充信息（可选）"""
    deep_df = pd.read_csv(archive_dir / date / 'deep_itm_all.csv') if (archive_dir / date / 'deep_itm_all.csv').exists() else None
    call_df, put_df = load_moneyness_by_type(archive_dir, date)
    daily_records = []
    for stock in all_stocks_set:
        # 深度实值
        has_deep, deep_turn, vol_oi = check_deep_itm_daily(deep_df, stock, CONFIG['deep_turnover_threshold'])
        # ATM比值
        atm_anom, atm_ratio, atm_call_turn, atm_put_turn, _, _ = check_atm_ratio_daily(call_df, put_df, stock, CONFIG['atm_ratio_threshold'])
        # OTM比值
        otm_anom, otm_ratio_turn, otm_ratio_oi, otm_call_turn, otm_put_turn, otm_call_oi, otm_put_oi = check_otm_ratio_daily(call_df, put_df, stock, CONFIG['otm_ratio_threshold'])

        daily_records.append({
            'date': date,
            'stock_owner': stock,
            'has_deep_itm': has_deep,
            'deep_turnover': deep_turn,
            'deep_vol_oi_ratio': vol_oi,
            'atm_anomaly': atm_anom,
            'atm_ratio': atm_ratio,
            'atm_call_turnover': atm_call_turn,
            'atm_put_turnover': atm_put_turn,
            'otm_anomaly': otm_anom,
            'otm_ratio_turnover': otm_ratio_turn,
            'otm_ratio_oi': otm_ratio_oi,
            'otm_call_turnover': otm_call_turn,
            'otm_put_turnover': otm_put_turn,
            'otm_call_oi': otm_call_oi,
            'otm_put_oi': otm_put_oi,
        })
    return pd.DataFrame(daily_records)

def compute_window_stats(df, window_dates, metric_col, bool_condition=None, value_col=None):
    """通用窗口统计：对每个股票，在窗口日期内统计满足条件的次数和比例"""
    # 准备空结果字典
    stocks = df['stock_owner'].unique()
    results = []
    for stock in stocks:
        stock_df = df[df['stock_owner'] == stock]
        # 只保留窗口内的日期
        stock_df = stock_df[stock_df['date'].isin(window_dates)]
        total_days = len(window_dates)
        if total_days == 0:
            continue
        if bool_condition is not None:
            condition_true = stock_df[bool_condition]
            active_days = len(condition_true)
            pct = active_days / total_days
            # 附加指标：比如深度实值的成交额总和、平均vol/oi等
            extra = {}
            if value_col and value_col in stock_df.columns:
                extra['total_value'] = condition_true[value_col].sum()
                extra['avg_value'] = condition_true[value_col].mean()
            if 'deep_vol_oi_ratio' in stock_df.columns:
                extra['median_vol_oi_ratio'] = condition_true['deep_vol_oi_ratio'].median()
            results.append({
                'stock_owner': stock,
                'active_days': active_days,
                'total_days': total_days,
                'pct': pct,
                **extra
            })
        else:
            # 对于比值类，需要额外处理
            pass
    return pd.DataFrame(results)

def main():
    archive_dir = CONFIG['archive_dir']
    data = load_all_archive_dates(archive_dir)
    if not data:
        print("未找到archive数据，请先运行option_anomaly_system/main.py")
        return

    all_dates = sorted(data.keys())
    window_dates = all_dates[-CONFIG['window_days']:] if len(all_dates) >= CONFIG['window_days'] else all_dates
    print(f"分析窗口日期: {window_dates}")

    # 收集所有出现的股票
    all_stocks = set()
    for date in data.keys():
        deep_df = data[date].get('deep_itm_all')
        if deep_df is not None and not deep_df.empty:
            all_stocks.update(deep_df['stock_owner'].unique())
        call_df, put_df = load_moneyness_by_type(archive_dir, date)
        if call_df is not None:
            all_stocks.update(call_df['stock_owner'].unique())
        if put_df is not None:
            all_stocks.update(put_df['stock_owner'].unique())
    print(f"共涉及 {len(all_stocks)} 只股票")

    # 逐日生成每日汇总表（为了窗口统计，需要所有日期的记录）
    daily_summaries = []
    for date in all_dates:
        # 构建当日所有股票的记录（即使无异常也记录）
        # 简便方式：从已有数据中提取该日所有出现过的股票，加上其他可能
        stocks_in_date = set()
        deep_df = data[date].get('deep_itm_all')
        if deep_df is not None:
            stocks_in_date.update(deep_df['stock_owner'].unique())
        call_df, put_df = load_moneyness_by_type(archive_dir, date)
        if call_df is not None:
            stocks_in_date.update(call_df['stock_owner'].unique())
        if put_df is not None:
            stocks_in_date.update(put_df['stock_owner'].unique())
        # 获取该日这些股票的指标
        for stock in stocks_in_date:
            # 深度实值
            deep_df_day = deep_df[deep_df['stock_owner'] == stock] if deep_df is not None and not deep_df.empty else None
            has_deep, deep_turn, vol_oi = check_deep_itm_daily(deep_df_day, stock, CONFIG['deep_turnover_threshold'])
            # ATM/OTM
            call_df_day = call_df[call_df['stock_owner'] == stock] if call_df is not None else pd.DataFrame()
            put_df_day  = put_df[put_df['stock_owner'] == stock] if put_df is not None else pd.DataFrame()
            atm_anom, atm_ratio, atm_call_turn, atm_put_turn, _, _ = check_atm_ratio_daily(call_df_day, put_df_day, stock, CONFIG['atm_ratio_threshold'])
            otm_anom, otm_ratio_turn, otm_ratio_oi, otm_call_turn, otm_put_turn, otm_call_oi, otm_put_oi = check_otm_ratio_daily(call_df_day, put_df_day, stock, CONFIG['otm_ratio_threshold'])
            daily_summaries.append({
                'date': date,
                'stock_owner': stock,
                'has_deep_itm': has_deep,
                'deep_turnover': deep_turn,
                'deep_vol_oi_ratio': vol_oi,
                'atm_anomaly': atm_anom,
                'atm_ratio': atm_ratio,
                'atm_call_turnover': atm_call_turn,
                'atm_put_turnover': atm_put_turn,
                'otm_anomaly': otm_anom,
                'otm_ratio_turnover': otm_ratio_turn,
                'otm_ratio_oi': otm_ratio_oi,
                'otm_call_turnover': otm_call_turn,
                'otm_put_turnover': otm_put_turn,
                'otm_call_oi': otm_call_oi,
                'otm_put_oi': otm_put_oi,
            })
    daily_df = pd.DataFrame(daily_summaries)
    if daily_df.empty:
        print("没有生成任何每日记录，请检查数据")
        return

    # ---------- 需求1: DEEP_ITM 持续高活跃 ----------
    deep_window = daily_df[daily_df['date'].isin(window_dates)]
    deep_stats = deep_window.groupby('stock_owner').agg(
        active_days=('has_deep_itm', 'sum'),
        total_deep_turnover=('deep_turnover', 'sum'),
        avg_deep_turnover=('deep_turnover', 'mean'),
        median_vol_oi_ratio=('deep_vol_oi_ratio', lambda x: x[x>0].median() if (x>0).any() else 0)
    ).reset_index()
    deep_stats['total_days'] = len(window_dates)
    deep_stats['pct'] = deep_stats['active_days'] / len(window_dates)
    deep_stats = deep_stats[deep_stats['pct'] >= CONFIG['min_pct_active']]
    deep_stats = deep_stats.sort_values(['pct', 'total_deep_turnover'], ascending=[False, False])
    # 添加排序额外指标
    deep_stats['importance'] = deep_stats['pct'] * deep_stats['total_deep_turnover'] / 1e6  # 简单评分
    # 保存
    CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)
    deep_stats.to_csv(CONFIG['output_dir'] / 'deep_itm_persistence.csv', index=False)
    print(f"需求1: {len(deep_stats)} 个标的")

    # ---------- 需求2: ATM 比值异常（要求窗口内异常天数比例≥min_pct_atm_otm）----------
    atm_window = daily_df[daily_df['date'].isin(window_dates)]
    atm_stats = atm_window[atm_window['atm_anomaly']].groupby('stock_owner').agg(
        atm_anomaly_days=('atm_anomaly', 'sum'),
        avg_atm_ratio=('atm_ratio', 'mean'),
        total_atm_call_turn=('atm_call_turnover', 'sum'),
        total_atm_put_turn=('atm_put_turnover', 'sum')
    ).reset_index()
    atm_stats['total_days'] = len(window_dates)
    atm_stats['pct'] = atm_stats['atm_anomaly_days'] / len(window_dates)
    atm_stats = atm_stats[atm_stats['pct'] >= CONFIG['min_pct_atm_otm']]
    # 排序：按异常天数比例降序，再按总成交额降序
    atm_stats = atm_stats.sort_values(['pct', 'total_atm_call_turn'], ascending=[False, False])
    # 为每个股票补充IV对比信息（取窗口内最近一天的IV比值，若存在）
    iv_records = []
    for stock in atm_stats['stock_owner']:
        # 取该股在窗口内最后一次异常日的IV比值
        for date in reversed(window_dates):
            factors = load_factors_contract(archive_dir, date)
            if factors is not None:
                iv_ratio, match_cnt, reliable = compute_iv_ratio_for_bucket(archive_dir, date, stock, 'ATM', use_strike_match=True)
                if reliable:
                    iv_records.append({'stock_owner': stock, 'iv_ratio': iv_ratio, 'iv_match_cnt': match_cnt})
                    break
        else:
            iv_records.append({'stock_owner': stock, 'iv_ratio': np.nan, 'iv_match_cnt': 0})
    iv_df = pd.DataFrame(iv_records)
    atm_stats = atm_stats.merge(iv_df, on='stock_owner', how='left')
    atm_stats.to_csv(CONFIG['output_dir'] / 'atm_ratio_anomaly.csv', index=False)
    print(f"需求2: {len(atm_stats)} 个标的")

    # ---------- 需求3: DEEP_OTM 比值异常（成交额或持仓量）----------
    otm_window = daily_df[daily_df['date'].isin(window_dates)]
    otm_anomaly_df = otm_window[otm_window['otm_anomaly']]
    if not otm_anomaly_df.empty:
        otm_stats = otm_anomaly_df.groupby('stock_owner').agg(
            otm_anomaly_days=('otm_anomaly', 'sum'),
            avg_otm_ratio_turn=('otm_ratio_turnover', 'mean'),
            avg_otm_ratio_oi=('otm_ratio_oi', 'mean'),
            total_otm_call_turn=('otm_call_turnover', 'sum'),
            total_otm_put_turn=('otm_put_turnover', 'sum'),
            total_otm_call_oi=('otm_call_oi', 'sum'),
            total_otm_put_oi=('otm_put_oi', 'sum')
        ).reset_index()
        otm_stats['total_days'] = len(window_dates)
        otm_stats['pct'] = otm_stats['otm_anomaly_days'] / len(window_dates)
        otm_stats = otm_stats[otm_stats['pct'] >= CONFIG['min_pct_atm_otm']]
        otm_stats = otm_stats.sort_values(['pct', 'total_otm_call_turn'], ascending=[False, False])
        # IV 对比
        iv_otm_records = []
        for stock in otm_stats['stock_owner']:
            for date in reversed(window_dates):
                iv_ratio, match_cnt, reliable = compute_iv_ratio_for_bucket(archive_dir, date, stock, 'DEEP_OTM', use_strike_match=True)
                if reliable:
                    iv_otm_records.append({'stock_owner': stock, 'iv_ratio': iv_ratio, 'iv_match_cnt': match_cnt})
                    break
            else:
                iv_otm_records.append({'stock_owner': stock, 'iv_ratio': np.nan, 'iv_match_cnt': 0})
        iv_otm_df = pd.DataFrame(iv_otm_records)
        otm_stats = otm_stats.merge(iv_otm_df, on='stock_owner', how='left')
        otm_stats.to_csv(CONFIG['output_dir'] / 'deep_otm_ratio_anomaly.csv', index=False)
        print(f"需求3: {len(otm_stats)} 个标的")
    else:
        print("需求3: 无符合条件的标的")
        pd.DataFrame().to_csv(CONFIG['output_dir'] / 'deep_otm_ratio_anomaly.csv', index=False)

    # ---------- 需求4: DEEP_ITM 交易停止事件 ----------
    # 定义：窗口内前80%日期有deep_itm，后20%连续无deep_itm（或额度骤降）
    # 简化版：检测最后连续无deep_itm的天数 >= stop_consecutive_days，且之前活跃天数比例>=min_pct_active
    stop_candidates = []
    for stock in deep_stats['stock_owner']:
        stock_series = deep_window[deep_window['stock_owner'] == stock].sort_values('date')
        if len(stock_series) < 2:
            continue
        # 计算最后连续无deep_itm的天数
        has_deep_rev = stock_series['has_deep_itm'].values[::-1]
        consec_zero = 0
        for val in has_deep_rev:
            if not val:
                consec_zero += 1
            else:
                break
        if consec_zero >= CONFIG['stop_consecutive_days']:
            # 检查前期活跃比例是否达标
            before = stock_series.iloc[:-consec_zero] if consec_zero > 0 else stock_series
            active_before = before['has_deep_itm'].sum()
            total_before = len(before)
            if total_before > 0 and (active_before / total_before) >= CONFIG['min_pct_active']:
                stop_candidates.append({
                    'stock_owner': stock,
                    'stop_consecutive_days': consec_zero,
                    'active_days_before': active_before,
                    'total_days_before': total_before,
                    'pct_before': active_before / total_before,
                    'last_active_date': stock_series[stock_series['has_deep_itm']]['date'].max() if not stock_series[stock_series['has_deep_itm']].empty else None,
                    'stop_date_start': stock_series.iloc[-consec_zero]['date'] if consec_zero>0 else None
                })
    stop_df = pd.DataFrame(stop_candidates)
    if not stop_df.empty:
        stop_df = stop_df.sort_values(['stop_consecutive_days', 'pct_before'], ascending=[False, False])
        stop_df.to_csv(CONFIG['output_dir'] / 'deep_itm_stop_events.csv', index=False)
        print(f"需求4: {len(stop_df)} 个标的")
    else:
        pd.DataFrame().to_csv(CONFIG['output_dir'] / 'deep_itm_stop_events.csv', index=False)

    # ... 以上代码保持不变（加载数据、计算统计等） ...
    # 仅替换生成 Markdown 报告的部分

    # ---------- 生成 Markdown 报告（修复表格格式）----------
    md_lines = []
    md_lines.append("# 期权持续性异常行为监测报告")
    md_lines.append(f"**分析窗口**: {window_dates[0]} 至 {window_dates[-1]} (共{len(window_dates)}个交易日)")
    md_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append("")

    # 1. 深度实值持续高活跃
    md_lines.append("## 1. 深度实值(DEEP_ITM)持续高活跃标的")
    md_lines.append(
        f"阈值: 窗口内活跃天数 ≥ {int(CONFIG['min_pct_active'] * 100)}% （大额成交≥${CONFIG['deep_turnover_threshold']:,}）")
    if not deep_stats.empty:
        # 表头
        md_lines.append("| 股票 | 活跃天数 | 总天数 | 比例 | 总成交额(百万$) | 中位数成交量/持仓量 |")
        md_lines.append("|------|----------|--------|------|----------------|---------------------|")
        for _, row in deep_stats.head(20).iterrows():
            md_lines.append(
                f"| {row['stock_owner']} | {row['active_days']} | {row['total_days']} | {row['pct']:.1%} | {row['total_deep_turnover'] / 1e6:.2f} | {row['median_vol_oi_ratio']:.2f} |")
    else:
        md_lines.append("无符合条件的标的。")
    md_lines.append("")

    # 2. ATM 异常
    md_lines.append("## 2. ATM 看涨/看跌成交额异常标的")
    md_lines.append(
        f"阈值: 调用/认沽成交额比 > {CONFIG['atm_ratio_threshold']} 或 < 1/{CONFIG['atm_ratio_threshold']}，窗口内异常天数≥{int(CONFIG['min_pct_atm_otm'] * 100)}%")
    if 'atm_stats' in locals() and not atm_stats.empty:
        md_lines.append(
            "| 股票 | 异常天数 | 比例 | 平均比值 | 总Call成交额(百万$) | 总Put成交额(百万$) | IV比值(最近) |")
        md_lines.append(
            "|------|----------|------|----------|---------------------|---------------------|-------------|")
        for _, row in atm_stats.head(20).iterrows():
            iv_str = f"{row['iv_ratio']:.2f}" if pd.notna(row['iv_ratio']) else "N/A"
            md_lines.append(
                f"| {row['stock_owner']} | {row['atm_anomaly_days']} | {row['pct']:.1%} | {row['avg_atm_ratio']:.2f} | {row['total_atm_call_turn'] / 1e6:.2f} | {row['total_atm_put_turn'] / 1e6:.2f} | {iv_str} |")
    else:
        md_lines.append("无符合条件的标的。")
    md_lines.append("")

    # 3. DEEP_OTM 异常
    md_lines.append("## 3. DEEP_OTM 看涨/看跌成交额及持仓量异常标的")
    md_lines.append(
        f"阈值: 成交额比或持仓量比 > {CONFIG['otm_ratio_threshold']} 或 < 1/{CONFIG['otm_ratio_threshold']}，窗口内异常天数≥{int(CONFIG['min_pct_atm_otm'] * 100)}%")
    if 'otm_stats' in locals() and not otm_stats.empty:
        md_lines.append(
            "| 股票 | 异常天数 | 比例 | 平均成交额比 | 平均持仓量比 | 总Call成交额(百万$) | 总Put成交额(百万$) | IV比值(最近) |")
        md_lines.append(
            "|------|----------|------|--------------|--------------|---------------------|---------------------|-------------|")
        for _, row in otm_stats.head(20).iterrows():
            iv_str = f"{row['iv_ratio']:.2f}" if pd.notna(row['iv_ratio']) else "N/A"
            md_lines.append(
                f"| {row['stock_owner']} | {row['otm_anomaly_days']} | {row['pct']:.1%} | {row['avg_otm_ratio_turn']:.2f} | {row['avg_otm_ratio_oi']:.2f} | {row['total_otm_call_turn'] / 1e6:.2f} | {row['total_otm_put_turn'] / 1e6:.2f} | {iv_str} |")
    else:
        md_lines.append("无符合条件的标的。")
    md_lines.append("")

    # 4. 停止事件
    md_lines.append("## 4. DEEP_ITM 交易停止事件")
    md_lines.append(
        f"定义: 窗口前期活跃比例≥{int(CONFIG['min_pct_active'] * 100)}%，末段连续{CONFIG['stop_consecutive_days']}天无大额深度实值交易")
    if 'stop_df' in locals() and not stop_df.empty:
        md_lines.append("| 股票 | 连续停止天数 | 前期活跃天数 | 前期比例 | 最后活跃日期 | 停止起始日 |")
        md_lines.append("|------|--------------|--------------|----------|--------------|------------|")
        for _, row in stop_df.head(20).iterrows():
            md_lines.append(
                f"| {row['stock_owner']} | {row['stop_consecutive_days']} | {row['active_days_before']}/{row['total_days_before']} | {row['pct_before']:.1%} | {row['last_active_date']} | {row['stop_date_start']} |")
    else:
        md_lines.append("无符合条件的标的。")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("**说明**: IV比值仅在有相近执行价和到期日的配对合约时计算，否则显示N/A。")
    md_lines.append("重要性排序规则: 按比例降序，比例相同按总成交额降序。")

    report_path = CONFIG['output_dir'] / 'persistence_anomaly_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    print(f"Markdown报告已生成: {report_path}")
    print("全部任务完成！")

if __name__ == '__main__':
    main()