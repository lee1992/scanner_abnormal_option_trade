"""
统计指定结束日期的多个 markdown 报告文件中出现频率 > 1 的标的
"""
import re
from pathlib import Path
from collections import Counter
import pandas as pd


def analyze_repeated_stocks(
    output_dir: str = r".\output",
    end_date: str = "2026-05-09",
    top_n: int = None,
    report_type: str = "top_diff",
) -> pd.DataFrame:
    """
    统计指定结束日期的 markdown 文件中重复出现的标的

    Args:
        output_dir: 输出目录路径
        end_date: 要统计的结束日期，格式 YYYY-MM-DD，函数会自动转换为 YYYYMMDD
        top_n: 统计的文件数量（从最新的倒数 n 个），None 表示统计所有文件
        report_type: 报告类型 'top_diff' 或 'top_ratio'

    Returns:
        DataFrame，包含标的名称和出现次数，按出现次数降序排列

    示例:
        >>> result = analyze_repeated_stocks(end_date="2026-05-09", top_n=5)
        >>> print(result)
    """

    # 路径转换：YYYY-MM-DD → YYYYMMDD
    date_normalized = end_date#.replace("-", "")

    # 构造查找路径
    important_dir = Path(output_dir) / "important"
    if not important_dir.exists():
        print(f"❌ 目录不存在: {important_dir}")
        return pd.DataFrame()

    # 匹配文件名模式：Report_top_diff_YYYY-MM-DD_YYYYMMDD.md
    pattern = rf"Report_{report_type}_\d{{4}}-\d{{2}}-\d{{2}}_{date_normalized}\.md"

    # 获取所有匹配的文件
    files = sorted([f for f in important_dir.glob("*.md") if re.match(pattern, f.name)])

    if not files:
        print(f"❌ 未找到与模式匹配的文件: {pattern}")
        print(f"   在目录: {important_dir}")
        print(f"   可用文件: {list(important_dir.glob('*.md'))}")
        return pd.DataFrame()

    print(f"✓ 找到 {len(files)} 个文件，结束日期: {end_date}")
    print(f"  报告类型: {report_type}")

    # 如果指定了 top_n，只取最新的 n 个（按开始日期从大到小）
    if top_n and len(files) > top_n:
        files = sorted(files, key=lambda f: f.name, reverse=True)[:top_n]
        files = sorted(files)  # 再按名称排序以便后续显示

    stock_counter = Counter()
    file_stocks_map = {}  # 用于显示每个文件的标的

    # 遍历每个文件，提取标的名称
    for file_path in files:
        stocks_in_file = set()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 匹配 ### N. STOCK_CODE 的模式
            pattern_stock = r"^###\s+\d+\.\s+([A-Z\.]+)\s*$"
            matches = re.findall(pattern_stock, content, re.MULTILINE)

            if not matches:
                print(f"⚠  {file_path.name}: 未找到标的")
                continue

            stocks_in_file = set(matches)
            file_stocks_map[file_path.name] = sorted(list(stocks_in_file))

            # 计数
            for stock in stocks_in_file:
                stock_counter[stock] += 1

            print(f"✓ {file_path.name}: 找到 {len(stocks_in_file)} 个标的")

        except Exception as e:
            print(f"❌ 读取文件失败 {file_path.name}: {e}")
            continue

    # 筛选出现次数 > 1 的标的
    repeated_stocks = {stock: count for stock, count in stock_counter.items() if count > 1}

    if not repeated_stocks:
        print(f"\n⚠  没有找到出现超过1次的标的")
        return pd.DataFrame()

    # 转换为 DataFrame，按出现次数降序
    result_df = pd.DataFrame(
        list(repeated_stocks.items()),
        columns=["标的代码", "出现次数"]
    ).sort_values("出现次数", ascending=False).reset_index(drop=True)

    result_df.index = result_df.index + 1

    # 打印统计信息
    print(f"\n{'═' * 70}")
    print(f"  📊 重复出现的标的统计")
    print(f"{'═' * 70}")
    print(f"  分析文件数: {len(files)}")
    print(f"  总标的数（去重）: {len(stock_counter)}")
    print(f"  重复出现的标的数（>1次）: {len(repeated_stocks)}")
    print(f"\n{result_df.to_string()}\n")

    # 打印文件级别的详情（可选）
    print(f"{'─' * 70}")
    print(f"  📄 各文件标的详情:")
    print(f"{'─' * 70}")
    for filename in sorted(file_stocks_map.keys(), reverse=True):
        stocks = file_stocks_map[filename]
        print(f"  {filename}")
        print(f"    → {', '.join(stocks[:10])}" + ("..." if len(stocks) > 10 else ""))

    print(f"{'═' * 70}\n")

    return result_df


def get_available_end_dates(output_dir: str = r".\output") -> list:
    """
    获取 important 目录中所有可用的结束日期

    Returns:
        按日期从大到小排序的日期列表，格式 YYYY-MM-DD
    """
    important_dir = Path(output_dir) / "important"
    if not important_dir.exists():
        return []

    pattern = r"Report_top_diff_\d{4}-\d{2}-\d{2}_(\d{4}-\d{2}-\d{2})\.md"
    dates = set()

    for file in important_dir.glob("*.md"):
        match = re.search(pattern, file.name)
        if match:
            dates.add(match.group(1))

    return sorted(list(dates), reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 主程序示例
# ═══════════════════════════════════════════════════════════════════════════════
def analysis_repeat_func(end_date,top_n=5):
    print("=" * 70)
    print("  方式 1：指定结束日期 (2026-05-09)，统计倒数 5 个文件")
    print("=" * 70 + "\n")
    result = analyze_repeated_stocks(
        output_dir=r".\output",
        end_date=end_date,
        top_n=top_n,
        report_type="top_diff"
    )
    return result

if __name__ == "__main__":
    import sys

    # 方式 1：指定结束日期和文件数量
    print("=" * 70)
    print("  方式 1：指定结束日期 (2026-05-09)，统计倒数 5 个文件")
    print("=" * 70 + "\n")
    result = analyze_repeated_stocks(
        output_dir=r".\output",
        end_date="2026-05-09",
        top_n=5,
        report_type="top_diff"
    )

    # 方式 2：列出所有可用的结束日期
    print("\n" + "=" * 70)
    print("  所有可用的结束日期:")
    print("=" * 70)
    available_dates = get_available_end_dates()
    if available_dates:
        for i, date in enumerate(available_dates, 1):
            print(f"  {i}. {date}")
    else:
        print("  （未找到文件）")

    # 方式 3：交互式选择
    print("\n" + "=" * 70)
    print("  方式 3：交互式分析")
    print("=" * 70)
    if available_dates:
        print(f"\n  最新的结束日期: {available_dates[0]}")
        user_end_date = available_dates[0]  # 或改为 input() 进行交互
        user_top_n = 3  # 或改为 int(input("  请输入统计的文件数量: "))

        print(f"\n  分析结束日期: {user_end_date}，倒数 {user_top_n} 个文件\n")
        result = analyze_repeated_stocks(
            output_dir=r".\output",
            end_date=user_end_date,
            top_n=user_top_n,
            report_type="top_diff"
        )

        # 导出结果为 CSV（可选）
        if not result.empty:
            output_csv = Path(r".\output") / f"repeated_stocks_{user_end_date}.csv"
            result.to_csv(output_csv, encoding='utf-8-sig', index=True)
            print(f"✓ 结果已导出: {output_csv}\n")
