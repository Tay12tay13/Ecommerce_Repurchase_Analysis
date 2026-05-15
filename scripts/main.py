import sys
import os

# 将项目根目录加入 Python 路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.data_loader import load_raw_data, clean_data
from scripts.metrics import extract_purchase, save_purchase_data
from scripts.analysis import (
    load_purchase_data,
    purchase_frequency_segmentation,
    user_segmentation_by_speed, plot_speed_segmentation,
    first_purchase_behavior, plot_behavior_comparison,
    post_purchase_behavior, plot_post_behavior,
    repurchase_interval, plot_repurchase_interval,
    category_repurchase_rate, plot_top_categories,
    category_same_day_repurchase_rate,
    cross_purchase_speed,
    category_speed_analysis,
    plot_purchase_speed_trend,
    plot_category_speed_bar
)

def main():
    print("电商用户大促期间即时复购分析项目")
    print("-" * 50)

    # 1. 加载与清洗
    print("1. 加载原始数据...")
    df = load_raw_data()
    print(f"   原始数据行数: {len(df)}")
    print("2. 清洗数据...")
    df_clean = clean_data(df)
    print(f"   清洗后行数: {len(df_clean)}")

    # 2. 提取购买记录
    print("3. 提取购买记录...")
    purchase = extract_purchase(df_clean)
    print(f"   购买记录行数: {len(purchase)}")
    print(f"   购买用户数: {purchase['user_id'].nunique() if not purchase.empty else 0}")

    if purchase.empty:
        print("错误：未找到购买记录。")
        return

    # 3. 计算复购率（当日/次日）
    print("4. 计算复购率...")
    total_users = purchase['user_id'].nunique()
    intervals = repurchase_interval(purchase)
    same_day_users = (intervals == 0).sum()
    next_day_users = (intervals == 1).sum()
    same_day_rate = same_day_users / total_users
    next_day_rate = next_day_users / total_users
    print(f"   总购买用户数: {total_users}")
    print(f"   当日复购用户数: {same_day_users}")
    print(f"   当日复购率: {same_day_rate:.2%}")
    print(f"   次日复购用户数: {next_day_users}")
    print(f"   次日复购率: {next_day_rate:.2%}")

    # 4. 保存购买数据
    save_purchase_data(purchase)

    # 5. 深入分析
    print("\n--- 深入分析 ---")
    purchase = load_purchase_data()

    # 5.1 购买次数分层
    print("\n=== 用户购买次数分层 ===")
    freq_seg = purchase_frequency_segmentation(purchase)
    print(freq_seg)

    # 5.2 复购速度分层（仅复购用户）
    print("\n=== 复购速度分层（仅针对有复购用户） ===")
    speed_seg = user_segmentation_by_speed(purchase)
    print(speed_seg.value_counts())
    plot_speed_segmentation(speed_seg)

    # 5.3 行为对比（高复购≥3 vs 低复购1）
    print("\n=== 首次购买前行为对比（高复购≥3次 vs 低复购1次） ===")
    pre_compare = first_purchase_behavior(df_clean, purchase)
    print(pre_compare)
    plot_behavior_comparison(pre_compare)

    print("\n=== 购买后行为对比（高复购≥3次 vs 低复购1次） ===")
    post_compare = post_purchase_behavior(df_clean, purchase)
    print(post_compare)
    plot_post_behavior(post_compare)

    # 5.4 购买间隔分布
    print("\n=== 购买间隔分析 ===")
    intervals_all = repurchase_interval(purchase)
    print(f"平均复购间隔: {intervals_all.mean():.2f} 天")
    print(f"中位数复购间隔: {intervals_all.median():.2f} 天")
    plot_repurchase_interval(intervals_all)

    # 5.5 商品类目分析（整体复购率 + 当日复购率）
    # 5.5 商品类目分析（整体复购率 + 当日复购率）
    print("\n=== 商品类目整体复购率分析 ===")
    cat_rep = category_repurchase_rate(purchase, min_users=50)
    print(cat_rep.head(10))
    plot_top_categories(cat_rep, title='整体复购率最高的类目', filename='top_categories.png')

    print("\n=== 商品类目当日复购率分析（核心） ===")
    cat_same = category_same_day_repurchase_rate(purchase, min_users=30)
    print(cat_same.head(10))
    plot_top_categories(cat_same, top_n=10, title='当日复购率最高的类目', value_col='same_day_rate',
                        filename='same_day_top_categories.png')

    print("\n分析完成！")

    # 6.交叉分析
    print("\n=== 购买次数 × 复购速度 交叉分析 ===")
    cross = cross_purchase_speed(purchase)
    print(cross)

    print("\n=== 商品类目 × 复购速度 交叉分析（仅显示积极比例最高的前10） ===")
    cat_speed = category_speed_analysis(purchase)
    print(cat_speed.head(10))

    # 交叉分析可视化
    print("\n=== 交叉分析可视化 ===")
    # 购买次数趋势图（限制横轴范围，避免过长）
    plot_purchase_speed_trend(cross)
    # 商品类目条形图（前10）
    plot_category_speed_bar(cat_speed, top_n=10)


if __name__ == '__main__':
    main()