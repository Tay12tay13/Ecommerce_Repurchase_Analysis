# scripts/analysis.py 深入分析与可视化
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .config import TABLES_DIR, FIGURES_DIR
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_purchase_data():
    path = os.path.join(TABLES_DIR, 'purchase_data.csv')
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'])
    return df

# ---------- 购买次数分层 ----------
def purchase_frequency_segmentation(purchase):
    """统计高(≥3)、中(=2)、低(=1)购买次数的用户数"""
    user_counts = purchase.groupby('user_id').size()
    high = (user_counts >= 3).sum()
    mid = (user_counts == 2).sum()
    low = (user_counts == 1).sum()
    return pd.Series([high, mid, low], index=['高复购(≥3次)', '中复购(2次)', '低复购(1次)'])

# ---------- 复购速度分层 ----------
def repurchase_interval(purchase):
    """计算有二次购买用户的首次到第二次间隔（天）"""
    purchase = purchase.sort_values(['user_id', 'time'])
    first_two = purchase.groupby('user_id').head(2)
    first_two = first_two[first_two.duplicated(subset='user_id', keep=False)]
    intervals = first_two.groupby('user_id').apply(
        lambda g: (g['time'].iloc[1] - g['time'].iloc[0]).days
    )
    return intervals

def user_segmentation_by_speed(purchase):
    """仅针对有复购用户，按间隔≤1天（积极）和>1天（延迟）分层"""
    intervals = repurchase_interval(purchase)
    seg = {}
    for uid, interval in intervals.items():
        seg[uid] = '积极复购型(≤1天)' if interval <= 1 else '延迟复购型(>1天)'
    return pd.Series(seg, name='segment_speed')

def plot_speed_segmentation(seg_series):
    plt.figure(figsize=(8,5))
    seg_series.value_counts().plot(kind='bar')
    plt.title('复购用户复购速度分层')
    plt.xlabel('类型')
    plt.ylabel('用户数')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'speed_segmentation.png'), dpi=150)
    plt.show()

# ---------- 行为对比（高复购≥3 vs 低复购1） ----------
def first_purchase_behavior(df_clean, purchase):
    """对比高复购(≥3)和低复购(1)用户在首次购买前的行为"""
    # 确保 date 列是日期类型
    df_clean['date'] = pd.to_datetime(df_clean['date']).dt.date
    purchase['date'] = pd.to_datetime(purchase['date']).dt.date

    user_counts = purchase.groupby('user_id').size()
    high_users = user_counts[user_counts >= 3].index
    low_users = user_counts[user_counts == 1].index

    first_date = purchase.groupby('user_id')['date'].min().reset_index()
    first_date.columns = ['user_id', 'first_date']
    merged = df_clean.merge(first_date, on='user_id')
    merged['first_date'] = pd.to_datetime(merged['first_date']).dt.date
    before = merged[merged['date'] < merged['first_date']]

    def avg_actions(user_list):
        sub = before[before['user_id'].isin(user_list)]
        if sub.empty:
            return pd.Series(index=['pv', 'cart', 'fav'], data=0)
        return sub.groupby('user_id')['behavior_type'].value_counts().unstack(fill_value=0).mean()

    high_avg = avg_actions(high_users)
    low_avg = avg_actions(low_users)
    compare = pd.DataFrame({'高复购用户(≥3次)': high_avg, '低复购用户(1次)': low_avg}).fillna(0)
    return compare

def post_purchase_behavior(df_clean, purchase, window_days=7):
    # 强制转换日期列
    df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
    purchase['date'] = pd.to_datetime(purchase['date'], errors='coerce')

    user_counts = purchase.groupby('user_id').size()
    high_users = user_counts[user_counts >= 3].index
    low_users = user_counts[user_counts == 1].index

    first_date = purchase.groupby('user_id')['date'].min().reset_index()
    first_date.columns = ['user_id', 'first_date']
    merged = df_clean.merge(first_date, on='user_id')
    merged['first_date'] = pd.to_datetime(merged['first_date'], errors='coerce')

    # 计算间隔天数（确保两列都是 datetime）
    merged['days_since_first'] = (merged['date'] - merged['first_date']).dt.days
    after = merged[(merged['days_since_first'] > 0) & (merged['days_since_first'] <= window_days)]

    def avg_actions(user_list):
        sub = after[after['user_id'].isin(user_list)]
        if sub.empty:
            return pd.Series(index=['pv', 'cart', 'fav', 'buy'], data=0)
        return sub.groupby('user_id')['behavior_type'].value_counts().unstack(fill_value=0).mean()

    high_avg = avg_actions(high_users)
    low_avg = avg_actions(low_users)
    compare = pd.DataFrame({'高复购用户(≥3次)': high_avg, '低复购用户(1次)': low_avg}).fillna(0)
    return compare

def plot_behavior_comparison(compare):
    compare.plot(kind='bar', figsize=(8,5))
    plt.title('首次购买前平均行为次数对比')
    plt.xlabel('行为类型')
    plt.ylabel('平均次数')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'behavior_comparison.png'), dpi=150)
    plt.show()

def plot_post_behavior(compare):
    compare.plot(kind='bar', figsize=(8,5))
    plt.title('首次购买后7天内平均行为次数对比')
    plt.xlabel('行为类型')
    plt.ylabel('平均次数')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'post_behavior.png'), dpi=150)
    plt.show()

# ---------- 商品类目分析 ----------
def category_repurchase_rate(purchase, min_users=50):
    user_cat = purchase.groupby(['user_id', 'item_category']).size().reset_index(name='count')
    repurchase_users = purchase.groupby('user_id').size()[lambda x: x >= 2].index
    user_cat['is_repurchase'] = user_cat['user_id'].isin(repurchase_users)
    cat_rep = user_cat.groupby('item_category').agg(
        total_users=('user_id', 'nunique'),
        repurchase_users=('is_repurchase', 'sum')
    )
    cat_rep['repurchase_rate'] = cat_rep['repurchase_users'] / cat_rep['total_users']
    cat_rep = cat_rep[cat_rep['total_users'] >= min_users]
    return cat_rep.sort_values('repurchase_rate', ascending=False)

def category_same_day_repurchase_rate(purchase, min_users=30):
    intervals = repurchase_interval(purchase)
    same_day_users = intervals[intervals == 0].index
    user_cat = purchase.groupby(['user_id', 'item_category']).size().reset_index(name='cnt')
    user_cat['is_same_day'] = user_cat['user_id'].isin(same_day_users)
    cat_same = user_cat.groupby('item_category').agg(
        total_users=('user_id', 'nunique'),
        same_day_users=('is_same_day', 'sum')
    )
    cat_same['same_day_rate'] = cat_same['same_day_users'] / cat_same['total_users']
    cat_same = cat_same[cat_same['total_users'] >= min_users]
    return cat_same.sort_values('same_day_rate', ascending=False)

def plot_top_categories(cat_rep, top_n=10, title='整体复购率最高的类目', value_col='repurchase_rate', filename='top_categories.png'):
    top = cat_rep.head(top_n)
    plt.figure(figsize=(10,6))
    plt.barh(top.index.astype(str), top[value_col])
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel('复购率' if value_col == 'repurchase_rate' else '当日复购率')
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存至 {save_path}")
    plt.show()

# ---------- 购买间隔分布图 ----------
def plot_repurchase_interval(intervals):
    plt.figure(figsize=(10,6))
    plt.hist(intervals, bins=range(0, intervals.max()+2), edgecolor='black')
    plt.title('用户首次复购间隔分布')
    plt.xlabel('间隔天数')
    plt.ylabel('用户数')
    plt.xticks(range(0, intervals.max()+1, 1))
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'repurchase_interval.png'), dpi=150)
    plt.show()

# ---------- 交叉分析 ----------
def cross_purchase_speed(purchase):
    """
    交叉分析：购买次数分层 vs 复购速度分层
    返回 DataFrame：每个购买次数分段中，不同复购速度的用户数及比例
    """
    # 获取每个用户的购买次数
    user_counts = purchase.groupby('user_id').size().rename('purchase_count')
    # 获取复购速度分层（仅针对有复购的用户）
    speed_seg = user_segmentation_by_speed(purchase)  # 返回 Series: user_id -> 速度类型
    # 合并两个 Series
    df = pd.DataFrame({'purchase_count': user_counts, 'speed': speed_seg})
    # 交叉表（行=购买次数，列=速度类型）
    cross = pd.crosstab(df['purchase_count'], df['speed'])
    # 添加总计列和比例列
    cross['total'] = cross.sum(axis=1)
    for col in cross.columns[:-1]:
        cross[f'{col}_ratio'] = cross[col] / cross['total']
    return cross

def category_speed_analysis(purchase):
    """
    分析每个商品类目的用户中，积极复购型和延迟复购型的数量和比例
    只统计至少有过一次购买的用户，并按积极复购比例降序排列
    """
    # 获取复购速度分层（仅针对有复购的用户）
    speed_seg = user_segmentation_by_speed(purchase)  # Series: user_id -> speed
    # 获取每个用户首次购买的商品类目（取时间最早的记录）
    first_purchase = purchase.sort_values('time').groupby('user_id').first().reset_index()
    # 只保留需要的列（确保列名正确）
    first_purchase = first_purchase[['user_id', 'item_category']]
    # 合并（注意 speed_seg 的索引是 user_id）
    merged = first_purchase.merge(speed_seg.rename('speed'), left_on='user_id', right_index=True, how='inner')
    # 交叉表
    cross_cat = pd.crosstab(merged['item_category'], merged['speed'])
    cross_cat['total'] = cross_cat.sum(axis=1)
    # 只保留样本量≥30的类目，避免小样本噪音
    cross_cat = cross_cat[cross_cat['total'] >= 30]
    cross_cat['积极复购比例'] = cross_cat['积极复购型(≤1天)'] / cross_cat['total']
    cross_cat = cross_cat.sort_values('积极复购比例', ascending=False)
    return cross_cat


# ===================== 交叉分析可视化 =====================
def plot_purchase_speed_trend(cross_df):
    """
    绘制购买次数 vs 积极复购比例的趋势折线图
    cross_df: cross_purchase_speed 返回的 DataFrame，需包含 '积极复购型(≤1天)_ratio' 列
    """
    # 提取数据（排除购买次数过高的行，只保留前20次足够）
    x = cross_df.index.values
    y = cross_df['积极复购型(≤1天)_ratio'].values

    # 可选：限制 x 轴范围到 20 以内，避免尾部稀疏
    if len(x) > 20:
        x = x[:20]
        y = y[:20]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6, color='#2c7bb6')
    plt.xlabel('购买次数')
    plt.ylabel('积极复购型占比')
    plt.title('不同购买次数下用户复购速度分布')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(min(x), max(x) + 1, 2))
    # 添加数据标签
    for xi, yi in zip(x, y):
        plt.text(xi, yi + 0.02, f'{yi:.1%}', ha='center', va='bottom', fontsize=9)
    save_path = os.path.join(FIGURES_DIR, 'purchase_speed_trend.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存至 {save_path}")
    plt.show()


def plot_category_speed_bar(cat_speed_df, top_n=10):
    """
    绘制商品类目积极复购比例的横向条形图
    cat_speed_df: category_speed_analysis 返回的 DataFrame
    top_n: 显示前 top_n 个类目
    """
    top = cat_speed_df.head(top_n).copy()
    # 按比例升序排列，使条形图从高到低显示
    top = top.sort_values('积极复购比例', ascending=True)
    plt.figure(figsize=(10, 8))
    bars = plt.barh(top.index.astype(str), top['积极复购比例'], color='#fdae61')
    plt.xlabel('积极复购比例')
    plt.title(f'积极复购比例最高的前 {top_n} 个商品类目')
    # 添加数值标签
    for bar, ratio in zip(bars, top['积极复购比例']):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{ratio:.1%}', ha='left', va='center', fontsize=9)
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'category_speed_top10.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存至 {save_path}")
    plt.show()