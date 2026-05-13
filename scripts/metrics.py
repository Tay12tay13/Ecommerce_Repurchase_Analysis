# 购买记录提取
import pandas as pd
import os
from .config import TABLES_DIR

def extract_purchase(df):
    """
    从全量数据中提取购买记录，并计算每个用户的首次购买时间及天数差
    """
    purchase = df[df['behavior_type'] == 'buy'].copy()
    if purchase.empty:
        return purchase

    # 确保 time 列是 datetime 类型
    purchase['time'] = pd.to_datetime(purchase['time'])

    # 每个用户首次购买时间
    first_time = purchase.groupby('user_id')['time'].min().reset_index()
    first_time.columns = ['user_id', 'first_time']
    purchase = purchase.merge(first_time, on='user_id')
    # 计算天数差（整数）
    purchase['days_diff'] = (purchase['time'] - purchase['first_time']).dt.days
    return purchase

def save_purchase_data(purchase):
    """保存购买明细数据（用于后续分析）"""
    if purchase.empty:
        print("无购买数据，不保存。")
        return
    path = os.path.join(TABLES_DIR, 'purchase_data.csv')
    purchase.to_csv(path, index=False)
    print(f"购买数据已保存至 {path}")