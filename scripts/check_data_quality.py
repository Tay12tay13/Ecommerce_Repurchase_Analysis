# check_data_quality.py 检查数据质量
import pandas as pd
from scripts.data_loader import load_raw_data, clean_data
from scripts.metrics import extract_purchase

# 1. 加载并清洗数据（使用项目已有函数）
df = load_raw_data()
df_clean = clean_data(df)
purchase = extract_purchase(df_clean)

print("=== 数据质量检查 ===\n")

# 2. 行为类型分布
print("1. 行为类型分布：")
print(df_clean['behavior_type'].value_counts())
print()

# 3. 购买数据中的用户单日最大购买次数
print("2. 单个用户单日最大购买次数（购买行为）：")
daily_purchase = purchase.groupby(['user_id', 'date']).size()
max_daily = daily_purchase.max()
print(f"最大值: {max_daily}")
print(f"超过10次的天数记录数: {(daily_purchase > 10).sum()}")
print()

# 4. 数据时间范围
print("3. 数据时间范围（原始清洗后）：")
print(f"开始时间: {df_clean['time'].min()}")
print(f"结束时间: {df_clean['time'].max()}")
print(f"涉及天数: {df_clean['time'].dt.date.nunique()}")
print()

# 5. 购买记录的日期连续性（是否有明显断档）
print("4. 购买记录按日期分布（前10天）：")
purchase_daily = purchase.groupby(purchase['date']).size()
print(purchase_daily.head(10))
print()

# 6. 用户购买次数分布（检查极值）
user_purchase_counts = purchase.groupby('user_id').size()
print("5. 用户购买次数统计：")
print(user_purchase_counts.describe(percentiles=[.5, .9, .99]))
print(f"购买次数最多用户: {user_purchase_counts.max()} 次")