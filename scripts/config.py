#配置文件
import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'UserBehavior.csv')

# 输出目录
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
TABLES_DIR = os.path.join(OUTPUT_DIR, 'tables')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')

# 参数
SAMPLE_ROWS = 1000000          # 读取前100万行（可改为 None 读取全部）

# 确保输出目录存在
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)