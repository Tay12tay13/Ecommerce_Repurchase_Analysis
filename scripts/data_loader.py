#数据加载与清洗
import pandas as pd
from .config import RAW_DATA_PATH, SAMPLE_ROWS

def load_raw_data(nrows=None):
    """加载原始数据，指定列名（注意列顺序）"""
    if nrows is None:
        nrows = SAMPLE_ROWS
    columns = ['user_id', 'item_id', 'item_category', 'behavior_type', 'time']
    df = pd.read_csv(RAW_DATA_PATH, nrows=nrows, header=None, names=columns)
    return df

def clean_data(df):
    """清洗：时间转换（秒级时间戳）、去重"""
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['date'] = df['time'].dt.date
    df['hour'] = df['time'].dt.hour
    df = df.drop_duplicates()
    return df