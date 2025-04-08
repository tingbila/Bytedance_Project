# config/data_config.py

# 原始字段名
COLUMN_NAMES = [
    "uid", "user_city", "item_id", "author_id", "item_city", "channel",
    "finish", "like", "music_id", "device", "time", "duration_time"
]

# 离散特征
CATEGORICAL_COLS = [
    "uid", "user_city", "item_id", "author_id", "item_city", "channel", "music_id", "device"
]

# 数值特征
NUMERIC_COLS = ["time", "duration_time"]

# 目标列
TARGET_COLS = ["finish", "like"]
