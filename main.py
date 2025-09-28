import pandas as pd
from pathlib import Path

input_path = Path('data/data_v1.parquet')
v2_path = Path('data/data_v2.parquet')
train_path = Path('data/data_train.parquet')
test_path = Path('data/data_test.parquet')

if not input_path.exists():
    raise FileNotFoundError(f"Input file {input_path} does not exist")

data = pd.read_parquet(input_path)
columns_to_keep = ['open', 'high', 'low', 'close', 'volume', 'amount', 'label']
data_v2 = data[columns_to_keep].copy()
data_v2.to_parquet(v2_path)
print(f"Filtered data saved to {v2_path}, shape: {data_v2.shape}")

data_v2['date'] = data_v2.index.get_level_values(0)

train_mask = (data_v2['date'] >= '2018-01-01') & (data_v2['date'] <= '2023-12-31')
test_mask = (data_v2['date'] >= '2024-01-01') & (data_v2['date'] <= '2025-12-31')

train_data = data_v2[train_mask].drop(columns=['date'])
test_data = data_v2[test_mask].drop(columns=['date'])

train_data.to_parquet(train_path)
test_data.to_parquet(test_path)

print(f"Training data (2018-2023) saved to {train_path}, shape: {train_data.shape}")
print(f"Test data (2024-2025) saved to {test_path}, shape: {test_data.shape}")