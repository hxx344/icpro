import pandas as pd

df = pd.read_parquet('data/underlying/eth_index_60.parquet')
print(df.iloc[0]['close'])
