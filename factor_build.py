import numpy as np
import pandas as pd
import factors as F
import inspect
import warnings

warnings.filterwarnings("ignore")


def run(data_path, output_path):
    df = pd.read_parquet(data_path)
    df = df.sort_index()
    print(f"Computing from: {data_path}")
    print(f"Data loaded successfully: {df.shape}")
    if 'vwap' not in df.columns:
        if 'amount' in df.columns and 'volume' in df.columns:
            df['vwap'] = (df['amount'] / df['volume']).replace([np.inf, -np.inf], 0).fillna(0)
        else:
            raise ValueError("Data missing 'vwap' and cannot be computed from 'amount'/'volume'")
    funcs = []
    for name, func in inspect.getmembers(F, inspect.isfunction):
        if name.startswith('alpha'):
            funcs.append((name, func))
    factors = []
    for name, func in funcs:
        print(f"Computing factor: {name}")
        fac = func(df)
        fac = pd.Series(fac, index=df.index)
        fac.name = name
        factors.append(fac)

    factor_df = pd.concat(factors, axis=1)

    factor_df['label'] = df['label']
    factor_df = factor_df.dropna(how='any')

    print(f"Factor computation completed: {factor_df.shape}")

    factor_df.to_parquet(output_path)
    print(f"Factor library saved to: {output_path}")


if __name__ == '__main__':
    data_train = './data/data_train.parquet'
    output_train = './data/factor_train.parquet'
    run(data_train, output_train)

    data_test = './data/data_test.parquet'
    output_test = './data/factor_test.parquet'
    run(data_test, output_test)
