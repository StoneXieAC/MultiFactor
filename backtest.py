import pandas as pd
import numpy as np
from scipy.stats import spearmanr


def run(df, n=5):
    rets_l = []
    rets_ls = []

    for date, subdf in df.groupby(level=0):
        subdf = subdf.copy()

        subdf['group'] = pd.qcut(subdf['factor'], n, labels=False, duplicates='drop')

        top = subdf[subdf['group'] == n - 1]['label'].mean()
        bottom = subdf[subdf['group'] == 0]['label'].mean()

        rets_l.append(top)
        rets_ls.append(top - bottom)

    daily_l = pd.Series(rets_l, name='l')
    daily_ls = pd.Series(rets_ls, name='ls')

    ann_l = (1 + daily_l).prod() ** (252 / len(daily_l)) - 1
    ann_ls = (1 + daily_ls).prod() ** (252 / len(daily_ls)) - 1

    sharpe_l = (
        daily_l.mean() / daily_l.std() * np.sqrt(252) if daily_l.std() > 0 else np.nan
    )
    sharpe_ls = (
        daily_ls.mean() / daily_ls.std() * np.sqrt(252) if daily_ls.std() > 0 else np.nan
    )

    ic_series = df.groupby(level=0).apply(
        lambda x: spearmanr(x['factor'], x['label'])[0]
    )
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    ic_ir = ic_mean / ic_std

    return ic_mean, ic_std, ic_ir, ann_l, ann_ls, sharpe_l, sharpe_ls
