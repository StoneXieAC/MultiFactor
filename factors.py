import pandas as pd
import numpy as np

def cs_rank(s):
    """
    横截面百分位排序
    """
    return s.groupby(level=0).rank(pct=True)

def alpha01(df):
    """过去5日 (最高价 / 收盘价) 的均值"""
    ratio = df['high'] / df['close']
    return ratio.groupby(level=1).transform(lambda x: x.rolling(5).mean())

def alpha02(df):
    """过去5日 (最低价 / 收盘价) 的均值"""
    ratio = df['low'] / df['close']
    return ratio.groupby(level=1).transform(lambda x: x.rolling(5).mean())

def alpha03(df):
    """短期反转因子：昨日收益率的相反数"""
    return -df.groupby(level=1)['label'].shift(1)

def alpha04(df):
    """开盘价与成交量的秩相关（10日窗口，取负）"""
    def corr_open_vol(g):
        return g['open'].rank(pct=True).rolling(10).corr(g['volume'].rank(pct=True))
    out = df.groupby(level=1).apply(corr_open_vol).reset_index(level=0, drop=True)
    return -out

def alpha05(df):
    """日内波幅强度与价格变动 + 价相关性"""
    intraday_change = (df['close'] - df['open'])
    intraday_vol = (df['close'] - df['open']).abs().rolling(10).std()
    corr_close_open = df.groupby(level=1).apply(
        lambda g: g['close'].rolling(10).corr(g['open'])
    ).reset_index(level=0, drop=True)
    score = intraday_vol + intraday_change + corr_close_open
    return -score.groupby(level=0).rank(pct=True)

def alpha06(df):
    """VWAP 与成交量秩相关（5日窗口，取负）"""
    vwap = df['amount'] / df['volume']
    def corr_vwap_vol(g):
        return vwap.loc[g.index].rank(pct=True).rolling(5).corr(g['volume'].rank(pct=True))
    out = df.groupby(level=1).apply(corr_vwap_vol).reset_index(level=0, drop=True)
    return -out

def alpha07(df):
    """6日动量收益率（%）"""
    return (df['close'] - df.groupby(level=1)['close'].shift(6)) / df.groupby(level=1)['close'].shift(6) * 100

def alpha08(df):
    """高价波动率与成交量的关系"""
    high_std_rank = df.groupby(level=1)['high'].rolling(10).std().reset_index(
        level=0, drop=True
    ).groupby(level=0).rank(pct=True)
    corr_high_vol = df.groupby(level=1).apply(
        lambda g: g['high'].rolling(10).corr(g['volume'])
    ).reset_index(level=0, drop=True)
    return -high_std_rank * corr_high_vol

def alpha09(df):
    """开盘价与成交量相关性（10日窗口）"""
    out = df.groupby(level=1).apply(lambda g: g['open'].rolling(10).corr(g['volume']))
    return out.reset_index(level=0, drop=True).set_axis(df.index)

def alpha10(df):
    """成交量变化与收盘变化符号组合"""
    vol_chg = df.groupby(level=1)['volume'].diff()
    close_chg = df.groupby(level=1)['close'].diff()
    return np.sign(vol_chg) * (-close_chg)

def alpha11(df):
    """日内绝对波幅std + 涨跌 + 收开价相关性"""
    vol = df.groupby(level=1).apply(
        lambda g: (g['close'] - g['open']).abs().rolling(5).std()
    ).reset_index(level=0, drop=True)
    corr = df.groupby(level=1).apply(
        lambda g: g['close'].rolling(10).corr(g['open'])
    ).reset_index(level=0, drop=True)
    return -cs_rank(vol.set_axis(df.index) + (df['close'] - df['open']) + corr.set_axis(df.index))

def alpha12(df):
    """高价std × 高价与成交量相关性"""
    vol = df.groupby(level=1)['high'].transform(lambda x: x.rolling(10).std())
    corr = df.groupby(level=1).apply(
        lambda g: g['high'].rolling(10).corr(g['volume'])
    ).reset_index(level=0, drop=True)
    return -cs_rank(vol) * corr.set_axis(df.index)