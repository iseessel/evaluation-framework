import pdb
from google.cloud import bigquery, storage
from datetime import datetime
import pandas as pd
import numpy as np
import os

"""
Creates price features.

NOTE: This is identical to price_features_v2 except we are prediction_Vol

Experimentation Todos:
    1. Implement pivot points.
    2. Add more permnos. Currently only using current SP permnos -- Can also use old sp permnos. ** Might want to only use stocks in S&P at that time **
    3. Experiment with different lags.
    4. Implement trend line volatility.
    5. Global Z.

Implementation Details Todos:
    1. Use calendar details instead of -253 trading days.
    2. First 29 day of z scores are null.
    3. 30 null targets.
"""

START_DATE = '1980-01-01'
TRADING_DAYS = 253
TIME_LAG = 1
RAW_FEATURES = [
    'permno',
    'prediction_date',
    'date',
    'gain_loss',
    'beta',
    'beta_bull',
    'beta_bear',
    'market_correlation',
    'average_daily_return',
    'returns_bull',
    'returns_bear',
    'volatility',
    'kurtosis'
]

FINAL_FEATURES = [
    'permno', 'date', 'prediction_date',
    # 'average_daily_return_global_z', 'beta_global_z',
    # 'beta_bear_global_z', 'beta_bull_global_z', 'gain_loss_global_z',
    # 'kurtosis_global_z', 'market_correlation_global_z', 'returns_bear_global_z',
    # 'returns_bull_global_z', 'volatility_global_z',
    'average_daily_return_local_z', 'beta_local_z', 'beta_bear_local_z',
    'beta_bull_local_z', 'gain_loss_local_z', 'kurtosis_local_z',
    'market_correlation_local_z', 'returns_bear_local_z', 'returns_bull_local_z',
    'volatility_local_z'
]

print(f"Calculating the following features: { FINAL_FEATURES }")

"""
Using daily returns adjusted for cash dividends.
Fetch returns series from Bigquery.
    1. All S&P stocks since 1980 (1505 different stocks).
    2. Remove permnos with 5 or more missing prices (35 permnos).
    3. Remove any rows where prc is null (NB: Returns are always null the first day a stock goes public).
"""

QUERY = f"""
    SELECT
      permno,
      date,
      ret,
      COALESCE((
      SELECT
        MIN(date)
      FROM
        `silicon-badge-274423.financial_datasets.sp_timeseries_daily` std2
      WHERE
        std1.permno = std2.permno
-- Predict 6 months in the future. Note, may not always be exactly 6 months due to weekends/holidays.
        AND std2.date >= DATE_ADD(std1.date, INTERVAL 6 MONTH)), DATE_ADD(std1.date, INTERVAL 6 MONTH))  as prediction_date
    FROM
      `silicon-badge-274423.financial_datasets.sp_timeseries_daily` std1
    WHERE
      permno IN (
      SELECT
        DISTINCT permno
      FROM
        `silicon-badge-274423.features.price_features_v0`
        # SELECT PERMNO FROM `silicon-badge-274423.financial_datasets.sp_constituents_historical` WHERE finish > '{START_DATE}'
        )
      AND date > '{START_DATE}'
      AND permno NOT IN (
      SELECT
        permno
      FROM
        `silicon-badge-274423.financial_datasets.sp_timeseries_daily`
      WHERE
        date > '{START_DATE}'
        AND prc IS NULL
      GROUP BY
        permno
      HAVING
        COUNT(*) > 5 )
      AND prc IS NOT NULL
 """

# Fetch Stock Prices
print("Fetching stock prices. May take a few minutes.")
client = bigquery.Client(project='silicon-badge-274423')
returns_df = client.query(QUERY).to_dataframe()
returns_df = returns_df.sort_values(by=['permno', 'date']).reset_index()

# Adjusted returns are null the first day they are reported.
returns_df.loc[returns_df['ret'].isna(), 'ret'] = 0

# Fetch S&P returns. Use "Value-Weighted Return (includes distributions)".
QUERY = f"""
    SELECT
        PARSE_DATE("%Y%m%d", CAST(caldt as string)) as date, vwretd
    FROM
        `silicon-badge-274423.financial_datasets.sp_index_daily_returns`
    WHERE
        PARSE_DATE("%Y%m%d", CAST(caldt as string)) > '{START_DATE}'
    ORDER BY
      date
"""
print("Fetching S&P returns.")
sp_df = client.query(QUERY).to_dataframe()
sp_df = sp_df.sort_values(by=['date']).reset_index()

"""
    Set 1980-01-01 returns to 0, since we care about returns since 1980.
"""
sp_df.loc[0, 'vwretd'] = 0
merged_df = returns_df.merge(sp_df, on='date', how='left')

"""
Create target volatility.
"""


def fix_nested_index(series, indeces):
    series = series.reset_index(level=[0, 1])
    series = series.drop('permno', axis=1).set_index('level_1')
    series.columns = indeces

    return series


merged_df['log_ret'] = np.log(1 + merged_df['ret'])

by_permno = merged_df.groupby('permno')
six_mos = int(TRADING_DAYS / 2)
target_volatility = by_permno['log_ret'].rolling(
    window=six_mos).std() * np.sqrt(six_mos)
target_volatility = fix_nested_index(target_volatility, ['target_volatility'])
merged_df['target_volatility'] = target_volatility

vols = merged_df[['target_volatility', 'permno', 'date']]
vols.columns = ['target', 'permno', 'date']

merged_df = merged_df.merge(vols, how='left', left_on=[
                            'prediction_date', 'permno'], right_on=['date', 'permno'])
merged_df = merged_df[['permno', 'date_x',
                       'prediction_date', 'ret', 'vwretd', 'target', 'log_ret']]
merged_df.columns = ['permno', 'date', 'prediction_date',
                     'ret', 'vwretd', 'target', 'log_ret']

"""
    Gain Loss %: (Stock[-1] - Stock[0]) * 100 / (Stock[0])
"""
merged_df['cum_ret_stock'] = merged_df['ret'] + 1
by_permno = merged_df.groupby('permno')
pdb.set_trace()
merged_df['cum_ret_stock'] = by_permno.cum_ret_stock.cumprod()
merged_df['gain_loss'] = by_permno.cum_ret_stock.pct_change(
    periods=TRADING_DAYS)

by_permno = merged_df.groupby('permno')

print("Finished calculating gain_loss.")

"""
    Beta = cov(Stock, Market) / Var(Market)
"""
#


def beta(stock_col, market_col):
    cov = by_permno[[stock_col, market_col]].rolling(
        TRADING_DAYS, min_periods=1).cov()
    cov = cov.groupby(level=[0, 1]).last()[stock_col]
    # Need min periods, since bull and bear returns will have many NaN entries (When the market went up/down).
    var = by_permno[market_col].rolling(TRADING_DAYS, min_periods=1).var()

    beta = cov / var
    beta = fix_nested_index(beta, ['beta'])

    return beta


merged_df['beta'] = beta('ret', 'vwretd')

print("Finished calculating beta.")

"""
    'Beta-Bull': Beta of stock when market is up.
    'Beta-Bear 'Beta of stock when market is down.
"""

# Get Bull and Bear Lists of returns.
merged_df['ret_sp_bull'] = merged_df['vwretd']
merged_df.loc[merged_df['ret_sp_bull'] < 0, 'ret_sp_bull'] = None

merged_df['ret_stock_bull'] = merged_df['ret']
merged_df.loc[merged_df['ret_stock_bull'] < 0, 'ret_stock_bull'] = None

merged_df['ret_sp_bear'] = merged_df['vwretd']
merged_df.loc[merged_df['ret_sp_bear'] >= 0, 'ret_sp_bear'] = None

merged_df['ret_stock_bear'] = merged_df['ret']
merged_df.loc[merged_df['ret_stock_bear'] >= 0, 'ret_stock_bear'] = None

merged_df['is_bull'] = merged_df['ret_sp_bull'].notna()
merged_df['is_bear'] = merged_df['ret_sp_bear'].notna()

by_permno = merged_df.groupby('permno')

merged_df['beta_bull'] = beta('ret_stock_bull', 'ret_sp_bull')
merged_df['beta_bear'] = beta('ret_stock_bear', 'ret_sp_bear')

print("Finished calculating beta_bull and beta_bear.")

"""
    'Returns Bull': sum(ReturnsMeanData[3]) / len(ReturnsMeanData[3]),

    NB: Need min periods=1, since ret_stock_bull will have many NaNs.
    We don't use min_periods above, so when we drop nas, we will drop everything properly.
"""

sum_bull = by_permno['ret_stock_bull'].rolling(
    TRADING_DAYS, min_periods=1).sum()
num_bull_days = by_permno.rolling(253)['is_bull'].sum()
returns_bull = sum_bull / num_bull_days
returns_bull = fix_nested_index(returns_bull, ['returns_bull'])

merged_df['returns_bull'] = returns_bull

print("Finished calculating returns_bull.")

"""
    'Returns Bear': sum(ReturnsMeanData[2]) / len(ReturnsMeanData[2])
"""
sum_bear = by_permno['ret_stock_bear'].rolling(
    TRADING_DAYS, min_periods=1).sum()
num_bear_days = by_permno.rolling(253)['is_bear'].sum()
returns_bear = sum_bear / num_bear_days
returns_bear = fix_nested_index(returns_bear, ['returns_bear'])

merged_df['returns_bear'] = returns_bear
merged_df = merged_df.drop(columns=[
                           'is_bull', 'is_bear', 'ret_sp_bear', 'ret_sp_bull', 'ret_stock_bull', 'ret_stock_bear'])

print("Finished calculating returns_bear.")

"""
    Market Correlation: covar(StockS, MarketS) / (math.sqrt(var(StockS) * var(MarketS))),
"""
cov = by_permno[['ret', 'vwretd']].rolling(TRADING_DAYS).cov()
cov = cov.groupby(level=[0, 1]).last()['ret']
stock_var = by_permno['ret'].rolling(TRADING_DAYS).var()
market_var = by_permno['vwretd'].rolling(TRADING_DAYS).var()

market_correlation = cov / np.sqrt(stock_var * market_var)
market_correlation = fix_nested_index(
    market_correlation, ['market_correlation'])
merged_df['market_correlation'] = market_correlation

print("Finished calculating market_correlation.")

"""
    'Average Daily Return': sum(StockS) / len(StockS),
"""

average_daily_return = by_permno['ret'].rolling(
    TRADING_DAYS).sum() / TRADING_DAYS
average_daily_return = fix_nested_index(
    average_daily_return, ['average_daily_return'])
merged_df['average_daily_return'] = average_daily_return

print("Finished calculating average_daily_return.")

"""
    Kurtosis. 4th moment measure of the “tailedness”
    i.e. descriptor of shape of probability distribution of a real-valued random variable.
    In simple terms, one can say it is a measure of how heavy tail is compared to a normal distribution.

    https://www.geeksforgeeks.org/scipy-stats-kurtosis-function-python/
"""

kurt = by_permno['ret'].rolling(TRADING_DAYS).kurt()
kurt = fix_nested_index(kurt, ['kurtosis'])
merged_df['kurtosis'] = kurt

print("Finished calculating kurtosis.")

"""
    Volatility: https://stackoverflow.com/questions/31287552/logarithmic-returns-in-pandas-dataframe
"""

volatility = by_permno['log_ret'].rolling(
    window=TRADING_DAYS).std() * np.sqrt(TRADING_DAYS)
volatility = fix_nested_index(volatility, ['volatility'])
merged_df['volatility'] = volatility

print("Finished calculating Volatility.")
print(f"Finished with {RAW_FEATURES}")

"""
    Z-score transform these windows and add to dataframe.

    1. Global Z score for last 365 days.
    2. Permno Z score for last 365 days.

    NB: If stock has been listed within last 2 years. It will only have price features for last year, since the window is a year.
    We will consider these price features if there are a minimum of 30 periods to compare.
"""

# Drop all rows without values except prediction_date
merged_df = merged_df.dropna(
    subset=[n for n in merged_df if n not in ['prediction_date', 'target']])
numeric_features = set(RAW_FEATURES) - \
    set(['permno', 'date', 'prediction_date'])
features_df = merged_df[numeric_features]

# Global ZScore. TODO: Add Global Zscore. Need to change index to days.
# col_mean = features_df.rolling(window=TRADING_DAYS, min_periods=30).mean()
# col_std = features_df.rolling(window=TRADING_DAYS, min_periods=30).std()
# zscore = (features_df - col_mean)/col_std
# global_cols = [str + '_global_z' for str in zscore.columns]
# zscore.columns = global_cols
# merged_df = merged_df.merge(zscore, how='left', left_index=True, right_index=True)

# Local ZScore (Zscore by_permno)
by_permno = merged_df[numeric_features.union(set(['permno']))].groupby('permno')[
    list(numeric_features)]
col_mean = by_permno.rolling(window=TRADING_DAYS, min_periods=30).mean()
col_mean = fix_nested_index(col_mean, col_mean.columns.tolist())
col_std = by_permno.rolling(window=TRADING_DAYS, min_periods=30).std()
col_std = fix_nested_index(col_std, col_std.columns.tolist())
zscore = (features_df - col_mean) / (col_std)

local_cols = [str + '_local_z' for str in zscore.columns]
zscore.columns = local_cols
merged_df = merged_df.merge(
    zscore, how='left', left_index=True, right_index=True)

# print("Finished calculating local and global z-scores.")
print("Finished calculating local z-scores.")
final_df = merged_df[FINAL_FEATURES + ['target']]

"""
    Upload to Bigquery.
    TODO: This was stalling indefinitely. Have uploaded to bigquery manually.
"""
# TODO: Make this a utils class. Improve this method.
final_df.to_csv('price_features_vol_v3.csv', index=False)
# job_config = bigquery.LoadJobConfig(
#   source_format=bigquery.SourceFormat.CSV, skip_leading_rows=1,
#   autodetect=True,
#   write_disposition='WRITE_TRUNCATE'
# )
#
# with open('temp.csv', "rb") as source_file:
#   job = client.load_table_from_file(source_file, 'silicon-badge-274423.features.price_features_vol_v3', job_config=job_config)
#
# job.result()  # Waits for the job to complete.
#
# os.remove("temp.csv")
#
# print("Uploaded to Bigquery")

"""
    Upload pickle.
    TODO: This was stalling indefinitely. Have uploaded to GCS manually.
"""
final_df.to_pickle("./price_features_vol_v3")
#
# def upload_to_bucket(blob_name, path_to_file, bucket_name):
#     """ Upload data to a bucket"""
#
#     # Explicitly use service account credentials by specifying the private key file.
#     storage_client = storage.Client(project='silicon-badge-274423')
#
#     # print(buckets = list(storage_client.list_buckets())
#
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(blob_name)
#     blob.upload_from_filename(path_to_file)
#
#     #returns a public url
#     return blob.public_url
#
# upload_to_bucket('features/price_features_v2.pkl', './temp.pkl', 'feats')
#
# os.remove("temp.pkl")

"""
    The above code was adapated from the below, created by Aaron Kreiner.
"""
# import datetime as dt
# import pandas_datareader.data as web
# import numpy as np
# import math
# import csv
# import time
# start_time = time.time()
#
# class Stock(object):
#     def __init__(self, ticker, name="Stock"):
#         self.name = name
#         self.ticker = ticker
#         self.prices = None
#
#     def __str__(self):
#         return self.name, self.ticker
#
#     def gen_prices(self, start, end):
#         df = web.DataReader(self.ticker, "yahoo", start, end)
#         self.prices = df
#
#     def standardize(stock):
#         return list(map(lambda i: 0 if stock[i] == 0 else (stock[i + 1] - stock[i]) / (stock[i]) * 100, range(len(stock) - 1)))
#
#     def covar(A, B):
#       if len(A) == 0:
#         return 0
#       else:
#         return np.cov(A, B)[0][1] * (len(A) - 1) / len(A)
#
#     def var(A):
#       return np.var(A)
#
#     def seperate(Market, Stock):
#       BearMarket = list()
#       BullMarket = list()
#       BearStock = list()
#       BullStock = list()
#       Market = standardize(Market)
#       Stock = standardize(Stock)
#       for i in range(len(Market)):
#           if Market[i] >= 0:
#               BullMarket.append(Market[i])
#               BullStock.append(Stock[i])
#           else:
#               BearMarket.append(Market[i])
#               BearStock.append(Stock[i])
#       return [BearMarket, BullMarket, BearStock, BullStock]
#
#     def stratBetaCalc(BearMarket, BullMarket, BearStock, BullStock):
#       BetaBear = covar(BearMarket, BearStock) / var(BearMarket)
#       BetaBull = covar(BullMarket, BullStock) / var(BullMarket)
#       return [BetaBull, BetaBear]
#
#     def stratBeta(Stock, Market):
#       BetaList = seperate(Market, Stock)
#       return stratBetaCalc(BetaList[0], BetaList[1], BetaList[2], BetaList[3])
#
#     def StockStat(StockName, Stock, Market):
#       Stock, Market = trim(Stock, Market)
#       StockS = standardize(Stock)
#       MarketS = standardize(Market)
#       BetaDirList = stratBeta(Stock, Market)
#       ReturnsMeanData = seperate(Market, Stock)
#       StockInfo = {
#         'Stock': StockName,
#         #'Start Price': Stock[0],
#         #'Finish Price': Stock[-1],
#         #'Period High':max(Stock) ,
#         #'Period Low': min(Stock),
#         'Gain/Loss %': str((Stock[-1] - Stock[0]) * 100 / (Stock[0])),
#         'Beta': beta(Stock, Market),
#         'Beta-Bull': BetaDirList[0],
#         'Beta-Bear':BetaDirList[1],
#         'Market Correlation': covar(StockS, MarketS) / (math.sqrt(var(StockS) * var(MarketS))),
#         'Average Daily Return': sum(StockS) / len(StockS),
#         'Returns Bull': sum(ReturnsMeanData[3]) / len(ReturnsMeanData[3]),
#         'Returns Bear': sum(ReturnsMeanData[2]) / len(ReturnsMeanData[2])
#       }
#
#       return list(StockInfo.values()) #Dicts maintian correct order in python 3.6 lol very bad but i'm lazy
#
