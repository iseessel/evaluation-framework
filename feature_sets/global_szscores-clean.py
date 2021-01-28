#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pdb
from google.cloud import bigquery, storage
from datetime import datetime
import pandas as pd
import numpy as np
import os

"""
Ignore Warnings
"""

import warnings
warnings.filterwarnings("ignore")

import os


"""
Setting Credentials for Google Authentication
"""
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=r"/Users/akreiner/desktop/My First Project-d29575d47503.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=r"C:\Users\DELL\Desktop\NYU\DSGA 1001\Project\My First Project-3e25a41d4780.json"


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

START_DATE = '1970-01-01'
TRADING_DAYS = 253
six_mos = int(TRADING_DAYS/2)
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
    'kurtosis',
    'RSI_3', 'RSI_7', 'RSI_14', 'RSI_28', 'RSI_84', 'RSI_168',
     'pivot_max_2_indicator',
 'pivot_max_2_days_since_last',
 'pivot_max_2_returns_since_last',
 'pivot_max_2_any_pivot_indicator',
 'pivot_max_2_rolling_3mo',
 'pivot_max_2_rolling_6mo',
 'pivot_max_2_rolling_1yr',
 'pivot_max_2_rolling_2yr',
 'pivot_max_3_indicator',
 'pivot_max_3_days_since_last',
 'pivot_max_3_returns_since_last',
 'pivot_max_3_any_pivot_indicator',
 'pivot_max_3_rolling_3mo',
 'pivot_max_3_rolling_6mo',
 'pivot_max_3_rolling_1yr',
 'pivot_max_3_rolling_2yr',
 'pivot_max_5_indicator',
 'pivot_max_5_days_since_last',
 'pivot_max_5_returns_since_last',
 'pivot_max_5_any_pivot_indicator',
 'pivot_max_5_rolling_3mo',
 'pivot_max_5_rolling_6mo',
 'pivot_max_5_rolling_1yr',
 'pivot_max_5_rolling_2yr',
 'pivot_max_10_indicator',
 'pivot_max_10_days_since_last',
 'pivot_max_10_returns_since_last',
 'pivot_max_10_any_pivot_indicator',
 'pivot_max_10_rolling_3mo',
 'pivot_max_10_rolling_6mo',
 'pivot_max_10_rolling_1yr',
 'pivot_max_10_rolling_2yr',
 'pivot_max_15_indicator',
 'pivot_max_15_days_since_last',
 'pivot_max_15_returns_since_last',
 'pivot_max_15_any_pivot_indicator',
 'pivot_max_15_rolling_3mo',
 'pivot_max_15_rolling_6mo',
 'pivot_max_15_rolling_1yr',
 'pivot_max_15_rolling_2yr',
 'pivot_max_20_indicator',
 'pivot_max_20_days_since_last',
 'pivot_max_20_returns_since_last',
 'pivot_max_20_any_pivot_indicator',
 'pivot_max_20_rolling_3mo',
 'pivot_max_20_rolling_6mo',
 'pivot_max_20_rolling_1yr',
 'pivot_max_20_rolling_2yr',
 'pivot_min_2_indicator',
 'pivot_min_2_days_since_last',
 'pivot_min_2_returns_since_last',
 'pivot_min_2_any_pivot_indicator',
 'pivot_min_2_rolling_3mo',
 'pivot_min_2_rolling_6mo',
 'pivot_min_2_rolling_1yr',
 'pivot_min_2_rolling_2yr',
 'pivot_min_3_indicator',
 'pivot_min_3_days_since_last',
 'pivot_min_3_returns_since_last',
 'pivot_min_3_any_pivot_indicator',
 'pivot_min_3_rolling_3mo',
 'pivot_min_3_rolling_6mo',
 'pivot_min_3_rolling_1yr',
 'pivot_min_3_rolling_2yr',
 'pivot_min_5_indicator',
 'pivot_min_5_days_since_last',
 'pivot_min_5_returns_since_last',
 'pivot_min_5_any_pivot_indicator',
 'pivot_min_5_rolling_3mo',
 'pivot_min_5_rolling_6mo',
 'pivot_min_5_rolling_1yr',
 'pivot_min_5_rolling_2yr',
 'pivot_min_10_indicator',
 'pivot_min_10_days_since_last',
 'pivot_min_10_returns_since_last',
 'pivot_min_10_any_pivot_indicator',
 'pivot_min_10_rolling_3mo',
 'pivot_min_10_rolling_6mo',
 'pivot_min_10_rolling_1yr',
 'pivot_min_10_rolling_2yr',
 'pivot_min_15_indicator',
 'pivot_min_15_days_since_last',
 'pivot_min_15_returns_since_last',
 'pivot_min_15_any_pivot_indicator',
 'pivot_min_15_rolling_3mo',
 'pivot_min_15_rolling_6mo',
 'pivot_min_15_rolling_1yr',
 'pivot_min_15_rolling_2yr',
 'pivot_min_20_indicator',
 'pivot_min_20_days_since_last',
 'pivot_min_20_returns_since_last',
 'pivot_min_20_any_pivot_indicator',
 'pivot_min_20_rolling_3mo',
 'pivot_min_20_rolling_6mo',
 'pivot_min_20_rolling_1yr',
 'pivot_min_20_rolling_2yr',
 'pivot_cutoff_indicator_2_rolling_3mo',
 'pivot_cutoff_indicator_2_rolling_6mo',
 'pivot_cutoff_indicator_2_rolling_1yr',
 'pivot_cutoff_indicator_2_rolling_2yr',
 'pivot_cutoff_indicator_3_rolling_3mo',
 'pivot_cutoff_indicator_3_rolling_6mo',
 'pivot_cutoff_indicator_3_rolling_1yr',
 'pivot_cutoff_indicator_3_rolling_2yr',
 'pivot_cutoff_indicator_5_rolling_3mo',
 'pivot_cutoff_indicator_5_rolling_6mo',
 'pivot_cutoff_indicator_5_rolling_1yr',
 'pivot_cutoff_indicator_5_rolling_2yr',
 'pivot_cutoff_indicator_10_rolling_3mo',
 'pivot_cutoff_indicator_10_rolling_6mo',
 'pivot_cutoff_indicator_10_rolling_1yr',
 'pivot_cutoff_indicator_10_rolling_2yr',
 'pivot_cutoff_indicator_15_rolling_3mo',
 'pivot_cutoff_indicator_15_rolling_6mo',
 'pivot_cutoff_indicator_15_rolling_1yr',
 'pivot_cutoff_indicator_15_rolling_2yr',
 'pivot_cutoff_indicator_20_rolling_3mo',
 'pivot_cutoff_indicator_20_rolling_6mo',
 'pivot_cutoff_indicator_20_rolling_1yr',
 'pivot_cutoff_indicator_20_rolling_2yr',
]

FINAL_FEATURES = [
#     'index_x',
 'permno',
 'date',
#  'ret',
#  'adjusted_vol',
 'prediction_date',
#  'bidlo',
#  'askhi',
#  'adjusted_prc',
#  'target',
#  'index_y',
#  'vwretd',
#  'RSI_3',
#  'RSI_7',
#  'RSI_14',
#  'RSI_28',
#  'RSI_84',
#  'RSI_168',
 'pivot_max_2_indicator',
#  'pivot_max_2_days_since_last',
#  'pivot_max_2_returns_since_last',
 'pivot_max_2_any_pivot_indicator',
#  'pivot_max_2_rolling_3mo',
#  'pivot_max_2_rolling_6mo',
#  'pivot_max_2_rolling_1yr',
#  'pivot_max_2_rolling_2yr',
 'pivot_max_3_indicator',
#  'pivot_max_3_days_since_last',
#  'pivot_max_3_returns_since_last',
 'pivot_max_3_any_pivot_indicator',
#  'pivot_max_3_rolling_3mo',
#  'pivot_max_3_rolling_6mo',
#  'pivot_max_3_rolling_1yr',
#  'pivot_max_3_rolling_2yr',
 'pivot_max_5_indicator',
#  'pivot_max_5_days_since_last',
#  'pivot_max_5_returns_since_last',
 'pivot_max_5_any_pivot_indicator',
#  'pivot_max_5_rolling_3mo',
#  'pivot_max_5_rolling_6mo',
#  'pivot_max_5_rolling_1yr',
#  'pivot_max_5_rolling_2yr',
 'pivot_max_10_indicator',
#  'pivot_max_10_days_since_last',
#  'pivot_max_10_returns_since_last',
 'pivot_max_10_any_pivot_indicator',
#  'pivot_max_10_rolling_3mo',
#  'pivot_max_10_rolling_6mo',
#  'pivot_max_10_rolling_1yr',
#  'pivot_max_10_rolling_2yr',
 'pivot_max_15_indicator',
#  'pivot_max_15_days_since_last',
#  'pivot_max_15_returns_since_last',
 'pivot_max_15_any_pivot_indicator',
#  'pivot_max_15_rolling_3mo',
#  'pivot_max_15_rolling_6mo',
#  'pivot_max_15_rolling_1yr',
#  'pivot_max_15_rolling_2yr',
 'pivot_max_20_indicator',
#  'pivot_max_20_days_since_last',
#  'pivot_max_20_returns_since_last',
 'pivot_max_20_any_pivot_indicator',
#  'pivot_max_20_rolling_3mo',
#  'pivot_max_20_rolling_6mo',
#  'pivot_max_20_rolling_1yr',
#  'pivot_max_20_rolling_2yr',
 'pivot_min_2_indicator',
#  'pivot_min_2_days_since_last',
#  'pivot_min_2_returns_since_last',
 'pivot_min_2_any_pivot_indicator',
#  'pivot_min_2_rolling_3mo',
#  'pivot_min_2_rolling_6mo',
#  'pivot_min_2_rolling_1yr',
#  'pivot_min_2_rolling_2yr',
 'pivot_min_3_indicator',
#  'pivot_min_3_days_since_last',
#  'pivot_min_3_returns_since_last',
 'pivot_min_3_any_pivot_indicator',
#  'pivot_min_3_rolling_3mo',
#  'pivot_min_3_rolling_6mo',
#  'pivot_min_3_rolling_1yr',
#  'pivot_min_3_rolling_2yr',
 'pivot_min_5_indicator',
#  'pivot_min_5_days_since_last',
#  'pivot_min_5_returns_since_last',
 'pivot_min_5_any_pivot_indicator',
#  'pivot_min_5_rolling_3mo',
#  'pivot_min_5_rolling_6mo',
#  'pivot_min_5_rolling_1yr',
#  'pivot_min_5_rolling_2yr',
 'pivot_min_10_indicator',
#  'pivot_min_10_days_since_last',
#  'pivot_min_10_returns_since_last',
#  'pivot_min_10_any_pivot_indicator',
#  'pivot_min_10_rolling_3mo',
# #  'pivot_min_10_rolling_6mo',
# #  'pivot_min_10_rolling_1yr',
# #  'pivot_min_10_rolling_2yr',
#  'pivot_min_15_indicator',
#  'pivot_min_15_days_since_last',
#  'pivot_min_15_returns_since_last',
 'pivot_min_15_any_pivot_indicator',
#  'pivot_min_15_rolling_3mo',
#  'pivot_min_15_rolling_6mo',
#  'pivot_min_15_rolling_1yr',
#  'pivot_min_15_rolling_2yr',
 'pivot_min_20_indicator',
#  'pivot_min_20_days_since_last',
#  'pivot_min_20_returns_since_last',
 'pivot_min_20_any_pivot_indicator',
#  'pivot_min_20_rolling_3mo',
#  'pivot_min_20_rolling_6mo',
#  'pivot_min_20_rolling_1yr',
#  'pivot_min_20_rolling_2yr',
 'pivot_cutoff_indicator_2_rolling_3mo',
 'pivot_cutoff_indicator_2_rolling_6mo',
 'pivot_cutoff_indicator_2_rolling_1yr',
 'pivot_cutoff_indicator_2_rolling_2yr',
 'pivot_cutoff_indicator_3_rolling_3mo',
 'pivot_cutoff_indicator_3_rolling_6mo',
 'pivot_cutoff_indicator_3_rolling_1yr',
 'pivot_cutoff_indicator_3_rolling_2yr',
 'pivot_cutoff_indicator_5_rolling_3mo',
 'pivot_cutoff_indicator_5_rolling_6mo',
 'pivot_cutoff_indicator_5_rolling_1yr',
 'pivot_cutoff_indicator_5_rolling_2yr',
 'pivot_cutoff_indicator_10_rolling_3mo',
 'pivot_cutoff_indicator_10_rolling_6mo',
 'pivot_cutoff_indicator_10_rolling_1yr',
 'pivot_cutoff_indicator_10_rolling_2yr',
 'pivot_cutoff_indicator_15_rolling_3mo',
 'pivot_cutoff_indicator_15_rolling_6mo',
 'pivot_cutoff_indicator_15_rolling_1yr',
 'pivot_cutoff_indicator_15_rolling_2yr',
 'pivot_cutoff_indicator_20_rolling_3mo',
 'pivot_cutoff_indicator_20_rolling_6mo',
 'pivot_cutoff_indicator_20_rolling_1yr',
 'pivot_cutoff_indicator_20_rolling_2yr',
#  'log_ret',
#  'target_vol',
#  'cum_ret_stock',
#  'gain_loss',
#  'beta',
#  'beta_bull',
#  'beta_bear',
#  'returns_bull',
#  'returns_bear',
#  'market_correlation',
#  'average_daily_return',
#  'kurtosis',
#  'volatility',
 'pivot_max_2_rolling_2yr_local_z',
 'returns_bear_local_z',
 'pivot_min_20_returns_since_last_local_z',
 'pivot_min_10_rolling_6mo_local_z',
 'market_correlation_local_z',
 'pivot_min_10_returns_since_last_local_z',
 'pivot_min_5_rolling_2yr_local_z',
 'pivot_min_15_returns_since_last_local_z',
 'pivot_min_20_rolling_2yr_local_z',
 'beta_local_z',
 'pivot_max_5_rolling_1yr_local_z',
 'kurtosis_local_z',
 'volatility_local_z',
 'pivot_min_10_rolling_1yr_local_z',
 'pivot_min_2_rolling_6mo_local_z',
 'pivot_min_20_rolling_1yr_local_z',
 'pivot_max_3_returns_since_last_local_z',
 'pivot_max_20_rolling_3mo_local_z',
 'pivot_max_5_rolling_2yr_local_z',
 'pivot_max_15_rolling_3mo_local_z',
 'pivot_min_15_rolling_2yr_local_z',
 'gain_loss_local_z',
 'pivot_max_3_rolling_3mo_local_z',
 'pivot_min_10_rolling_2yr_local_z',
 'pivot_max_20_returns_since_last_local_z',
 'RSI_168_local_z',
 'pivot_max_10_rolling_1yr_local_z',
 'beta_bull_local_z',
 'pivot_min_20_rolling_6mo_local_z',
 'pivot_min_15_rolling_1yr_local_z',
 'pivot_min_3_rolling_2yr_local_z',
 'pivot_max_2_rolling_6mo_local_z',
 'average_daily_return_local_z',
 'pivot_min_15_rolling_6mo_local_z',
 'pivot_min_2_rolling_3mo_local_z',
 'pivot_min_10_rolling_3mo_local_z',
 'pivot_max_2_rolling_1yr_local_z',
 'RSI_14_local_z',
 'pivot_min_2_days_since_last_local_z',
 'pivot_max_20_days_since_last_local_z',
 'pivot_max_15_rolling_6mo_local_z',
 'beta_bear_local_z',
 'pivot_max_15_rolling_1yr_local_z',
 'pivot_max_10_rolling_2yr_local_z',
 'pivot_max_15_days_since_last_local_z',
 'pivot_max_20_rolling_6mo_local_z',
 'pivot_min_5_days_since_last_local_z',
 'RSI_84_local_z',
 'returns_bull_local_z',
 'pivot_max_3_rolling_2yr_local_z',
 'pivot_max_5_returns_since_last_local_z',
 'pivot_min_2_returns_since_last_local_z',
 'pivot_min_20_rolling_3mo_local_z',
 'pivot_max_3_days_since_last_local_z',
 'RSI_3_local_z',
 'pivot_max_15_rolling_2yr_local_z',
 'pivot_max_5_rolling_3mo_local_z',
 'pivot_max_2_rolling_3mo_local_z',
 'pivot_max_20_rolling_2yr_local_z',
 'pivot_min_5_returns_since_last_local_z',
 'pivot_max_20_rolling_1yr_local_z',
 'pivot_min_3_rolling_6mo_local_z',
 'pivot_max_3_rolling_1yr_local_z',
 'pivot_min_3_returns_since_last_local_z',
 'pivot_min_2_rolling_1yr_local_z',
 'pivot_min_10_days_since_last_local_z',
 'pivot_min_3_rolling_3mo_local_z',
 'pivot_min_5_rolling_1yr_local_z',
 'pivot_min_3_rolling_1yr_local_z',
 'pivot_max_5_days_since_last_local_z',
 'pivot_min_15_rolling_3mo_local_z',
 'pivot_max_5_rolling_6mo_local_z',
 'pivot_min_5_rolling_3mo_local_z',
 'pivot_min_3_days_since_last_local_z',
 'RSI_7_local_z',
 'pivot_max_10_days_since_last_local_z',
 'pivot_max_3_rolling_6mo_local_z',
 'pivot_min_5_rolling_6mo_local_z',
 'pivot_max_10_rolling_3mo_local_z',
 'pivot_min_20_days_since_last_local_z',
 'pivot_max_2_returns_since_last_local_z',
 'pivot_max_15_returns_since_last_local_z',
 'RSI_28_local_z',
 'pivot_max_2_days_since_last_local_z',
 'pivot_min_2_rolling_2yr_local_z',
 'pivot_min_15_days_since_last_local_z',
 'pivot_max_10_returns_since_last_local_z',
 'pivot_max_10_rolling_6mo_local_z',
 'pivot_max_2_rolling_2yr_global_z',
 'returns_bear_global_z',
 'pivot_min_20_returns_since_last_global_z',
 'pivot_min_10_rolling_6mo_global_z',
 'market_correlation_global_z',
 'pivot_min_10_returns_since_last_global_z',
 'pivot_min_5_rolling_2yr_global_z',
 'pivot_min_15_returns_since_last_global_z',
 'pivot_min_20_rolling_2yr_global_z',
 'beta_global_z',
 'pivot_max_5_rolling_1yr_global_z',
 'kurtosis_global_z',
 'volatility_global_z',
 'pivot_min_10_rolling_1yr_global_z',
 'pivot_min_2_rolling_6mo_global_z',
 'pivot_min_20_rolling_1yr_global_z',
 'pivot_max_3_returns_since_last_global_z',
 'pivot_max_20_rolling_3mo_global_z',
 'pivot_max_5_rolling_2yr_global_z',
 'pivot_max_15_rolling_3mo_global_z',
 'pivot_min_15_rolling_2yr_global_z',
 'gain_loss_global_z',
 'pivot_max_3_rolling_3mo_global_z',
 'pivot_min_10_rolling_2yr_global_z',
 'pivot_max_20_returns_since_last_global_z',
 'RSI_168_global_z',
 'pivot_max_10_rolling_1yr_global_z',
 'beta_bull_global_z',
 'pivot_min_20_rolling_6mo_global_z',
 'pivot_min_15_rolling_1yr_global_z',
 'pivot_min_3_rolling_2yr_global_z',
 'pivot_max_2_rolling_6mo_global_z',
 'average_daily_return_global_z',
 'pivot_min_15_rolling_6mo_global_z',
 'pivot_min_2_rolling_3mo_global_z',
 'pivot_min_10_rolling_3mo_global_z',
 'pivot_max_2_rolling_1yr_global_z',
 'RSI_14_global_z',
 'pivot_min_2_days_since_last_global_z',
 'pivot_max_20_days_since_last_global_z',
 'pivot_max_15_rolling_6mo_global_z',
 'beta_bear_global_z',
 'pivot_max_15_rolling_1yr_global_z',
 'pivot_max_10_rolling_2yr_global_z',
 'pivot_max_15_days_since_last_global_z',
 'pivot_max_20_rolling_6mo_global_z',
 'pivot_min_5_days_since_last_global_z',
 'RSI_84_global_z',
 'returns_bull_global_z',
 'pivot_max_3_rolling_2yr_global_z',
 'pivot_max_5_returns_since_last_global_z',
 'pivot_min_2_returns_since_last_global_z',
 'pivot_min_20_rolling_3mo_global_z',
 'pivot_max_3_days_since_last_global_z',
 'RSI_3_global_z',
 'pivot_max_15_rolling_2yr_global_z',
 'pivot_max_5_rolling_3mo_global_z',
 'pivot_max_2_rolling_3mo_global_z',
 'pivot_max_20_rolling_2yr_global_z',
 'pivot_min_5_returns_since_last_global_z',
 'pivot_max_20_rolling_1yr_global_z',
 'pivot_min_3_rolling_6mo_global_z',
 'pivot_max_3_rolling_1yr_global_z',
 'pivot_min_3_returns_since_last_global_z',
 'pivot_min_2_rolling_1yr_global_z',
 'pivot_min_10_days_since_last_global_z',
 'pivot_min_3_rolling_3mo_global_z',
 'pivot_min_5_rolling_1yr_global_z',
 'pivot_min_3_rolling_1yr_global_z',
 'pivot_max_5_days_since_last_global_z',
 'pivot_min_15_rolling_3mo_global_z',
 'pivot_max_5_rolling_6mo_global_z',
 'pivot_min_5_rolling_3mo_global_z',
 'pivot_min_3_days_since_last_global_z',
 'RSI_7_global_z',
 'pivot_max_10_days_since_last_global_z',
 'pivot_max_3_rolling_6mo_global_z',
 'pivot_min_5_rolling_6mo_global_z',
 'pivot_max_10_rolling_3mo_global_z',
 'pivot_min_20_days_since_last_global_z',
 'pivot_max_2_returns_since_last_global_z',
 'pivot_max_15_returns_since_last_global_z',
 'RSI_28_global_z',
 'pivot_max_2_days_since_last_global_z',
 'pivot_min_2_rolling_2yr_global_z',
 'pivot_min_15_days_since_last_global_z',
 'pivot_max_10_returns_since_last_global_z',
 'pivot_max_10_rolling_6mo_global_z']

print(f"Calculating the following features: { FINAL_FEATURES }")

"""
Using daily returns adjusted for cash dividends.
Fetch returns series from Bigquery.
    1. All S&P stocks since 1980 (1505 different stocks).
    2. Remove permnos with 5 or more missing prices (35 permnos).
    3. Remove any rows where prc is null (NB: Returns are always null the first day a stock goes public).
"""

"""
@TODO: Change QUERY to get data for all the permnos instead of just two permnos, 
change line 580 delete AND sp_daily_features.permno IN ('14593','13407') 
"""

QUERY = f"""
WITH
  sp_daily_features AS (
  SELECT
    date,
    std1.permno,
    askhi,
    bidlo,
    ticker,
    ret,
    CASE
      # When prc is 0 it is not reported. Sometimes price is negative when its an estimate of the closing price.
      WHEN prc = 0 THEN NULL
    ELSE
    ABS(prc/cfacpr)
  END
    AS adjusted_prc,
    COALESCE((
      SELECT
        MIN(date)
      FROM
        `silicon-badge-274423.financial_datasets.sp_timeseries_daily` std2
      WHERE
        std1.permno = std2.permno
        -- Predict 6 months in the future. Note, may not always be exactly 6 months due to weekends/holidays.
        AND std2.date >= DATE_ADD(std1.date, INTERVAL 6 MONTH)),
      DATE_ADD(std1.date, INTERVAL 6 MONTH)) AS prediction_date
  FROM
    `silicon-badge-274423.financial_datasets.sp_timeseries_daily` std1
  LEFT JOIN
     `silicon-badge-274423.financial_datasets.sp_constituents_historical` historical
  ON
    std1.permno = historical.permno
  WHERE
    std1.date <= historical.finish
  ORDER BY
    std1.permno,
    date )
SELECT
  DISTINCT
  sp_daily_features.permno,
  sp_daily_features.date,
  sp_daily_features.ret,
  sp_daily_features.prediction_date,
  sp_daily_features.bidlo,
  sp_daily_features.askhi,
  sp_daily_features.adjusted_prc
FROM
  sp_daily_features
WHERE
    sp_daily_features.date >= '{ START_DATE }' AND sp_daily_features.permno IN ('14593','13407')
"""

# Fetch Stock Prices
print("Fetching stock prices. May take a few minutes.")
client = bigquery.Client(project='silicon-badge-274423')
returns_df = client.query(QUERY).to_dataframe()

"""
    FOR TESTING: SAVE returns_df as csv, comment out the above lines, and then load the csv here.
"""


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


# Saving the CSV file to Unit Test the code for Pivot Points
# merged_df.to_csv("merged_df_new.csv")
"""
    Create Pivot Points.
"""

# Step 1. Create isPivot indicator variables for time series for different.

# Step 2. Use rolling function to create pivot points for each time series day.
# Create a dictionary of key- permnos and value - dataframes of that permno

ls_permno = list(set(merged_df['permno']))
merged_df = merged_df[merged_df['permno'].isin(ls_permno)]

dic_permno = {}
for permn in ls_permno:
    dic_permno[permn] = merged_df[merged_df['permno'] == permn]

def computeRSI (data, time_window):
    diff = data.diff(1).dropna()        # diff in one field(one day)

    #this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff

    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[ diff>0 ]

    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[ diff < 0 ]

    # check pandas documentation for ewm
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
    # values are related to exponential decay
    # we set com=time_window-1 so we get decay alpha=1/time_window
    up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()

    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    return rsi

ls_windowsize = [3,7,14,28,28*3,28*6]

for permn in ls_permno:
    df = dic_permno[permn]
    for window_size in ls_windowsize:
        df['RSI'+"_"+str(window_size)] = computeRSI(df['adjusted_prc'], window_size)

def computePivotMax (df, time_window):
    dfcopy=df.copy()

    pivots=[]
    dates=[]
    ls_index_pivot = []
    ls_pivot_indicator = []
    counter=0
    lastPivot=0

    Rangev=[0]*time_window*2
    dateRangev=[0]*time_window*2

    dfcopy["pivot_max_"+str(time_window)+"_indicator"]=0

    ls_days_since_last_pivot = []

    return_since_last_pivot = []

    ls_is_any_pivot = []

    flag_pivot = False

    lastPivotDate = None

    lastPivotPrice = None

    for i in dfcopy.index:
        currentMax=max(Rangev,default=0)
        value=round(dfcopy["askhi"][i],2)

            #value=round(df["Low"][i],2)

        Rangev=Rangev[1:time_window*2-1]
        Rangev.append(value)

        dateRangev=dateRangev[1:time_window*2-1]
        date = dfcopy["date"][i]
        dateRangev.append(i)

        if currentMax==max(Rangev,default=0):
            counter=counter+1
        else:
            counter=0
        if counter==time_window:
            flag_pivot = True
            lastPivot=currentMax
            dateloc=Rangev.index(lastPivot)
            lastDate=dateRangev[dateloc]
            pivots.append(lastPivot)
            dates.append(lastDate)
            ls_index_pivot.append(i)

            lastPivotDate = dfcopy["date"][i]
            lastPivotPrice = dfcopy["adjusted_prc"][i]
            dfcopy["pivot_max_"+str(time_window)+"_indicator"][i]=1

        if flag_pivot == False:
            ls_days_since_last_pivot.append(0)
            return_since_last_pivot.append(0)
            ls_is_any_pivot.append(1)
        else:
            days_since = (pd.to_datetime(dfcopy["date"][i]) - pd.to_datetime(lastPivotDate)).days
            ls_days_since_last_pivot.append(days_since)
            if lastPivotPrice == 0 or lastPivotPrice == None or dfcopy["adjusted_prc"][i] == None:
                return_since = 0
            else:
                return_since = (dfcopy["adjusted_prc"][i] - lastPivotPrice)/lastPivotPrice
            return_since_last_pivot.append(return_since)
            ls_is_any_pivot.append(0)

    dic_pivot_returns = {"pivot_indictor":dfcopy["pivot_max_"+str(time_window)+"_indicator"],
    "days_since_pivot":ls_days_since_last_pivot,
    "return_since_pivot":return_since_last_pivot,
    "ls_is_any_pivot":ls_is_any_pivot}
    return dic_pivot_returns

def computePivotMin (df, time_window):
    dfcopy=df.copy()

    pivots=[]
    dates=[]
    ls_index_pivot = []
    ls_pivot_indicator = []
    counter=0
    lastPivot=0

    Rangev=[0]*time_window*2
    dateRangev=[0]*time_window*2

    dfcopy["pivot_min_"+str(time_window)+"_indicator"]=0

    ls_days_since_last_pivot = []

    return_since_last_pivot = []

    ls_is_any_pivot = []

    flag_pivot = False

    lastPivotDate = None

    lastPivotPrice = None

    for i in dfcopy.index:
        currentMin=min(Rangev,default=0)
        value=round(dfcopy["bidlo"][i],2)

        Rangev=Rangev[1:time_window*2-1]
        Rangev.append(value)

        dateRangev=dateRangev[1:time_window*2-1]
        date = dfcopy["date"][i]
        dateRangev.append(i)

        if currentMin==min(Rangev,default=0):
            counter=counter+1
        else:
            counter=0
        if counter==time_window:
            flag_pivot = True
            lastPivot=currentMin
            dateloc=Rangev.index(lastPivot)
            lastDate=dateRangev[dateloc]
            pivots.append(lastPivot)
            dates.append(lastDate)
            ls_index_pivot.append(i)

            lastPivotDate = dfcopy["date"][i]
            lastPivotPrice = dfcopy["adjusted_prc"][i]
            dfcopy["pivot_min_"+str(time_window)+"_indicator"][i]=1

        if flag_pivot == False:
            ls_days_since_last_pivot.append(0)
            return_since_last_pivot.append(0)
            ls_is_any_pivot.append(1)
        else:
            days_since = (pd.to_datetime(dfcopy["date"][i]) - pd.to_datetime(lastPivotDate)).days
            ls_days_since_last_pivot.append(days_since)
            if lastPivotPrice == 0 or lastPivotPrice == None or dfcopy["adjusted_prc"][i] == None:
                return_since = 0
            else:
                return_since = (dfcopy["adjusted_prc"][i] - lastPivotPrice)/lastPivotPrice
            return_since_last_pivot.append(return_since)
            ls_is_any_pivot.append(0)

    dic_pivot_returns = {"pivot_indictor":dfcopy["pivot_min_"+str(time_window)+"_indicator"],
    "days_since_pivot":ls_days_since_last_pivot,
    "return_since_pivot":return_since_last_pivot,
    "ls_is_any_pivot":ls_is_any_pivot}
    return dic_pivot_returns

ls_time_windows = [2,3,5,10,15,20]
TRADING_DAYS = 253
for permn in ls_permno:
    df = dic_permno[permn]
    for time_window in ls_time_windows:
        #pivot points for max implementation
        dic_pivot_returns = computePivotMax(df, time_window)
        # Shifting down by time window to get rid of look ahead bias
        df["pivot_max_"+str(time_window)+"_indicator"] = dic_pivot_returns["pivot_indictor"]
        df["pivot_max_"+str(time_window)+"_indicator"] = df["pivot_max_"+str(time_window)+"_indicator"].shift(time_window)
        df["pivot_max_"+str(time_window)+"_days_since_last"] = dic_pivot_returns["days_since_pivot"]
        df["pivot_max_"+str(time_window)+"_days_since_last"] = df["pivot_max_"+str(time_window)+"_days_since_last"].shift(time_window)
        df["pivot_max_"+str(time_window)+"_returns_since_last"] = dic_pivot_returns["return_since_pivot"]
        df["pivot_max_"+str(time_window)+"_returns_since_last"] = df["pivot_max_"+str(time_window)+"_returns_since_last"].shift(time_window)
        df["pivot_max_"+str(time_window)+"_any_pivot_indicator"] = dic_pivot_returns["ls_is_any_pivot"]
        df["pivot_max_"+str(time_window)+"_any_pivot_indicator"] = df["pivot_max_"+str(time_window)+"_any_pivot_indicator"].shift(time_window)
        df["pivot_max_"+str(time_window)+"_rolling"+"_3mo"]= df["pivot_max_"+str(time_window)+"_indicator"].rolling(window=int(TRADING_DAYS/4)).sum()
        #################################################################################################
        ######### Subtracting 3 to give some leeway for unexpected holidays and STUFF ###################
        #################################################################################################
        df["pivot_max_"+str(time_window)+"_rolling"+"_6mo"]= df["pivot_max_"+str(time_window)+"_indicator"].rolling(window=int(TRADING_DAYS/2), min_periods = int(TRADING_DAYS/2) - 3).sum()
        df["pivot_max_"+str(time_window)+"_rolling"+"_1yr"]= df["pivot_max_"+str(time_window)+"_indicator"].rolling(window=int(TRADING_DAYS), min_periods = int(TRADING_DAYS/2) - 3).sum()
        df["pivot_max_"+str(time_window)+"_rolling"+"_2yr"]= df["pivot_max_"+str(time_window)+"_indicator"].rolling(window=int(TRADING_DAYS*2), min_periods = int(TRADING_DAYS/2) - 3).sum()
#         df["pivot_cutoff_indicator"]


ls_time_windows = [2,3,5,10,15,20]
TRADING_DAYS = 253
for permn in ls_permno:
    df = dic_permno[permn]
    for time_window in ls_time_windows:
        #pivot points for min implementation
        dic_pivot_returns = computePivotMin(df, time_window)
        # Shifting down by time window to get rid of look ahead bias
        df["pivot_min_"+str(time_window)+"_indicator"] = dic_pivot_returns["pivot_indictor"]
        df["pivot_min_"+str(time_window)+"_indicator"] = df["pivot_min_"+str(time_window)+"_indicator"].shift(time_window)
        df["pivot_min_"+str(time_window)+"_days_since_last"] = dic_pivot_returns["days_since_pivot"]
        df["pivot_min_"+str(time_window)+"_days_since_last"] = df["pivot_min_"+str(time_window)+"_days_since_last"].shift(time_window)
        df["pivot_min_"+str(time_window)+"_returns_since_last"] = dic_pivot_returns["return_since_pivot"]
        df["pivot_min_"+str(time_window)+"_returns_since_last"] = df["pivot_min_"+str(time_window)+"_returns_since_last"].shift(time_window)
        df["pivot_min_"+str(time_window)+"_any_pivot_indicator"] = dic_pivot_returns["ls_is_any_pivot"]
        df["pivot_min_"+str(time_window)+"_any_pivot_indicator"] = df["pivot_min_"+str(time_window)+"_any_pivot_indicator"].shift(time_window)
        df["pivot_min_"+str(time_window)+"_rolling"+"_3mo"]= df["pivot_min_"+str(time_window)+"_indicator"].rolling(window=int(TRADING_DAYS/4)).sum()
        #################################################################################################
        ######### Subtracting 3 to give some leeway for unexpected holidays and STUFF ###################
        #################################################################################################
        df["pivot_min_"+str(time_window)+"_rolling"+"_6mo"]= df["pivot_min_"+str(time_window)+"_indicator"].rolling(window=int(TRADING_DAYS/2), min_periods = int(TRADING_DAYS/2) - 3).sum()
        df["pivot_min_"+str(time_window)+"_rolling"+"_1yr"]= df["pivot_min_"+str(time_window)+"_indicator"].rolling(window=int(TRADING_DAYS), min_periods = int(TRADING_DAYS/2) - 3).sum()
        df["pivot_min_"+str(time_window)+"_rolling"+"_2yr"]= df["pivot_min_"+str(time_window)+"_indicator"].rolling(window=int(TRADING_DAYS*2), min_periods = int(TRADING_DAYS/2) - 3).sum()

ls_time_windows = [2,3,5,10,15,20]
TRADING_DAYS = 253
for permn in ls_permno:
    df = dic_permno[permn]
    length_df = len(df)
    for time_window in ls_time_windows:
        #pivot points for min implementation
        if length_df-int(TRADING_DAYS/4) + 1 < 0:
            df["pivot_cutoff_indicator_"+str(time_window)+"_rolling"+"_3mo"] = [1]*(length_df)
        if length_df-int(TRADING_DAYS/4) + 1 >= 0:
            df["pivot_cutoff_indicator_"+str(time_window)+"_rolling"+"_3mo"] = [1]*(int(TRADING_DAYS/4)-1) + [0]*(length_df-int(TRADING_DAYS/4) + 1)
        if length_df-int(TRADING_DAYS/2) + 1 < 0:
            df["pivot_cutoff_indicator_"+str(time_window)+"_rolling"+"_6mo"] = [1]*(length_df)
        if length_df-int(TRADING_DAYS/2) + 1 >= 0:
            df["pivot_cutoff_indicator_"+str(time_window)+"_rolling"+"_6mo"] = [1]*(int(TRADING_DAYS/2)-1) + [0]*(length_df-int(TRADING_DAYS/2) + 1)
        if length_df-int(TRADING_DAYS) + 1 < 0:
            df["pivot_cutoff_indicator_"+str(time_window)+"_rolling"+"_1yr"] = [1]*(length_df)
        if length_df-int(TRADING_DAYS) + 1 >= 0:
            df["pivot_cutoff_indicator_"+str(time_window)+"_rolling"+"_1yr"] = [1]*(int(TRADING_DAYS)-1) + [0]*(length_df-int(TRADING_DAYS) + 1)
        if length_df-int(TRADING_DAYS*2) + 1 < 0:
            df["pivot_cutoff_indicator_"+str(time_window)+"_rolling"+"_2yr"] = [1]*(length_df)
        if length_df-int(TRADING_DAYS*2) + 1 >= 0:
            df["pivot_cutoff_indicator_"+str(time_window)+"_rolling"+"_2yr"] = [1]*(int(TRADING_DAYS*2)-1) + [0]*(length_df-int(TRADING_DAYS*2) + 1)

df_final = pd.DataFrame()
for i in range(len(ls_permno)):
    permn = ls_permno[i]
    df_temp = dic_permno[permn]
    df_final = df_final.append(df_temp,ignore_index= True)

merged_df = df_final

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
target_volatility = by_permno['log_ret'].rolling(
window=six_mos).std() * np.sqrt(six_mos)
target_volatility = fix_nested_index(target_volatility, ['target_vol'])
merged_df['target_vol'] = target_volatility

vols = merged_df[['target_vol', 'permno', 'date']]
vols.columns = ['target_vol', 'permno', 'date']

merged_df = merged_df.merge(vols, how='left', left_on=[
                            'prediction_date', 'permno'], right_on=['date', 'permno'])

# THIS IS AN ARTIFACT FROM BEFORE ADDING RSI + PIVOT POINTS
# merged_df = merged_df[['permno', 'date_x',
#                        'prediction_date', 'ret', 'vwretd', 'target_vol_y', 'log_ret']]
#
# merged_df.columns = ['permno', 'date', 'prediction_date',
#                      'ret', 'vwretd', 'target_vol', 'log_ret']
#

merged_df = merged_df.rename(columns={'date_x': 'date', 'target_vol_y': 'target_vol'})
merged_df = merged_df.drop(columns=['target_vol_x', 'date_y'], axis=1)
merged_df['target_vol'] = merged_df.groupby('permno').target_vol.ffill()


"""
    Gain Loss %: (Stock[-1] - Stock[0]) * 100 / (Stock[0])
"""
merged_df['cum_ret_stock'] = merged_df['ret'] + 1
by_permno = merged_df.groupby('permno')
merged_df['cum_ret_stock'] = by_permno.cum_ret_stock.cumprod()
merged_df['gain_loss'] = by_permno.cum_ret_stock.pct_change(
    periods=TRADING_DAYS)

"""
    Returns Target Goes Here
"""
cum_ret = merged_df[['permno', 'date', 'cum_ret_stock']]

merged_df = merged_df.merge(cum_ret, how='left', left_on=[
                            'prediction_date', 'permno'], right_on=['date', 'permno'])

# We need targets for stocks that are delisted 6 months in advanced for testing. 
merged_df['cum_ret_stock_y'] = merged_df.groupby('permno').cum_ret_stock_y.ffill()

merged_df['target'] = (merged_df['cum_ret_stock_y'] -
                       merged_df['cum_ret_stock_x']) / (merged_df['cum_ret_stock_x'])

merged_df = merged_df.rename(columns={'date_x': 'date', 'cum_ret_stock_x': 'cum_ret_stock'})
merged_df = merged_df.drop(columns=['cum_ret_stock_y', 'date_y'], axis=1)

# THIS IS AN ARTIFACT FROM BEFORE ADDING RSI + PIVOT POINTS
# merged_df = merged_df[['permno', 'date_x', 'prediction_date',
#                        'ret', 'vwretd', 'cum_ret_stock_x', 'target', 'target_vol', 'log_ret', 'gain_loss']]
# merged_df.columns = ['permno', 'date', 'prediction_date',
#                      'ret', 'vwretd', 'cum_ret_stock', 'target', 'target_vol', 'log_ret', 'gain_loss']


# by_permno = merged_df.groupby('permno')


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

# Get stock returns on days when market has gone up.
merged_df['ret_stock_bull'] = merged_df['ret']
merged_df.loc[merged_df['ret_sp_bull'].isna(), 'ret_stock_bull'] = None

merged_df['ret_sp_bear'] = merged_df['vwretd']
merged_df.loc[merged_df['ret_sp_bear'] >= 0, 'ret_sp_bear'] = None

# Get stock returns on days when market has gone down.
merged_df['ret_stock_bear'] = merged_df['ret']
merged_df.loc[merged_df['ret_sp_bear'].isna(), 'ret_stock_bear'] = None

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
    2. permno Z score for last 365 days.
    NB: If stock has been listed within last 2 years. It will only have price features for last year, since the window is a year.
    We will consider these price features if there are a minimum of 30 periods to compare.
"""

# Drop all rows without values except prediction_date
merged_df = merged_df.dropna(
    subset=[n for n in merged_df if n not in ['prediction_date', 'target']])
numeric_features = set(RAW_FEATURES) -     set(['permno', 'date', 'prediction_date'])

# Indicator variables should not be z transformed
import re
regex = re.compile(r'indicator')
filtered_indicator = [i for i in list(numeric_features) if not regex.search(i)]

# features_df = merged_df[numeric_features]
features_df = merged_df[filtered_indicator]

# Global ZScore. TODO: Add Global Zscore. Need to change index to days.
# col_mean = features_df.rolling(window=TRADING_DAYS, min_periods=30).mean()
# col_std = features_df.rolling(window=TRADING_DAYS, min_periods=30).std()
# zscore = (features_df - col_mean)/col_std
# global_cols = [str + '_global_z' for str in zscore.columns]
# zscore.columns = global_cols
# merged_df = merged_df.merge(zscore, how='left', left_index=True, right_index=True)


# In[8]:


############ ISAAC CODE COMMENTED OUT #################################
# # Local ZScore (Zscore by_permno)

# by_permno = merged_df[numeric_features.union(set(['permno']))].groupby('permno')[
#     list(numeric_features)]
by_permno = merged_df[numeric_features.union(set(['permno']))].groupby('permno')[
    filtered_indicator]
col_mean = by_permno.rolling(window=TRADING_DAYS, min_periods=30).mean()
col_mean = fix_nested_index(col_mean, col_mean.columns.tolist())
col_std = by_permno.rolling(window=TRADING_DAYS, min_periods=30).std()
col_std = fix_nested_index(col_std, col_std.columns.tolist())
zscore = (features_df - col_mean) / (col_std)

local_cols = [str + '_local_z' for str in zscore.columns]
zscore.columns = local_cols
merged_df = merged_df.merge(
    zscore, how='left', left_index=True, right_index=True)
############################################################

### Calculating Global Z-Scores
for col in filtered_indicator:
    by_date = merged_df.groupby(['date'])
    col_mean = by_date[col].transform('mean')
    col_std = by_date[col].transform('std')
    merged_df[col+"_global_z"] = (merged_df[col] - col_mean)/col_std
    merged_df[col+"_global_z"] = merged_df[col+"_global_z"].fillna(0)

# Saving File with ALl the features and transformed features to lookup for trends
merged_df.to_csv("Final_Features.csv")

# Creating a dataframe with only the relevant features for training

final_df = merged_df[FINAL_FEATURES + ['target', 'target_vol']]
final_df = final_df.dropna()

"""
    Upload to Bigquery.
    TODO: This was stalling indefinitely. Have uploaded to bigquery manually.
"""

# TODO: Make this a utils class. Improve this method.
final_df.to_csv('features_v11.csv', index=False)


# ##########################################################
# # Global ZScore (Zscore by_date)- Aaron

# # TODO: Anything that are indicators shouldn't have a zscore.
# by_date = merged_df[numeric_features.union(set(['date', 'permno']))].groupby('date')[list(numeric_features) + ['permno']]
# import pdb; pdb.set_trace()
# col_mean = by_date.rolling(window=1, min_periods=1).mean()
# col_mean = fix_nested_index(col_mean, col_mean.columns.tolist())
# col_std = by_date.rolling(window=1, min_periods=1).std()
# col_std = fix_nestecopd_index(col_std, col_std.columns.tolist())

# import pdb; pdb.set_trace()
# zscore = (features_df - col_mean) / (col_std)
# local_cols = [str + '_global_z' for str in zscore.columns]
# zscore.columns = local_cols
# merged_df = merged_df.merge(
#     zscore, how='left', left_index=True, right_index=True)
# import pdb; pdb.set_trace()
# ########################################################################

# # print("Finished calculating local and global z-scores.")
# print("Finished calculating local z-scores.")

# final_df = merged_df[FINAL_FEATURES + ['target', 'target_vol']]
# final_df = final_df.dropna()

# """
#     Upload to Bigquery.
#     TODO: This was stalling indefinitely. Have uploaded to bigquery manually.
# """

# # TODO: Make this a utils class. Improve this method.
# final_df.to_csv('features_v11.csv', index=False)

