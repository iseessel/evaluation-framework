"""
    All other feature sets have been deprecated.

    We will now just use the "features_vx" standard for ease of use.

    We are now using permno instead of permno.
"""

import numpy as np
import pandas as pd
from google.cloud import bigquery
from tensorflow.keras.preprocessing import timeseries_dataset_from_array

START_DATE = '1970-01-01'
TRADING_DAYS = 253

QUERY = f"""
WITH
  sp_daily_features AS (
  SELECT
    date,
    std1.permno,
    ticker,
    ret,
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
  sp_daily_features.prediction_date
FROM
  sp_daily_features
WHERE
    sp_daily_features.date >= '{ START_DATE }'
"""

WINDOW_SIZE = 50

print("Fetching data from Bigquery. Could take a few minutes.")
client = bigquery.Client(project='silicon-badge-274423')
df = client.query(QUERY).to_dataframe()

# Forward fill target prices for when stocks become delisted and there is no 6 month later target.
df = df.sort_values(by=['permno', 'date']).reset_index()

"""
    Create target (Returns) Here
"""
# Adjusted returns are null the first day they are reported.
df.loc[df['ret'].isna(), 'ret'] = 0

df = df.dropna()

by_permno = df.groupby('permno')
df['cum_ret_stock'] = df['ret'] + 1
df['cum_ret_stock'] = by_permno.cum_ret_stock.cumprod()

cum_ret = df[['permno', 'date', 'cum_ret_stock']]

df = df.merge(cum_ret, how='left', left_on=[
                            'prediction_date', 'permno'], right_on=['date', 'permno'])

# We need targets for stocks that are delisted 6 months in advanced for testing.
df['cum_ret_stock_y'] = df.groupby('permno').cum_ret_stock_y.ffill()

df['target'] = (df['cum_ret_stock_y'] -
                       df['cum_ret_stock_x']) / (df['cum_ret_stock_x'])

df = df.rename(columns={'date_x': 'date', 'cum_ret_stock_x': 'cum_ret_stock'})
df = df.drop(columns=['date_y', 'cum_ret_stock_y'])

"""
Create target volatility.
"""

def fix_nested_index(series, indeces):
    series = series.reset_index(level=[0, 1])
    series = series.drop('permno', axis=1).set_index('level_1')
    series.columns = indeces

    return series

df['log_ret'] = np.log(1 + df['ret'])

by_permno = df.groupby('permno')
six_mos = int(TRADING_DAYS / 2)
target_volatility = by_permno['log_ret'].rolling(
    window=six_mos).std() * np.sqrt(six_mos)

target_volatility = fix_nested_index(target_volatility, ['target_vol'])
df['target_vol'] = target_volatility

# https://stackoverflow.com/questions/40084931/taking-subarrays-from-numpy-array-with-given-stride-stepsize

def strided_app(a, L=WINDOW_SIZE, S=1):
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))


def rolling_returns(a):
    # Set first day of window's returns to one, since we are calculating rolling returns from "first" day of window.
    def f(x): return (x - a[0]) / (a[0])
    return f(a)

dates = np.empty((0), 'datetime64')
prediction_dates = np.empty((0), 'datetime64')
permno = np.empty((0), 'object')
rolling_window = np.empty((0, WINDOW_SIZE), 'float32')
targets = np.empty((0), 'float32')

i = 0
for _, row in df.groupby('permno'):
    row = row.sort_values(by='date')

    # When a stock has just been listed, the permno will not have a long enough price series to be included.
    if len(row.cum_ret_stock.to_numpy()) < 50:
        continue

    cum_rets = strided_app(row.cum_ret_stock.to_numpy())

    # Dates, permno, and target dates are offset by WINDOW_SIZE.
    dates = np.append(dates, row.date.to_numpy()[(WINDOW_SIZE - 1):])
    permno = np.append(permno, row.permno.to_numpy()[(WINDOW_SIZE - 1):])
    prediction_dates = np.append(
        prediction_dates, row.prediction_date.to_numpy()[(WINDOW_SIZE - 1):])

    t = row.target.to_numpy()[(WINDOW_SIZE - 1):]
    rets = []
    targs = []
    for j, cum_rets in enumerate(cum_rets):
        # Set first day to zero returns, as it is our starting point.
        window_returns = rolling_returns(cum_rets)
        rets.append(window_returns)

        # Target is from T - 50, NOT T. This is in order to capture the momentum.
        target = (1 + window_returns[-1]) * (1 + t[j])

        targs.append(target)

    targets = np.append(targets, np.array(targs),)
    rolling_window = np.append(rolling_window, np.array(rets), axis=0)
    print(f"Done with {i} / { len(df.groupby('permno')) }")
    i += 1

# Sort the dates and the other arrays.
x = dates.argsort()

dates, permno, rolling_window, targets, prediction_dates = dates[
    x], permno[x], rolling_window[x], targets[x], prediction_dates[x]
features_df = pd.DataFrame({
    'date': dates,
    'permno': permno,
    'adjusted_rets': rolling_window.tolist(),
    'target': targets,
    'prediction_date': prediction_dates
})

df = df[['date', 'permno', 'target_vol']]
features_df = pd.merge(features_df, df,  how='left', left_on=['permno','prediction_date'], right_on = ['permno','date'])
features_df['target_vol'] = features_df.groupby('permno').target_vol.ffill()
features_df = features_df[['date_x', 'permno', 'adjusted_rets', 'target', 'prediction_date', 'target_vol']]
features_df = features_df.dropna()
features_df = features_df.rename(columns={'date_x': 'date'})

features_df.date = features_df.date.astype('string')
features_df.prediction_date = features_df.prediction_date.astype('string')

"""
    TODO: This was stalling indefinitely. Have uploaded to GCS manually.
"""
import pdb; pdb.set_trace()
with open('./features_v9.json', 'w') as f:
    f.write(features_df.to_json(orient='records', lines=True))

# print("Uploading data to bigquery.")
# job_config = bigquery.LoadJobConfig(
#     autodetect=True,
#     write_disposition='WRITE_TRUNCATE'
# )
#
# with open('price_features_v9.json', "rb") as source_file:
#     job = client.load_table_from_file(
#         source_file, 'silicon-badge-274423.features.features_v9', job_config=job_config)
#
# job.result()  # Waits for the job to complete.
