from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from google.cloud import bigquery
import pandas as pd
import numpy as np

WINDOW_SIZE = 50

# TODO: This query created the table we are querying below.
QUERY = """
SELECT
    *
FROM
    `silicon-badge-274423.features.price_features_v0`
"""

print("Fetching data from Bigquery. Could take a few minutes.")
client = bigquery.Client(project='silicon-badge-274423')
df = client.query(QUERY).to_dataframe()
df = df.dropna()

arr = df.adjusted_prc.to_numpy()

# https://stackoverflow.com/questions/40084931/taking-subarrays-from-numpy-array-with-given-stride-stepsize
def strided_app(a, L = WINDOW_SIZE, S=1):
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

def rolling_returns(a):
    f = lambda x: (x - a[0]) / (a[0])
    return f(a)

dates = np.empty((0), 'datetime64')
prediction_dates =  np.empty((0), 'datetime64')
permnos = np.empty((0), 'object')
rolling_window = np.empty((0,WINDOW_SIZE), 'float32')
targets = np.empty((0), 'float32')

# TODO: Can potentially optimize this.
i = 0
for permno, row in df.groupby('permno'):
    row = row.sort_values(by='date')
    # Sometimes a permno will not have a long enough price series to be included.
    if len(row.adjusted_prc.to_numpy()) < 50:
        continue
    prcs = strided_app(row.adjusted_prc.to_numpy())

    # Dates, permnos, and target dates are offset by WINDOW_SIZE.
    dates = np.append(dates, row.date.to_numpy()[(WINDOW_SIZE - 1):])
    permnos = np.append(permnos, row.permno.to_numpy()[(WINDOW_SIZE - 1):])
    prediction_dates = np.append(prediction_dates, row.prediction_date.to_numpy()[(WINDOW_SIZE - 1):])

    t = row.target.to_numpy()[(WINDOW_SIZE - 1):]
    rets = []
    targs = []
    for j, prc in enumerate(prcs):
        rets.append(rolling_returns(prc))
        targs.append(((t[j] - prc[0]) / prc[0]))

    targets = np.append(targets, np.array(targs),)
    rolling_window = np.append(rolling_window, np.array(rets), axis=0)
    print(f"Done with {i} / { len(df.groupby('permno')) }")
    i += 1

import pdb; pdb.set_trace()
# Sort the dates and the other arrays.
x = dates.argsort()
dates, permnos, rolling_window, targets, prediction_dates = dates[x], permnos[x], rolling_window[x], targets[x], prediction_dates[x]
features_df = pd.DataFrame({
    'date': dates,
    'permno': permnos,
    'adjusted_rets': rolling_window.tolist(),
    'target': targets,
    'prediction_date': prediction_dates
})

features_df.date = features_df.date.astype('string')
features_df.prediction_date = features_df.date.astype('string')

"""
    Upload pickle.
    TODO: This was stalling indefinitely. Have uploaded to GCS manually.
"""
features_df.to_pickle("./price_features_v1.pkl")
with open('./price_features_v1.json', 'w') as f:
    f.write(features_df.to_json(orient='records', lines=True))
