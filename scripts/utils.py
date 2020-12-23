"""
    Create union all queries for multiple datasets.

    TODO: Make this a script we can use easily.
"""

# import os
# from datetime import datetime
# import pandas as pd
# import numpy as np
# import pdb
# from google.cloud import bigquery
#
# v0_dsets = [
#     "silicon-badge-274423.portfolio.boosted_tree_features_vol_v4_light_prod",
#     "silicon-badge-274423.portfolio.boosted_tree_features_vol_v4_light_prod_005",
#     "silicon-badge-274423.portfolio.boosted_tree_features_vol_v4_light_prod_01",
#     "silicon-badge-274423.portfolio.boosted_tree_features_vol_v4_light_prod_01_bonds_unconstrained",
#     "silicon-badge-274423.portfolio.fb_prophet_sp_daily_features_v0_prod_t_005",
#     "silicon-badge-274423.portfolio.fb_prophet_sp_daily_features_v0_prod_t_01",
#     "silicon-badge-274423.portfolio.fb_prophet_sp_daily_features_v0_prod_t_01_bonds_unconstrained",
#     "silicon-badge-274423.portfolio.lstm_model_price_features_vol_v4_prod",
#     "silicon-badge-274423.portfolio.lstm_model_price_features_vol_v4_prod_005",
#     "silicon-badge-274423.portfolio.lstm_model_price_features_vol_v4_prod_01",
#     "silicon-badge-274423.portfolio.lstm_model_price_features_vol_v4_prod_01_bonds_unconstrained",
#     "silicon-badge-274423.portfolio.lstm_model_price_features_vol_v4_prod_relu",
#     "silicon-badge-274423.portfolio.lstm_model_price_features_vol_v4_prod_relu_005",
#     "silicon-badge-274423.portfolio.lstm_model_price_features_vol_v4_prod_relu_01",
#     "silicon-badge-274423.portfolio.lstm_model_price_features_vol_v4_prod_relu_01_bonds_unconstrained",
#     "silicon-badge-274423.portfolio.lstm_model_price_features_vol_v5_prod",
#     "silicon-badge-274423.portfolio.lstm_model_price_features_vol_v5_prod_005",
#     "silicon-badge-274423.portfolio.lstm_model_price_features_vol_v5_prod_01",
#     "silicon-badge-274423.portfolio.lstm_model_price_features_vol_v5_prod_01_bonds_unconstrained"
# ]
#
# v1_dsets = [
#     'silicon-badge-274423.portfolio.boosted_tree_price_features_vol_v7_prod_100_bonds_False',
#     'silicon-badge-274423.portfolio.boosted_tree_price_features_vol_v7_prod_100_bonds_True',
#     'silicon-badge-274423.portfolio.boosted_tree_price_features_vol_v7_prod_10_bonds_False',
#     'silicon-badge-274423.portfolio.boosted_tree_price_features_vol_v7_prod_10_bonds_True',
#     'silicon-badge-274423.portfolio.boosted_tree_price_features_vol_v7_prod_5_bonds_False',
#     'silicon-badge-274423.portfolio.boosted_tree_price_features_vol_v7_prod_5_bonds_True',
#     'silicon-badge-274423.portfolio.lstm_model_price_features_vol_v8_prod_100_bonds_False',
#     'silicon-badge-274423.portfolio.lstm_model_price_features_vol_v8_prod_100_bonds_True',
#     'silicon-badge-274423.portfolio.lstm_model_price_features_vol_v8_prod_10_bonds_False',
#     'silicon-badge-274423.portfolio.lstm_model_price_features_vol_v8_prod_10_bonds_True',
#     'silicon-badge-274423.portfolio.lstm_model_price_features_vol_v8_prod_5_bonds_False',
#     'silicon-badge-274423.portfolio.lstm_model_price_features_vol_v8_prod_5_bonds_True',
#     'silicon-badge-274423.portfolio.lstm_model_relu_price_features_vol_v7_prod_100_bonds_False',
#     'silicon-badge-274423.portfolio.lstm_model_relu_price_features_vol_v7_prod_100_bonds_True',
#     'silicon-badge-274423.portfolio.lstm_model_relu_price_features_vol_v7_prod_10_bonds_False',
#     'silicon-badge-274423.portfolio.lstm_model_relu_price_features_vol_v7_prod_10_bonds_True',
#     'silicon-badge-274423.portfolio.lstm_model_relu_price_features_vol_v7_prod_5_bonds_False',
#     'silicon-badge-274423.portfolio.lstm_model_relu_price_features_vol_v7_prod_5_bonds_True',
#     'silicon-badge-274423.portfolio.lstm_model_relu_price_features_vol_v8_prod_100_bonds_False',
#     'silicon-badge-274423.portfolio.lstm_model_relu_price_features_vol_v8_prod_100_bonds_True',
#     'silicon-badge-274423.portfolio.lstm_model_relu_price_features_vol_v8_prod_10_bonds_False',
#     'silicon-badge-274423.portfolio.lstm_model_relu_price_features_vol_v8_prod_10_bonds_True',
#     'silicon-badge-274423.portfolio.lstm_model_relu_price_features_vol_v8_prod_5_bonds_False',
#     'silicon-badge-274423.portfolio.lstm_model_relu_price_features_vol_v8_prod_5_bonds_True',
#     'silicon-badge-274423.portfolio.lstm_model_tanh_price_features_vol_v7_prod_100_bonds_False',
#     'silicon-badge-274423.portfolio.lstm_model_tanh_price_features_vol_v7_prod_100_bonds_True',
#     'silicon-badge-274423.portfolio.lstm_model_tanh_price_features_vol_v7_prod_10_bonds_False',
#     'silicon-badge-274423.portfolio.lstm_model_tanh_price_features_vol_v7_prod_10_bonds_True',
#     'silicon-badge-274423.portfolio.lstm_model_tanh_price_features_vol_v7_prod_5_bonds_False',
#     'silicon-badge-274423.portfolio.lstm_model_tanh_price_features_vol_v7_prod_5_bonds_True'
# ]
#
# query = ""
# for dset in v0_dsets:
#     query = query + \
#         f"SELECT *, dataset as features_dataset, '{dset}' as dataset FROM `{dset}` UNION ALL "
#
# print(query)

"""
    Create results_analysis daily_price_series_v1
"""

v0_dsets = [
    "silicon-badge-274423.portfolio.boosted_tree_features_vol_v4_light_prod",
    "silicon-badge-274423.portfolio.boosted_tree_features_vol_v4_light_prod_005",
    "silicon-badge-274423.portfolio.boosted_tree_features_vol_v4_light_prod_01",
    "silicon-badge-274423.portfolio.boosted_tree_features_vol_v4_light_prod_01_bonds_unconstrained",
    "silicon-badge-274423.portfolio.fb_prophet_sp_daily_features_v0_prod_t_005",
    "silicon-badge-274423.portfolio.fb_prophet_sp_daily_features_v0_prod_t_01",
    "silicon-badge-274423.portfolio.fb_prophet_sp_daily_features_v0_prod_t_01_bonds_unconstrained",
    "silicon-badge-274423.portfolio.lstm_model_price_features_vol_v4_prod",
    "silicon-badge-274423.portfolio.lstm_model_price_features_vol_v4_prod_005",
    "silicon-badge-274423.portfolio.lstm_model_price_features_vol_v4_prod_01",
    "silicon-badge-274423.portfolio.lstm_model_price_features_vol_v4_prod_01_bonds_unconstrained",
    "silicon-badge-274423.portfolio.lstm_model_price_features_vol_v4_prod_relu",
    "silicon-badge-274423.portfolio.lstm_model_price_features_vol_v4_prod_relu_005",
    "silicon-badge-274423.portfolio.lstm_model_price_features_vol_v4_prod_relu_01",
    "silicon-badge-274423.portfolio.lstm_model_price_features_vol_v4_prod_relu_01_bonds_unconstrained",
    "silicon-badge-274423.portfolio.lstm_model_price_features_vol_v5_prod",
    "silicon-badge-274423.portfolio.lstm_model_price_features_vol_v5_prod_005",
    "silicon-badge-274423.portfolio.lstm_model_price_features_vol_v5_prod_01",
    "silicon-badge-274423.portfolio.lstm_model_price_features_vol_v5_prod_01_bonds_unconstrained"
]

for dset in dsets:
    QUERY = f"""
        SELECT
            b.*, sdf.ret, sdf.date as daily_date
        FROM
          `{dset}` b
        LEFT JOIN
          `silicon-badge-274423.financial_datasets.sp_timeseries_daily` sdf
        ON
        # Need to get a row for every day bond reports.
          b.permno = sdf.permno OR (b.permno = 'bond' AND sdf.permno = '14593')
        WHERE
            (sdf.date > b.date and sdf.date <= b.prediction_date)
    """


def weighted_average(group):
    weights = group['weight']
    cum_ret = group['ret']
    return np.average(cum_ret, weights=weights)


client = bigquery.Client(project='silicon-badge-274423')
stock_df = client.query(QUERY).to_dataframe()
stock_df = stock_df.sort_values(by=['date', 'permno', 'daily_date'])
min_date = '2009-01-01'

# Get bond discount rate
bonds = stock_df[stock_df.permno == 'bond']
grouped = bonds.groupby(['date'])
num_rows = grouped.size()
rates = ((1 + grouped.actual_ret.min())**(1 / num_rows)) - 1

# TODO: Can improve this
for date, rate in rates.items():
    stock_df.loc[((stock_df.date) == date) & (
        stock_df.permno == 'bond'), 'ret'] = rate

weighted_avg = stock_df.groupby(
    ['daily_date']).apply(func=weighted_average)
weighted_avg = 1 + weighted_avg
df = weighted_avg.cumprod()
df = pd.DataFrame({'email': df.index, 'list': df.values})
df['dset'] = dset

df.to_csv('temp.csv', index=False)

job_config = bigquery.LoadJobConfig(
    source_format=bigquery.SourceFormat.CSV, skip_leading_rows=1,
    autodetect=True,
    write_disposition='WRITE_APPEND'
)

with open('temp.csv', "rb") as source_file:
    job = client.load_table_from_file(
        source_file, 'silicon-badge-274423.results_analysis.daily_price_series_v0', job_config=job_config)

job.result()  # Waits for the job to complete.

os.remove("temp.csv")
