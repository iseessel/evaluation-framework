"""
    TODO: Put code in here that creates the results_analysis for given dataset.
"""

"""
    Create results_analysis daily_price_series_vx
"""

import pdb
from google.cloud import bigquery
import numpy as np
import pandas as pd
import datetime
import os
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

v1_dsets = [
    'silicon-badge-274423.portfolio.boosted_tree_price_features_vol_v7_prod_100_bonds_False',
    'silicon-badge-274423.portfolio.boosted_tree_price_features_vol_v7_prod_100_bonds_True',
    'silicon-badge-274423.portfolio.boosted_tree_price_features_vol_v7_prod_10_bonds_False',
    'silicon-badge-274423.portfolio.boosted_tree_price_features_vol_v7_prod_10_bonds_True',
    'silicon-badge-274423.portfolio.boosted_tree_price_features_vol_v7_prod_5_bonds_False',
    'silicon-badge-274423.portfolio.boosted_tree_price_features_vol_v7_prod_5_bonds_True',
    'silicon-badge-274423.portfolio.lstm_model_price_features_vol_v8_prod_100_bonds_False',
    'silicon-badge-274423.portfolio.lstm_model_price_features_vol_v8_prod_100_bonds_True',
    'silicon-badge-274423.portfolio.lstm_model_price_features_vol_v8_prod_10_bonds_False',
    'silicon-badge-274423.portfolio.lstm_model_price_features_vol_v8_prod_10_bonds_True',
    'silicon-badge-274423.portfolio.lstm_model_price_features_vol_v8_prod_5_bonds_False',
    'silicon-badge-274423.portfolio.lstm_model_price_features_vol_v8_prod_5_bonds_True',
    'silicon-badge-274423.portfolio.lstm_model_relu_price_features_vol_v7_prod_100_bonds_False',
    'silicon-badge-274423.portfolio.lstm_model_relu_price_features_vol_v7_prod_100_bonds_True',
    'silicon-badge-274423.portfolio.lstm_model_relu_price_features_vol_v7_prod_10_bonds_False',
    'silicon-badge-274423.portfolio.lstm_model_relu_price_features_vol_v7_prod_10_bonds_True',
    'silicon-badge-274423.portfolio.lstm_model_relu_price_features_vol_v7_prod_5_bonds_False',
    'silicon-badge-274423.portfolio.lstm_model_relu_price_features_vol_v7_prod_5_bonds_True',
    'silicon-badge-274423.portfolio.lstm_model_relu_price_features_vol_v8_prod_100_bonds_False',
    'silicon-badge-274423.portfolio.lstm_model_relu_price_features_vol_v8_prod_100_bonds_True',
    'silicon-badge-274423.portfolio.lstm_model_relu_price_features_vol_v8_prod_10_bonds_False',
    'silicon-badge-274423.portfolio.lstm_model_relu_price_features_vol_v8_prod_10_bonds_True',
    'silicon-badge-274423.portfolio.lstm_model_relu_price_features_vol_v8_prod_5_bonds_False',
    'silicon-badge-274423.portfolio.lstm_model_relu_price_features_vol_v8_prod_5_bonds_True',
    'silicon-badge-274423.portfolio.lstm_model_tanh_price_features_vol_v7_prod_100_bonds_False',
    'silicon-badge-274423.portfolio.lstm_model_tanh_price_features_vol_v7_prod_100_bonds_True',
    'silicon-badge-274423.portfolio.lstm_model_tanh_price_features_vol_v7_prod_10_bonds_False',
    'silicon-badge-274423.portfolio.lstm_model_tanh_price_features_vol_v7_prod_10_bonds_True',
    'silicon-badge-274423.portfolio.lstm_model_tanh_price_features_vol_v7_prod_5_bonds_False',
    'silicon-badge-274423.portfolio.lstm_model_tanh_price_features_vol_v7_prod_5_bonds_True'
]

v2_dsets = [
    'silicon-badge-274423.portfolio.features_v10_100_bonds_False',
    'silicon-badge-274423.portfolio.features_v10_100_bonds_True',
    'silicon-badge-274423.portfolio.features_v10_10_bonds_False',
    'silicon-badge-274423.portfolio.features_v10_10_bonds_True',
    'silicon-badge-274423.portfolio.features_v10_5_bonds_False',
    'silicon-badge-274423.portfolio.features_v10_5_bonds_True'
]

client = bigquery.Client(project='silicon-badge-274423')


"""
    Create union tables for the stock returns and volatility predictions.
"""

v2_dsets_preds = ['silicon-badge-274423.stock_model_evaluation.features_v10']

query = ""
for i, dset in enumerate(v2_dsets_preds):
    query = query + f"SELECT *, '{dset}' as features_dataset FROM `{dset}`"
    if i != len(v2_dsets_preds) - 1:
         query = query + "UNION ALL "

table_id = 'silicon-badge-274423.results.stock_model_evaluation_v2'
job_config = bigquery.QueryJobConfig(
    allow_large_results=True, destination=table_id, write_disposition='WRITE_TRUNCATE'
)

# Start the query, passing in the extra configuration.
query_job = client.query(query, job_config=job_config)  # Make an API request.
query_job.result()  # Wait for the job to complete.

print("Query results loaded to the table {}".format(table_id))

print(query)

# """
#     Create union tables for the portfolio choices.
# """
query = ""
for i, dset in enumerate(v2_dsets):
    query = query + f"SELECT *, '{dset}' as portfolio_dataset FROM `{dset}`"
    if i != len(v2_dsets) - 1:
         query = query + " UNION ALL "

table_id = 'silicon-badge-274423.results.portfolio_v2'
job_config = bigquery.QueryJobConfig(
    allow_large_results=True, destination=table_id, write_disposition='WRITE_TRUNCATE'
)

# Start the query, passing in the extra configuration.
query_job = client.query(query, job_config=job_config)  # Make an API request.
query_job.result()  # Wait for the job to complete.

print("Query results loaded to the table {}".format(table_id))

print(query)

"""
    Create daily timeseries for all the results.
"""

#TODO: Very small amount of periods have weights summing a bit above 1, probably
# due to rounding errors. 

final_df = pd.DataFrame()
for dset in v2_dsets:
    print(f"Calculating daily returns for: { dset }")
    QUERY = f"""
        WITH
            portfolio AS (
                SELECT
                  *,
                  (
                  SELECT
                    COALESCE(MIN(date), '2019-12-31')
                  FROM
                      `{dset}` b2
                  WHERE
                    b2.date > b.date AND b2.permno = 'ALL') AS trading_date
                FROM
                  `{dset}` b
                ORDER BY
                  date
            )

        SELECT
            portfolio.*, sdf.ret, sdf.date as daily_date
        FROM
            portfolio
        LEFT JOIN
          `silicon-badge-274423.financial_datasets.sp_timeseries_daily` sdf
        ON
          # Need to get a row for every day bond reports.
          portfolio.permno = sdf.permno OR (portfolio.permno = 'bond' AND sdf.permno = '20482')
        WHERE
            (sdf.date >= portfolio.date AND sdf.date < portfolio.trading_date)
    """

    def weighted_average(group):
        weights = group['weight']
        cum_ret = group['ret']
        return np.average(cum_ret, weights=weights)

    stock_df = client.query(QUERY).to_dataframe()
    stock_df = stock_df.sort_values(by=['date', 'permno', 'daily_date'])

    # Get bond discount rate
    bonds = stock_df[stock_df.permno == 'bond']
    grouped = bonds.groupby(['date'])
    num_rows = grouped.size()
    six_mo_ret = grouped.actual_ret.max()
    rates = ((1 + six_mo_ret)**(1 / num_rows)) - 1

    # TODO: Can improve this
    for date, rate in rates.items():
        stock_df.loc[((stock_df.date) == date) & (
            stock_df.permno == 'bond'), 'ret'] = rate

    # When stocks are delisted, they will have no timeseries, or (very rarely) returns aren't listed
    # We will count these returns as zero.
    stock_df['ret'] = stock_df['ret'].fillna(0)
    stock_df = stock_df.drop_duplicates()

    weighted_avg = stock_df.groupby(['daily_date']).apply(func=weighted_average)
    weighted_avg = 1 + weighted_avg
    df = weighted_avg.cumprod()
    df = pd.DataFrame({'date': df.index, 'cumulative_returns': df.values})
    df['dataset'] = dset

    final_df = final_df.append(df)

final_df.to_csv('temp.csv', index=False)

job_config = bigquery.LoadJobConfig(
    source_format=bigquery.SourceFormat.CSV, skip_leading_rows=1,
    autodetect=True,
    write_disposition='WRITE_TRUNCATE'
)

with open('temp.csv', "rb") as source_file:
    job = client.load_table_from_file(
        source_file, 'silicon-badge-274423.results_analysis.daily_price_series_v2', job_config=job_config)

job.result()  # Waits for the job to complete.

os.remove("temp.csv")
