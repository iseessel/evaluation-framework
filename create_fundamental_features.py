from google.cloud import bigquery
import pandas as pd
import pdb

QUERY = """
  WITH financial_ratios_firm_level as (
  SELECT
    permno,
    PARSE_DATE('%Y%m%d',
        public_date) as public_date,
    bm,
    evm,
    roa,
    roe,
    capital_ratio,
    short_debt,
    cash_ratio,
    quick_ratio
  FROM
    `silicon-badge-274423.financial_datasets.financial_ratios_firm_level`
  WHERE
    permno IN (
    SELECT
      DISTINCT permno
    FROM
      `silicon-badge-274423.features.sp_daily_features`)
    AND PARSE_DATE('%Y%m%d',
      public_date) > '1979-01-01'
    AND PARSE_DATE('%Y%m%d',
      public_date) < '2020-01-01'
  ORDER BY
    public_date, permno
  ), sp_daily_features as (
  SELECT
    *
  FROM
    `silicon-badge-274423.features.sp_daily_features`
  )

  SELECT
    *
  FROM
    sp_daily_features as sdf
  FULL OUTER JOIN
    financial_ratios_firm_level as fl ON sdf.permno = fl.permno AND sdf.date = fl.public_date
"""

FEATURE_LIST = [
  "permno_1",
  "public_date",
  "bm",
  "evm",
  "roa",
  "roe",
  "capital_ratio",
  "short_debt",
  "cash_ratio",
  "quick_ratio"
]

# First join to next known date for public_dates that don't report.
client = bigquery.Client(project='silicon-badge-274423')
df = client.query(QUERY).to_dataframe()

# One's without date (i.e. public_date is on a weekend)
without_price = df[df.date.isnull()]
with_price = df[df.date.notnull()]

# for i, row in without_price.iterrows():
#   # Get latest known date
#   if i % 100 == 0:
#     print(f"Finished with { i }/{ len(without_price) }")
#
#   temp = with_price[with_price.date >= row.public_date]
#   temp = temp[temp.permno == row.permno_1]
#   min_date = temp.date.min()
#
#   # Set values here.
#   idx = temp.index[temp['date'] == min_date][0]
#   for feature in FEATURE_LIST:
#     with_price.at[idx, feature] = row[feature]

# Fill in ones with a date from above.
result_df = pd.DataFrame()
permno_list = df.permno.unique()
for i, permno in enumerate(permno_list):
  print(f"Finished with { i }/{ len(permno_list) }")

  df = with_price[with_price.permno == permno].sort_values(by='date')
  df = df.ffill()
  # TODO: What do we do with missing values here?
  df = df.dropna(subset=FEATURE_LIST)
  # df = df.dropna(subset=FEATURE_LIST, thresh=len(df.columns) - 3)

  result_df = result_df.append(df)
  if i == 10:
    break

import pdb; pdb.set_trace()
result_df = result_df.sort_values(by=['date', 'permno'])
