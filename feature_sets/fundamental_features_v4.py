from google.cloud import bigquery
import pandas as pd
import os

"""
  This creates fundamental features.
  TODO: Adapt this.
"""

# The table queried below (silicon-badge-274423.financial_datasets.sp_price_fundamentals) was created from this query.
QUERY = """
WITH
  sp_daily_features AS (
  SELECT
    *
  FROM
    `silicon-badge-274423.features.sp_daily_features` ),
  financial_ratios_firm_level AS (
  SELECT
    permno,
    PARSE_DATE('%Y%m%d',
      public_date) AS public_date,
    bm,
    evm,
    roa,
    roe,
    capital_ratio,
    short_debt,
    cash_ratio,
    quick_ratio,
    (
    SELECT
      MIN(date)
    FROM
      sp_daily_features s
    WHERE
      s.permno = f.permno
      AND s.date >= PARSE_DATE('%Y%m%d',
        public_date) ) AS next_date
  FROM
    `silicon-badge-274423.financial_datasets.financial_ratios_firm_level` f
  WHERE
    permno IN (
    SELECT
      PERMNO
    FROM
      `silicon-badge-274423.financial_datasets.sp_constituents_historical`
    WHERE
      finish > '1980-01-01'))
  SELECT
    *
  FROM
    sp_daily_features AS sdf
  FULL OUTER JOIN
    financial_ratios_firm_level AS fl
  ON
    sdf.permno = fl.permno
    AND sdf.date = fl.next_date
"""

FEATURE_LIST = [
  "bm",
  "evm",
  "roa",
  "roe",
  "capital_ratio",
  "short_debt",
  "cash_ratio",
  "quick_ratio"
]

QUERY = """
  SELECT
    date, permno, public_date, ticker, adjusted_prc, adjusted_vol, exchcd, bm, evm, roa, roe, capital_ratio, short_debt, cash_ratio, quick_ratio
  FROM
    `silicon-badge-274423.financial_datasets.sp_price_fundamentals`
"""

# First join to next known date for public_dates that don't report.
client = bigquery.Client(project='silicon-badge-274423')
features_df = client.query(QUERY).to_dataframe()

# Fill in ones with a date from above.
result_df = pd.DataFrame()
permno_list = features_df.permno.unique()
for i, permno in enumerate(permno_list):
  df = features_df[features_df.permno == permno].sort_values(by='date')
  df = df.ffill()

  # TODO: Should we relax the missing values constraint?
  df = df.dropna(subset=FEATURE_LIST)
  # df = df.dropna(subset=FEATURE_LIST, thresh=len(df.columns) - 3)

  result_df = result_df.append(df)
  print(f"Finished with { i }/{ len(permno_list) }")

result_df = result_df.sort_values(by=['date', 'permno'])

result_df.to_csv('temp.csv', index=False)
job_config = bigquery.LoadJobConfig(
    source_format=bigquery.SourceFormat.CSV, skip_leading_rows=1,
    autodetect=True,
    write_disposition='WRITE_TRUNCATE'
)

with open('temp.csv', "rb") as source_file:
    job = client.load_table_from_file(source_file, "silicon-badge-274423.features.sp_daily_fund_features_v1", job_config=job_config)

job.result()  # Waits for the job to complete.

os.remove("temp.csv")
