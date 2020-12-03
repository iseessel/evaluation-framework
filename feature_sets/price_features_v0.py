from google.cloud import bigquery

"""
Note: This feature set is deprecated.
"""

QUERY = """
WITH
  sp_daily_features AS (
  SELECT
    date,
    permno,
    ticker,
    CASE
      # When prc is 0 it is not reported. Sometimes price is negative when its an estimate of the closing price.
      WHEN prc = 0 THEN NULL
    ELSE
    ABS(prc/cfacpr)
  END
    AS adjusted_prc,
    CASE
    # Volume is set to -99 if the value is missing. A volume of zero usually indicates that there were no trades during the time period and is usually paired with bid/ask quotes in price fields.
      WHEN vol >= 0 THEN vol/cfacshr
    ELSE
    NULL
  END
    AS adjusted_vol,
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
  WHERE
    PERMNO IN (
    SELECT
      PERMNO
    FROM
      `silicon-badge-274423.financial_datasets.sp_constituents_historical`
    WHERE
      # Select only current S&P Stocks. Opportunity to add more.
      finish = (
      SELECT
        MAX(finish)
      FROM
        `silicon-badge-274423.financial_datasets.sp_constituents_historical` ) )
    AND date >= '1980-01-01'
  ORDER BY
    permno,
    date )
SELECT
  sp_daily_features.permno,
  sp_daily_features.date,
  sp_daily_features.adjusted_prc,
  sp_daily_features.adjusted_vol,
  sp_daily_features.prediction_date,
  (CASE
      WHEN std.prc = 0 THEN NULL
    ELSE
    ABS(std.prc/std.cfacpr) END) AS target
FROM
  sp_daily_features
LEFT JOIN
  `silicon-badge-274423.financial_datasets.sp_timeseries_daily` std
ON
  std.permno = sp_daily_features.permno
  AND std.date = sp_daily_features.prediction_date
"""

client = bigquery.Client(project='silicon-badge-274423')
df = client.query(QUERY).to_dataframe()

df.date = df.date.astype('string')
df.prediction_date = df.prediction_date.astype('string')

"""
    Upload to Bigquery.
    TODO: This was stalling indefinitely. Have uploaded to bigquery manually.
"""
# TODO: Make this a utils class. Improve this method.
df.to_csv('temp.csv', index=False)
# job_config = bigquery.LoadJobConfig(
#   source_format=bigquery.SourceFormat.CSV, skip_leading_rows=1,
#   autodetect=True,
#   write_disposition='WRITE_TRUNCATE'
# )
#
# with open('temp.csv', "rb") as source_file:
#   job = client.load_table_from_file(source_file, 'silicon-badge-274423.features.price_features_v0', job_config=job_config)
#
# job.result()  # Waits for the job to complete.
#
# os.remove("temp.csv")
#
# print("Uploaded to Bigquery")
