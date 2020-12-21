"""
WITH
  sp_daily_features AS (
  SELECT
    date,
    std1.permno,
    ticker,
    ret,
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
  LEFT JOIN
     `silicon-badge-274423.financial_datasets.sp_constituents_historical` historical
  ON
    std1.permno = historical.permno
  WHERE
    std1.date >= historical.start AND std1.date <= historical.finish
  ORDER BY
    std1.permno,
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
