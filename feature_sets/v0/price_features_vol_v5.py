"""
The below query created v5.

SELECT
  v1.*, v3.target as target_vol
FROM
  `silicon-badge-274423.features.price_features_v1` v1
INNER JOIN
  `silicon-badge-274423.features.price_features_vol_v3` v3
ON v1.permno = v3.permno AND v1.date = v3.date
"""
