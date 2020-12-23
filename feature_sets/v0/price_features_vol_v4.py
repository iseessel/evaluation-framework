"""
The below query created v4.

SELECT
  v2.*, v3.target as target_vol
FROM
  `silicon-badge-274423.features.price_features_v2` v2
INNER JOIN
  `silicon-badge-274423.features.price_features_vol_v3` v3
ON v2.permno = v3.permno AND v2.date = v3.date
"""
