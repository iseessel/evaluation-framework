from google.cloud import bigquery

"""
How to test permnos.
"""

client = bigquery.Client(project='silicon-badge-274423')
all_permnos = client.query(QUERY).to_dataframe()['permno'].tolist()

DATASETS = [
    "boosted_tree_features_vol_v4_light_prod",
    "boosted_tree_features_vol_v4_light_prod_005",
    "boosted_tree_features_vol_v4_light_prod_01",
    "fb_prophet_sp_daily_features_v0_prod_t",
    "fb_prophet_sp_daily_features_v0_prod_t_005",
    "fb_prophet_sp_daily_features_v0_prod_t_01",
    "lstm_model_price_features_vol_v4_prod",
    "lstm_model_price_features_vol_v4_prod_005",
    "lstm_model_price_features_vol_v4_prod_01",
    "lstm_model_price_features_vol_v4_prod_relu",
    "lstm_model_price_features_vol_v4_prod_relu_005",
    "lstm_model_price_features_vol_v4_prod_relu_01",
    "lstm_model_price_features_vol_v5_prod",
    "lstm_model_price_features_vol_v5_prod_005",
    "lstm_model_price_features_vol_v5_prod_01"
]

QUERY = """
SELECT
  *,
  'boosted_tree_features_vol_v4_light_prod' AS dset
FROM
  `silicon-badge-274423.portfolio.boosted_tree_features_vol_v4_light_prod`
UNION ALL
SELECT
  *,
  'boosted_tree_features_vol_v4_light_prod_005' AS dset
FROM
  `silicon-badge-274423.portfolio.boosted_tree_features_vol_v4_light_prod_005`
UNION ALL
SELECT
  *,
  'boosted_tree_features_vol_v4_light_prod_01' AS dset
FROM
  `silicon-badge-274423.portfolio.boosted_tree_features_vol_v4_light_prod_01`
UNION ALL
SELECT
  *,
  'fb_prophet_sp_daily_features_v0_prod_t' AS dset
FROM
  `silicon-badge-274423.portfolio.fb_prophet_sp_daily_features_v0_prod_t`
UNION ALL
SELECT
  *,
  'fb_prophet_sp_daily_features_v0_prod_t_005' AS dset
FROM
  `silicon-badge-274423.portfolio.fb_prophet_sp_daily_features_v0_prod_t_005`
UNION ALL
SELECT
  *,
  'fb_prophet_sp_daily_features_v0_prod_t_01' AS dset
FROM
  `silicon-badge-274423.portfolio.fb_prophet_sp_daily_features_v0_prod_t_01`
UNION ALL
SELECT
  *,
  'lstm_model_price_features_vol_v4_prod' AS dset
FROM
  `silicon-badge-274423.portfolio.lstm_model_price_features_vol_v4_prod`
UNION ALL
SELECT
  *,
  'lstm_model_price_features_vol_v4_prod_005' AS dset
FROM
  `silicon-badge-274423.portfolio.lstm_model_price_features_vol_v4_prod_005`
UNION ALL
SELECT
  *,
  'lstm_model_price_features_vol_v4_prod_01' AS dset
FROM
  `silicon-badge-274423.portfolio.lstm_model_price_features_vol_v4_prod_01`
UNION ALL
SELECT
  *,
  'lstm_model_price_features_vol_v4_prod_relu_01' AS dset
FROM
  `silicon-badge-274423.portfolio.lstm_model_price_features_vol_v4_prod_01`
UNION ALL
SELECT
  *,
  'lstm_model_price_features_vol_v4_prod_relu' AS dset
FROM
  `silicon-badge-274423.portfolio.lstm_model_price_features_vol_v4_prod_relu`
UNION ALL
SELECT
  *,
  'lstm_model_price_features_vol_v4_prod_relu_005' AS dset
FROM
  `silicon-badge-274423.portfolio.lstm_model_price_features_vol_v4_prod_relu_005`
UNION ALL
SELECT
  *,
  'lstm_model_price_features_vol_v5_prod' AS dset
FROM
  `silicon-badge-274423.portfolio.lstm_model_price_features_vol_v5_prod`
UNION ALL
SELECT
  *,
  'lstm_model_price_features_vol_v5_prod_005' AS dset
FROM
wHERE
    date >= '2009-12-31' and DATE <= '2019-12-30'
"""

for (i, dset) in enumerate(DATASETS):
    q = f"""
        SELECT
            *, '{dataset}' as dataset
        FROM
            `silicon-badge-274423.portfolio.lstm_model_price_features_vol_v4_prod`
    """

    if i != len(DATASETS) - 1:
        q = q + "UNION ALL"
