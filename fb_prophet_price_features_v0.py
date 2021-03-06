from models.fb_prophet import FBProphet
from stock_predictions import EvaluationFramework
from models.lstm import LSTMModel
from google.cloud import bigquery
import os

"""
    NB: This may no longer work with refactor.
    Results of this model were not good, so not prioritizing.
"""

# Separate permnos into 10 chunks for parallel training


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    return [lst[i:i + n] for i in range(0, len(lst), n)]


QUERY = """
  SELECT
      DISTINCT permno
  FROM
      `silicon-badge-274423.features.price_features_v0`
"""

client = bigquery.Client(project='silicon-badge-274423')
all_permnos = client.query(QUERY).to_dataframe()['permno'].tolist()
chunked_permno = chunks(all_permnos, 51)
chunk_num = int(os.environ['CHUNK_NUMBER'])
permnos = chunked_permno[chunk_num]
dataset = 'silicon-badge-274423.features.price_features_v0'

args = {
    'client': bigquery.Client(project='silicon-badge-274423'),
    'model': FBProphet,
    'permnos': permnos,
    'dataset': dataset,
    'features': ['adjusted_prc'],
    'start': '1980-01-01',
    'end': '2019-12-31',
    'offset': '2000-01-01',
    'increments': 180,
    'hypers': {},
    'evaluation_timeframe': [180],
    'evaluation_table_id': f'silicon-badge-274423.stock_model_evaluation.fbprophet_sp_daily_features_{chunk_num}',
    'pooled': False
}

preds = EvaluationFramework(**args)
eval = preds.eval()

"""
NOTE: This query was run after this executed, because now we are predictin returns NOT anything else.

WITH
  fb_prophet AS (
  SELECT
    permno,
    train_end AS date,
    prediction_date,
    (adjusted_prc_pred - adjusted_prc_train_end) / adjusted_prc_train_end AS returns_prediction,
    (adjusted_prc_actual - adjusted_prc_train_end) / adjusted_prc_train_end AS returns_actual,
    std_dev_pred/adjusted_prc_pred AS vol_prediction,
    model,
    dataset,
    train_start,
    train_end
  FROM
    `silicon-badge-274423.stock_model_evaluation.fb_prophet_sp_daily_features_v0_prod` v0 )
SELECT
  *,
  POWER(returns_prediction - returns_actual, 2) AS returns_mse,
  returns_prediction - returns_actual AS returns_MAPE,
  returns_prediction * returns_actual > 0,
FROM
  fb_prophet
"""
