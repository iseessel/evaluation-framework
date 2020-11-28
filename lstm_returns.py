from models.lstm import LSTMModel
from stock_predictions import StockPredictions
from google.cloud import bigquery
import os

QUERY = """
  SELECT
      DISTINCT permno
  FROM
      `silicon-badge-274423.features.sp_daily_featuresr`
"""

client = bigquery.Client(project='silicon-badge-274423')
all_permnos = client.query(QUERY).to_dataframe()['permno'].tolist()
dataset = 'silicon-badge-274423.features.sp_daily_features'

args = {
    'client': bigquery.Client(project='silicon-badge-274423'),
    'model': LSTMModel,
    'permnos': all_permnos,
    'dataset': dataset,
    'features': ['adjusted_prc'],
    'start': '1980-01-01',
    'end': '2019-12-31',
    'offset': '2000-01-01',
    'increments': 180,
    'hypers': {},
    'evaluation_timeframe': [180],
    'evaluation_table_id': f'silicon-badge-274423.stock_model_evaluation.lstm_sp_daily_features',
    'pooled': True
}

preds = StockPredictions(**args)
eval = preds.eval()
