from models.lstm import LSTMModel
from stock_predictions import EvaluationFramework
from google.cloud import bigquery
import os

QUERY = """
  SELECT
      DISTINCT permno,
      COALESCE((
      SELECT
        MIN(date)
      FROM
        `silicon-badge-274423.financial_datasets.sp_timeseries_daily` std2
      WHERE
        std1.permno = std2.permno
-- Predict 6 months in the future. Note, may not always be exactly 6 months due to weekends/holidays.
        AND std2.date >= DATE_ADD(std1.date, INTERVAL 6 MONTH)), DATE_ADD(std1.date, INTERVAL 6 MONTH))  as prediction_date
  FROM
      `silicon-badge-274423.features.price_features_v0` std1
"""

client = bigquery.Client(project='silicon-badge-274423')
all_permnos = client.query(QUERY).to_dataframe()['permno'].tolist()
dataset = 'silicon-badge-274423.features.price_features_v0'

args = {
    'client': bigquery.Client(project='silicon-badge-274423'),
    'model': LSTMModel,
    'permnos': all_permnos,
    'dataset': dataset,
    'features': ['adjusted_prc', 'adjusted_vol'],
    'start': '1980-01-01',
    'end': '2019-12-31',
    'offset': '2000-01-01',
    'increments': 180,
    'hypers': {},
    'evaluation_timeframe': [180],
    'evaluation_table_id': f'silicon-badge-274423.stock_model_evaluation.lstm_sp_daily_features',
    'pooled': True
}

preds = EvaluationFramework(**args)
eval = preds.eval()
