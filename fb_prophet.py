from models.fb_prophet import FBProphet
from stock_predictions import StockPredictions
from models.lstm import LSTMModel
from google.cloud import bigquery
import os

# Separate permnos into 10 chunks for parallel training
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    return [lst[i:i + n] for i in range(0, len(lst), n)]

QUERY = """
  SELECT
      DISTINCT permno
  FROM
      `silicon-badge-274423.features.sp_daily_features`
"""

client = bigquery.Client(project='silicon-badge-274423')
all_permnos = client.query(QUERY).to_dataframe()['permno'].tolist()
chunked_permno = chunks(all_permnos, 51)
chunk_num = int(os.environ['CHUNK_NUMBER'])
permnos = chunked_permno[chunk_num]
dataset = 'silicon-badge-274423.features.sp_daily_features'

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

preds = StockPredictions(**args)
eval = preds.eval()
