from models.fb_prophet import FBProphet
from stock_predictions import StockPredictions
from models.lstm import LSTMModel
from google.cloud import bigquery

"""
Training FB Prophet on the aapl stock.
"""

args = {
    'client': bigquery.Client(project='silicon-badge-274423'),
    'model': FBProphet,
    'permnos': ["10104","10107","10138","10145","10516","10696","10909","11404","11403", "14593"],
    'dataset': 'silicon-badge-274423.features.sp_daily_features',
    'features': ['adjusted_prc'],
    'start': '1980-01-01',
    'end': '2019-12-31',
    'offset': '2000-01-01',
    'increments': 180,
    'hypers': {},
    'evaluation_timeframe': [180],
   'evaluation_table_id': 'silicon-badge-274423.stock_model_evaluation.fbprophet_sp_daily_features',
    'pooled': False
}

preds = StockPredictions(**args)
eval = preds.eval()

"""
  Training for LSTM Model.
"""

# from google.cloud import bigquery
#
# args = {
#     'client': bigquery.Client(project='silicon-badge-274423'),
#     'model': LSTMModel,
#     'permnos': ["10104","10107"],
#     'dataset': 'silicon-badge-274423.features.sp_daily_features',
#     'evaluation_table_id': 'silicon-badge-274423.stock_model_evaluation.lstm_sp_daily_features',
#     'features': ['adjusted_prc','adjusted_vol'],
#     'start': '1990-01-01',
#     'end': '2000-12-31',
#     'offset': '2000-01-01',
#     'increments': 180,
#     'hypers': {'a':1},
#     'evaluation_timeframe': [180],
#     'pooled': True
# }
#
# preds = StockPredictions(**args)
# hello = preds.eval()
