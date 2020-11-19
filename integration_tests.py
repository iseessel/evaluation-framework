from models.fb_prophet import FBProphet
from stock_predictions import StockPredictions
from google.cloud import bigquery

"""
Training FB Prophet on the aapl stock.
"""

args = {
    'client': bigquery.Client(project='silicon-badge-274423'),
    'model': FBProphet,
    'permnos': ['14593'],
    'dataset': 'silicon-badge-274423.features.sp_daily_features',
    'features': ['adjusted_prc'],
    'start': '1990-01-01',
    'end': '2000-12-31',
    'offset': '2000-01-01',
    'increments': 180,
    'hypers': {},
    'evaluation_timeframe': [180]
}

preds = StockPredictions(**args)
eval = preds.eval()
print(eval)
