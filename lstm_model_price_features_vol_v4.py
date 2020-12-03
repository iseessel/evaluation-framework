# TODO Move to evaluations folder. Couldn't figure out how to import models, etc.,
from models.lstm_model import LSTMModel
from evaluation_framework import EvaluationFramework
from google.cloud import bigquery
import numpy as np

"""
    Receives X_train, Y_train, and X_train as Pandas Dataframe.
    Returns as Dict of Numpy Arrays
"""

DATASET = 'silicon-badge-274423.features.price_features_vol_v4'

FEATURES = [
    'average_daily_return_local_z', 'beta_local_z', 'beta_bear_local_z',
    'beta_bull_local_z', 'gain_loss_local_z', 'kurtosis_local_z',
    'market_correlation_local_z', 'returns_bear_local_z', 'returns_bull_local_z',
    'volatility_local_z'
]


def glue(x_train, y_train, x_test, y_test, y_train_vol, y_test_vol):
    x_train = x_train[FEATURES].to_numpy()
    x_train = x_train.reshape(-1, 1, x_train.shape[1])

    y_train = y_train.to_numpy().reshape(-1, 1)

    # Need a way of mapping a y_test example to a prediction date
    permno_dates = {
        'permno': x_test.permno.to_numpy(),
        'date': x_test.date.to_numpy(),
        'prediction_date': x_test.prediction_date.to_numpy()
    }

    x_test = x_test[FEATURES].to_numpy()
    x_test = x_test.reshape(-1, 1, x_test.shape[1])

    y_test = y_test.to_numpy().reshape(-1, 1)

    if y_train_vol is not None:
        y_train_vol = y_train_vol.to_numpy().reshape(-1, 1)

    if y_test_vol is not None:
        y_test_vol = y_test_vol.to_numpy().reshape(-1, 1)

    return (x_train, y_train, x_test, y_test, permno_dates, y_train_vol, y_test_vol)


QUERY = f"""
  SELECT
        DISTINCT permno
  FROM
        `{ DATASET }`
"""

client = bigquery.Client(project='silicon-badge-274423')
all_permnos = client.query(QUERY).to_dataframe()['permno'].tolist()

args = {
    'client': bigquery.Client(project='silicon-badge-274423'),
    'model': LSTMModel,
    'permnos': all_permnos,
    'dataset': DATASET,
    'features': FEATURES,
    'start': '2000-01-01',
    'end': '2019-12-31',
    'offset': '2010-01-01',
    'increments': 6,
    'hypers': {},
    'evaluation_timeframe': [180],
    'evaluation_table_id': f'silicon-badge-274423.stock_model_evaluation.lstm_model_price_features_vol_v4_prod',
    'pooled': True,
    'glue': glue,
    'options': {
        'returns_from_t': True
    }
}

preds = EvaluationFramework(**args)
eval = preds.eval()
