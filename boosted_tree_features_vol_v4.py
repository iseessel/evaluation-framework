# TODO Move to evaluations folder. Couldn't figure out how to import models, etc.,
from models.boosted_tree import BoostedTree
from evaluation_framework import EvaluationFramework
from google.cloud import bigquery
import numpy as np

"""
    Receives X_train, Y_train, and X_train as Pandas Dataframe.
    Returns as Dict of Numpy Arrays
"""


def glue(x_train, y_train, x_test, y_test, y_train_vol, y_test_vol):
    import pdb
    pdb.set_trace()
    # x_train = np.array(x_train.adjusted_rets.tolist())
    # x_train = x_train.reshape(-1, x_train.shape[1], 1)
    #
    # y_train = y_train.to_numpy().reshape(-1, 1)
    #
    # # Need a way of mapping a y_test example to a prediction date
    # permno_dates = {
    #     'permno': x_test.permno.to_numpy(),
    #     'date': x_test.date.to_numpy(),
    #     'prediction_date': x_test.prediction_date.to_numpy()
    # }
    #
    # x_test = np.array(x_test.adjusted_rets.tolist())
    # x_test = x_test.reshape(-1, x_test.shape[1], 1)
    #
    # y_test = y_test.to_numpy().reshape(-1, 1)
    #
    # if y_train_vol is not None:
    #     y_train_vol = y_train_vol.to_numpy().reshape(-1, 1)
    #
    # if y_test_vol is not None:
    #     y_test_vol = y_test_vol.to_numpy().reshape(-1, 1)
    #
    # return (x_train, y_train, x_test, y_test, permno_dates, y_train_vol, y_test_vol)


DATASET = 'silicon-badge-274423.features.price_features_vol_v4'

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
    'features': ['adjusted_rets', 'date', 'permno', 'prediction_date'],
    'start': '1980-01-01',
    'end': '2000-06-30',
    'offset': '2000-01-01',
    'increments': 180,
    'hypers': {},
    'evaluation_timeframe': [180],
    'evaluation_table_id': f'silicon-badge-274423.stock_model_evaluation.lstm_model_price_features_vol_v4_test',
    'pooled': True,
    'glue': glue,
    'options': {
        'returns_from_t': True
    }
}

preds = EvaluationFramework(**args)
eval = preds.eval()
