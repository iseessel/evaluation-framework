# TODO Move to evaluations folder. Couldn't figure out how to import models, etc.,
from models.lstm_v2 import LSTMV2
from evaluation_framework import EvaluationFramework
from google.cloud import bigquery
import numpy as np

"""
    Receives X_train, Y_train, and X_train as Pandas Dataframe.
    Returns as Dict of Numpy Arrays
"""


def glue(x_train, y_train, x_test, y_test, permnos_test):
    x_train = np.array(x_train.adjusted_rets.tolist())
    x_train = x_train.reshape(-1, x_train.shape[1], 1)

    y_train = y_train.to_numpy().reshape(-1, 1)

    x_test = np.array(x_test.adjusted_rets.tolist())
    x_test = x_test.reshape(-1, x_test.shape[1], 1)

    y_test = y_test.to_numpy().reshape(-1, 1)

    permnos_test = permnos_test.to_numpy()

    return (x_train, y_train, x_test, y_test, permnos_test)


DATASET = 'silicon-badge-274423.features.price_features_v1'

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
    'model': LSTMV2,
    'permnos': all_permnos,
    'dataset': DATASET,
    'features': ['adjusted_rets', 'date', 'permno', 'prediction_date'],
    'start': '1980-01-01',
    'end': '2019-12-31',
    'offset': '2000-01-01',
    'increments': 180,
    'hypers': {},
    'evaluation_timeframe': [180],
    'evaluation_table_id': f'silicon-badge-274423.stock_model_evaluation.lstm_v2_price_features_v1_test1',
    'pooled': True,
    'glue': glue
}

preds = EvaluationFramework(**args)
eval = preds.eval()
