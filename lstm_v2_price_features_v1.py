# TODO Move to evaluations folder. Couldn't figure out how to import models, etc.,
from models.lstm_v2 import LSTMV2
from evaluation_framework.evaluation_framework import EvaluationFramework

"""
    Receives X_train, Y_train, and X_train as Pandas Dataframe.
    Returns as Dict of Numpy Arrays
"""
def glue(train_df, test_df):
    X_train = np.array(train_df.adjusted_ret.tolist())
    X_train = X_train.reshape(-1, X_train.shape[1], 1)

    Y_train = train_df.target.to_numpy()
    Y_train = Y_train.reshape(-1,1)

    X_test = np.array(test_df.adjusted_ret.tolist())
    X_test = X_train.reshape(-1, X_train.shape[1], 1)

    Y_test = test_df.target.to_numpy()
    Y_test = Y_test.reshape(-1,1)

    return (X_train, Y_train, X_test, Y_test)

DATASET = 'silicon-badge-274423.features.price_features_v1'

QUERY = f"""
  SELECT
        DISTINCT permno
  FROM
        `{ DATASET }`
"""

client = bigquery.Client(project='silicon-badge-274423')
all_permnos = client.query(QUERY).to_dataframe()['permno'].tolist()
permnos = chunked_permno[chunk_num]
dataset = 'silicon-badge-274423.features.price_features_v0'

args = {
    'client': bigquery.Client(project='silicon-badge-274423'),
    'model': LSTMV2,
    'permnos': all_permnos,
    'dataset': dataset,
    'features': ['adjusted_prc'],
    'start': '1980-01-01',
    'end': '2019-12-31',
    'offset': '2000-01-01',
    'increments': 180,
    'hypers': {},
    'evaluation_timeframe': [180],
    'evaluation_table_id': f'silicon-badge-274423.stock_model_evaluation.fbprophet_sp_daily_features_{chunk_num}',
    'pooled': False,
    'glue': glue
}

preds = EvaluationFramework(**args)
eval = preds.eval()
