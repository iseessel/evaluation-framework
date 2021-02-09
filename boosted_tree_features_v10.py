# TODO Move to evaluations folder. Couldn't figure out how to import models, etc.,
from models.boosted_tree import BoostedTree
from stock_pickers.non_linear_optimization_v0 import NonLinearOptimization
from google.cloud import bigquery
from utils import scikit_learn_glue, STANDARD_WEIGHT_CONSTRAINTS
from experiment import Experiment

FEATURES_DATASET = 'silicon-badge-274423.features.features_v10'
PREDICTIONS_DATASET = f'silicon-badge-274423.stock_model_evaluation.boosted_tree_features_v10'
FEATURES = [
    'average_daily_return_local_z', 'beta_local_z', 'beta_bear_local_z',
    'beta_bull_local_z', 'gain_loss_local_z', 'kurtosis_local_z',
    'market_correlation_local_z', 'returns_bear_local_z', 'returns_bull_local_z',
    'volatility_local_z'
]

PERMNO_QUERY = f"""
  SELECT
        DISTINCT permno
  FROM
        `{ FEATURES_DATASET }`
"""

client = bigquery.Client(project='silicon-badge-274423')
all_permnos = client.query(PERMNO_QUERY).to_dataframe()['permno'].tolist()

evaluation_framework_kwargs = {
    'client': bigquery.Client(project='silicon-badge-274423'),
    'model': BoostedTree,
    'permnos': all_permnos,
    'features_dataset': FEATURES_DATASET,
    'features': FEATURES,
    'start': '1970-01-01',
    'end': '2019-12-31',
    'offset': '1980-01-01',
    'increments': 6,
    'hypers': {},
    'evaluation_timeframe': [180],
    'evaluation_table_id': PREDICTIONS_DATASET,
    'pooled': True,
    'glue': scikit_learn_glue,
    'options': {
        'returns_from_t': True
    }
}

portfolio_creator_kwargs = {
    'weight_constraints': STANDARD_WEIGHT_CONSTRAINTS,
    'predictions_dataset': PREDICTIONS_DATASET,
    'stock_picker': NonLinearOptimization
}

kwargs = {
    'evaluation_framework_kwargs': None,
    'portfolio_creator_kwargs': portfolio_creator_kwargs
}

experiment = Experiment(**kwargs)
experiment.run()
