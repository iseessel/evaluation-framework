"""
    Runs a full experiment start to finish.

    1. Train model and predict stock prices.
    2. Choose portfolios for each time period.
    3. Evaluate portfolio according to several key metrics.
"""

from portfolio_creator import PortfolioCreator
from evaluation_framework import EvaluationFramework
from google.cloud import bigquery
import tensorflow as tf
import numpy as np

class Experiment:
    def __init__(self, **kwargs):
        self.evaluation_framework_kwargs = kwargs['evaluation_framework_kwargs']
        self.portfolio_creator_kwargs = kwargs['portfolio_creator_kwargs']

    def run(self):
        print("####################################################")
        print(f"Beggining new Experiment!")
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        print(f"Evaluation Framework Kwargs: { self.evaluation_framework_kwargs }\n")
        print(f"Portfolio Creator Kwargs : { self.portfolio_creator_kwargs }")
        print("####################################################\n")

        if self.evaluation_framework_kwargs:
            preds = EvaluationFramework(**self.evaluation_framework_kwargs)
            eval = preds.eval()

        portfolio_datasets = []
        if self.portfolio_creator_kwargs:
            portfolio_datasets = self.__create_portfolios()

    def __create_portfolios(self):
        weight_constraints = self.portfolio_creator_kwargs['weight_constraints']
        stock_picker = self.portfolio_creator_kwargs['stock_picker']
        predictions_dataset = self.portfolio_creator_kwargs['predictions_dataset']


        portfolio_datasets = []
        for constraint in weight_constraints:
            stock_constraint, bond_constraint = constraint
            dataset = predictions_dataset.split('.')[0]
            dset_name = f"{ dataset }_stocks_{str(int(stock_constraint[1] * 100))}_bonds_{str(int(bond_constraint[1] * 100))}"
            target_table = f'silicon-badge-274423.portfolio.{ dset_name }'

            portfolio_datasets.append(target_table)

            print("####################################################")
            print(f"Stock Constraints: { constraint[0] }. Bond Constraints: { constraint[1] }")
            print(f"Predictions Dataset: { predictions_dataset }")
            print(f"Target Table: { target_table}")
            print("####################################################\n")

            kwargs = {
                'stock_picker': stock_picker,
                'dataset': predictions_dataset,
                'client': bigquery.Client(project='silicon-badge-274423'),
                'num_candidate_stocks': 40,
                'target_table_id': target_table,
                'options': {
                    'constraint': constraint
                }
            }

            x = PortfolioCreator(**kwargs)
            x.pick_stocks()

        return portfolio_datasets
