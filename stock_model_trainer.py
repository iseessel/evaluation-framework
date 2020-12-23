import pandas as pd
from datetime import datetime, timedelta, timezone
import json


class StockModelTrainer:
    """
      Class is used to train and evaluate one stock given one model for a given train/test dataset and evaluation timeframe.

      :param model: Model Instance (see model.py and fb_prophet.py for examples).
      :param permno: Int representing the Permno of the stock.
      :param train: Training dataframe. Date as a datetime column. The last value of train is considered "t" or the "prediction_date".
      :param test: Test dataframe. Same format as train, except the future values.

      e.g. With the following parameters:
        Model = FBPRophet, Permno = AAPL, train = train_dataframe, test = test_dataframe

        We can:
          1. Train model on train data.
          2. Predict it for t + 1, t + 7, t + 30, t + 90
          3. Evaluate the predictions based on the test data.
            a. Squared Error (MSE)
            b. Average Percent Error (MAPE)
            c. Correct Direction
            d. Within prediction interval.
          4. Returns Data Frame.
    """

    def __init__(self, **kwargs):
        self.model = kwargs['model']
        # TODO: Support pooled models using multiple stock data.
        self.y_test = kwargs['y_test']
        self.y_test_vol = kwargs['y_test_vol']
        self.hypers = kwargs['hypers']
        self.dataset = kwargs['dataset']
        self.train_start = kwargs['train_start']
        self.train_end = kwargs['train_start']

    def fit(self):
        return self.model.fit()

    def evaluate(self):
        evaluation_df = pd.DataFrame()
        predictions = self.model.predict()

        # Sometimes model will already have return_target.
        # E.g. if there is a special transformation needed to obtain returns from time T.
        if not 'return_target' in predictions.columns:
            # Order will be preserved.
            predictions['return_target'] = self.y_test.reshape(-1)

        if not 'vol_target' in predictions.columns:
            # Order will be preserved.
            predictions['vol_target'] = self.y_test_vol.reshape(-1)

        predictions['returns_correct_direction'] = (
            predictions.return_prediction * predictions.return_target) >= 0

        predictions['model'] = self.model.__class__.__name__
        predictions['train_start'] = self.train_start
        predictions['train_end'] = self.train_end
        predictions['dataset'] = self.dataset

        cols = ['permno', 'date', 'prediction_date', 'return_prediction', 'return_target', 'vol_prediction', 'vol_target',
                'returns_correct_direction', 'model', 'train_start', 'train_end', 'dataset']
        predictions = predictions[cols]

        return predictions
