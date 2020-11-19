import pandas as pd
from datetime import datetime, timedelta
import json

class StockModelTrainer:
  """
    Class is used to train and evaluate one stock given one model for a given train/test dataset and evaluation timeframe.

    :param model: Model Instance (see model.py and fb_prophet.py for examples).
    :param permno: Int representing the Permno of the stock.
    :param train: Training dataframe. Date as a datetime column. The last value of train is considered "t" or the "prediction_date".
    :param test: Test dataframe. Same format as train, except the future values.
    :param evaluation_timeframe: Array<Int>. Timeframes to predict (see below).

    e.g. With the following parameters:
      Model = FBPRophet, Permno = AAPL, train = train_dataframe, test = test_dataframe, evaluation_timeframe = [1,7,30,90].

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
    self.permno = kwargs['permno'] #TODO: Support pooled models using multiple stock data.
    self.train = kwargs['train']
    self.test = kwargs['test']
    self.evaluation_timeframe = sorted(kwargs['evaluation_timeframe'])
    self.hypers = kwargs['hypers']

  def fit(self):
    return self.model.fit(self.train)

  def evaluate(self):
    predictions = self.model.predict(self.evaluation_timeframe, features=self.train)

    # Collate the predictions with the test data.
    test_data = self.__interpolate_test(predictions)
    # import pdb; pdb.set_trace()
    test_data = test_data.loc[predictions['date'].dt.strftime('%Y-%m-%d')]
    # import pdb; pdb.set_trace()

    test_data['date'] = test_data.index.to_timestamp()
    test_data = pd.merge(test_data, predictions, on='date')

    test_data['permno'] = self.train['permno']
    test_data['ticker'] = self.train['ticker']
    test_data['model'] = self.model.__class__.__name__
    test_data['hypers'] = json.dumps(self.hypers)
    test_data['MSE'] = (test_data.adjusted_prc - test_data.adjusted_prc_pred)**2 # Mean Square Error.
    test_data['MAPE'] = abs((test_data.adjusted_prc - test_data.adjusted_prc_pred)/test_data.adjusted_prc) # Mean Average Percent Error.

    # Get last "known" price, in order to predict whether or not we predicted the correct direction.
    # TODO: Threshold for this; e.g. +0.2%, -0.1%? Only use correct direction of the portfolio selection.
    test_data['prediction_date'] =  self.train['date'].max()
    test_data['adjusted_prc_last'] = self.train.loc[self.train['date'] == self.train['date'].max()]['adjusted_prc'].iloc[0]
    test_data['correct_direction'] = ((test_data.adjusted_prc - test_data.adjusted_prc_last) * (test_data.adjusted_prc_pred - test_data.adjusted_prc_last) > 0) # Guessed Correct Direction.
    # TODO: Potentially allow for non-normally distributed error.
    test_data['within_pred_int'] = (((test_data.adjusted_prc_pred + 2 * test_data.std_dev) > test_data.adjusted_prc) &  ((test_data.adjusted_prc_pred - 2 * test_data.std_dev) < test_data.adjusted_prc)) #Actual within Confident Interval.

    # TODO: Potentially predict the volatility for each of the stocks. For the framework we can calculate explicitly.
    # 6 months from now what is the predicted "volatility of the stock" => (Standard deviation of returns daily * root(N))
    # => Log(returns)

    test_data = test_data[['permno', 'ticker', 'model', 'hypers', 'date', 'adjusted_prc_last', 'adjusted_prc_pred', 'std_dev', 'adjusted_prc', 'MSE', 'MAPE', 'correct_direction', 'within_pred_int']]
    return test_data

  # TODO: Improve missing data handling. e.g. Every 6 months.
  def __interpolate_test(self, predictions):
    test_data = self.test
    test_data = test_data.set_index('date')
    test_data.index = pd.to_datetime(test_data.index)
    test_data.index = test_data.index.to_period("D")

    min_date = self.test['date'].min()
    max_date = self.test['date'].max()
    idx = pd.period_range(min_date, max_date)
    test_data = test_data.reindex(idx)
    # TODO: Logic to only forward fill.
    # Interpolate by finding the nearest value in case the first and the last value or NaN.
    test_data = test_data.ffill().bfill()

    return test_data
