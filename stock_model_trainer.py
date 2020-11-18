from google.cloud import bigquery
import pandas as pd
from datetime import datetime, timedelta

# TODO: Make this an attribute on the class.
client = bigquery.Client(project='silicon-badge-274423')

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

  def fit(self):
    return self.model.fit(self.train)

  def evaluate(self):
    predictions = self.model.predict(self.evaluation_timeframe, features=self.train)

    # Collate the predictions with the test data.
    test_data = self.__interpolate_test(predictions)
    test_data = test_data.loc[predictions['date'].dt.strftime('%Y-%m-%d')]

    test_data['date'] = test_data.index.to_timestamp()
    test_data = pd.merge(test_data, predictions, on='date')

    test_data['MSE'] = (test_data.adjusted_prc - test_data.adjusted_prc_pred)**2 # Mean Square Error.
    test_data['MAPE'] = abs((test_data.adjusted_prc - test_data.adjusted_prc_pred)/test_data.adjusted_prc) # Mean Average Percent Error.

    # Get last "known" price, in order to predict whether or not we predicted the correct direction.
    test_data['prediction_date'] =  self.train['date'].max()
    test_data['adjusted_prc_last'] = self.train.loc[self.train['date'] == self.train['date'].max()]['adjusted_prc'].iloc[0]
    test_data['correct_direction'] = ((test_data.adjusted_prc - test_data.adjusted_prc_last) * (test_data.adjusted_prc_pred - test_data.adjusted_prc_last) > 0) # Guessed Correct Direction.
    # TODO: Potentially allow for non-normally distributed error.
    test_data['within_pred_int'] = (((test_data.adjusted_prc_pred + 2 * test_data.std_dev) > test_data.adjusted_prc) &  ((test_data.adjusted_prc_pred - 2 * test_data.std_dev) < test_data.adjusted_prc)) #Actual within Confident Interval.

    test_data = test_data[['date', 'adjusted_prc_last', 'adjusted_prc_pred', 'std_dev', 'adjusted_prc', 'MSE', 'MAPE', 'correct_direction', 'within_pred_int']]
    return test_data

  def __interpolate_test(self, predictions):
    test_data = self.test
    test_data.set_index('date', inplace=True)
    test_data.index = pd.to_datetime(test_data.index)
    test_data.index = test_data.index.to_period("D")

    min_date = predictions['date'].min()
    max_date = predictions['date'].max()
    idx = pd.period_range(min_date, max_date)
    test_data = test_data.reindex(idx)
    # Interpolate linearly. If first or last value is NaN, the value will be left as NaN.
    test_data = test_data.interpolate()
    # Interpolate by finding the nearest value in case the first and the last value ar NaN.
    test_data = test_data.bfill().ffill()

    return test_data
