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
    self.permnos = kwargs['permnos'] #TODO: Support pooled models using multiple stock data.
    self.train = kwargs['train']
    self.test = kwargs['test']
    self.evaluation_timeframe = sorted(kwargs['evaluation_timeframe'])
    self.hypers = kwargs['hypers']
    self.dataset = kwargs['dataset']

  def fit(self):
    return self.model.fit()

  def evaluate(self):
    evaluation_df = pd.DataFrame()

    # Evaluate for each permno
    for train_data, test_data in zip(self.train, self.test):
      print(f"Permno: {self.permnos}.")
      print(f"Min Train Date: {str(train_data['date'].min())}.")
      print(f"Max Train Date: {str(train_data['date'].max())}.")
      print(f"Max Test Date: {str(test_data['date'].max())}.")

      predictions = self.model.predict(self.evaluation_timeframe)

      # Collate the predictions with the test data.
      test_data = self.__interpolate_test(predictions, test_data)
      test_data = test_data.loc[predictions['date'].dt.strftime('%Y-%m-%d')]

      test_data['date'] = test_data.index.to_timestamp()
      test_data['adjusted_prc_actual'] = test_data['adjusted_prc']
      test_data = pd.merge(test_data, predictions, on=['date', 'permno'])

      # test_data['permno'] = self.permno
      test_data['ticker'] = train_data['ticker'][train_data['ticker'].notnull()].iloc[0]
      test_data['model'] = self.model.__class__.__name__
      test_data['hypers'] = json.dumps(self.hypers)

      test_data['train_start'] = str(train_data['date'].min())
      test_data['train_end'] = str(train_data['date'].max())
      test_data['prediction_date'] = test_data['date'].astype(str)
      test_data['dataset'] = self.dataset
      test_data['features'] = ','.join(train_data.columns.values.tolist())

      test_data['MSE'] = (test_data.adjusted_prc_actual - test_data.adjusted_prc_pred)**2 # Mean Square Error.
      test_data['MAPE'] = abs((test_data.adjusted_prc_actual - test_data.adjusted_prc_pred)/test_data.adjusted_prc_actual) # Mean Average Percent Error.
      # Get last "known" price, in order to predict whether or not we predicted the correct direction.
      # TODO: Threshold for this; e.g. +0.2%, -0.1%? Only use correct direction of the portfolio selection.
      test_data['adjusted_prc_train_end'] = train_data.loc[train_data['date'] == train_data['date'].max()]['adjusted_prc'].iloc[0]
      test_data['correct_direction'] = ((test_data.adjusted_prc_actual - test_data.adjusted_prc_train_end) * (test_data.adjusted_prc_pred - test_data.adjusted_prc_train_end) > 0) # Guessed Correct Direction.
      # TODO: Potentially allow for non-normally distributed error.
      test_data['within_pred_int'] = (((test_data.adjusted_prc_pred + 2 * test_data.std_dev_pred) > test_data.adjusted_prc_actual) &  ((test_data.adjusted_prc_pred - 2 * test_data.std_dev_pred) < test_data.adjusted_prc_actual)) #Actual within Confident Interval.

      evaluation_df = evaluation_df.append(test_data)

    evaluation_df = evaluation_df[
      [
        'permno', 'ticker', 'model', 'dataset', 'features', 'hypers', 'train_start',
        'train_end', 'prediction_date', 'adjusted_prc_train_end', 'adjusted_prc_pred', 'std_dev_pred',
        'adjusted_prc_actual', 'MSE', 'MAPE', 'correct_direction', 'within_pred_int'
      ]
    ]

    return evaluation_df

  # TODO: Improve missing data handling. e.g. Every 6 months.
  def __interpolate_test(self, predictions, test_data):
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
