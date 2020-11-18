from google.cloud import bigquery
import pandas as pd
from datetime import datetime, timedelta

# # Authenticate with Bigquery. Please see https://colab.research.google.com/notebooks/bigquery.ipynb#scrollTo=SeTJb51SKs_W.
# from google.colab import auth
# auth.authenticate_user()

client = bigquery.Client(project='silicon-badge-274423')

class StockModelTrainer:
  # TODO: Write class documentation for this.
  def __init__(self, **kwargs):
    # Model must have train method, and predict method.
    self.model = kwargs['model']
    # Which stock we are training on. TODO: Support multiple stocks
    self.permno = kwargs['permno']
    # Array of integers representing number of days after train_end.
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
    test_data['adjusted_prc_last'] = self.train.loc[self.train['date'] == self.train['date'].max()]['adjusted_prc'].iloc[0]
    test_data['correct_direction'] = ((test_data.adjusted_prc - test_data.adjusted_prc_last) * (test_data.adjusted_prc_pred - test_data.adjusted_prc_last) > 0) # Guessed Correct Direction.
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
    # Interpolate linearly If first or last value is NaN, the value will be left as NaN.
    test_data = test_data.interpolate()
    # Interpolate by finding the nearest value for the first and the last value.
    test_data = test_data.bfill().ffill()

    return test_data
