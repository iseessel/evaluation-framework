from fbprophet import Prophet
import pandas as pd

class FBProphet(Model):
  """
    Thin wrapper around the FBProphet Class. Used to fit the model on a set
  """

  def __init__(self, **kwargs):
    self.hypers = kwargs.get('hypers', {})
    self.trained_model = None

  """
    Fits the model with the test data.

    :param data: Pandas Dataframe. Column date(type: datetime) and adjusted_prc(type: float).
    :return: self
  """
  def fit(self, data):
    # Prepare the data for the FB Prophet train method.
    prophet_df = data.copy(deep=True)
    prophet_df.rename(columns={'adjusted_prc':'y', 'date': 'ds'}, inplace=True)
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

    # Set prediction interval to 95%
    self.trained_model = Prophet(**self.hypers, interval_width = 0.95).fit(prophet_df)
    return self
  """
    Predicts the future given the trained model.

    :param periods_ahead: List<Int>. ex. periods_ahead = [7, 30, 90] Considering our last known value is at time t, we will predict [t + 7, t + 30, t + 90].
    :param features: List<string>. Facebook Prophet predict methods do not take features, as you must fit. Argument here is used for monkey patching.
  """
  def predict(self, periods_ahead, features=None):
    if not self.trained_model:
      raise("Need to Train the model!")

    future = self.trained_model.make_future_dataframe(periods=sorted(periods_ahead)[-1], include_history=False, freq='D')
    forecast = self.trained_model.predict(future)
    forecast = forecast[['ds', 'yhat', 'yhat_upper', 'yhat_lower']]

    # Assume that the prediction is normally distributed with mean y. Hence std_dev = (yhat_upper - yhat_lower) / 4
    forecast['std_dev'] = (forecast.yhat_upper - forecast.yhat_lower)/4
    forecast.rename(columns={'yhat':'adjusted_prc_pred', 'ds': 'date'}, inplace=True)
    forecast = forecast[['adjusted_prc_pred', 'date', 'std_dev']]

    # Return only the periods we care about.
    return forecast.iloc[[x-1 for x in periods_ahead]]
