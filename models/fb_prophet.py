from fbprophet import Prophet
import pandas as pd

class FBProphet:
  def __init__(self, **kwargs):
    self.hypers = kwargs.get('hypers', {})
    self.trained_model = None

  def fit(self, data):
    # Prepare the data for the FB Prophet train method.
    prophet_df = data.copy(deep=True)
    prophet_df.rename(columns={'adjusted_prc':'y', 'date': 'ds'}, inplace=True)
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

    # Set prediction interval to 95%
    self.trained_model = Prophet(**self.hypers, interval_width = 0.95).fit(prophet_df)
    return self.trained_model

  def predict(self, periods_ahead, features=None):
    if not self.trained_model:
      raise("Need to Train the model!")

    future = self.trained_model.make_future_dataframe(periods=periods_ahead[-1], include_history=False, freq='D')
    forecast = self.trained_model.predict(future)
    forecast = forecast[['ds', 'yhat', 'yhat_upper', 'yhat_lower']]

    # Assume that the prediction is normally distributed with mean y. Hence std_dev = (yhat_upper - yhat_lower) / 4
    forecast['std_dev'] = (forecast.yhat_upper - forecast.yhat_lower)/4
    forecast.rename(columns={'yhat':'adjusted_prc_pred', 'ds': 'date'}, inplace=True)
    forecast = forecast[['adjusted_prc_pred', 'date', 'std_dev']]

    # Return only the periods we care about.
    return forecast.iloc[[x-1 for x in periods_ahead]]
