class Model:
    def __init__(self, **kwargs):
      self.hypers = kwargs.get('hypers', {})

    def fit(self, data):
      raise('#fit method not yet implemented!')

    def predict(self, periods_ahead, features=None):
      raise('#predict method not yet implemented.')
