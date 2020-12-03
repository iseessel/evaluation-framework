from sklearn.ensemble import GradientBoostingRegressor
from datetime import timedelta
import pandas as pd


class BoostedTree:
    def __init__(self, **kwargs):
        # self.hypers = kwargs.get('hypers', {})
        self.trained_model = {}
        self.x_train = kwargs['x_train']
        self.x_test = kwargs['x_test']
        self.y_train = kwargs['y_train']
        self.y_train_vol = kwargs['y_train_vol']
        self.y_test = kwargs['y_test']
        self.permno_dates = kwargs['permno_dates']

    def fit(self):
        model = self.__get_model()
        # Epochs and batch size have been tested in jupyter notebook.
        model.fit(self.x_train, self.y_train)
        self.trained_model['price'] = model

        if self.y_train_vol is not None:
            # Epochs and batch size have been tested in jupyter notebook.
            model = self.__get_model()
            model.fit(self.x_train, self.y_train_vol)
            self.trained_model['vol'] = model

        return True

    def predict(self):
        return_prediction = self.trained_model['price'].predict(self.x_test)
        vol_prediction = self.trained_model['vol'].predict(self.x_test)
        actual_returns = self.y_test.reshape(-1)

        predictions_dic = {
            'permno': self.permno_dates['permno'],
            'date': self.permno_dates['date'],
            'prediction_date': self.permno_dates['prediction_date'],
            'return_prediction': return_prediction,
            'return_target': actual_returns,
            'vol_prediction': vol_prediction
        }

        predictions_df = pd.DataFrame(predictions_dic)

        return predictions_df

    def __get_model(self):
        return GradientBoostingRegressor(loss="ls", n_estimators=100, verbose=1m)
