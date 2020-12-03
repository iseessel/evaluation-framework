from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import BatchNormalization
import datetime as dt
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf
from datetime import timedelta


class LSTMModel:
    def __init__(self, **kwargs):
        # self.hypers = kwargs.get('hypers', {})
        self.trained_model = {}
        self.x_train = kwargs['x_train']
        self.y_train = kwargs['y_train']
        self.x_test = kwargs['x_test']
        self.y_test = kwargs['y_test']
        self.y_train_vol = kwargs['y_train_vol']
        self.permno_dates = kwargs['permno_dates']
        self.options = kwargs['options']

    def fit(self):
        model = self.__get_model()
        # Epochs and batch size have been tested in jupyter notebook.
        model.fit(self.x_train, self.y_train, epochs=1, batch_size=1024)
        self.trained_model['price'] = model

        if self.y_train_vol is not None:
            # Epochs and batch size have been tested in jupyter notebook.
            model = self.__get_model()
            model.fit(self.x_train, self.y_train_vol,
                      epochs=3, batch_size=1024)
            self.trained_model['vol'] = model

        return True

    def predict(self):
        if self.options['returns_from_t']:
            predicted_returns, actual_returns = self.__predict_returns_from_t()
        else:
            predicted_returns, actual_returns = self.__predict_returns_t_minus_window()

        vol_prediction = None
        if self.y_train_vol is not None:
            vol_prediction = self.trained_model['vol'].predict(
                self.x_test).reshape(-1)

        predictions_dic = {
            'permno': self.permno_dates['permno'],
            'date': self.permno_dates['date'],
            'prediction_date': self.permno_dates['prediction_date'],
            'return_prediction': predicted_returns,
            'return_target': actual_returns,
            'vol_prediction': vol_prediction
            # TODO: Add in standard deviation
        }

        predictions_df = pd.DataFrame(predictions_dic)

        return predictions_df

    def __predict_returns_from_t(self):
        predicted_returns = self.trained_model['price'].predict(
            self.x_test).reshape(-1)
        actual_returns = self.y_test.reshape(-1)

        return predicted_returns, actual_returns

    def __predict_returns_from_t_minus_window(self):
        predicted_returns = self.trained_model['price'].predict(self.x_test)

        # Get true returns. We are predicting returns from T - 50, but we need returns from T.
        predicted_returns_from_t = []
        actual_returns_from_t = []
        for i, (predicted_ret, target_ret) in enumerate(zip(predicted_returns, self.y_test)):
            p_ret = 1 + predicted_ret[0]
            last_return = 1 + self.x_test[i][-1][0]
            return_from_t = (p_ret - last_return) / (last_return)
            predicted_returns_from_t.append(return_from_t)

            target_ret = 1 + target_ret[0]
            return_from_t = (target_ret - last_return) / (last_return)
            actual_returns_from_t.append(return_from_t)

        return predicted_returns_from_t, actual_returns_from_t

    def __get_model(self):
        inputshape = self.x_train.shape[1:]
        mod = Sequential()
        mod.add(LSTM(units=64, return_sequences=True, input_shape=inputshape))
        mod.add(Dropout(0.2))
        mod.add(BatchNormalization())
        mod.add(LSTM(units=64, return_sequences=True))
        mod.add(Dropout(0.1))
        mod.add(BatchNormalization())

        mod.add((LSTM(units=64)))
        mod.add(Dropout(0.1))
        mod.add(BatchNormalization())
        mod.add((Dense(units=16, activation='relu')))
        mod.add(BatchNormalization())
        # mod.add((Dense(units = 4, activation='tanh')))
        mod.add((Dense(units=1, activation='relu')))
        mod.compile(loss=MeanSquaredError(), optimizer='adam', metrics=[
                    'accuracy', 'mean_squared_error'])
        mod.summary()

        return mod
