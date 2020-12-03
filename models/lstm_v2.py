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


class LSTMV2:
    def __init__(self, **kwargs):
        # self.hypers = kwargs.get('hypers', {})
        self.trained_model = {}
        self.x_train = kwargs['x_train']
        self.x_test = kwargs['x_test']
        self.y_train = kwargs['y_train']
        self.permno_dates = kwargs['permno_dates']

    def fit(self):
        model = self.__get_model()
        self.trained_model['price'] = model.fit(
            self.x_train, self.y_train, epochs=1, batch_size=1024)

        # TODO: Need to get the vol variables here.
        # model = self.__get_model()
        # self.trained_model['vol'] = model.fit(
        #     self.x_train, self.y_train, epochs=1, batch_size=1024)

    def predict(self, periods_ahead, window_size=50):
        predicted_returns = self.trained_model['price'].predict(self.x_test)

        # Get true returns. We are predicting from returns T - 50, but we need returns from T.
        real_rets = []
        for (i, predicted_ret) in enumerate(predicted_returns):
            predicted_ret = 1 + predicted_ret
            last_return = self.x_test[i]
            real_return = (predicted_ret - last_return) / (last_return)
            real_rets.append(real_return)

        predictions_dic = {
            'permno': self.permno_dates[0],
            'date': self.permno_dates[1],
            'adjusted_prc_pred': self.y_train
        }

        pdb.set_trace()
        return
        # predicted_vol = self.trained_model['vol'].predict(X_test)

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

    def __tilted_loss(self, q, y, f):
        e = (y - f)
        return K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)
