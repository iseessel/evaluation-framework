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
import tensorflow as tf
from datetime import timedelta

class LSTMV2:
    def __init__(self, **kwargs):
        # self.hypers = kwargs.get('hypers', {})
        self.trained_model = {}
        self.permnos = kwargs['permnos']
        self.train_x = kwargs['train_x']
        self.train_y = kwargs['train_y']
        self.test_x = kwargs['test_x']
        self.quantiles = [0.05, 0.5, 0.95]

    def get_model(self, inputshape, quantile):
        mod=Sequential()
        # mod.add(LSTM(units = 64, return_sequences = True, input_shape = (X_train_lstm.shape[1], 2)))
        # mod.add(LSTM(units = 64, return_sequences = True, input_shape = (X_train_lstm.shape[1], num_features)))
        print("Input Shape: ",inputshape)
        mod.add(LSTM(units = 64, return_sequences = True, input_shape = inputshape))
        mod.add(Dropout(0.2))
        mod.add(BatchNormalization())
        mod.add(LSTM(units = 64, return_sequences = True))
        mod.add(Dropout(0.1))
        mod.add(BatchNormalization())

        mod.add((LSTM(units = 64)))
        mod.add(Dropout(0.1))
        mod.add(BatchNormalization())
        mod.add((Dense(units = 16, activation='tanh')))
        mod.add(BatchNormalization())
        # mod.add((Dense(units = 4, activation='tanh')))
        mod.add((Dense(units = 1, activation='tanh')))
        mod.compile(loss=lambda y,f: self.__tilted_loss(quantile,y,f), optimizer='adam', metrics=['accuracy','mean_squared_error'])
        mod.summary()

        return mod
