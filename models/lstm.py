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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
import tensorflow as tf
from datetime import timedelta
import pdb

class LSTMModel:
    def __init__(self, **kwargs):
        # self.hypers = kwargs.get('hypers', {})
        self.trained_model = None
        self.stock_scaler = {}
        self.permnos = kwargs['permnos']
        self.train = kwargs['train']
        self.options = kwargs.get('options',{})

    def CreateTrainData(self, data, permno, window_size = 50):
        trainset = {}
        """
        Target is our label, which is adjusted_prc
        We only need adjusted_prc and adjusted_vol for training,
        so we are only keeping them
        Preparing Data
        """
        # features = ['adjusted_prc','adjusted_vol']
        features_not_to_include = ['adjusted_prc','date','permno','ticker']
        price = ['adjusted_prc']
        features = ['adjusted_vol']
        # features = [col for col in data.columns if col not in features_not_to_include]
        data_lstm = data.copy(deep=True)
        # Other features are here.
        df_train_lstm = data_lstm[features]
        df_price = data_lstm[price]
        df_train_target_lstm = df_price['adjusted_prc']
        df_date = data['date'][:-window_size]
        """
        Scaling the training data
        """

        scaler = MinMaxScaler(feature_range = (0, 1))
        # Reshaping because it only have one dimension, so error is thrown
        train_target_set_lstm = df_train_target_lstm.values.reshape(-1, 1)
        # train_set_lstm = df_train_lstm.values
        if len(features) == 1:
            train_set_lstm = df_train_lstm.values.reshape(-1, 1)
        elif len(features) > 1:
            train_set_lstm = df_train_lstm.values
        train_set_price_lstm = df_price.values.reshape(-1, 1)
        # TODO: Do some research on whether or not we need to fit transform returns?
        # NOt scaling price
        # train_set_price_scaled_lstm = scaler.fit_transform(train_set_price_lstm)
        # TODO: Does it map by column name or by the 0th row.
        training_set_scaled_lstm = scaler.fit_transform(train_set_lstm)
        # TODO: Should transform based on price_scaled_lstm
        # NOt scaling price
        # train_target_set_scaled_lstm = scaler.fit_transform(train_target_set_lstm)

        self.stock_scaler[permno] = scaler
        # stock_scaler[stock] = scaler

        """
        Creating training data
        Using a 50 day window, i.e. Target Value for 50 datapoints will be t+50
        Dimension of X_train_lstm : (n x 50 x 2)
        n -> number of training sets, in this case 7000-50
        50 -> Window size of training set, i.e 50 days
        2 -> Two features ['adjusted_prc','adjusted_vol']

        Dimension of y_train_lstm : (n x 1)
        n -> number of targets, in this case 7000-50
        1 -> Label, i.e, ['adjusted_prc']
        """
        X_train_lstm = []
        X_train_price_lstm = []
        y_train_lstm = []

        # Prediction offset to get the days ahead price
        prediction_offset = self.options['prediction_offset'] - 1
        for i in range(window_size, len(train_set_lstm) - prediction_offset):
            # Taking 50 day data for training
            X_train_lstm.append(training_set_scaled_lstm[i-window_size:i,:])
            X_train_price_lstm.append(train_set_price_lstm[i-window_size:i,:])

            # Target value is the price on day 50 + 1 i.e. 51st Day
            # TODO: Add in days ahead calculation for this.
            y_train_lstm.append(train_target_set_lstm[i + prediction_offset,:])

        # NOt creating a numpy array, difficulty in passing to dataframe while creating one
        X_train_lstm, X_train_price_lstm, y_train_lstm = np.array(X_train_lstm), np.array(X_train_price_lstm), np.array(y_train_lstm)
        print(X_train_lstm.shape)
        print(X_train_price_lstm.shape)
        print(y_train_lstm.shape)

        X_train_returns = []
        X_train_features = []
        y_train_returns = []
        i = 0
        for train_sample,price_sample,train_target in zip(X_train_lstm,X_train_price_lstm,y_train_lstm):
            if price_sample[0]!=0:
                X_train_features.append(train_sample)
                X_train_returns.append((price_sample - price_sample[0])/price_sample[0])
                y_train_returns.append((train_target - price_sample[0])/price_sample[0])
        X_train_returns = np.array(X_train_returns)
        X_train_features = np.array(X_train_features)
        y_train_returns = np.array(y_train_returns)

        trainset["Returns"] = X_train_returns
        trainset["Features"] = X_train_features
        trainset["Label_Returns"] = y_train_returns

        all_features = []
        for returns, feat in zip(X_train_returns,X_train_features):
            f = feat
            p = returns
            f = pd.DataFrame(f,columns=features)
            p = pd.DataFrame(p,columns=["Return"])

            feat_price = pd.concat([f, p], axis=1).to_numpy()

            all_features.append(feat_price)


        trainset["Ret_Feat"] = np.array(all_features)
        trainset['date'] = df_date

        return trainset

    def get_model(self, inputshape):
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
        mod.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy','mean_squared_error'])
        mod.summary()

        return mod

    def fit(self):
        # Number of features and input shape shouldn't change -- therefore we can use the first dataframe for shape.
        trainset = self.CreateTrainData(self.train[0], self.train[0].permno[0])

        num_features = trainset["Ret_Feat"].shape[-1] - 1
        inputshape = (trainset["Features"].shape[1], num_features)

        # Create one model for all stocks as this is a "pooled-stock" model.
        model = self.get_model(inputshape)

        # TODO: Potentially remove this -- when we want to train on the cloud.
        # callback = tf.keras.callbacks.ModelCheckpoint(filepath='baseline/RNN_model.h5',
        #                                    monitor='mean_squared_error',
        #                                    verbose=1,
        #                                    save_best_only=True,
        #                                    save_weights_only=False,
        #                                    mode='auto',
        #                                    save_freq='epoch')

        permno_train_dic = {}
        for permno_data in self.train:
            # The permno for stock
            permno_string = permno_data.permno[0]
            trainset = self.CreateTrainData(permno_data, permno_string)
            permno_train_dic[permno_string] = trainset

        merged_stock_array = []

        for permno in permno_train_dic.keys():
            trainset = permno_train_dic[permno]
            for date, features, label in zip(trainset['date'],trainset['Returns'],trainset['Label_Returns']):
                merged_stock_array.append([date, features, label])

        merged_stock_array.sort(key = lambda x: x[0])
        merged_stock_df = pd.DataFrame(merged_stock_array,columns=['date','features','label'])

        num_features = trainset["Ret_Feat"].shape[-1] - 1
        inputshape = (trainset["Features"].shape[1], num_features)

        X_train = np.array(merged_stock_df["features"].tolist())
        Y_train = np.array(merged_stock_df["label"].tolist())

        # TODO: Experiment with # of epochs when more features.
        # model.fit(X_train, Y_train, epochs = 1, batch_size = 64,callbacks=[callback])

        model.fit(X_train, Y_train, epochs=1, batch_size=256)
        # model.fit(X_train, Y_train, epochs=2, batch_size=64)

        print(f"Training Successful for ALL STOCKS from {merged_stock_df['date'].min()} to {merged_stock_df['date'].max()}! HURRAYYY !")

        self.trained_model = model
        return self

    def ___prices_to_returns(self, window_price_data):
        # TODO: Support multiple predictions per permno.
        # Right now we are only predicting once per permno, per time period.
        price_0 = window_price_data.iloc[0]
        returns = [(price- price_0)/price_0 for price in window_price_data]
        return np.array([returns]).reshape(1, -1, 1)

    def ___create_last_window_data(self, data, feature_list):
      testset = {}

      testset['Returns'] = self.___prices_to_returns(data['adjusted_prc'])
      testset['date'] = data['date']
      # TODO: Scaling the features

      df_test = data[feature_list]
      if len(feature_list) == 1:
          test_set = df_test.values.reshape(-1, 1)
      elif len(feature_list) > 1:
          test_set = df_test.values

      # TODO: Z-Transform features here.
      testset['Features'] = np.array(test_set).reshape(1, -1, test_set.shape[-1])

      # Collate features and returns into Ret_Feat.
      all_features = []
      for returns, feat in zip(testset['Returns'], testset['Features']):
          f = feat
          p = returns
          f = pd.DataFrame(f,columns=feature_list)
          p = pd.DataFrame(p,columns=["Return"])

          feat_price = pd.concat([f, p], axis=1).to_numpy()

          all_features.append(feat_price)

      # TODO: ORDER OF RET FEAT MATTERS ** BE CAREFUL **.
      testset["Ret_Feat"] = np.array(all_features)

      # f = pd.DataFrame(testset['Features'], columns=feature_list)
      # p = pd.DataFrame(testset['Returns'], columns=["Return"])
      # all_features = []
      # feat_price = pd.concat([f, p], axis=1).to_numpy()
      # testset["Ret_Feat"] = np.array([feat_price])
      return testset

    def predict(self, periods_ahead, window_size = 50):
        #TODO: Centralize this, to be the same list in CreateTrain and here
        features_not_to_include = ['adjusted_prc', 'date', 'permno', 'ticker']
        # ls_features = [col for col in data.columns if col not in features_not_to_include]
        price = ['adjusted_prc']
        ls_features = ['adjusted_vol']

        permno_test_dic = {}
        first_permno_price = {}
        for permno_data in self.train:
            permno_string = permno_data.permno[0]
            # The permno for stock
            permno_data = permno_data.sort_values(by=['date'])
            last_test_window = permno_data[-window_size:]
            min_date = last_test_window.date.min()

            key = str(min_date) + ',' +str(permno_string)
            permno_dic = {}
            permno_dic['adjusted_prc'] = last_test_window[last_test_window['date'] == min_date]['adjusted_prc'].iloc[0]
            permno_dic['last_date'] = last_test_window.date.max()
            first_permno_price[key] = permno_dic

            testset = self.___create_last_window_data(last_test_window, ls_features)

            # Ret_Feat numpy (50, 2)
            # Feat numpy (50, 1): just volume
            # date: dataframe.
            # Returns numpy (50, 1)
            permno_test_dic[permno_string] = testset

        merged_stock_array = []

        for permno in permno_test_dic.keys():
            testset = permno_test_dic[permno]
            for date, returns, ret_feat in zip(testset['date'], testset['Returns'], testset['Ret_Feat']):
                merged_stock_array.append([date, returns, ret_feat, permno])

        merged_stock_array.sort(key=lambda x: x[0])
        merged_stock_df = pd.DataFrame(merged_stock_array, columns=['date', 'returns', 'ret_feat', 'permno'])

        # TODO: Add in other features here. X_All_Features_Test
        ############# WHEN USING ONLY PRICE DATA ######################
        X_test = np.array(merged_stock_df["returns"].tolist())
        predicted_stock_price = self.trained_model.predict(X_test)
        ###############################################################

        ############# WHEN USING ALL FEATURES ######################
        # X_test = np.array(merged_stock_df["ret_feat"].tolist())
        # predicted_stock_price = self.trained_model.predict(X_All_Features_Test)
        ###############################################################

        predictions_dic  = {
          'permno': [],
          'date': [],
          'adjusted_prc_pred': [],
          'std_dev_pred': []
        }

        # Convert stock prediction returns into stock prediction prices. Format dataframe for #stock_model_trainer.
        for permno, date, predicted_returns in zip(merged_stock_df['permno'], merged_stock_df['date'], predicted_stock_price):
          # Prediction date is 180 days (prediction_offset) after last known date of the stock.
          prediction_date = first_permno_price[str(date) + ',' + str(permno)]['last_date'] + timedelta(self.options['prediction_offset'])
          initial_price = first_permno_price[str(date) + ',' + str(permno)]['adjusted_prc']
          predicted_price = initial_price + (initial_price * predicted_returns)

          # TODO: Refactor to train different percentile models.
          predictions_dic['std_dev_pred'].append(None)
          predictions_dic['permno'].append(permno)
          predictions_dic['date'].append(prediction_date)
          predictions_dic['adjusted_prc_pred'].append(predicted_price[0])

        predictions_df = pd.DataFrame(predictions_dic)

        return predictions_df
