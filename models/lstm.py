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

class LSTMModel:
    def __init__(self, **kwargs):
#         self.hypers = kwargs.get('hypers', {})
        self.trained_model = None
        self.stock_scaler = {}
        self.permnos = kwargs['permnos']

    def CreateTrainData(self,data,permno,window_size = 50):
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
#         features = [col for col in data.columns if col not in features_not_to_include]
        data_lstm = data.copy(deep=True)
        # Other features are here.
        df_train_lstm = data_lstm[features]
        df_price = data_lstm[price]
        df_train_target_lstm = df_price['adjusted_prc']
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
        train_set_price_scaled_lstm = scaler.fit_transform(train_set_price_lstm)
        training_set_scaled_lstm = scaler.fit_transform(train_set_lstm)
        train_target_set_scaled_lstm = scaler.fit_transform(train_target_set_lstm)

        self.stock_scaler[permno] = scaler
#         stock_scaler[stock] = scaler

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

        for i in range(window_size,len(train_set_lstm)):
            # Taking 50 day data for training
            X_train_lstm.append(training_set_scaled_lstm[i-window_size:i,:])
            X_train_price_lstm.append(train_set_price_scaled_lstm[i-window_size:i,:])

            # Target value is the price on day 50 + 1 i.e. 51st Day
            # TODO: Add in days ahead calculation for this.
            y_train_lstm.append(train_target_set_scaled_lstm[i,:])

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
    #     trainset["Ret_Feat"] = np.vstack((X_train_returns,X_train_features))
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
        return trainset

#     def get_model(self,num_features):
    def get_model(self,inputshape):
        mod=Sequential()
    #     mod.add(LSTM(units = 64, return_sequences = True, input_shape = (X_train_lstm.shape[1], 2)))
#         mod.add(LSTM(units = 64, return_sequences = True, input_shape = (X_train_lstm.shape[1], num_features)))
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
    #     mod.add((Dense(units = 4, activation='tanh')))
        mod.add((Dense(units = 1, activation='tanh')))
        mod.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy','mean_squared_error'])
        mod.summary()

        return mod

    def fit(self, data):
      # Number of features and input shape shouldn't change -- therefore we can use the first dataframe for shape.
      trainset = self.CreateTrainData(data[0], data[0].permno[0])
      #         num_features = trainset["Ret_Feat"].shape[-1]
      num_features = trainset["Ret_Feat"].shape[-1] - 1
      inputshape = (trainset["Features"].shape[1], num_features)
      # Create one model for all stocks as this is a "pooled-stock" model.
      model = self.get_model(inputshape)

      # TODO: Potentially remove this -- when we want to train on the cloud.
      callback = tf.keras.callbacks.ModelCheckpoint(filepath='baseline/RNN_model.h5',
                                         monitor='mean_squared_error',
                                         verbose=1,
                                         save_best_only=True,
                                         save_weights_only=False,
                                         mode='auto',
                                         save_freq='epoch')

      for permno_data in data:
        trainset = self.CreateTrainData(permno_data, permno_data.permno[0])
  #         num_features = trainset["Ret_Feat"].shape[-1]
        num_features = trainset["Ret_Feat"].shape[-1] - 1
        inputshape = (trainset["Features"].shape[1], num_features)
  #         model = self.get_model(num_features)

  #         model.fit(stock_trainset[stock]["Ret_Feat"], stock_trainset[stock]["Label_Returns"], epochs = 10, batch_size = 64,callbacks=[callback])
        # TODO: Experiment with # of epochs when more features.
        model.fit(trainset["Returns"], trainset["Label_Returns"], epochs = 1, batch_size = 64,callbacks=[callback])
        # print(permno_data)
        print(f"Training Successful for { permno_data.permno[0] }")

      self.trained_model = model
      return self

    def predict(self, periods_ahead, features=None, window_size = 50):
        #TODO: Test scalers are done appropriately.
        pred_true = {}

        """
        Splitting the data into train and test, data from 0-7000 will
        be used as train and the rest as test.
        Target is our label, which is adjusted_prc
        We only need adjusted_prc and adjusted_vol for training,
        so we are only keeping them
        """
        print(features)
        features_not_to_include = ['adjusted_prc','date','permno','ticker']
        price = ['adjusted_prc']
        ls_features = ['adjusted_vol']
#         features = [col for col in data.columns if col not in features_not_to_include]
#         data_lstm_test = data.copy(deep=True)
        data_lstm_test = features
        df_test = data_lstm_test[ls_features]
        df_price = data_lstm_test[price]
        df_target_test = df_price['adjusted_prc']

        """
        Scaling the training data
        """
        # TODO: Get the scaler for the correct permno.
        scaler = self.stock_scaler
        # Reshaping because it only have one dimension, so error is thrown
        target_set_test = df_target_test.values.reshape(-1, 1)

        if len(ls_features) == 1:
            test_set = df_test.values.reshape(-1, 1)
        elif len(ls_features) >1:
            test_set = df_test.values
        test_set_price = df_price.values.reshape(-1, 1)

        test_price_scaled = scaler.fit_transform(test_set_price)
        test_set_scaled = scaler.fit_transform(test_set)
        target_set_test_scaled = scaler.fit_transform(target_set_test)

        """
        Creating testing data
        Using a 50 day window, i.e. Target Value for 50 datapoints will be t+50
        Dimension of X_train_lstm : (n x 50 x 2)
        n -> number of training sets, in this case 7000-50
        50 -> Window size of training set, i.e 50 days
        2 -> Two features ['adjusted_prc','adjusted_vol']

        Dimension of y_train_lstm : (n x 1)
        n -> number of targets, in this case 7000-50
        1 -> Label, i.e, ['adjusted_prc']
        """
        X_test = []
        X_test_price = []
        y_test = []
        final_dates = []
        for i in range(window_size,len(test_set)):
            # Taking 50 day data for training
            X_test.append(test_set_scaled[i-window_size:i,:])
            X_test_price.append(test_price_scaled[i-window_size:i,:])
            final_dates.append(data_lstm_test['date'][i])
            # Target value is the price on day 50 + 1 i.e. 51st Day
            # TODO: Make Y test Modular as per above.
            y_test.append(target_set_test_scaled[i,:])

        X_test, X_test_price, y_test = np.array(X_test), np.array(X_test_price), np.array(y_test)
        print(X_test.shape)
        print(X_test_price.shape)
        print(y_test.shape)


        X_test_returns = []
        X_test_price_returns = []
        y_test_returns = []
        i = 0
        for test_sample,price_sample,test_target in zip(X_test,X_test_price,y_test):
            if price_sample[0]!=0:
                X_test_returns.append(test_sample)
                X_test_price_returns.append((price_sample - price_sample[0])/price_sample[0])
                y_test_returns.append((test_target - price_sample[0])/price_sample[0])
        X_test_returns = np.array(X_test_returns)
        X_test_price_returns = np.array(X_test_price_returns)
        y_test_returns = np.array(y_test_returns)


        all_features = []

        for returns, feat in zip(X_test_price_returns,X_test_returns):
            f = feat
            p = returns
            f = pd.DataFrame(f,columns=ls_features)
            p = pd.DataFrame(p,columns=["Return"])

            feat_price = pd.concat([f, p], axis=1).to_numpy()

            all_features.append(feat_price)



        X_All_Features_Test = np.array(all_features)
        print(X_All_Features_Test.shape)

#         predicted_stock_price = self.trained_model.predict(X_All_Features_Test)

        # TODO: Add in other features here. X_All_Features_Test
        import pdb; pdb.set_trace()

        ############# WHEN USING ONLY PRICE DATA ######################
        predicted_stock_price = self.trained_model.predict(X_test_price_returns)
        ###############################################################

        final_preds = []
        final_true_target = []
        for predict, price_sample, true_test_return in zip(predicted_stock_price,X_test_price,y_test_returns):
            if test_sample[0]!=0:
                final_preds.append(predict*price_sample[0] + price_sample[0])
                final_true_target.append(true_test_return*price_sample[0] + price_sample[0])
        final_preds = np.array(final_preds)
        final_true_target = np.array(final_true_target)

        final_predicted_stock_price = scaler.inverse_transform(final_preds)
        final_predicted_stock_price = final_predicted_stock_price.reshape(-1)

#         dates_final = features['date'][window_size:]
        dates_final = final_dates[:final_predicted_stock_price.shape[0]]
        print("Shape of Dates: ",np.array(dates_final).shape)
        print("Shape of Predictions: ",final_predicted_stock_price.shape)

        import pdb; pdb.set_trace()
        df = pd.DataFrame({'adjusted_prc_pred':final_predicted_stock_price},index = pd.to_datetime(features['date'][window_size:len(features['date'])-1]))

#         df['date'] = features['date'][window_size:len(features['date'])-1]
#         df['date'] = pd.to_datetime(dates_final)
        df['date'] = pd.PeriodIndex(dates_final,dtype='period[D]',  freq='D')
        print(type(df['date']))
#         df['date'] = pd.to_datetime(df['date'])
# #         df['datetime'] = pd.to_datetime(df['date'])
# #         df = df.set_index('datetime')
#         df = df.set_index('date')
#         df['date'] = dates_final
#         df['date'] = pd.to_datetime(df['date'])
        print(df)
        return df
#         true_target = scaler.inverse_transform(final_true_target)

        # Storing Predictions and True Labels
#         stock_preds_true[stock] = {"Predictions":final_predicted_stock_price,"TruePrice":true_target}

#       raise('#predict method not yet implemented.')
