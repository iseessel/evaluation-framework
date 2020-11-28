

    def predict(self, periods_ahead, window_size=50):

        for permno_data in self.train:
            permno_data = permno_data.sort_values(by=['date'])
            last_test_window = permno_data[-window_size:]
            last_return_window = self.___prices_to_returns(last_test_window['adjusted_prc'])

        # TODO: Test scalers are done appropriately.
        pred_true = {}

        """
        Splitting the data into train and test, data from 0-7000 will
        be used as train and the rest as test.
        Target is our label, which is adjusted_prc
        We only need adjusted_prc and adjusted_vol for training,
        so we are only keeping them
        """
        print(features)
        features_not_to_include = ['adjusted_prc', 'date', 'permno', 'ticker']
        price = ['adjusted_prc']
        ls_features = ['adjusted_vol']
        # TODO: Uncomment this line when passing in blacklisted features.
        # ls_features = [col for col in data.columns if col not in features_not_to_include]
        # data_lstm_test = data.copy(deep=True)
        data_lstm_test = features
        df_test = data_lstm_test[ls_features]
        df_price = data_lstm_test[price]
        df_target_test = df_price['adjusted_prc']

        """
        Scaling the testing data.
        """
        # TODO: Get the scaler for the correct permno.
        scaler = self.stock_scaler
        # Reshaping because it only have one dimension, so error is thrown
        target_set_test = df_target_test.values.reshape(-1, 1)

        if len(ls_features) == 1:
            test_set = df_test.values.reshape(-1, 1)
        elif len(ls_features) > 1:
            test_set = df_test.values
        test_set_price = df_price.values.reshape(-1, 1)

        # import pdb; pdb.set_trace()
        # TODO: Different transforms for future.
        # test_price_scaled = scaler.fit_transform(test_set_price)
        # TODO: transform instead of fit_transform to let go of overfit/leak
        test_set_scaled = scaler.transform(test_set)
        # target_set_test_scaled = scaler.fit_transform(target_set_test)

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
        # Number of days ahead for which you want the prediction.
        prediction_offset = self.options['prediction_offset'] - 1
        for i in range(window_size, len(test_set) - prediction_offset):
            # Taking 50 day data for training
            X_test.append(test_set_scaled[i - window_size:i, :])
            X_test_price.append(test_set_price[i - window_size:i, :])
            final_dates.append(data_lstm_test['date'][i])
            # Target value is the price on day 50 + 1 i.e. 51st Day
            # TODO: Make Y test Modular as per above.

            y_test.append(target_set_test[i + prediction_offset, :])

        X_test, X_test_price, y_test = np.array(X_test), np.array(X_test_price), np.array(y_test)
        print(X_test.shape)
        print(X_test_price.shape)
        print(y_test.shape)

        X_test_returns = []
        X_test_price_returns = []
        y_test_returns = []
        i = 0
        for test_sample, price_sample, test_target in zip(X_test, X_test_price, y_test):
            if price_sample[0] != 0:
                X_test_returns.append(test_sample)
                X_test_price_returns.append((price_sample - price_sample[0]) / price_sample[0])
                y_test_returns.append((test_target - price_sample[0]) / price_sample[0])
        X_test_returns = np.array(X_test_returns)
        X_test_price_returns = np.array(X_test_price_returns)
        y_test_returns = np.array(y_test_returns)

        all_features = []

        for returns, feat in zip(X_test_price_returns, X_test_returns):
            f = feat
            p = returns
            f = pd.DataFrame(f, columns=ls_features)
            p = pd.DataFrame(p, columns=["Return"])

            feat_price = pd.concat([f, p], axis=1).to_numpy()

            all_features.append(feat_price)

        X_All_Features_Test = np.array(all_features)
        print(X_All_Features_Test.shape)

        #         predicted_stock_price = self.trained_model.predict(X_All_Features_Test)

        # TODO: Add in other features here. X_All_Features_Test
        import pdb;
        pdb.set_trace()

        ############# WHEN USING ONLY PRICE DATA ######################
        predicted_stock_price = self.trained_model.predict(X_test_price_returns)
        ###############################################################

        final_preds = []
        final_true_target = []
        for predict, price_sample, true_test_return in zip(predicted_stock_price, X_test_price, y_test_returns):
            if test_sample[0] != 0:
                final_preds.append(predict * price_sample[0] + price_sample[0])
                final_true_target.append(true_test_return * price_sample[0] + price_sample[0])
        final_preds = np.array(final_preds)
        final_true_target = np.array(final_true_target)

        # NOt scaling price
        # final_predicted_stock_price = scaler.inverse_transform(final_preds)
        final_predicted_stock_price = final_predicted_stock_price.reshape(-1)

        #         dates_final = features['date'][window_size:]
        dates_final = final_dates[:final_predicted_stock_price.shape[0]]
        print("Shape of Dates: ", np.array(dates_final).shape)
        print("Shape of Predictions: ", final_predicted_stock_price.shape)

        import pdb;
        pdb.set_trace()
        df = pd.DataFrame({'adjusted_prc_pred': final_predicted_stock_price},
                          index=pd.to_datetime(features['date'][window_size:len(features['date']) - 1]))

        #         df['date'] = features['date'][window_size:len(features['date'])-1]
        #         df['date'] = pd.to_datetime(dates_final)
        df['date'] = pd.PeriodIndex(dates_final, dtype='period[D]', freq='D')
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
