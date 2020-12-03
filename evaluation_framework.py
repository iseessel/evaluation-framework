import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from stock_model_trainer import StockModelTrainer
from google.cloud import bigquery
import os


class EvaluationFramework:
    """
      Class is used to train and evaluate multiple stocks given one model for the entire lifecycle provided.

      :param model: Model Class. (see model.py and fb_prophet.py for examples).
      :param permnos: Array<Int>. Represents the stock Permno.
      :param dataset: String. The full table name of the features dataset.
      :param storage_bucket: String. Storage bucket of the features pickle. ** Can only use storage_bucket or dataset.
      :param features: Array<String>. Array of strings of features. Must match column names in bigquery dataset.
      :param hypers: Dict<Any>. Dictionary of hyper parameters. Will be fed into the Model class above.
      :param start: String. Date to start training. E.g. '1980-01-01' will start the evaluation the first of January.
      :param end: String. Date to end training. E.g. '2020-01-01' will end training 2000-01-01
      :param offset: String. Date to start evaluation. E.g. start='1980-01-01', offset='2000-01-01'. Our first training will be from 1980 to 2000.
      :param increments: Int. How many days after offset to retrain the model.
      :param evaluation_timeframe: Array<Int> Which time horizons we are evaluating the model on.

      E.g. For Params.
        Model = FBPRophet, Permno = ['AAPL', ..., 'GOOG'], dataset ='my_dataset', features = ['adjusted_prc'], hypers={}, start = '1980-01-01', end = '2020-01-01', offset = '2000-01-01', increments=180, evaluation_timeframe = [180].

        First we will train each model from start (1980-01-01) to offset (2000-01-01) and evaluate the 180 days prediction.
        Next we will retrain each model every 6 months until end ('2020-01-01') and evaluate the 180 days prediction.
    """

    def __init__(self, **kwargs):
        self.client = kwargs['client']
        self.model = kwargs['model']
        self.permnos = kwargs['permnos']
        self.dataset = kwargs['dataset']
        self.features = kwargs['features']
        self.hypers = kwargs['hypers']
        self.start = datetime.strptime(kwargs['start'], '%Y-%m-%d').date()
        self.end = datetime.strptime(kwargs['end'], '%Y-%m-%d').date()
        self.offset = datetime.strptime(kwargs['offset'], '%Y-%m-%d').date()
        self.increments = kwargs['increments']
        self.evaluation_timeframe = sorted(kwargs['evaluation_timeframe'])
        self.evaluation_table_id = kwargs['evaluation_table_id']
        self.glue = kwargs['glue']
        self.pooled = kwargs['pooled']

    def eval(self):
        if self.pooled:
            self.__eval_multi_stock()
        else:
            self.__eval_single_stock()

    def __create_stock_model_trainer(self, x_train, y_train, x_test, y_test, permnos_test):
        kwargs = {
            # 'hypers': self.hypers,
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'permnos_test': permnos_test,
        }

        model = self.model(**kwargs)
        kwargs = {
            'model': model,
            'y_test': y_test,
            'evaluation_timeframe': self.evaluation_timeframe,
            'hypers': self.hypers,
            'dataset': self.dataset
        }

        return StockModelTrainer(**kwargs)

    def __load_stock_results(self, stock_results):
        # Prepare evaluation data to be uploaded to Bigquery.
        df = pd.DataFrame(stock_results[1:], columns=stock_results[0])

        df.to_csv('temp.csv', index=False)
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.CSV, skip_leading_rows=1,
            autodetect=True,
            write_disposition='WRITE_TRUNCATE'
        )

        with open('temp.csv', "rb") as source_file:
            job = self.client.load_table_from_file(
                source_file, self.evaluation_table_id, job_config=job_config)

        job.result()  # Waits for the job to complete.

        os.remove("temp.csv")

        return df

    def __get_train_test(self, df, train_start, train_end):
        # Get dataframe with last known predictions.
        train_df = df[df.date >= train_start]
        train_df = train_df[train_df.prediction_date <= train_end]

        x_train = train_df.drop('target', axis=1)[self.features]
        y_train = train_df['target']

        # TODO: Refactor test dataset for different testing strategies.
        # Get prediction dates.
        test_df = df[self.features]
        test_df = df[df.date >= train_start]
        test_df = df[df.date <= train_end]
        idx = test_df.groupby(['permno'])['prediction_date'].transform(
            max) == test_df['prediction_date']
        test_df = test_df[idx]

        x_test = test_df.drop('target', axis=1)[self.features]
        y_test = test_df['target']

        permnos_test = test_df.permno

        return (x_train, y_train, x_test, y_test, permnos_test)

    def __eval_multi_stock(self):
        stock_data = self.__get_stock_data()

        timeframes = self.__get_timeframes()
        for time in timeframes:
            print(f"Starting Timeframe: {self.start} - {time}")
            x_train, y_train, x_test, y_test, permnos_test = self.__get_train_test(
                stock_data, self.start, time)
            print(
                f"X train: From ({x_train.date.min()} - {x_train.date.max() }). Num examples: { len(x_train) }\n"
                f"Y train. Num examples: { len(y_train) }\n"
                f"X test: From ({x_test.date.min()} - {x_test.date.max() }). Num examples: { len(x_test) }\n"
                f"Y test: Num examples: { len(x_test) }\n"
                f"Permnos: Num examples: { len(x_test) }\n"
            )

            # Apply custome glue function to get data ready for model.
            x_train, y_train, x_test, y_test, permnos_test = self.glue(
                x_train, y_train, x_test, y_test, permnos_test)

            trainer = self.__create_stock_model_trainer(
                x_train, y_train, x_test, y_test, permnos_test)

            trainer.fit()
            # df=trainer.evaluate()
            #
            # if len(stock_results) == 0:
            #     stock_results.append(df.columns.values.tolist())
            #
            # stock_results=stock_results + df.values.tolist()
            #
            # return self.__load_stock_results(stock_results)

    # TODO: This needs to be refactored. This will *NOT* work.

    def __eval_single_stock(self):
        timeframes = self.__get_stock_timeframes()

        stock_data = self.__get_stock_data()

        # TODO: Opportunity to parallelize this.
        # https://stackoverflow.com/questions/3033952/threading-pool-similar-to-the-multiprocessing-pool.
        stock_results = []
        for permno, timeframes in timeframes.items():
            stock_result = []

            for time in timeframes:
                train, test = self.__get_train_test(
                    stock_data, [permno], self.start, time, self.evaluation_timeframe[-1])
                trainer = self.__create_stock_model_trainer(
                    [permno], train, test, self.start, time)

                trainer.fit()
                df = trainer.evaluate()
                if len(stock_results) == 0:
                    stock_results.append(df.columns.values.tolist())

                stock_result = stock_result + df.values.tolist()

            stock_results = stock_results + stock_result

        return self.__load_stock_results(stock_results)

    # TODO: Decide whether or not to Retrieve this per stock (May run into memory constraints.)
    def __get_stock_data(self):
        features = set(['permno', 'date', 'prediction_date',
                        'target']).union(self.features)
        features = ','.join(features)
        QUERY = f"""
        SELECT
            { features }
        FROM
            `{ self.dataset }`
        WHERE
            permno IN UNNEST({ self.permnos }) AND
              date >= '{ self.start.strftime('%Y-%m-%d') }' AND
              date <= '{ self.__eval_end(self.end).strftime('%Y-%m-%d') }'
        ORDER BY
            date
      """
        print("Fetching features dataset")
        print(QUERY)

        df = self.client.query(QUERY).to_dataframe()

        # TODO: Memoize this.
        return df

    # TODO: Improve this.
    def __get_stock_timeframes(self):
        QUERY = f"""
        SELECT
            permno, MIN(date) as min_date, MAX(date) as max_date
        FROM
            `{ self.dataset }`
        WHERE
            permno IN UNNEST({ self.permnos }) AND
            date >= '{ self.start.strftime('%Y-%m-%d') }' AND
            date <= '{ self.__eval_end(self.end).strftime('%Y-%m-%d') }'
        GROUP BY
            permno
      """

        df = self.client.query(QUERY).to_dataframe()
        dates = self.__get_timeframes()

        result = {}
        for s in df.iterrows():
            stock = s[1]
            # Only choose the stocks for which we can evaluate the future properly.
            max_date = stock.max_date - \
                timedelta(days=self.evaluation_timeframe[-1])

            stock_dates = [date for date in dates if date >=
                           stock.min_date and date <= max_date]
            result[stock.permno] = stock_dates

        return result

    # TODO: Improve performance of this method.
    # def __get_train_test(self, df, permnos, start, end, last_pred_days):
    #   # Add 7 in case time occurrs on a holiday and/or weekend.
    #   eval_end = end + timedelta(days=last_pred_days + 7)
    #   df = df[df.permno.isin(permnos)]
    #
    #   train = []
    #   test = []
    #   i = 0
    #   for permno in permnos:
    #     i = i + 1
    #     train_permno = df[(df['date'] >= start) & (df['date'] <= end) & (df['permno'] == permno)].reset_index()
    #     test_permno = df[(df['date'] > end) & (df['date'] <= eval_end) & (df['permno'] == permno)].reset_index()
    #     if not train_permno.empty and not test_permno.empty:
    #       train.append(train_permno)
    #       test.append(test_permno)
    #
    #   return train, test

    # TODO: Improve timeframes to months (instead of days).
    def __get_timeframes(self):
        dates = []

        curr_date = self.offset
        while curr_date <= self.end:
            dates.append(curr_date)
            curr_date = curr_date + timedelta(days=self.increments)

        return dates

    def __eval_end(self, end):
        eval_end = end + timedelta(days=self.evaluation_timeframe[-1])
        return eval_end
