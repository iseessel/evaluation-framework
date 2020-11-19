import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from stock_model_trainer import StockModelTrainer
from google.cloud import bigquery
import os

class StockPredictions:
    """
      Class is used to train and evaluate multiple stocks given one model for the entire lifecycle provided.

      :param model: Model Class. (see model.py and fb_prophet.py for examples).
      :param permnos: Array<Int>. Represents the stock Permno.
      :param dataset: String. The full table name of the features dataset.
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

    def eval(self):
      timeframes = self.__get_stock_timeframes()

      stock_data = self.__get_stock_data()

      # TODO: Opportunity to parallelize this.
      stock_results = []
      for permno, timeframes in timeframes.items():
        stock_result = []

        for time in timeframes:
          train, test = self.__get_train_test(stock_data, permno, self.start, time, self.evaluation_timeframe[-1])
          kwargs = {
            'model': self.model(**self.hypers),
            'permno': permno,
            'train': train,
            'test': test,
            'evaluation_timeframe': self.evaluation_timeframe,
            'hypers': self.hypers,
            'dataset': self.dataset
          }

          trainer = StockModelTrainer(**kwargs)
          trainer.fit()
          df = trainer.evaluate()
          if len(stock_results) == 0:
            stock_results.append(df.columns.values.tolist())

          stock_result = stock_result + df.values.tolist()
          print(stock_result)

        stock_results = stock_results + stock_result

      # Prepare evaluation data to be uploaded to Bigquery.
      df = pd.DataFrame(stock_results[1:], columns=stock_results[0])

      df.to_csv('temp.csv', index=False)
      job_config = bigquery.LoadJobConfig(
          source_format=bigquery.SourceFormat.CSV, skip_leading_rows=1,
          autodetect=True,
          write_disposition='WRITE_TRUNCATE'
      )

      with open('temp.csv', "rb") as source_file:
          job = self.client.load_table_from_file(source_file, self.evaluation_table_id, job_config=job_config)

      job.result()  # Waits for the job to complete.

      os.remove("temp.csv")

      return df


    # TODO: Decide whether or not to Retrieve this per stock (May run into memory constraints.)
    def __get_stock_data(self):
      QUERY = f"""
        SELECT
            { ','.join(self.features) + ',date,permno,ticker' }
        FROM
            `{ self.dataset }`
        WHERE
            permno IN UNNEST({ self.permnos }) AND
              date >= '{ self.start.strftime('%Y-%m-%d') }' AND
              date <= '{ self.__eval_end(self.end).strftime('%Y-%m-%d') }'
        ORDER BY
            date
      """

      df = self.client.query(QUERY).to_dataframe()

      #TODO: Memoize this.
      return df

    # TODO: Cover case where we can train but we can't evaluate because we don't have the data.
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
        max_date = stock.max_date - timedelta(days=self.evaluation_timeframe[-1])

        stock_dates = [date for date in dates if date >= stock.min_date and date <= max_date]
        result[stock.permno] = stock_dates

      return result

    def __get_train_test(self, df, permno, start, end, last_pred_days):
      # Add 7 in case time occurrs on a holiday and/or weekend.
      eval_end = end + timedelta(days=last_pred_days + 7)
      df = df[df['permno'] == permno]

      train = df[(df['date'] >= start) & (df['date'] <= end)]
      test = df[(df['date'] > end) & (df['date'] <= eval_end)]

      return train, test

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
