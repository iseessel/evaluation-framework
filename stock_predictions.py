class StockPredictions:
    def __init__(self, **kwargs):
      self.model = kwargs['model']
      self.permnos = kwargs['permnos']
      self.dataset = kwargs['dataset']
      self.features = kwargs['features']
      self.hypers = kwargs['hypers']

      # Start date of evaluation.
      self.start = datetime.strptime(kwargs['start'], '%Y-%m-%d').date()
      # End date of evaluation.
      self.end = datetime.strptime(kwargs['end'], '%Y-%m-%d').date()
      # Offset of first training model.
      self.offset = datetime.strptime(kwargs['offset'], '%Y-%m-%d').date()
      # How often are we retraining and repicking the stocks.
      self.increments = kwargs['increments']
      # Which prediction timeframes are we interested in.
      self.evaluation_timeframe = sorted(kwargs['evaluation_timeframe'])

    def eval(self):
      timeframes = self.__get_stock_timeframes()

      stock_data = self.__get_stock_data()

      # TODO: Opportunity to parallelize this.
      for permno, timeframes in timeframes.items():
        for time in timeframes:
          train, test = self.__get_train_test(stock_data, permno, self.start, time, self.evaluation_timeframe[-1])

          kwargs = {
            'model': self.model(**self.hypers),
            'permno': permno,
            'train': train,
            'test': test,
            'evaluation_timeframe': self.evaluation_timeframe
          }


          trainer = StockModelTrainer(**kwargs)
          print(type(trainer))
          print(type(kwargs['model']))
          trainer.fit()
          return trainer.evaluate()

        # Write prediction results to bigquery.

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
              date <= '{ self.end.strftime('%Y-%m-%d') }'
        ORDER BY
            date
      """

      # Get all necessary data and split it into train and test dataframes.
      df = client.query(QUERY).to_dataframe()

      return df

    def __get_stock_timeframes(self):
      QUERY = f"""
        SELECT
            permno, MIN(date) as min_date, MAX(date) as max_date
        FROM
            `{ self.dataset }`
        WHERE
            permno IN UNNEST({ self.permnos }) AND date >= '{ self.start.strftime('%Y-%m-%d') }' AND date <= '{ self.end.strftime('%Y-%m-%d') }'
        GROUP BY
            permno
      """

      df = client.query(QUERY).to_dataframe()
      dates = self.__get_timeframes()

      result = {}
      for s in df.iterrows():
        stock = s[1]
        # Only choose the stock dates that are within the correct time period.
        stock_dates = [date for date in dates if date >= stock.min_date and date <= stock.max_date]
        result[stock.permno] = stock_dates

      return result

    def __get_train_test(self, df, permno, start, end, last_pred_days):
      eval_end = end + timedelta(days=last_pred_days)
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
