"""
  Gets stock predictions from Biquery and creates a portfolio
"""
from stock_pickers.non_linear_optimization import NonLinearOptimization
from google.cloud import bigquery


class PortfolioCreator:
    def __init__(self, **kwargs):
        self.stock_picker = kwargs['stock_picker']
        self.dataset = kwargs['dataset']
        self.client = kwargs['client']

    def pick_stocks(self):
        # Retrieve stocks from bigquery.
        QUERY = f"""
      SELECT
          permno, ticker, train_end, prediction_date, adjusted_prc_train_end, adjusted_prc_pred, std_dev_pred
      FROM
        `{ self.dataset }`
    """

        all_predictions = self.client.query(QUERY).to_dataframe()
        all_stock_picks = {}
        # Feed in predictions to the stock picker for each prediction date.
        for name, group in all_predictions.groupby('prediction_date'):
            kwargs = {
                'predictions': group,
                'client': self.client
            }
            stock_picker = self.stock_picker(**kwargs)
            stock_picks = stock_picker.pick()
            all_stock_picks[name.strftime('%Y-%m-%d')] = stock_picks

        # TODO: Save stock predictions in bigquery.


kwargs = {
    'stock_picker': NonLinearOptimization,
    'dataset': 'silicon-badge-274423.stock_model_evaluation.fbprophet_sp_daily_features',
    'client': bigquery.Client(project='silicon-badge-274423')
}
x = PortfolioCreator(**kwargs)
x.pick_stocks()
