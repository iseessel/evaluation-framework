"""
    Gets stock predictions from Biquery and creates a portfolio
"""

from stock_pickers.non_linear_optimization_v0 import NonLinearOptimization
from google.cloud import bigquery
from dateutil.relativedelta import relativedelta
import pandas as pd
import os
import random
import datetime

class PortfolioCreator:
    def __init__(self, **kwargs):
        self.stock_picker = kwargs['stock_picker']
        self.dataset = kwargs['dataset']
        self.client = kwargs['client']
        self.num_candidate_stocks = kwargs['num_candidate_stocks']
        self.target_table_id = kwargs['target_table_id']
        self.options = kwargs['options']

    def pick_stocks(self):
        predictions = self.__get_predictions()
        all_permnos = predictions.permno.astype('string').unique().to_numpy()
        daily_returns = self.__get_target_returns(
            all_permnos, predictions.date.min() - relativedelta(years=1))
        bonds_df = self.__get_bond_returns(predictions.date.min())

        weight_results = [['date', 'prediction_date', 'permno', 'weight', 'actual_ret',
                           'dataset', 'num_candidate_stocks', 'stock_picker']]

        # Feed in predictions to the stock picker for each prediction date.
        for date, group in predictions.groupby('date'):

            # Small bug: Sometimes a stock delists slightly before prediction date. Skip these dates, as they will have a small number of stocks in consideration.
            if len(group) <= 10:
                print("\n####################################################")
                print(f"Skipping date: { date }, as there are { len(group) } stocks")
                print("####################################################\n")
                continue

            # SMALL BUG WHERE OCASSIONALLY TWO PREDICTIONS FOR A PERMNO
            group = group.drop_duplicates(subset=['permno'], keep='last')


            print("\n####################################################")
            print(f"Starting stock picking for: { date.strftime('%Y-%m-%d') }")
            print(f"Considering { len(group) } stocks")
            print("####################################################\n")

            bond_return = bonds_df[bonds_df.date == date]
            prediction_date = group.prediction_date.min()
            bond_return = bond_return.ret.iloc[0]

            # Important to order by permnos, as correlation matrix must have same ordering as group.
            group = group.sort_values('permno')

            # Only choose top 50 stocks, ranked by (predicted_ret)/(predicted_vol)
            candidate_returns = self.__rank_by_sharpe_ratio(group, bond_return)
            correlation_matrix = self.__create_correlation_matrix(
                daily_returns, date, candidate_returns.permno.tolist())
            permnos = group.permno.astype('string').unique().to_numpy()

            kwargs = {
                'predictions': candidate_returns,
                'client': self.client,
                'correlation_matrix': correlation_matrix,
                'bond_return': bond_return,
                'options': self.options
            }

            stock_picker = self.stock_picker(**kwargs)
            stock_picks = stock_picker.pick()

            cum_ret = 0
            for permno, weight in stock_picks.items():
                if weight == 0:
                    continue
                elif permno == 'bond':
                    cum_ret = cum_ret + (bond_return * weight)

                    data = [date.strftime('%Y-%m-%d'), prediction_date, permno, weight, bond_return,
                            self.dataset, self.num_candidate_stocks, str(self.stock_picker)]

                    weight_results.append(data)
                else:
                    prediction_date = group[group.permno ==
                                            permno].prediction_date.iloc[0]

                    df = daily_returns[daily_returns.permno == permno]

                    # TODO: This logic doesn't perfectly match the daily return calculations, as prediction dates may differ
                    # from trading dates by a day or two. Is "roughly" accurate.
                    actual_ret = predictions[(predictions.permno == permno) & (predictions.date == date)].iloc[0]['return_target']
                    cum_ret = cum_ret + (actual_ret * weight)

                    data = [date.strftime('%Y-%m-%d'), prediction_date.strftime('%Y-%m-%d'), permno, weight, actual_ret,
                            self.dataset, self.num_candidate_stocks, str(self.stock_picker)]

                    weight_results.append(data)

            weight_results.append([date.strftime(
                '%Y-%m-%d'), prediction_date, 'ALL', 1, cum_ret, self.dataset, self.num_candidate_stocks, str(self.stock_picker)])

        self.__upload_weights_to_bigquery(weight_results)

    def __upload_weights_to_bigquery(self, weight_results):
        my_df = pd.DataFrame(weight_results)
        hash = random.getrandbits(128)
        my_df.to_csv(f'temp_{hash}.csv', index=False, header=False)

        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.CSV, skip_leading_rows=1,
            autodetect=True,
            write_disposition='WRITE_TRUNCATE'
        )

        with open(f'temp_{hash}.csv', "rb") as source_file:
            job = self.client.load_table_from_file(
                source_file, self.target_table_id, job_config=job_config)

        job.result()  # Waits for the job to complete.

        os.remove(f"temp_{hash}.csv")

    def __get_bond_returns(self, min_date):
        # Add some padding in case we need to forward fill.
        min_date = min_date - relativedelta(years=1)

        QUERY = f"""
            SELECT
                observation_date as date, ret
            FROM
                `silicon-badge-274423.financial_datasets.bond_yield`
            WHERE
                observation_date >= '{min_date}'
        """
        bonds_df = self.client.query(QUERY).to_dataframe()
        bonds_df = bonds_df.sort_values(by='date')

        # Forward fill bond data.
        max_date = bonds_df.date.max()
        bonds_df = bonds_df.set_index('date')
        bonds_df.index = pd.to_datetime(bonds_df.index)
        bonds_df.index = bonds_df.index.to_period("D")
        idx = pd.period_range(min_date, max_date)
        bonds_df = bonds_df.reindex(idx)
        bonds_df.ret = bonds_df.ret.ffill()

        # Reset index to match returns dataframe.
        bonds_df['date'] = bonds_df.index
        bonds_df.date = bonds_df.date.apply(lambda x: x.to_timestamp().date())
        bonds_df = bonds_df.reset_index()

        return bonds_df

    def __get_target_returns(self, permnos, min_date):
        QUERY = f"""
            SELECT
                permno, date, ret
            FROM
                `silicon-badge-274423.financial_datasets.sp_timeseries_daily` std2
            WHERE
                permno in UNNEST({ permnos.tolist() }) AND date >= '{min_date}'
        """
        df = self.client.query(QUERY).to_dataframe()

        idx = df.groupby(['permno'])['date'].transform(
            min) == df['date']
        # Set first known date of returns to 0, as we want to calculate from timeT.
        df.loc[idx, 'ret'] = 0
        df.loc[df['ret'].isna(), 'ret'] = 0

        df = df.sort_values(by=['date', 'permno'])
        # Calculate cumulative adjusted returns.
        df['cum_ret'] = df['ret'] + 1
        by_permno = df.groupby('permno')
        df['cum_ret'] = by_permno.cum_ret.cumprod()

        df['permno'] = df['permno'].astype('str')

        return df

    def __get_predictions(self):
        QUERY = f"""
            SELECT
                pred.permno, date,  prediction_date, return_prediction, return_target, vol_prediction
            FROM
                `{ self.dataset }` pred
            INNER JOIN
                `silicon-badge-274423.financial_datasets.sp_constituents_historical` sp
            ON
                CAST(sp.permno as INT64) = pred.permno

            --   Only get stocks that are in the S&P at prediction time. Otherwise we introduce leakage into the process.
            WHERE
                start <= date
            ORDER BY
                date, permno
        """

        preds_df = self.client.query(QUERY).to_dataframe()
        preds_df = preds_df.drop_duplicates()
        preds_df['permno'] = preds_df['permno'].astype('string')

        return preds_df

    def __create_correlation_matrix(self, daily_returns, curr_date, permnos):
        # NB: Order of Correlation matrix and __rank_by_sharpe_ratio below must match.
        df = daily_returns[daily_returns['permno'].isin(
            permnos)].sort_values(by='permno')

        df = df.pivot_table(index='date', columns='permno',
                            values='ret').reset_index()

        # Calculate correlation of the last year.
        min_date = curr_date - relativedelta(years=1)
        df = df[df['date'] >= min_date]

        # Get correlation matrix.
        correlations = df.corr(method='pearson')

        # Add the correlations of treasury rate.
        correlations['bond'] = 0
        correlations.loc['bond'] = 0
        correlations['bond']['bond'] = 1
        correlations.reset_index(level=0, inplace=True)

        # To list of list with column headers.
        return [correlations.columns.tolist()] + correlations.values.tolist()

    def __rank_by_sharpe_ratio(self, predictions, bond_return):
        predictions['sharpe'] = (
            predictions['return_prediction'] / predictions['vol_prediction'])
        predictions = predictions.sort_values(
            by='sharpe', ascending=False)
        predictions = predictions[0:self.num_candidate_stocks].sort_values(
            by='permno')

        predictions.permno = predictions.permno.astype('str')
        predictions = predictions[[
            'permno', 'return_prediction', 'vol_prediction']]
        predictions.loc[len(predictions)] = ['bond', bond_return, 0]

        return predictions

# DATASETS = [
#     'features_v9'
# ]
#
# # [stock constraint, bond weight constraint]
# weight_constraints = [
#     [(0, 1), (0,1)],
#     [(0, 1), (0,0)],
#     # For example this constrains stocks to 10% of the portfolio and leaves bonds unconstrained.
#     [(0, 0.1), (0, 1)],
#     [(0, 0.1), (0, 0.1)],
#     [(0, 0.1), (0, 0)],
#     [(0, 0.05), (0, 1)],
#     [(0, 0.05), (0, 0.05)],
#     [(0, 0.05), (0, 0)]
# ]
#
# # weight_constraints = [
# #     (0, 1),
# #     (0, 0.1),
# #     (0, 0.05)
# # ]
#
# constrain_bonds = [True, False]
# dset = []
#
# for dataset in DATASETS:
#     for constraint in weight_constraints:
#         stock_constraint, bond_constraint = constraint
#         dset_name = f"{dataset}_stocks_{str(int(stock_constraint[1] * 100))}_bonds_{str(int(bond_constraint[1] * 100))}"
#         target_table = f'silicon-badge-274423.portfolio.{ dset_name }'
#         predictions_dataset = f'silicon-badge-274423.stock_model_evaluation.{dataset}'
#
#         dset.append(target_table)
#
#         print("####################################################")
#         print(f"Stock Constraints: { constraint[0] }. Bond Constraints: { constraint[1] }")
#         print(f"Predictions Dataset: { predictions_dataset }")
#         print(f"Target Table: { target_table}")
#         print("####################################################\n")
#
#         kwargs = {
#             'stock_picker': NonLinearOptimization,
#             'dataset': predictions_dataset,
#             'client': bigquery.Client(project='silicon-badge-274423'),
#             'num_candidate_stocks': 40,
#             'target_table_id': target_table,
#             'options': {
#                 'constraint': constraint
#             }
#         }
#
#         x = PortfolioCreator(**kwargs)
#         x.pick_stocks()
