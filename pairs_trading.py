'''
Module to implement pairs trading strategy.
'''
import datetime
import logging
import numpy as np
import pandas as pd
import quantstats as qs
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
from functools import cache
from config import DB_CONNECTION_STR, DB_NAME, EQ_DAILY_COLLECTION
from database.database_wrapper import DatabaseWrapper
from utils import calculate_correlations

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
pd.options.mode.chained_assignment = None


class DataMixin:
    '''
    Mixin class to read data from database for a given set of tickers.
    '''

    @cache
    def load_data_from_db(self) -> pd.DataFrame:
        '''
        Reads ticker data from database.
        :return: Dataframe of closing price for the tickers.
        '''
        if self.tickers:
            query = {'Symbol': {'$in': self.tickers}, 'Date': {'$gte': self.start, '$lte': self.end}}
        else:
            query = {'Date': {'$gte': self.start, '$lte': self.end}}
        projection = {'Symbol': 1, 'Date': 1, 'Close': 1, '_id': 0}
        self.data_df = pd.DataFrame(self.db.read(EQ_DAILY_COLLECTION, query, projection))
        self.data_df = pd.crosstab(self.data_df['Date'], self.data_df['Symbol'], self.data_df['Close'],
                                   aggfunc=lambda x: x)
        return self.data_df


class BackTesting(DataMixin):
    '''
    Class to backtest Pairs trading strategy.
    '''

    def __init__(self,
                 start: datetime.datetime,
                 end: datetime.datetime,
                 strategy_params: dict,
                 upper_std: int = 2,
                 lower_std: int = 2,
                 rolling_window: int = 5
                 ):
        self.start = start
        self.end = end
        self.upper_std = upper_std
        self.lower_std = lower_std
        self.strategy_params = strategy_params
        self.rolling_window = rolling_window
        self.tickers = list(set([item for key in self.strategy_params.keys() for item in key]))
        self.db = DatabaseWrapper(DB_CONNECTION_STR, DB_NAME)

    def calculate_spread_bollinger_bands(self, ticker1: str, ticker2: str, alpha: float, beta: float) -> pd.DataFrame:
        '''
        Calculates Bollinger Bands for the ticker pair testing period spread.
        :param ticker1: Ticker of Security 1
        :param ticker2: Ticker of Security 2
        :param alpha: Alpha calculated for the pair in Pairs trading strategy run
        :param beta: Beta calculated for the pair in Pairs trading strategy run
        :return: Dataframe with Upper and Lower Bollinger Bands
        '''
        data = self.data_df[[ticker1, ticker2]]
        spread = data[ticker1] - (alpha + beta * data[ticker2])
        data['Spread'] = spread
        data['Mean'] = data['Spread'].rolling(window=self.rolling_window, min_periods=1).mean()
        data['Std'] = data['Spread'].rolling(window=self.rolling_window, min_periods=1).std().fillna(0.)
        data['BB_Upper'] = data['Mean'] + self.upper_std * data['Std']
        data['BB_Lower'] = data['Mean'] - self.lower_std * data['Std']
        return data

    def calculate_signal(self, bb_data: pd.DataFrame) -> pd.DataFrame:
        '''
        Calculates Long/Short/NoSignal
        :param bb_data: Dataframe with Bollinger Bands for the pair.
        :return: Dataframe with Signal column added to input Dataframe
        '''
        # If T-1 spread is less than T-1 Lower band and T0 spread is higher than T0 Lower band
        # then we buy the spread
        bb_data.loc[(bb_data.shift(1)['Spread'] <= bb_data.shift(1)['BB_Lower']) & (
                    bb_data['Spread'] >= bb_data['BB_Lower']), 'Signal'] = 'Long'
        # If T-1 spread is greater than T-1 Higher band and T0 spread is less than T0 Higher band
        # then we short the spread
        bb_data.loc[(bb_data.shift(1)['Spread'] >= bb_data.shift(1)['BB_Upper']) & (
                    bb_data['Spread'] <= bb_data['BB_Upper']), 'Signal'] = 'Short'
        # if none of the above than no signal
        bb_data = bb_data.fillna({'Signal': 'NoSignal'})
        # Assumption 1st position is always Long
        bb_data.iloc[0, 7] = 'Long'
        return bb_data

    def calculate_trade_decision(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        Calculates decision to close the position or not.
        :param data: Dataframe with ticker data
        :return: Dataframe with position Decision column
        '''
        decision = ['Long']
        for i in range(1, len(data)):
            if decision[i - 1] in ['TakeProfit', 'StopLoss', 'NoSignal']:
                decision.append(data['Signal'].iloc[i])
            else:
                if decision[i - 1] == 'Long':
                    # if you hit the mean, you take profit
                    take_profit_price = data.iloc[i, 3]
                    # if you hit the lower band, you stop loss
                    stop_loss_price = data.iloc[i, 6]
                    if data.iloc[i, 2] >= take_profit_price:
                        decision.append('TakeProfit')
                    elif data.iloc[i, 2] <= stop_loss_price:
                        decision.append('StopLoss')
                    else:
                        decision.append(decision[i - 1])
                if decision[i - 1] == 'Short':
                    # if you hit the mean, you take profit
                    take_profit_price = data.iloc[i, 3]
                    # if you hit the upper band, you stop loss
                    stop_loss_price = data.iloc[i, 5]
                    if data.iloc[i, 2] <= take_profit_price:
                        decision.append('TakeProfit')
                    elif data.iloc[i, 2] >= stop_loss_price:
                        decision.append('StopLoss')
                    else:
                        decision.append(decision[i - 1])
        data['Decision'] = decision
        return data

    def calculate_portfolio_value(self, data: pd.DataFrame, hedge_ratio: float) -> pd.DataFrame:
        '''
        Calculates amount of stocks to long/short in the pair.
        :param data: Dataframe with pair data
        :param hedge_ratio: Hedge Ratio of the pair
        :return: Dataframe with Ticker1 and Ticker2 quantity columns.
        '''
        data.loc[data['Decision'] == 'Long', ['Ticker1_Qty', 'Ticker2_Qty']] = [100, 100 * round(hedge_ratio, 2) * -1]
        data.loc[data['Decision'] == 'Short', ['Ticker1_Qty', 'Ticker2_Qty']] = [-100, -100 * round(hedge_ratio, 2) * -1]
        data.loc[data['Decision'].isin(['TakeProfit', 'StopLoss']), 'Ticker1_Qty'] = data['Ticker1_Qty'].shift(1)
        data.loc[data['Decision'].isin(['TakeProfit', 'StopLoss']), 'Ticker2_Qty'] = data['Ticker2_Qty'].shift(1)
        data.loc[data['Decision'] == 'NoSignal', ['Ticker1_Qty', 'Ticker2_Qty']] = [0, 0]
        return data

    def calculate_portfolio_results(self, ticker1: str, ticker2: str, data: pd.DataFrame) -> tuple:
        '''
        Calculates portfolio returns, sharpe, sortino, max drawdown, cagr and volatility.
        :param ticker1: Ticker1 Symbol
        :param ticker2: Ticker2 Symbol
        :param data: pair Dataframe
        :return: tuple of portfolio results and pair spread, Bollinger bands, returns, cumulative returns, rolling correlation
        '''
        data['PortfolioValue'] = data[ticker1] * data['Ticker1_Qty'] + data[ticker2] * data['Ticker2_Qty']
        data.loc[data.shift(1)['Decision'] == 'Long', 'Returns'] = data['PortfolioValue'] / data[
            'PortfolioValue'].shift(1) - 1
        data.loc[data.shift(1)['Decision'].isin(['TakeProfit', 'StopLoss', 'NoSignal']), 'Returns'] = 0.0
        data.loc[data.shift(1)['Decision'] == 'Short', 'Returns'] = 1 - data['PortfolioValue'] / data[
            'PortfolioValue'].shift(1)
        data['Returns'] = data['Returns'].fillna(0.0).replace([-1.0, -np.inf, np.inf], 0.0)
        portfolio_value = data[data['PortfolioValue'] != 0.0]['PortfolioValue']
        portfolio_returns = data[data['Returns'] != 0.0]['Returns']
        if len(portfolio_returns) <= 1:
            sharpe_ratio = sortino_ratio = max_drawdown = cagr = volatility = 0
        else:
            sharpe_ratio = qs.stats.sharpe(portfolio_returns)
            sortino_ratio = qs.stats.sortino(portfolio_returns)
            max_drawdown = qs.stats.max_drawdown(portfolio_value)
            cagr = qs.stats.cagr(portfolio_returns)
            volatility = qs.stats.volatility(portfolio_returns)
        result = pd.DataFrame(data=[(ticker1, ticker2, cagr, volatility, max_drawdown, sharpe_ratio, sortino_ratio), ],
                              columns=['Ticker1', 'Ticker2', 'CAGR', 'Vol', 'Max Drawdown', 'Sharpe', 'Sortino'])
        timeseries = data[[ticker1, ticker2, 'Spread', 'BB_Upper', 'BB_Lower', 'Returns']]
        timeseries['Cum_Returns'] = timeseries['Returns'].cumsum()
        timeseries['Rolling_Correlation'] = timeseries[ticker1].rolling(self.rolling_window, min_periods=1).corr(
            timeseries[ticker2]).fillna(0.0)
        return result, timeseries

    def run(self) -> dict:
        '''
        Run backtesting for the given pairs trading strategy params.
        :return: Backtesting results for the pairs included.
        '''
        # Load data from database for tickers for the testing period.
        self.load_data_from_db()
        res = {}
        for tickers, params in self.strategy_params.items():
            data = self.calculate_spread_bollinger_bands(tickers[0], tickers[1], params['alpha'], params['beta'])
            data = self.calculate_signal(data)
            data = self.calculate_trade_decision(data)
            data = self.calculate_portfolio_value(data, params['beta'])
            result = self.calculate_portfolio_results(tickers[0], tickers[1], data)
            res[tickers] = result
        return res


class PairsTrading(DataMixin):
    '''
    Implements Pairs Trading Strategy.
    Calculates top correlated ticker pairs among all the index components based on confidence level selected.
    Takes top correlated pairs and finds the co-integrated pairs using OLS/Kalman filter
    and Augmented Dickey Fuller (ADF) test.
    '''

    def __init__(self,
                 start: datetime.datetime,
                 end: datetime.datetime,
                 tickers: list = [],
                 confidence_level: float = 0.95,
                 num_top_corrs: int = 30,
                 calc_type: str = 'ols'):
        self.start = start
        self.end = end
        self.confidence_level = confidence_level
        self.tickers = tickers
        self.num_top_corrs = num_top_corrs
        self.calc_type = calc_type
        self.db = DatabaseWrapper(DB_CONNECTION_STR, DB_NAME)
        self.data_df = None
        self.corrs = None
        self.selected_tickers_corrs = None  # Top correlated pairs
        self.cointegrated_pairs = None  # Co-integrated pairs among top correlated pairs

    def get_top_tickers(self, data_df: pd.DataFrame) -> list:
        '''
        Returns top correlated pairs based on confidence level selected.
        :param data_df: Dataframe of closing ticker data
        :return: top correlated pairs
        '''
        logger.info('Calculating top correlated pairs!')
        self.corrs = calculate_correlations(data_df)
        pvalue_threshold = 1 - self.confidence_level
        self.selected_tickers_corrs = self.corrs[
            (self.corrs.Ticker1 != self.corrs.Ticker2) & (self.corrs.pvalue <= pvalue_threshold)]
        self.selected_tickers_corrs = self.selected_tickers_corrs.nlargest(self.num_top_corrs, 'Correlation')
        return self.selected_tickers_corrs

    def select_cointegrated_pairs(self) -> dict:
        '''
        Calculates co-integrated pairs using OLS and Kalman Filtering with
        Augmented Dickey Fuller (ADF) test.
        :return: co-integrated pairs dictionary with their parameters
        '''
        logger.info('Calculating top co-integrated pairs!')
        self.cointegrated_pairs = {}
        for idx, row in self.selected_tickers_corrs.iterrows():
            x = self.data_df[row['Ticker2']]
            y = self.data_df[row['Ticker1']]
            alpha = beta = spread = None
            if self.calc_type == 'OLS':
                x = sm.add_constant(x)
                reg = sm.OLS(y, x)
                res = reg.fit()
                spread = res.resid
                alpha = res.params[0]
                beta = res.params[1]
            elif self.calc_type == 'Kalman Filter':
                # TODO implement Kalman filtering logic
                pass
            # ADF test for cointegration
            test = ts.adfuller(spread)
            if test[1] <= 1 - self.confidence_level:
                logger.info(f"{row['Ticker1']} and {row['Ticker2']} are cointegrated!")
                self.cointegrated_pairs.update({(row['Ticker1'], row['Ticker2']): {
                    'alpha': alpha,
                    'beta': beta,
                    'spread': spread}
                })
        return self.cointegrated_pairs

    def run(self) -> dict:
        '''
        Runs Pairs Trading Strategy.
        :return: co-integrated pairs with their parameters
        '''
        logger.info('Running Pairs Trading Strategy!')
        data_df = self.load_data_from_db()
        top_tickers = self.get_top_tickers(data_df)
        res = self.select_cointegrated_pairs()
        return res


if __name__ == '__main__':
    start = datetime.datetime(2023, 10, 1)
    end = datetime.datetime(2023, 10, 18)
    test_start = datetime.datetime(2023, 10, 19)
    test_end = datetime.datetime(2023, 10, 30)
    obj = PairsTrading(start, end, tickers=['AAPL', 'MSFT', 'AMZN'])
    strategy = obj.run()
    backtest = BackTesting(test_start, test_end, strategy, 0.5, 0.5, rolling_window=2)
    res = backtest.run()
    backtest1 = BackTesting(start, end, strategy, 0.5, 0.5, rolling_window=2)
    res1 = backtest1.run()
    print(res)
