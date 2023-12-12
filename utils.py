'''
utils module containing common methods for the dashboards.
'''

import numpy as np
import pandas as pd
from config import GSPC_URL, NDX_URL, RUT_FILE_PATH, EQ_INDEX_COLLECTION
from scipy.stats import pearsonr
from database.database_wrapper import DatabaseWrapper
from database.data_loader import DataLoader


def load_index_components_in_db(db_str: str, db_name: str, collection_name: str = EQ_INDEX_COLLECTION):
    '''
    Reads Index components from web or csv and writes to the database
    :param db_str: Database Connection string
    :param db_name: Database name
    :param collection_name: Collection name to write to
    :return: None
    '''
    data_loader = DataLoader(db_str, db_name)
    # S&P500 components
    source_config = {'source': 'web', 'url': GSPC_URL,
                     'index_symbol': 'GSPC', 'col_to_pick': 'Symbol', 'scrapped_idx': 0, 'replace_tuple': 0}
    db_data_config = {'collection_name': collection_name, 'collection_index': [('Symbol', 1)]}
    data_loader.load_data_db(source_config, db_data_config)
    # Nasdaq-100
    source_config = {'source': 'web', 'url': NDX_URL, 'index_symbol': 'NDX',
                     'col_to_pick': 'Ticker', 'scrapped_idx': 4}
    data_loader.load_data_db(source_config, db_data_config)

    # Russell-2000
    source_config = {'source': 'csv', 'index_symbol': '^RUT', 'file_path': RUT_FILE_PATH}
    data_loader.load_data_db(source_config, db_data_config)


def get_index_components_from_db(db_str: str, db_name: str, collection_name: str, index_list: list) -> set:
    '''
    Reads components/tickers of indices in the index list and return components/tickers list
    :param db_str: Database connection string
    :param db_name: Database name
    :param collection_name: Collection Name
    :param index_list: list of indices
    :return: set of index components/tickers
    '''
    db = DatabaseWrapper(db_str, db_name)
    projection = {'Components': 1, '_id': 0}
    query = {'Symbol': {'$in': index_list}}
    components = db.read(collection_name, query, projection)
    tickers = set()
    for comp in components:
        tickers.update(comp['Components'])
    # Add index tickers as well
    tickers.update(index_list)
    return tickers


def get_ticker_data_from_db(db_str: str,
                            db_name: str,
                            collection_name: str,
                            ticker_list: list,
                            start_date: str,
                            end_date: str,
                            fillna_method: str = 'ffill') -> pd.DataFrame:
    '''
    Reads data for ticker list from the database for start to end date and return their Closing price.
    :param db_str: Database string
    :param db_name: Database name
    :param collection_name: Collection Name
    :param ticker_list: list of tickers
    :param start_date: Period start date
    :param end_date: Period end date
    :param fillna_method: method to use for filling NaN values in pandas Dataframe.
    :return: Dataframe with Close price for the tickers (tickers are the columns) for the dates.
    '''
    db = DatabaseWrapper(db_str, db_name)
    query = {'Symbol': {'$in': ticker_list}, 'Date': {'$gte': start_date, '$lte': end_date}}
    projection = {'Symbol': 1, 'Date': 1, 'Close': 1, '_id': 0}
    data_df = pd.DataFrame(db.read(collection_name, query, projection))
    data_df['Close'].fillna(method=fillna_method, inplace=True)
    data_df = pd.crosstab(data_df['Date'], data_df['Symbol'], data_df['Close'], aggfunc=lambda x: x)
    # Drop any ticker which does not have full market data
    data_df.fillna(method=fillna_method, inplace=True)
    data_df = data_df.dropna(axis=1)
    return data_df


def calculate_correlations(data_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculates correlations and pvalues for the tickers in the dataframe
    :param data_df: Pandas Dataframe with ticker closing data
    :return: Correlation and pvalue dataframe
    '''
    corrs = data_df.corr('pearson')
    shape = corrs.shape
    corrs = corrs.rename_axis(None).rename_axis(None, axis=1)
    corrs = corrs.where(np.triu(np.ones(corrs.shape), k=1).astype(bool))
    corrs = corrs.stack().reset_index()
    corrs.columns = ['Ticker1', 'Ticker2', 'Correlation']
    pvalues = data_df.corr(lambda x, y: pearsonr(x, y)[1]) - np.eye(*shape)
    pvalues = pvalues.rename_axis(None).rename_axis(None, axis=1)
    pvalues = pvalues.stack().reset_index()
    pvalues.columns = ['Ticker1', 'Ticker2', 'pvalue']
    corrs = pd.merge(corrs, pvalues, on=['Ticker1', 'Ticker2'])
    return corrs