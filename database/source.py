'''
Module to read data from different sources like: yfinance, web scrapping, and csv file.
'''
import datetime
import logging

import numpy
import pandas as pd
import yfinance as yf
from abc import ABC, abstractmethod
from typing import Union

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SOURCE_REGISTRY = {}


def register(source_type: str):
    def wrapper(cls):
        SOURCE_REGISTRY[source_type] = cls
        return cls

    return wrapper


class SourceInterface(ABC):
    '''
    Interface to read data from different types of sources registered to the interface.
    '''

    @staticmethod
    def source_obj(identifier: str, **kwargs: dict):
        obj = SOURCE_REGISTRY[identifier](**kwargs)
        logger.info(f'Returning Source Object: {obj}')
        return obj

    @abstractmethod
    def get_data(self, **data_kwargs):
        return NotImplementedError('Implement in child class!')


@register('yfinance')
class YahooFinance(SourceInterface):
    '''
    Get market data using yfinance module from Yahoo Finance.
    '''

    def __init__(self,
                 tickers: list,
                 start: Union[datetime.date, datetime.datetime],
                 end: Union[datetime.date, datetime.datetime],
                 data_type: str = 'ohlc'):
        '''
        :param tickers: list of tickers to get data from Yahoo Finance.
        :param start: Period start date
        :param end: Period end date
        :param data_type: type of data to get from Yahoo Finance
        '''
        self.tickers = tickers
        self.start = start
        self.end = end
        self.data_type = data_type

    def get_data(self, **data_kwargs) -> pd.DataFrame:
        '''
        Get data from yfinance for the tickers.
        :return: Pandas DataFrame.
        '''
        logger.info('Getting data from yfinance!')
        results = []
        for ticker in self.tickers:
            ticker_obj = yf.Ticker(ticker)
            if self.data_type == 'ohlc':
                ohlc_df = ticker_obj.history(start=self.start, end=self.end)
                ohlc_df['Symbol'] = ticker
                ohlc_df = ohlc_df.reset_index()
                logger.info('Data loaded for {} from source!'.format(ticker))
                results.append(ohlc_df)
        results = pd.concat(results)
        return results


@register('web')
class WebScrapping(SourceInterface):
    '''
    Get component data of the indexes (GSPC, NDX) from websites using web scrapping.
    '''

    def __init__(self, url: str, index_symbol: str, scrapped_idx: int, col_to_pick: int, replace_tuple: tuple = None):
        '''
        :param url: web url to scrap
        :param index_symbol: Ticker of index
        :param scrapped_idx: list index which contains relevant data in the website data list
        :param col_to_pick: Dataframe column Name
        :param replace_tuple: tuple to replace data in website data
        '''
        self.url = url
        self.index_symbol = index_symbol
        self.scrapped_idx = scrapped_idx
        self.col_to_pick = col_to_pick
        self.replace_tuple = replace_tuple

    def get_data(self, **data_kwargs) -> dict:
        '''
        Get data from the web url provided.
        :return: Dict of index symbol and its components.
        '''
        scrapped_data = pd.read_html(self.url)[self.scrapped_idx]
        scrapped_data = scrapped_data[self.col_to_pick]
        if self.replace_tuple:
            scrapped_data = scrapped_data.str.replace(self.replace_tuple[0], self.replace_tuple[1])
        components = {'Symbol': self.index_symbol, 'Components': scrapped_data.tolist()}
        logger.info('Scrapped web data from: {}'.format(self.url))
        return components


@register('csv')
class ReadCSV(SourceInterface):
    '''
    Read Russell 2000 (^RUT) component data from csv file. Assuming one of the column has its ticker list
    '''

    def __init__(self, file_path: str, index_symbol: str, header: int = 0, column: str = 'Ticker'):
        '''
        :param file_path: path of csv file to read
        :param index_symbol: Ticker of index
        :param header: row number which has header in the csv file
        :param column: Dataframe column name to pick
        '''
        self.file_path = file_path
        self.index_symbol = index_symbol
        self.header = header
        self.column = column

    def get_data(self, **data_kwargs) -> dict:
        '''
        Get data from the csv file.
        :return: Dict of index symbol and its components.
        '''
        file_data = pd.read_csv(self.file_path, header=self.header)
        file_data = file_data[self.column].dropna().to_list()
        components = {'Symbol': self.index_symbol, 'Components': file_data}
        logger.info(f'Loaded data from: {self.file_path}')
        return components


if __name__ == '__main__':
    kwargs = {'start': datetime.date(2023, 9, 1), 'end': datetime.date(2023, 10, 18),
              'tickers': ['AAPL', 'MSFT']}
    obj = SourceInterface.source_obj('yfinance', **kwargs)
    data = obj.get_data()
    print(data)
    web_config = {'url': 'https://en.wikipedia.org/wiki/Nasdaq-100', 'index_symbol': 'NDX', 'scrapped_idx': 4, 'col_to_pick': 'Ticker'}
    obj = SourceInterface.source_obj('web', **web_config)
    data = obj.get_data()
    print(data)
