'''
Module for reading data from different sources and write to the database.
'''

import logging
import pandas as pd
from typing import Union
from database.database_wrapper import DatabaseWrapper
from database.source import SourceInterface

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DataLoader:
    '''
    Class for reading data from different sources and loading into the database.
    '''

    def __init__(self, db_uri: str, db_name: str):
        self.db_uri = db_uri
        self.db_name = db_name
        self.db = DatabaseWrapper(self.db_uri, self.db_name)

    @classmethod
    def get_source_data(cls, source_config: dict) -> Union[dict, pd.DataFrame]:
        '''
        Gets source data using SourceInterface using source config.
        :param source_config: dictionary for the source
        :return: data from the source
        '''
        logger.info('Starting to get data from source: {}!'.format(source_config['source']))
        source = source_config.pop('source')
        source_obj = SourceInterface.source_obj(source, **source_config)
        source_data = source_obj.get_data()
        logger.info('Finished getting data from source: {}!'.format(source))
        return source_data

    def load_data_db(self, source_config: dict, db_data_config: dict):
        '''
        Gets data from source and write to the database.
        :param source_config: source config for SourceInterface
        :param db_data_config: config for database
        :return: None
        '''
        logger.info('Preparing to write data to db: {0} in {1}!'.format(self.db_name, db_data_config['collection_name']))
        source_data = self.get_source_data(source_config)
        self.db.write(db_data_config['collection_name'], db_data_config['collection_index'], source_data)
        logger.info('Data successfully written to db: {0} in {1}!'.format(self.db_name, db_data_config['collection_name']))


if __name__ == '__main__':
    # from pymongo import MongoClient
    CONNECTION_STRING = "mongodb://localhost:27017"
    # client = MongoClient(CONNECTION_STRING)
    # db = client['mktdata']
    # # components = db.index_components.find({'Symbol': {'$in': ['GSPC', 'NDX']}}, {'Components': 1, '_id': 0})
    # data = db.equities_daily.find({}, {'Symbol': 1, 'Close': 1, 'Date': 1, '_id': 0})
    # data_df = pd.DataFrame(data)
    # print(data_df)
    # tickers = set()
    # for comp in components:
    #     tickers.update(comp['Components'])
    # print(len(tickers))
    # print(list(components))
    # for i, x in enumerate(components):
    #     print(x)
    # data_loader = DataLoader(CONNECTION_STRING, 'mktdata')
    # source_config = {'source': 'yfinance', 'tickers': tickers,
    #                  'start': datetime.datetime(2023, 8, 1), 'end': datetime.datetime(2023, 10, 1)}
    # db_data_config = {'collection_name': 'equities_daily', 'collection_index': [('Symbol', 1), ('Date', 1)]}
    # source_config = {'source': 'web', 'url': 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
    #                  'index_symbol': 'GSPC','col_to_pick': 'Symbol', 'scrapped_idx': 0, 'replace_tuple': 0}
    # source_config = {'source': 'web', 'url': 'https://en.wikipedia.org/wiki/Nasdaq-100', 'index_symbol': 'NDX',
    #                  'col_to_pick': 'Ticker', 'scrapped_idx': 4}
    # db_data_config = {'collection_name': 'index_components', 'collection_index': [('Symbol', 1)]}
    # data_loader.load_data_db(source_config, db_data_config)


