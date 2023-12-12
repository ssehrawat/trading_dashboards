'''
Wrapper on Mongo db APIs for database create, read, write etc. operations
'''

import logging
from typing import Union
import pandas as pd
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.cursor import Cursor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DatabaseWrapper:
    '''
    Wrapper on Mongo db APIs for database create, read, write etc. operations
    '''

    def __init__(self, db_uri: str, db_name: str):
        self.db_uri = db_uri
        self.db_name = db_name
        client = MongoClient(self.db_uri)
        self.db = client[self.db_name]

    def get_database(self) -> Database:
        return self.db
    
    def create_collection(self, collection_name: str, collection_index: list) -> Collection:
        '''
        Creates a new collection or returns an existing one if it already exists
        :param collection_name: Name of the collection
        :param collection_index: Index of the collection
        :return: Mongo db collection
        '''
        if collection_name in self.db.list_collection_names():
            return self.db.get_collection(collection_name)
        collection = self.db.create_collection(collection_name)
        if collection_index:
            collection.create_index(collection_index, unique=True)
        return collection

    def write(self, collection_name: str, collection_index: list, data: Union[pd.DataFrame, list]):
        '''
        Writes data to the collection.
        :param collection_name: Name of the collection
        :param collection_index: Index of the collection
        :param data: Data to be written
        :return: None
        '''
        collection = self.create_collection(collection_name, collection_index)
        data = data.to_dict('records') if isinstance(data, pd.DataFrame) else data
        data = data if isinstance(data, list) else [data]
        try:
            collection.insert_many(data, ordered=False)
        except Exception as e:
            logger.info(e)

    def read(self, collection_name: str, query: dict = {}, projection: dict = {}) -> Cursor:
        '''
        Reads data from a collection.
        :param collection_name: Name of the collection
        :param query: Query to filter the data
        :param projection: Fields to return in the query
        :return: Results of the query
        '''
        collection = self.db[collection_name]
        if projection:
            return collection.find(query, projection)
        return collection.find(query)





