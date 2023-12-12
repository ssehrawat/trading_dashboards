'''
config file for the dashboards.
'''

import os

DB_CONNECTION_STR = 'mongodb://localhost:27017'
DB_NAME = 'mktdata'
EQ_DAILY_COLLECTION = 'equities_daily'
EQ_INDEX_COLLECTION = 'index_components'
GSPC_URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
NDX_URL = 'https://en.wikipedia.org/wiki/Nasdaq-100'
RUT_FILE_PATH = os.path.join(f"{os.path.abspath('')}", 'rut_components.csv')
INDEX_LIST = ['^GSPC', 'NDX', '^RUT']