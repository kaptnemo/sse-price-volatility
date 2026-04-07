## -*- coding: utf-8 -*-
"""
Baostock Helper Module
This module provides a helper class for interacting with the Baostock API,
including methods for logging in, querying daily and minute stock data,
and saving results to MongoDB.
"""

import baostock as bs
import pandas as pd
from pymongo import MongoClient
import contextlib

MONGO_HOST = "mongodb://root:cyw271828@localhost:27017/"
MONGO_DATABASE = "prosera"


@contextlib.contextmanager
def get_mongo_client():
    """Get a MongoDB client instance."""
    with MongoClient(MONGO_HOST) as client:
        yield client


class BaostockResult:
    """A class to encapsulate the result of a Baostock query.
    This class holds the data, error code, error message, table name,
    timestamp column, and tag columns for the result.
    It also provides a method to save the result to MongoDB.
    """
    def __init__(self, data, table_name, timestamp_column=None, tag_columns=None):
        self.data = data
        self.table_name = table_name
        self.tag_columns = tag_columns
        self.timestamp_column = timestamp_column

    def _save_to_mongo(self, collection_name: str, replace: bool, client: MongoClient):
        """Internal method to save data to MongoDB.
        Args:
            collection_name (str): The name of the MongoDB collection.
            replace (bool): Whether to replace the existing collection.
            client (MongoClient): The MongoDB client instance.
        """
        if not self.data.empty:
            if replace:
                client[MONGO_DATABASE].drop_collection(collection_name)
            mongo_collection = client[MONGO_DATABASE][collection_name]
            mongo_collection.insert_many(self.data.to_dict('records'))
        else:
            print("No data to save to MongoDB.")


    def save_to_mongo(self, collection_name: str = None, replace: bool = False, client: MongoClient = None):
        """save data to mongo, the collection name is the method name which get the data
        Args:
            collection_name (str): The name of the MongoDB collection to save the data.
            replace (bool): Whether to replace the existing collection.
            client (MongoClient): The MongoDB client instance. If not provided, a new client will be created.
            if client is provided, it will be used to save the data, and the close method will not be called.
        """
        collection_name = collection_name or self.table_name
        if client:
            self._save_to_mongo(collection_name, replace, client)
        else:
            with MongoClient(MONGO_HOST) as client:
                self._save_to_mongo(collection_name, replace, client)

    def save_to_csv(self, file_path: str):
        """Save the data to a CSV file.
        Args:
            file_path (str): The path to the CSV file where the data will be saved.
        """
        if not self.data.empty:
            self.data.to_csv(file_path, index=False)
        else:
            print("No data to save to CSV.")


class BaostockHelper:
    """A helper class for interacting with the Baostock API.
    This class provides methods for logging in, querying daily and minute stock data,
    and saving results to MongoDB.
    It can be used as a context manager to ensure proper login and logout.
    """

    def __enter__(self):
        """Initialize BaostockHelper and login to Baostock."""
        self.login()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Logout from Baostock when exiting the context."""
        # self.logout()
        pass

    @staticmethod
    def login():
        """Login to Baostock and return the login response."""
        lg = bs.login()
        if lg.error_code != '0':
            raise Exception(f"Login failed: {lg.error_msg}")
        return lg

    @staticmethod
    def logout():
        """Logout from Baostock."""
        bs.logout()

    def daily(self, code, start_date, end_date, adjustflag="2"):
        """Fetch daily data for a given stock code."""
        rs = bs.query_history_k_data_plus(
            code,
            "date,open,high,low,close,volume,amount",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag=adjustflag
        )
        if rs.error_code != '0':
            raise Exception(f"Query failed: {rs.error_msg}")

        # Convert the 'time' column to datetime format
        # and set the frequency
        df = rs.get_data()
        if not df.empty:
            df['code'] = code
            df['time'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
            df['frequency'] = "d"
            df.set_index('time', inplace=True)
        else:
            df = pd.DataFrame()

        return BaostockResult(df,
                              table_name="stock_ohlcv",
                              tag_columns=['code', 'frequency', 'date'])
    
    def minute(self, code, start_date, end_date, frequency="5", adjustflag="2"):
        """Fetch minute data for a given stock code."""
        rs = bs.query_history_k_data_plus(
            code,
            "date,time,open,high,low,close,volume,amount",
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            adjustflag=adjustflag,
        )
        if rs.error_code != '0':
            raise Exception(f"Query failed: {rs.error_msg}")
        
        # Convert the 'time' column to datetime format
        # and set the frequency
        df = rs.get_data()
        if not df.empty:
            df['code'] = code
            df['time'] = pd.to_datetime(df['time'], format='%Y%m%d%H%M%S000')
            df['frequency'] = f'{frequency}min'
            df.set_index('time', inplace=True)
        else:
            df = pd.DataFrame()

        return BaostockResult(df,
                              table_name="stock_ohlcv",
                              tag_columns=['code', 'frequency', 'date'])

    def all_stocks(self):
        """Fetch all stock codes available in Baostock."""
        rs_basic = bs.query_stock_basic()
        if rs_basic.error_code != '0':
            raise Exception(f"Query failed: {rs_basic.error_msg}")
        
        df_basic = rs_basic.get_data()
        df_stock = df_basic[df_basic['type'] == '1']

        return BaostockResult(df_stock,
                              table_name="stocks")



if __name__ == "__main__":

    # Example usage
    with BaostockHelper() as helper:
        daily_result = helper.daily("sh.600008", "2020-09-01", "2020-09-28")
        # daily_result.save_to_influx(daily_result.table_name)

        minute_result = helper.minute("sh.600008", "2020-09-01", "2020-09-28")
        # minute_result.save_to_influx(minute_result.table_name)

        print(1)
        # all_stocks_result = helper.all_stocks()
        # all_stocks_result.save_to_mongo("stocks", replace=True)
