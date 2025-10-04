from networksecurity.exception.exceptions import NetworkSecurityError
from networksecurity.logging.logger import logging
import os
import sys

from networksecurity.entity.config_entity import DataingestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from networksecurity.constant import training_pipeline
import pymongo



from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self,data_ingestion_config:DataingestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise NetworkSecurityError(e,sys) from e
        


    def export_collection_as_dataframe(self):
        logging.info('Export MongoDB collection as a pandas dataframe')
        try:
            database_name=self.data_ingestion_config.database_name
            collection_name=self.data_ingestion_config.collection_name
            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)

            db = self.mongo_client[database_name]              # database
            collection = db[collection_name]

            df=pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df=df.drop(columns=["_id"],axis=1)

            df.replace({'na':np.nan},inplace=True)
            logging.info(f"Connected to MongoDB database: {database_name} and collection: {collection_name}")
            
            return df
        except Exception as e:
            raise NetworkSecurityError(e,sys) from e
        


        
    def   export_data_into_feature_store(self,df:pd.DataFrame):
        """Export data into feature store"""
        try:
            feature_store_dir=os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir,exist_ok=True)
            df.to_csv(self.data_ingestion_config.feature_store_file_path,index=False,header=True)
            logging.info(f"Exported data into feature store at {self.data_ingestion_config.feature_store_file_path}")
            return df
        except Exception as e:
            raise NetworkSecurityError(e,sys) from e
        
    def split_data_as_train_test(self,df):
        """Split data into train and test set"""
        try:
            train_set,test_set=train_test_split(df,test_size=self.data_ingestion_config.test_size,random_state=42)
            train_dir=os.path.dirname(self.data_ingestion_config.train_file_path)
            test_dir=os.path.dirname(self.data_ingestion_config.test_file_path)
            os.makedirs(train_dir,exist_ok=True)
            os.makedirs(test_dir,exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.train_file_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_file_path,index=False,header=True)
            logging.info(f"Split data into train and test set at {self.data_ingestion_config.train_file_path} and {self.data_ingestion_config.test_file_path}")
        except Exception as e:
            raise NetworkSecurityError(e,sys) from e


    
        

    def initiate_data_ingestion(self):
        try:
            dataframe=self.export_collection_as_dataframe()
            df=self.export_data_into_feature_store(df=dataframe)
            self.split_data_as_train_test(df)

            dataingestionartifact=DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_file_path, 
                test_file_path=self.data_ingestion_config.test_file_path
            )
            return dataingestionartifact
        except Exception as e:
            raise NetworkSecurityError(e,sys) from e