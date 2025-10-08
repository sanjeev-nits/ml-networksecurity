from networksecurity.exception.exceptions import NetworkSecurityError
from networksecurity.logging.logger import logging
import sys
import os
from networksecurity.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file       

class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):  
        try:
            logging.info(f"{'>>'*20} Data Validation {'<<'*20}")
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config

            self.schema_info=read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityError(e,sys) from e
        
    @staticmethod
    def read_data(file_path):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityError(e,sys) from e
        
    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns=len(self.schema_info['columns'])
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Dataframe has columns: {dataframe.shape[1]}")
            if dataframe.shape[1]==number_of_columns:
                return True
            return False
        except Exception as e:
            raise NetworkSecurityError(e,sys) from e
        
    def detect_data_drift(self,base_df:pd.DataFrame,current_df,threshold=0.05)->bool:
        try:
            status=True
            report={}
            for column in base_df.columns:
                d1=base_df[column]
                d2=current_df[column]
                same_distribution=ks_2samp(d1,d2) ##compair the distribution of two colums
                if same_distribution.pvalue>threshold:
                    logging.info(f"Same distribution {column}")
                    report[column]={
                        "pvalue":float(same_distribution.pvalue),
                        "same_distribution":True
                    }
                else:
                    logging.info(f"Different distribution {column}")
                    report[column]={
                        "pvalue":float(same_distribution.pvalue),
                        "same_distribution":False
                    }
                    status=False
                    
            drift_report_file_path=self.data_validation_config.drift_report_file_path
            report_dir_path=os.path.dirname(drift_report_file_path)
            os.makedirs(report_dir_path,exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path,content=report,replace=True)
            return status
        except Exception as e:
            raise NetworkSecurityError(e,sys) from e    

    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            train_file_path=self.data_ingestion_artifact.train_file_path
            test_file_path=self.data_ingestion_artifact.test_file_path

            ##reading data from file
            train_df=DataValidation.read_data(train_file_path)
            test_df=DataValidation.read_data(test_file_path)


            ##validate number of columns
            status=self.validate_number_of_columns(train_df)
            if not status:
                raise Exception("Number of columns are not matching with schema file")
            
            ##lets cheaking data drift
            status=self.detect_data_drift(base_df=train_df,current_df=test_df)
            dir_path=os.path.dirname(self.data_validation_config.valid_data_dir)
            os.makedirs(dir_path,exist_ok=True)

            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_validation_config.valid_test_file_path), exist_ok=True)

            train_df.to_csv(self.data_validation_config.valid_train_file_path,index=False,header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path,index=False,header=True)

            return DataValidationArtifact(
    validation_status=True,
    valid_data_dir=self.data_validation_config.valid_data_dir,
    valid_train_file_path=self.data_validation_config.valid_train_file_path,
    valid_test_file_path=self.data_validation_config.valid_test_file_path,
    invalid_data_dir=self.data_validation_config.invalid_data_dir,
    drift_report_file_path=self.data_validation_config.drift_report_file_path
)

            
        except Exception as e:
            raise NetworkSecurityError(e,sys) from e 




