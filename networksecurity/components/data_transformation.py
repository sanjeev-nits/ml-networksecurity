import sys
import os
import numpy as np
import pandas as pd  

from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from networksecurity.constant.training_pipeline import TARGET_COLUMN,DATA_TRANSFOMATION_IMPUTER_OBJECT_FILE_NAME
from networksecurity.entity.artifact_entity import DataTransformationArtifact,DataValidationArtifact

from networksecurity.entity.config_entity import DataValidationConfig,DataTransformationConfig
from networksecurity.exception.exceptions import NetworkSecurityError
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import save_numpy_array_data,save_object



class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_validation_artifact=data_validation_artifact
            self.data_transformation_config=data_transformation_config
        except Exception as e:
            raise NetworkSecurityError(e,sys) from e
        
    @staticmethod
    def read_data(file_path):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityError(e,sys) from e
        
    def get_data_transformer_object(self)->Pipeline:
        try:
            logging.info("entre the get data transformer object method")
            imputer=KNNImputer(**DATA_TRANSFOMATION_IMPUTER_OBJECT_FILE_NAME)
            pipeline=Pipeline(steps=[
                ('Imputer',imputer)
            ])
            return pipeline
        except Exception as e:
            raise NetworkSecurityError(e,sys) from e
        
    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info("Reading training and test data")
            train_df=self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df=self.read_data(self.data_validation_artifact.valid_test_file_path)
            
            #training dataframe
            input_feature_train_df=train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df=train_df[TARGET_COLUMN].replace({-1:0,1:1})

            input_feature_test_df=test_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_test_df=test_df[TARGET_COLUMN].replace({-1:0,1:1})

            preprocesser=self.get_data_transformer_object()
            logging.info("Applying preprocessing object on training and testing dataframe")

            preprocesser_obj=preprocesser.fit(input_feature_train_df)
            input_feature_train_feature=preprocesser_obj.transform(input_feature_train_df)
            input_feature_test_feature=preprocesser_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_feature,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_feature,np.array(target_feature_test_df)]
            logging.info("Saved transformed training and testing array")
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path,preprocesser_obj)

            #prepare artifact
            data_transformation_artifact=DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                preprocessed_object_file_path=self.data_transformation_config.transformed_object_file_path
            )
            logging.info(f"Data transformation artifact:{data_transformation_artifact}")
            return data_transformation_artifact




        except Exception as e:
            raise NetworkSecurityError(e,sys) from e