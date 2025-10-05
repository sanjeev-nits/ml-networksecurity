from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_validation import DataValidationConfig
from networksecurity.exception.exceptions import NetworkSecurityError
import sys
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import TrainPipelineConfig,DataingestionConfig


if __name__=="__main__":
    try:
        training_pipeline_config=TrainPipelineConfig()
        data_ingestion_config=DataingestionConfig(training_pipeline_config)
        data_ingestion=DataIngestion(data_ingestion_config)
        logging.info("Initiate the data ingestion")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        print(dataingestionartifact)
        logging.info(f"Data Ingestion artifact:{dataingestionartifact}")
        logging.info("Data ingestion completed and moving to data validation")
        data_validation_config=DataValidationConfig(training_pipeline_config)
        data_validation=DataValidation(dataingestionartifact,data_validation_config)
        logging.info("Initiate the data validation")
        data_validation_artifact=data_validation.initiate_data_validation()
        logging.info('>>data validation completed  <<')
    except Exception as e:
        raise NetworkSecurityError(e,sys) from e
