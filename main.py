from networksecurity.components.data_ingestion import DataIngestion
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
    except Exception as e:
        raise NetworkSecurityError(e,sys) from e
