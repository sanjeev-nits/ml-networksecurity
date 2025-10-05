from datetime import datetime
import os

from networksecurity.constant import training_pipeline


class TrainPipelineConfig:
    def __init__(self):
        self.pipeline_name=training_pipeline.PIPELINE_NAME
        self.artifact_dir=os.path.join(
            os.getcwd(),
            training_pipeline.ARTIFACT_DIR,
            datetime.now().strftime("%m%d%Y__%H%M%S")
        )




class DataingestionConfig:
    def __init__(self,train_pipeline_config:TrainPipelineConfig):
        self.data_ingestion_dir:str=os.path.join(
            train_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path=os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            training_pipeline.FILE_NAME
        )
        self.train_file_path=os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TRAIN_FILE_NAME
        )
        self.test_file_path=os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TEST_FILE_NAME
        )
        self.collection_name=training_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.database_name=training_pipeline.DATA_INGESTION_DATABASE_NAME
        self.test_size=training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION

class DataValidationConfig:
    def __init__(self,train_pipeline_config:TrainPipelineConfig):
        self.data_validation_dir:str=os.path.join(
            train_pipeline_config.artifact_dir,
            training_pipeline.DATA_VALIDATION_DIR_NAME
        )
        self.valid_data_dir=os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_VALID_DIR
        )
        self.valid_train_file_path=os.path.join(
            self.valid_data_dir,
            training_pipeline.TRAIN_FILE_NAME
        )
        self.valid_test_file_path=os.path.join(
            self.valid_data_dir,
            training_pipeline.TEST_FILE_NAME
        )
        self.invalid_data_dir=os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_INVALID_DIR
        )
        self.drift_report_file_path=os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
        )
        