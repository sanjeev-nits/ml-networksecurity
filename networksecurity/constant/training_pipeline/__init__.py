import os 
import sys
import pandas as pd
import numpy as np




"""dafine common constant var for trainig pipeline"""

TARGET_COLUMN="Result"
PIPELINE_NAME:str="networksecurity"
ARTIFACT_DIR:str="artifact"
FILE_NAME:str="phisingData.csv"

TRAIN_FILE_NAME:str="train.csv"
TEST_FILE_NAME:str="test.csv"

SCHEMA_FILE_PATH=os.path.join("data_schema","schema.yaml")


""""Dta Ingestion Constants  statrt with DATA_INGESTION VAr name"""

DATA_INGESTION_COLLECTION_NAME="Networkdata"
DATA_INGESTION_DATABASE_NAME="SANJEEV"
DATA_INGESTION_DIR_NAME="data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR="feature_store"
DATA_INGESTION_INGESTED_DIR="ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION=0.2


"""Data VAlidation constants start with DATA_VALIDATION VAR name"""

DATA_VALIDATION_DIR_NAME="data_validation"
DATA_VALIDATION_VALID_DIR="validated"
DATA_VALIDATION_INVALID_DIR="invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR="drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME="report.yaml"



"""Data Transformation Constants start with DATA_TRANSFORMATION VAR name"""

DATA_TRANSFORMATION_DIR_NAME: str="data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str="transformed"
DATA_TRANSFORMATION_TRANSFROMED_OBJECT_DIR: str="transformed_object"
PREPROCESSING_OBJECT_FILE_NAME: str="preprocessing.pkl"

#knn imputer object file name
DATA_TRANSFOMATION_IMPUTER_OBJECT_FILE_NAME: dict={
    'missing_values': np.nan,
    'n_neighbors': 3,
    'weights': 'uniform',
}

"""Model Trainer Constants start with MODEL_TRAINER VAR name"""

SAVE_MODEL_DIR:str=os.path.join("saved_models")
MODEL_TRAINER_DIR_NAME: str="model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str="trained_model"
MODEL_FILE_NAME: str="model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float=0.7
MODEL_TRAINER_OVERFITTING_THRESHOLD: float=0.05