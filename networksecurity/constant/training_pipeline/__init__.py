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
