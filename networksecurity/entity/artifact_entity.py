from dataclasses import dataclass
import os
import sys
from networksecurity.constant.training_pipeline import *

@dataclass
class DataIngestionArtifact:
    train_file_path:str
    test_file_path:str

@dataclass
class DataValidationArtifact:
    validation_status:bool
    valid_data_dir:str
    valid_train_file_path:str
    valid_test_file_path:str
    invalid_data_dir:str
    drift_report_file_path:str

@dataclass
class DataTransformationArtifact:
    transformed_train_file_path:str
    transformed_test_file_path:str
    preprocessed_object_file_path:str
