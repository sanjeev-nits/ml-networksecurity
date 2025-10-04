from dataclasses import dataclass
import os
import sys
from networksecurity.constant.training_pipeline import *

@dataclass
class DataIngestionArtifact:
    train_file_path:str
    test_file_path:str