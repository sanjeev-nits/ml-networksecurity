import os
import sys

from networksecurity.constant.training_pipeline import SAVE_MODEL_DIR,MODEL_FILE_NAME

from networksecurity.exception.exceptions import NetworkSecurityError
from networksecurity.logging.logger import logging


class NetworkModel:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor=preprocessor
            self.model=model

        except Exception as e:
            raise NetworkSecurityError(e,sys) from e

    def predict(self,X):
        try:
            logging.info("Prediction started")
            x_transformed=self.preprocessor.transform(X)
            logging.info("Preprocessing completed")
            y_pred=self.model.predict(X)
            logging.info("Prediction completed")
            return y_pred
        except Exception as e:
            raise NetworkSecurityError(e,sys) from e
