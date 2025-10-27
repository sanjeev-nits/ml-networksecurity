import os
import sys
import os
import numpy as np
import pandas as pd  

from networksecurity.logging.logger import logging
from networksecurity.exception.exceptions import NetworkSecurityError


from networksecurity.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact
from networksecurity.entity.config_entity import   ModelTrainerConfig


from networksecurity.utils.ml_utils.model.estimater import NetworkModel
from networksecurity.utils.main_utils.utils import save_object,load_object,load_numpy_array_data,evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import dagshub
dagshub.init(repo_owner='sanjeevkumar814155', repo_name='ml-networksecurity', mlflow=True)


class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifacts:DataTransformationArtifact):

        try:
            self.model_trainer_config=model_trainer_config
            self.data_transfer_artifacts=data_transformation_artifacts
        except Exception as e:
            raise NetworkSecurityError(e,sys)
        
    def track_mlflow(self,best_model, classification_metric, X_train):
        """
        Logs model metrics and the trained model to MLflow.
        
        Args:
            best_model: The trained scikit-learn model object.
            classification_metric: An object containing classification metrics.
            X_train: A sample of your training data (e.g., a pandas DataFrame).
        """
        with mlflow.start_run():
            # Access metrics from the classification_metric object
            f1_score = classification_metric.f1_score
            precision_score = classification_metric.precision_score
            recall_score = classification_metric.recall_score

            # Log the metrics
            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision score", precision_score)
            mlflow.log_metric("recall_score", recall_score)

            # Infer the model signature from the training data
            predictions = best_model.predict(X_train)
            signature = infer_signature(X_train, predictions)

            # Log the model with the correct artifact_path, signature, and input example
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model",
                signature=signature,
                input_example=X_train[[0]] # Use the first row as a small input example
            )
        

    def train_model(self,x_train,y_train,x_test,y_test):

        models= {
            "Random Forest":RandomForestClassifier(verbose=1),
            "Decision Tree":DecisionTreeClassifier(),
            "AdaBoost":AdaBoostClassifier(),
            "Logistic Regression":LogisticRegression(verbose=1)
        }

        params={
            "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            "Random Forest":{
                # 'criterion':['gini', 'entropy', 'log_loss'],
                
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,32,128,256]
            },
            "Gradient Boosting":{
                # 'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.001],
                'subsample':[0.6,0.7,0.75],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,32,256]
            },
            "Logistic Regression":{},
            
            "AdaBoost":{
                'learning_rate':[.1,.01,.001],
                'n_estimators': [8,16,32,64,128,256]
            }
            
        }
        model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,params=params)
        
        best_model_score=max(sorted(model_report.values()))
        
        best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

        best_model=models[best_model_name]
        y_train_pred=best_model.predict(x_train)

        classification_train_metric= get_classification_score(y_true=y_train,y_pred=y_train_pred)
        logging.info(f"classification train metric:{classification_train_metric}")

        ## track exprement with the mlflow
        self.track_mlflow(best_model,classification_train_metric,x_train)
        


        y_test_pred=best_model.predict(x_test)

        classification_test_metric= get_classification_score(y_true=y_test,y_pred=y_test_pred)
        logging.info(f"classification test metric:{classification_test_metric}")
        self.track_mlflow(best_model,classification_test_metric,x_test)

        preprocessor=load_object(file_path=self.data_transfer_artifacts.preprocessed_object_file_path)

        model_dir_path=os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)
        
        Network_model=NetworkModel(preprocessor=preprocessor,model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj=Network_model)

        save_object("final_model/model.pkl",best_model)

        ##model trainer Artifacts
        model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=classification_train_metric,
                             test_metric_artifact=classification_test_metric
                             )
        
        logging.info(f"MODEL TRAINER ARTIFACT:{model_trainer_artifact}")




    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path=self.data_transfer_artifacts.transformed_train_file_path
            test_file_path=self.data_transfer_artifacts.transformed_test_file_path

            train_arr=load_numpy_array_data(train_file_path)
            test_arr=load_numpy_array_data(test_file_path)

            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            model=self.train_model(x_train,y_train,x_test,y_test)
        except Exception as e:
            raise NetworkSecurityError(e,sys)
        



        
