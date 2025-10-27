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

class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifacts:DataTransformationArtifact):

        try:
            self.model_trainer_config=model_trainer_config
            self.data_transfer_artifacts=data_transformation_artifacts
        except Exception as e:
            raise NetworkSecurityError(e,sys)
        

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
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.85,0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
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

        ## track the ml flow


        y_test_pred=best_model.predict(x_test)

        classification_test_metric= get_classification_score(y_true=y_test,y_pred=y_test_pred)

        preprocessor=load_object(file_path=self.data_transfer_artifacts.preprocessed_object_file_path)

        model_dir_path=os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)
        
        Network_model=NetworkModel(preprocessor=preprocessor,model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj=Network_model)

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
        
