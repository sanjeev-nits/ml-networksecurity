import sys
import os
from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.exception.exceptions import NetworkSecurityError
from networksecurity.logging.logger import logging

from networksecurity.utils.main_utils.utils import load_object,load_numpy_array_data,evaluate_models,save_object
from sklearn.metrics import f1_score,precision_score,recall_score


from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.ml_utils.model.estimater import NetworkModel

from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (RandomForestClassifier,
                              AdaBoostClassifier,
                              GradientBoostingClassifier)








class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,
                 data_transformation_artifact:DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityError(e,sys) from e   

    def train_model(self,x_train,y_train,x_test,y_test):
        try:
            models={
                "RandomForest":RandomForestClassifier(),
                "DecisionTree":DecisionTreeClassifier(),
                "LinearRegression":LogisticRegression(),
                "GradientBoosting":GradientBoostingClassifier(),
                "AdaBoost":AdaBoostClassifier()}
            
            params={
                "RandomForest":{
                    'n_estimators':[8,16,32,64,128,256]
                },
                "DecisionTree":{
                    'criterion':['gini','entropy','log_loss'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2']
                },
                "LinearRegression":{
                    'penalty':['l1','l2','elasticnet','none'],
                    'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                },
                "GradientBoosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "AdaBoost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators':[8,16,32,64,128,256]
                }
            }

            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,
                                              x_test=x_test,y_test=y_test,models=models,
                                              params=params)
            
            best_model_name,max_score=sorted(model_report.items(),key=lambda x:x[1],reverse=True)[0]
            best_model=models[best_model_name]

            y_train_pred=best_model.predict(x_train)

            classification_train_metric=get_classification_score(y_true=y_train,y_pred=y_train_pred)

            ## track the MLflow
            
            y_test_pred=best_model.predict(x_test)
            classification_test_metric=get_classification_score(y_true=y_test,y_pred=y_test_pred)

            preprocessor=load_object(file_path=self.data_transformation_artifact.preprocessed_object_file_path)
            model_dir_path=os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)

            network_model=NetworkModel(preprocessor=preprocessor,model=best_model)
            logging.info(f"Saving model at path: {self.model_trainer_config.trained_model_file_path}")
            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=network_model)

            ## model trainer artifact
            model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                                       train_metric_artifact=classification_train_metric,   
                                                         test_metric_artifact=classification_test_metric)
            
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact


        except Exception as e:
            raise NetworkSecurityError(e,sys) from e 
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path=self.data_transformation_artifact.transformed_train_file_path
            test_file_path=self.data_transformation_artifact.transformed_test_file_path

            logging.info(f"Loading transformed training dataset")
            train_array=load_numpy_array_data(file_path=train_file_path)
            test_array=load_numpy_array_data(file_path=test_file_path)
            logging.info(f"Loading transformed training dataset is completed")
            logging.info(f"Splitting training and testing input and target feature")
            x_train,y_train=train_array[:,:-1],train_array[:,-1]
            x_test,y_test=test_array[:,:-1],test_array[:,-1]

            model=self.train_model(x_train,y_train,x_test,y_test)
            return model

        except Exception as e:
            raise NetworkSecurityError(e,sys) from e