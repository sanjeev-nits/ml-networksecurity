import yaml
from networksecurity.exception.exceptions import NetworkSecurityError
import os,sys
from networksecurity.logging.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

import pandas as pd
import numpy as np

import pickle

def read_yaml_file(file_path:str)->dict:
    """
    Reads a YAML file and returns its contents as a dictionary.

    Args:
        file_path (str): The path to the YAML file. """
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise NetworkSecurityError(e, sys) from e

def write_yaml_file(file_path:str,content:object,replace:bool=False)->None:
    """
    Writes a dictionary to a YAML file.

    Args:
        file_path (str): The path to the YAML file.
        content (object): The content to write to the file.
        replace (bool): Whether to replace the file if it already exists. Default is False.
    """
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path, 'w') as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityError(e, sys) from e
    

def save_numpy_array_data(file_path:str,array:np.array)->None:
    """
    Saves a numpy array to a file.

    Args:
        file_path (str): The path to the file where the array will be saved.
        array (np.array): The numpy array to save.
    """
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            np.save(file_obj,array)
    except Exception as e:
        raise NetworkSecurityError(e, sys) from e
    
def save_object(file_path:str,obj:object)->None:
    """
    Saves a Python object to a file using pickle.

    Args:
        file_path (str): The path to the file where the object will be saved.
        obj (object): The Python object to save.
    """
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise NetworkSecurityError(e, sys) from e
    
def load_object(file_path:str)->object:
    """
    Loads a Python object from a file using pickle.

    Args:
        file_path (str): The path to the file from which the object will be loaded.

    Returns:
        object: The loaded Python object.
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityError(e, sys) from e
    
def load_numpy_array_data(file_path:str)->np.array:
    """
    Loads a numpy array from a file.

    Args:
        file_path (str): The path to the file from which the array will be loaded.

    Returns:
        np.array: The loaded numpy array.
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")
        with open(file_path,'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityError(e, sys) from e
    
def evaluate_models(x_train,y_train,x_test,y_test,models,params):
    try:
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            param=params[list(models.keys())[i]]

            gs=GridSearchCV(model,param,cv=3)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)

            y_train_pred=model.predict(x_train)
            y_test_pred=model.predict(x_test)

            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]]=test_model_score
    
            logging.info(f"{list(models.keys())[i]} : Train Score : {train_model_score} Test Score : {test_model_score}")
        return report
    except Exception as e:
        raise NetworkSecurityError(e, sys) from e