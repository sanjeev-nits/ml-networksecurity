import yaml
from networksecurity.exception.exceptions import NetworkSecurityError
import os,sys
from networksecurity.logging.logger import logging

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