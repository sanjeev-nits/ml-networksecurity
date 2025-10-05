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