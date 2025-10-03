from setuptools import setup, find_packages
import os
import sys
from typing import List

def get_requirements(filename: str) -> List[str]:
    """Read requirements from a file and return them as a list."""
    try:
        requirement_lst = [line.strip() for line in open(filename, 'r') if line.strip() and line.strip() != '-e .']

    except FileNotFoundError:
        print(f"Warning: {filename} not found.")

    return requirement_lst

setup(
    name="Network_Security",
    version="0.0.1",
    author="Sanjeev Kumar",
    author_email="sanjeev814155@gmail.com",
    description="A package for Network Security",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)