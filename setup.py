#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 09:27:52 2020

@author: tyrion
"""
from setuptools import setup

if __name__ == '__main__':
    
    setup(
        name='polyfit',   
        version='0.2', 
        description='Constrained polynomial regression',
        author='Daniel Schmitz',
        license='MIT',
        packages=['polyfit'],
        install_requires=[
            'numpy',
            'cvxpy',
            'scikit-learn'
        ],
        include_package_data=True,
        package_data={'': ['Example_Data.npz']}             
    )