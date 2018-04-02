# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 17:46:36 2018

@author: Dawie
"""
#compile renders package
#import py_compile
#py_compile.compile('renders.py', 'renders.pyc')

#http://www.ritchieng.com/machine-learning-project-customer-segments/#
# Import libraries necessary for this project
import numpy as np
import pandas as pd
import renders as rs
from IPython.display import display # Allows the use of display() for DataFrames

# Show matplotlib plots inline (nicely formatted in the notebook)
%matplotlib inline

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print ("Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))
except:
    print ("Dataset could not be loaded. Is the dataset missing?")
    
# Display a description of the dataset
stats = data.describe()
stats

# Using data.loc to filter a pandas DataFrame
data.loc[[100, 200, 300],:]

# Retrieve column names
# Alternative code:
# data.keys()
data.columns

# Fresh filter
fresh_q1 = 3127.750000
display(data.loc[data.Fresh < fresh_q1, :].head())