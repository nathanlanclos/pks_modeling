#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 10:31:41 2024

@author: jonah
"""

import pandas as pd

# Path to data folder
directory = '../GATSol/dataset/'

# Load CSVs
df_train = pd.read_pickle(directory + 'eSol_train.pkl')
df_test = pd.read_pickle(directory + 'eSol_test.pkl')

# Drop 'aac' and 'blosum62_embedding' columns
df_train.drop(columns=['aac', 'blosum62_embedding', 'flexibility', 'embedding'], inplace=True)
df_test.drop(columns=['aac', 'blosum62_embedding', 'flexibility', 'embedding'], inplace=True)

# Save the updated DataFrames
df_train.to_pickle(directory + 'eSol_train_updated.pkl')
df_test.to_pickle(directory + 'eSol_test_updated.pkl')