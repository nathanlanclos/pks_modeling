#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 20:03:41 2024

@author: jonah
"""

import pandas as pd
import protlearn.features
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np

# Path to data folder
directory = '../GATSol/dataset/'

# Load training and testing data (with embeddings)
df_train = pd.read_pickle(directory + 'eSol_Train.pkl')
df_test = pd.read_pickle(directory + 'eSol_Test.pkl')

# Concatenate dataframes
dfs = [df_train, df_test]

# Loop through dataframes and extract features, append to dataframe
for df in dfs:
    df['binary_solubility'] = (df['solubility'] > 0.5).astype(int) # Add binary solubility score
    # Create empty lists to store features
    molecular_weights = []
    aromaticities = []
    gravies = []
    isoelectric_points = []
    flexibilities = []
    lengths = []
    aacs = []
    ctdc_dict = {}  # Dictionary to store ctdc features

    for _, row in df.iterrows():
        sequence = row['sequence']
        analysis = ProteinAnalysis(sequence)
        
        # Extract basic features
        molecular_weights.append(analysis.molecular_weight())
        aromaticities.append(analysis.aromaticity())
        gravies.append(analysis.gravy())
        isoelectric_points.append(analysis.isoelectric_point())
        flexibilities.append(analysis.flexibility())
        lengths.append(float(protlearn.features.length(sequence)[0]))
        aacs.append(protlearn.features.aac(sequence))
    
        
        # Extract ctdc features
        ctdc_values, ctdc_labels = protlearn.features.ctdc(sequence)
        ctdc_values = np.squeeze(ctdc_values)
        
        # Append ctdc features to dictionary
        for i, label in enumerate(ctdc_labels):
            if label not in ctdc_dict:
                ctdc_dict[label] = []  # Initialize an empty list for the column
            ctdc_dict[label].append(ctdc_values[i])  # Append the value for the current sequence
    
    # Add basic features to the dataframe
    df['molecular_weight'] = molecular_weights
    df['aromaticity'] = aromaticities
    df['gravy'] = gravies
    df['isoelectric_point'] = isoelectric_points
    df['flexibility'] = flexibilities
    df['length'] = lengths
    df['aac'] = aacs

    # Add ctdc features to the dataframe
    for label, values in ctdc_dict.items():
        df[label] = values
        
# Pickle and save
df_test.to_pickle(directory + 'eSol_test.pkl')
df_train.to_pickle(directory + 'eSol_train.pkl')  