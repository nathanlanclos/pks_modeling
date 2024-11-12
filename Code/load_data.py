#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:12:35 2024

@author: jonah
"""
import torch
import esm
import pandas as pd
from tqdm import tqdm

# Path to data folder
directory = '../GATSol/dataset/'

# Load CSVs
df_train = pd.read_csv(directory + 'eSol_train.csv')

# Convert to list
sequences_train = [(row['gene'], row['sequence']) for _, row in df_train.iterrows()]

# Load the ESM-2 model and alphabet
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # Disable dropout for deterministic results

# Prepare a list to store embeddings
embeddings_list = []

# Loop through each sequence with tqdm to show progress
for batch_labels, batch_strs, batch_tokens in tqdm([batch_converter([seq]) for seq in sequences_train], desc="Generating Embeddings"):
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])
    embedding = results["representations"][33].mean(dim=1)  # Mean pooling
    embeddings_list.append(embedding.cpu().numpy())

# Add embeddings to the DataFrame and save
df_train['embedding'] = embeddings_list
df_train.to_pickle(directory + 'eSol_train.pkl')  # Save with embeddings