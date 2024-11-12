#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:12:35 2024

@author: jonah
"""
import torch
import esm
import pandas as pd

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

# Convert the sequences to batch format
batch_labels, batch_strs, batch_tokens = batch_converter(sequences_train)

# Generate embeddings
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33])
embeddings = results["representations"][33]

# Aggregate embeddings by averaging across sequence length
sequence_embeddings = embeddings.mean(dim=1)

# Save embeddings along with gene names and solubility values for downstream use
df_train['embedding'] = [embedding.cpu().numpy() for embedding in sequence_embeddings]
df_train.to_pickle(directory + "eSol_train_embeddings.pkl")  # Save as a pickle file for easy loading later