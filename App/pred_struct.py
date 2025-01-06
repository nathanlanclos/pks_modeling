#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 12:42:38 2025

@author: jonah
"""

## Adapted from code by Nathan Lanclos

import networkx as nx
import py3Dmol
import numpy as np
import torch
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser
from transformers.utils import send_example_telemetry
from transformers import AutoTokenizer, EsmForProteinFolding

# Load tokenizer and model
model_name = "facebook/esmfold_v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = EsmForProteinFolding.from_pretrained(model_name, low_cpu_mem_usage=True)

# Uncomment to switch the stem to float16 for memory optimization
model.esm = model.esm.half()

# Enable TensorFloat32 for faster matrix multiplications
torch.backends.cuda.matmul.allow_tf32 = True

# Set chunk size optimized for an 11GB GPU
model.trunk.set_chunk_size(64)

def run_tokenizer(protein_seq):
    input_ids = tokenizer([protein_seq], return_tensors="pt", add_special_tokens=False)['input_ids']
    outputs = model(input_ids)
    return input_ids, outputs

