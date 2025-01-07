#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:55:45 2025

@author: jonah
"""

import streamlit as st
import pandas as pd
import numpy as np
from pred_struct import *

st.write("""# Welcome to ggProt """)

seq = st.text_input("Enter your protein sequence:", placeholder="e.g., MSAGVITGVLLVFLLLGYLVYALINAEAF")

if seq:  # Run the tokenizer only if seq is not empty
    input_ids, outputs = pred_struct.run_tokenizer(seq)

st.write("""### Citation
Lanclos N, Weigand-Whittier J. Cool title. Cool journal (2025).""")