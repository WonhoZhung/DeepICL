#!/bin/bash

# 1. Create conda environment
conda create -n DeepICL python=3.7.13 -y
conda activate DeepICL

# 2. install packages dependent on mkl
conda install scipy=1.6.2 numpy=1.21.2 pandas=1.3.4 scikit-learn=1.0.2 seaborn=0.11.0 -y

# 3. install pytorch and torch_geometric
conda install pytorch=1.9.0 cudatoolkit=10.2 -c pytorch -y
conda install pyg=2.0.3 -c pyg -c conda-forge -y

# 4. install others
conda install -c rdkit rdkit=2022.09.1 -y
conda install -c conda-forge biopython=1.77 openbabel=3.1.1 -y
conda install -c conda-forge plip=2.2.2
