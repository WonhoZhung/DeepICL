# DeepICL

[//]: # (Badges)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is a repository of our paper, "A Protein-Ligand Interaction-focused 3D Molecular Generative Framework for Generalizable Structure-based Drug Design" ([ChemRxiv link](https://chemrxiv.org/engage/chemrxiv/article-details/6482d9dbbe16ad5c57af1937#)).

Inspired by how the practitioners manage to improve the potency of a ligand toward a target protein, we devised a strategy where prior knowledge of appropriate interactions navigates the ligand generation.
Our proposed model, DeepICL (**Deep** **I**nteraction-**C**onditioned **L**igand generative model), employs an interaction condition that captures the local pocket environment, to precisely control the generation process of a ligand inside a binding pocket.


## Install packages
You can install required packages by running following commands:
```
chmod +x install_packages.sh
bash install_packages.sh
```

## Data processing
First, download general set of the PDBbind dataset from [Link](http://www.pdbbind.org.cn/).
Then, run the following commands to process the data:
```
cd data
python preprocessing.py {PDBBIND_DATA_DIR} {PROCESSED_DATA_DIR} {NCPU}
```
If you are processing a data for sampling, you can follow the instructions in this [Demo](https://drive.google.com/file/d/10uxhu7vUuEkefOe7yb2FeE-6Ekdfp8qR/view?usp=sharing).


## Training DeepICL
For training DeepICL, run the following commands:
```
cd script
python -u train.py --world_size {NGPU} --save_dir {SAVE_DIR} --data_dir {DATA_DIR} --key_dir {KEY_DIR} --num_layers 6 --num_dense_layers 3 --num_hidden_feature 128 --dist_one_hot_param1 0 10 25 --dist_one_hot_param2 0 15 300 --lr 1e-3 --num_epochs 1001 --save_every 1 --k 8 --vae_loss_beta 0.2 --lr_decay 0.8 --lr_tolerance 4 --lr_min 1e-6 --conditional
```

## Ligand sampling
For sampling ligands via DeepICL, run the following commands:
```
cd script
python -u generate.py --ngpu 0 --ncpu {NCPU} --k 8 --data_dir {DATA_DIR} --key_dir {KEY_DIR} --restart_dir {SAVED_MODEL_DIR} --result_dir {RESULT_DIR} --num_layers 6 --num_dense_layers 3 --num_hidden_feature 128 --num_sample {NUM_SAMPLE} --max_num_add_atom 30 --dist_one_hot_param1 0 10 25 --dist_one_hot_param2 0 15 300 --temperature_factor1 0.1 --temperature_factor2 0.1 --radial_limits 0.9 2.2 --add_noise --pocket_coeff_max 10.0 --pocket_coeff_thr 2.5 --pocket_coeff_beta 0.91 --conditional --use_condition --verbose -y --memo {MEMO for sampling details}
```

Ligand elaboration with predefined core is demonstrated in this [Demo](https://drive.google.com/file/d/10uxhu7vUuEkefOe7yb2FeE-6Ekdfp8qR/view?usp=sharing).
