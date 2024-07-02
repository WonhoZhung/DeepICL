# DeepICL

[//]: # (Badges)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is a repository of our paper, "3D molecular generative framework for interaction-guided drug design", _Nat Commun_ 15, 2688 (2024). ([Link](https://doi.org/10.1038/s41467-024-47011-2)).

Inspired by how the practitioners manage to improve the potency of a ligand toward a target protein, we devised a strategy where prior knowledge of appropriate interactions navigates the ligand generation.
Our proposed model, DeepICL (**Deep** **I**nteraction-**C**onditioned **L**igand generative model), employs an interaction condition that captures the local pocket environment to precisely control the generation process of a ligand inside a binding pocket.

<p align="center">
  <img src="image/Figure 1.jpg" /> 
</p>


## Install packages
You can install the required packages by running the following commands:
```
chmod +x install_packages.sh
bash install_packages.sh
```
It will take a few minutes, have a coffee break!~☕️

## Data processing
First, download the general set of the PDBbind dataset from [Link](http://www.pdbbind.org.cn/).
Then, run the following commands to process the data:
```
cd data
python preprocessing.py {PDBBIND_DATA_DIR} {PROCESSED_DATA_DIR} {NCPU}
```
If you are processing data for sampling, you can follow the instructions in this [Demo](https://drive.google.com/file/d/10uxhu7vUuEkefOe7yb2FeE-6Ekdfp8qR/view?usp=sharing).


## Training DeepICL
For training DeepICL, run the following commands:
```
cd script
python -u train.py --world_size {NGPU} --save_dir {SAVE_DIR} --data_dir {DATA_DIR} --key_dir {KEY_DIR} --lr 1e-3 --num_epochs 1001 --save_every 1 --k 8 --lr_decay 0.8 --lr_tolerance 4 --lr_min 1e-6 --conditional
```
```
Arguments:
  --world_size      number of GPUs for DDP (int)
  --save_sir        directory where model .pt files will be saved (str)
  --data_dir        directory where processed data exist (str)
  --key_dir         directory where key .pkl files exist (str)
  --lr              learning rate (float)
  --num_epochs      number of epochs to train model (int)
  --save_every      period of saving model during training (int)
  --k               k in k-nearest neighbor (int)
  --lr_decay        lr scheduling parameter for decaying (float)
  --lr_tolerance    lr scheduling parameter of how many epochs to tolerate (int)
  --lr_min          lr scheduling parameter of minimum lr (float)
  --conditional     if true, the model uses interaction condition during training - for ablation (store_true)
```

## Ligand sampling
For sampling ligands via DeepICL, run the following commands:
```
cd script
python -u generate.py --ncpu {NCPU} --k 8 --data_dir {DATA_DIR} --key_dir {KEY_DIR} --restart_dir {SAVED_MODEL_DIR} --result_dir {RESULT_DIR} --num_layers 6 --num_dense_layers 3 --num_hidden_feature 128 --num_sample {NUM_SAMPLE} --max_num_add_atom 30 --dist_one_hot_param1 0 10 25 --dist_one_hot_param2 0 15 300 --temperature_factor1 0.1 --temperature_factor2 0.1 --radial_limits 0.9 2.2 --add_noise --pocket_coeff_max 10.0 --pocket_coeff_thr 2.5 --pocket_coeff_beta 0.91 --conditional --use_condition --verbose -y
```
```
Arguments:   
  --ncpu                number of CPUs for multiprocessing (int)   
  --data_dir            directory where processed data exist (str)   
  --key_dir             directory where key .pkl files exist (str)   
  --restart_dir         directory of the saved model .pt (str)   
  --result_dir          directory where the sampled ligands .sdf will be saved (str)
  --k                   k in k-nearest neighbor (int)   
  --num_sample          number of sampling for a single pocket (int)   
  --max_num_add_atom    maximum number of atoms to add during the sampling (int)   
  --temperature_factor1 temperature factor for controlling randomness of type selection (float)   
  --temperature_factor2 temperature factor for controlling randomness of position selection (float)
  --add_noise           if true, apply Gaussian noise to the initial position (store_true)   
  --conditional         if true, the model uses interaction condition for sampling (store_true)   
  --use_condition       deprecated, always true when --conditional is true (store_true)
  --use_scaffold        if true, use scaffold - WARNING: scaffold should be defined in data processing (store_true)   
  --verbose             if true, verbose mode that print logs (store_true)   
  --y                   if true, recreate the restart directory without asking (store_true)   
```

It took about a minute to generate 100 samples with 8 CPUs.

Ligand elaboration with a predefined core is demonstrated in this [Demo](https://drive.google.com/file/d/10uxhu7vUuEkefOe7yb2FeE-6Ekdfp8qR/view?usp=sharing).


## (Update!) Ligand sampling without reference ligand (pocket-only)
An interaction condition is set based on heuristics e.g. SMARTS patterns of amino acid residues.
Since there is no reference ligand, this method needs additional inforamtion about the [x,y,z] coordinate of the initial point.

Reference ligand-free sampling is demonstrated in this [Demo2](https://drive.google.com/file/d/1Sg2mIeFut66KjAhZBgM-1nPqSmc9g7lC/view?usp=sharing).

## (Update!) Ligands generated from CrossDocked2020 benchmark set
100 ligands are generated for each pocket in CrossDocked2020 benchmark set, following the splitting method of [Luo et al.](https://arxiv.org/pdf/2203.10446)
Note that DeepICL was trained on PDBbind v.2020, thus an overlap of pockets between the training set and CrossDocked2020 test set may exist.


You can download the generated ligands (in total of 10,000) at [Zenodo](https://zenodo.org/records/12623708?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjkyYTg5N2JjLTI5OGUtNDU3MS05ZGFlLWQ5ZDkxNWIwZDYxMyIsImRhdGEiOnt9LCJyYW5kb20iOiJlYjg3YjBiZDU3NDgzNjIyZGY3MzFjNWE3YmU5ZGQ2NiJ9.JWdxSstx1ktp2M9y4CTHq2mCecXWcGe_72WuTbdPtPH7TMMK2YSlUfGwcAMIzA6zatLQWMqV-Wjb5uEueK33qw).


## Citing this work
```
@article{Zhung2024,
  title = {3D molecular generative framework for interaction-guided drug design},
  volume = {15},
  ISSN = {2041-1723},
  url = {http://dx.doi.org/10.1038/s41467-024-47011-2},
  DOI = {10.1038/s41467-024-47011-2},
  number = {1},
  journal = {Nature Communications},
  publisher = {Springer Science and Business Media LLC},
  author = {Zhung,  Wonho and Kim,  Hyeongwoo and Kim,  Woo Youn},
  year = {2024},
  month = mar 
}
```
