import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm


def read_pkl(fn):
    with open(fn, "rb") as f:
        data = pickle.load(f)
    return data


def interaction_stat(data_list):
    info_list = []
    for data in tqdm(data_list):
        interaction_info = data[-1]
        interaction_cnt = interaction_info.sum(0)  # pipi, salt, hbond, hydro, none
        info_list.append(interaction_cnt)
    return np.stack(info_list, axis=0)


def mw_stat(data_list):
    mw_list = []
    for data in data_list:
        smi = data[-3]
        mol = Chem.MolFromSmiles(smi)
        mw = Descriptors.MolWt(mol)
        mw_list.append(mw)
    return np.array(mw_list)


def num_atom_stat(data_list):
    na_list = []
    for data in data_list:
        smi = data[-3]
        mol = Chem.MolFromSmiles(smi)
        na = mol.GetNumAtoms()
        na_list.append(na)
    return np.array(na_list)


fn_dir = "./data/"
fn_list = sorted(glob.glob(fn_dir + "*"))
key_list = [fn.split("/")[-1] for fn in fn_list]
pdb2aff_dict = read_pkl("./pdbbind_v2020_general_true_aff.pkl")
# pdb2aff_dict = read_pkl("./pdbbind_v2020_total_calc_affinity_dict.pkl")["ligand"]
pdb2aff_list = sorted(
    [(k, -1.36 * v) for k, v in pdb2aff_dict.items()], key=lambda x: x[1]
)
aff_thr_list = [-1.36 * i for i in range(3, 10)]
aff_range = zip(aff_thr_list[:-1], aff_thr_list[1:])

df = {"Range": [], "Type": [], "Count": [], "MW": [], "Normalized Count": []}
for aff_max, aff_min in aff_range:
    thr_key_list = [k for k, v in pdb2aff_list if aff_min < v < aff_max]
    data_list = [read_pkl(fn) for fn, k in zip(fn_list, key_list) if k in thr_key_list]
    stat = interaction_stat(data_list)  # pipi, salt, hbond, hydro, none
    na = num_atom_stat(data_list).reshape(-1, 1)
    mw = mw_stat(data_list).reshape(-1, 1)
    norm_stat = (10 * stat / na).mean(axis=0)
    stat = stat.mean(axis=0)
    mw = mw.mean()
    na = na.mean()

    for type, cnt, ncnt in zip(
        ["Pi-pi", "Saltbridge", "Hydrogen bond", "Hydrophobic"], stat[:4], norm_stat[:4]
    ):
        df["Range"].append(f"{aff_max:.1f}~{aff_min:.1f}")
        df["Type"].append(type)
        df["Count"].append(cnt)
        df["MW"].append(mw)
        df["Normalized Count"].append(ncnt)

df = pd.DataFrame(df)
# sns.barplot(data=df, x="Range", y="Count")
sns.barplot(data=df, x="Range", y="Count", hue="Type")
plt.show()
exit()

for k, stat in zip(key_list, interaction_stat(data_list)):
    print(k, stat)
