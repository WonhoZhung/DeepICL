import os
import pickle

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from tqdm import tqdm


def get_scaffold(mol):
    try:
        scf = GetScaffoldForMol(mol)
    except:
        return
    return scf


def get_unique_scaffold_list(mol_smi_list):
    scf_smi_set = set()
    tot_mol_list = []
    for mol_smi in tqdm(mol_smi_list):
        try:
            mol = Chem.MolFromSmiles(mol_smi)
            assert mol is not None
            scf = get_scaffold(mol)
            assert scf is not None
            scf_smi = Chem.MolToSmiles(scf, isomericSmiles=True, canonical=True)
            assert scf_smi is not ""
        except:
            continue
        scf_smi_set.add(scf_smi)
        tot_mol_list.append(mol)
    scf_smi_list = sorted(list(scf_smi_set))
    return scf_smi_list, tot_mol_list


def get_scaffold_freq_dict(mol_list):
    scf_smi_list = []
    for mol in tqdm(mol_list):
        try:
            scf = get_scaffold(mol)
            assert scf is not None
            scf_smi = Chem.MolToSmiles(scf, isomericSmiles=True, canonical=True)
            assert scf_smi is not ""
        except:
            continue
        scf_smi_list.append(scf_smi)
    scf_freq_dict = {k: 0 for k in scf_smi_list}
    for scf_smi in scf_smi_list:
        scf_freq_dict[scf_smi] += 1
    return scf_freq_dict


DATA_DIR = "./keys/train_smiles.txt"
RESULT_DIR = "./keys/train_scaffold.txt"
with open(DATA_DIR, "r") as f:
    mol_smi_list = [l.strip().split()[-1] for l in f.readlines()]
print("NUM MOLECULE:", len(mol_smi_list))
scf_smi_list, tot_mol_list = get_unique_scaffold_list(mol_smi_list)
print(len(tot_mol_list))
print(len(scf_smi_list))
print(f"{100 * len(scf_smi_list) / len(tot_mol_list):.2f} %")
exit()
tot_scf_freq_dict = get_scaffold_freq_dict(tot_mol_list)
with open("./keys/train_scaffold_freq_dict.pkl", "wb") as w:
    pickle.dump(tot_scf_freq_dict, w)

exit()
print("NUM SCAFFOLD:", len(scf_smi_list))
with open(RESULT_DIR, "w") as w:
    for scf_smi in scf_smi_list:
        w.write(f"{scf_smi}\n")
