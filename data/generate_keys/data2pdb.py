import os
import sys
import torch
import pickle
import traceback
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


ATOM_TYPES = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br']


def write_xyz(types, coords, msg="", fn=None, is_onehot=True):
    if isinstance(types, torch.Tensor):
        types = np.array(types.detach().cpu())
    if isinstance(coords, torch.Tensor):
        coords = np.array(coords.detach().cpu())
    xyz = ""
    xyz += f"{coords.shape[0]}\n"
    xyz += msg + '\n'
    for i in range(coords.shape[0]):
        if is_onehot:
            atom_type = ATOM_TYPES[np.argmax(types[i])]
        else:
            atom_type = types[i]
        xyz += f"{atom_type}\t{coords[i][0]}\t{coords[i][1]}\t{coords[i][2]}\n"
    if fn is not None:
        with open(fn, 'w') as w: w.writelines(xyz[:-1])
    return xyz[:-1]

def process_sdf(sdf_fn):
    with open(sdf_fn, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for i, line in enumerate(lines):
        if i < 3:
            new_lines.append(line)
        elif i == 3:
            new_lines.append(line)
            NA = int(line.strip().split()[0])
            NB = int(line.strip().split()[1])
        elif i < 4 + NA:
            new_line = line[:35] + "0  0  0  0  0  0  0  0  0  0  0  0\n"
            new_lines.append(new_line)
        elif i < 4 + NA + NB:
            new_line = line[:8] + "1  0  0  0  0\n"
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    with open(sdf_fn, 'w') as w:
        w.writelines(new_lines)
    return

def xyz_to_sdf(xyz_fn, sdf_fn, smi=None):
    os.system(f"obabel {xyz_fn} -O {sdf_fn} 2> /dev/null")
    if smi is not None:
        process_sdf(sdf_fn)
        temp = Chem.MolFromSmiles(smi)
        ref = Chem.SDMolSupplier(sdf_fn)[0]
        for a in ref.GetAtoms():
            a.SetNumExplicitHs(0)
            a.SetNumRadicalElectrons(0)
        ref = AllChem.AssignBondOrdersFromTemplate(temp, ref)
        Chem.AssignAtomChiralTagsFromStructure(ref)
        Chem.SanitizeMol(ref)
        #print(Chem.MolToSmiles(ref))
        writer = Chem.SDWriter(sdf_fn)
        writer.write(ref)
        writer.close()


def xyz_to_pdb(xyz_fn, pdb_fn):
    os.system(f"obabel {xyz_fn} -O {pdb_fn} 2> /dev/null")

def data2pdb(
        data_fn, 
        ligand_xyz_fn,
        ligand_pdb_fn,
        scaff_xyz_fn,
        scaff_pdb_fn, 
        pocket_xyz_fn,
        pocket_pdb_fn
        ):

    with open(data_fn, 'rb') as f:
        ligand_type, ligand_coord, pocket_type, pocket_coord, ligand_adj, ligand_prop, scaff_prop, \
        ligand_n, scaff_n, ligand_smi, scaff_smi = \
        pickle.load(f)

    write_xyz(ligand_type[:,:9], ligand_coord, fn=ligand_xyz_fn, is_onehot=True)
    write_xyz(ligand_type[:scaff_n,:9], ligand_coord[:scaff_n], fn=scaff_xyz_fn, is_onehot=True)
    write_xyz(pocket_type[:,:9], pocket_coord, fn=pocket_xyz_fn, is_onehot=True)
    xyz_to_sdf(ligand_xyz_fn, ligand_pdb_fn, ligand_smi)
    xyz_to_sdf(scaff_xyz_fn, scaff_pdb_fn, scaff_smi)
    xyz_to_pdb(pocket_xyz_fn, pocket_pdb_fn)
    
    os.unlink(ligand_xyz_fn)
    os.unlink(scaff_xyz_fn)
    os.unlink(pocket_xyz_fn)

    return


if __name__ == "__main__":

    import os
    for k in sorted(os.listdir("../generate_data/")):
        #print(k)
        #if k != "1wn6": 
        #    continue
        try:
            data2pdb(
                    f"../generate_data/{k}",
                    f"./reference/{k}_ligand_ref.xyz",
                    f"./reference/{k}_ligand_ref.sdf",
                    f"./reference/{k}_scaffold_ref.xyz",
                    f"./reference/{k}_scaffold_ref.sdf",
                    f"./reference/{k}_pocket_ref.xyz",
                    f"./reference/{k}_pocket_ref.pdb",
            )
        except:
            print(k, traceback.format_exc())

