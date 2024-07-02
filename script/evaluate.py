from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.RDLogger import DisableLog
DisableLog("rdApp.*")

import argparse
import glob
import pickle


def save_results(mol_list, fn):
    writer = Chem.SDWriter(fn)
    for mol in mol_list:
        writer.write(mol)
    writer.close()


def filter_generated_results(keys, result_dir, train_smiles=None, filter_level=0):
    total_mol_list = []
    val, uni, nov = 0, 0, 0
    N = len(keys)

    for key in keys:
        if key == "":
            gen_fns = glob.glob(f"{result_dir}/*.sdf")
            key = "total"
        else:
            print(f"Key: {key}")
            gen_fns = glob.glob(f"{result_dir}/{key}*.sdf")

        tot_n = len(gen_fns)
        if tot_n == 0:
            print("No generated molecules...")
            continue

        # 1. Check validity
        val_mol_list = []
        val_smi_dict = {}
        for fn in gen_fns:
            name = fn.split("/")[-1].split(".")[0]

            if "val" in fn or "uni" in fn or "nov" in fn:
                tot_n -= 1
                continue

            try:
                mol = Chem.SDMolSupplier(fn)[0]
                assert mol is not None  # Check mol object is generated
                smi = Chem.CanonSmiles(Chem.MolToSmiles(mol))
                assert "." not in smi  # Check mol is not fragmentized
            except Exception as e:
                continue

            for a in mol.GetAtoms():
                a.SetNumExplicitHs(a.GetNumRadicalElectrons())
                a.SetNumRadicalElectrons(0)
            mol = Chem.AddHs(mol, addCoords=True)

            val_mol_list.append(mol)
            val_smi_dict[name] = Chem.CanonSmiles(Chem.MolToSmiles(mol))
        val_n = len(val_smi_dict)

        print("# Generated molecule:", tot_n)
        print("Validity:", f"{100 * val_n / tot_n:.2f}%")
        val += 100 * val_n / tot_n

        if filter_level == 1:
            save_results(val_mol_list, f"{result_dir}/{key}_valid_results.sdf")
            total_mol_list += val_mol_list
            continue

        # 2. Check uniqueness
        uni_mol_list = []
        uni_smi_dict = {}
        val_keys = list(val_smi_dict.keys())
        for i in range(len(val_keys)):
            FLAG = 0
            uni_keys = list(uni_smi_dict.keys())
            for j in range(len(uni_keys)):
                if val_smi_dict[val_keys[i]] == uni_smi_dict[uni_keys[j]]:
                    FLAG = 1
            if not FLAG:  # New molecule
                uni_mol_list.append(val_mol_list[i])
                uni_smi_dict[val_keys[i]] = val_smi_dict[val_keys[i]]
        uni_n = len(uni_smi_dict)

        print("Uniqueness:", f"{100 * uni_n / val_n:.2f}%")
        uni += 100 * uni_n / val_n

        if filter_level == 2:
            save_results(uni_mol_list, f"{result_dir}/{key}_unique_results.sdf")
            total_mol_list += uni_mol_list
            continue

        # 3. Check novelty
        nov_mol_list = []
        nov_smi_dict = {}
        uni_keys = list(uni_smi_dict.keys())
        for i in range(len(uni_keys)):
            FLAG = 0
            for ts in train_smiles:
                if uni_smi_dict[uni_keys[i]] == ts:
                    FLAG = 1
            if not FLAG:
                nov_mol_list.append(uni_mol_list[i])
                nov_smi_dict[uni_keys[i]] = uni_smi_dict[uni_keys[i]]
        nov_n = len(nov_smi_dict)

        print("Novelty:", f"{100 * nov_n / uni_n:.2f}%")
        nov += 100 * nov_n / uni_n

        if filter_level == 3 or filter_level == 0:
            save_results(nov_mol_list, f"{result_dir}/{key}_novel_results.sdf")
            total_mol_list += nov_mol_list
            continue

    val /= N
    uni /= N
    nov /= N

    return total_mol_list, val, uni, nov


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--result_dir", help="directory for generated molecules", type=str)
    parser.add_argument("--key_dir", help="directory for keys", type=str)
    parser.add_argument("--smi_dir", help="directory for training set smiles", type=str)
    parser.add_argument(
        "--filter_level", 
        help="filter level, 0: All, 1: Valid, 2: Unique, 3: Novel", 
        type=int, 
        default=0
    )

    args = parser.parse_args()

    if args.key_dir is None:
        KEYS = [""]
    else:
        with open(args.key_dir, "rb") as f:
            KEYS = pickle.load(f)
        KEYS = sorted(KEYS)

    with open(args.smi_dir, "r") as f:
        SMIS = [Chem.CanonSmiles(s.strip().split()[-1]) for s in f.readlines()]

    tot_mol_list, val, uni, nov = filter_generated_results(
        KEYS, args.result_dir, SMIS, args.filter_level
    )

    print(f"TOTAL VALIDITY: {val:.2f}")
    if args.filter_level > 1 or args.filter_level == 0:
        print(f"TOTAL UNIQUNESS: {uni:.2f}")
    if args.filter_level > 2 or args.filter_level == 0:
        print(f"TOTAL NOVELTY: {nov:.2f}")
