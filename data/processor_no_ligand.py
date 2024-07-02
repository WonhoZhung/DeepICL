import glob
import os
import random
import tempfile
import traceback

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

RDLogger.DisableLog("rdApp.*")

import pickle

from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.PDBIO import Select
from scipy.spatial import distance_matrix


DATA_DIR = "" ## directory to PDBbind v2020 general set 
SAVE_DIR = "" ## directory where processed data will be saved

ATOM_TYPES = ["C", "N", "O", "F", "P", "S", "Cl", "Br"]
AA_TYPES = [
    "GLY", "ALA", "VAL", "LEU", "ILE",
    "PHE", "PRO", "MET", "TRP", "SER",
    "THR", "TYR", "CYS", "ARG", "HIS",
    "LYS", "ASN", "ASP", "GLN", "GLU"
]
INTERACTION_TYPES = ["pipi", "anion", "cation", "hbd", "hba", "hydro"]
HYDROPHOBICS = ["F", "CL", "BR", "I"]
HBOND_DONOR_SMARTS = ["[!#6;!H0]"]
HBOND_ACCEPTOR_SMARTS = [
    "[$([!#6;+0]);!$([F,Cl,Br,I]);!$([o,s,nX3]);!$([Nv5,Pv5,Sv4,Sv6])]"
]
SALT_ANION_SMARTS = ["[O;$([OH0-,OH][CX3](=[OX1])),$([OX1]=[CX3]([OH0-,OH]))]"]
SALT_CATION_SMARTS = [
    "[N;$([NX3H2,NX4H3+;!$(NC=[!#6]);!$(NC#[!#6])][#6])]",
    "[#7;$([NH2X3][CH0X3](=[NH2X3+,NHX2+0])[NHX3]),$([NH2X3+,NHX2+0]=[CH0X3]([NH2X3])[NHX3])]",
    "[#7;$([$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]1:[#6X3H]:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]:[#6X3]1),$([$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]1:[#6X3H]:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3]:[#6X3H]1)]",
]
DEGREES = [0, 1, 2, 3, 4]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
FORMALCHARGES = [-2, -1, 0, 1, 2, 3]
SYMBOL_TO_MASS = {
    "C": 12.0,
    "N": 14.0,
    "O": 16.0,
    "F": 19.0,
    "P": 31.0,
    "S": 32.1,
    "Cl": 35.5,
    "Br": 80.0,
}
MAX_ATOM_NUM = 50
MAX_ADD_ATOM_NUM = 30
SEED = 0

FORMAT = ["sdf", "mol2", "pdb", "xyz"]


def write_xyz(types, coords, msg="", fn=None, is_onehot=True):
    xyz = ""
    xyz += f"{coords.shape[0]}\n"
    xyz += msg + "\n"
    for i in range(coords.shape[0]):
        if is_onehot:
            atom_type = ATOM_TYPES[np.argmax(types[i])]
        else:
            atom_type = types[i]
        xyz += f"{atom_type}\t{coords[i][0]}\t{coords[i][1]}\t{coords[i][2]}\n"
    if fn is not None:
        with open(fn, "w") as w:
            w.writelines(xyz[:-1])
    return xyz[:-1]


def xyz_to_sdf(xyz_fn, sdf_fn):
    os.system(f"obabel {xyz_fn} -O {sdf_fn}")


def read_file(filename):
    extension = filename.split(".")[-1]
    if extension == "sdf":
        mol = Chem.SDMolSupplier(filename)[0]
    elif extension == "mol2":
        mol = Chem.MolFromMol2File(filename)
    elif extension == "pdb":
        mol = Chem.MolFromPDBFile(filename)
    elif extension == "xyz":
        filename2 = filename[:-4] + ".sdf"
        xyz_to_sdf(filename, filename2)
        mol = Chem.SDMolSupplier(filename2)[0]
    else:
        # print("Wrong file format...")
        return
    if mol is None:
        # print("No mol from file...")
        return
    return mol


def get_properties(mol, key="ligand", pdb_id=None):
    key = key
    pdb_id = pdb_id

    properties = {}

    # 1. Molecular weight
    mw = Descriptors.MolWt(mol)

    # 2. TPSA
    tpsa = Descriptors.TPSA(mol)

    # 3. LogP
    logp = Descriptors.MolLogP(mol)

    # 6. Free SASA
    sasa = calc_free_sasa(mol)

    properties.update(
        {
            "mw": np.array([mw]),
            "tpsa": np.array([tpsa]),
            "logp": np.array([logp]),
            "sasa": np.array([sasa]),
        }
    )

    return properties


def calc_free_sasa(mol):
    """
    Code from
    https://sunhwan.github.io/blog/2021/02/04/RDKit-Protein-Ligand-SASA.html
    """
    from rdkit.Chem import rdFreeSASA

    # compute ligand SASA
    mol_h = Chem.AddHs(mol, addCoords=True)

    # Get Van der Waals radii (angstrom)
    ptable = Chem.GetPeriodicTable()
    radii = [ptable.GetRvdw(atom.GetAtomicNum()) for atom in mol_h.GetAtoms()]

    # Compute solvent accessible surface area
    sasa = rdFreeSASA.CalcSASA(mol_h, radii)
    return sasa


class PocketDataProcessor:
    def __init__(
        self,
        data_dir=DATA_DIR,
        save_dir=SAVE_DIR,
        max_atom_num=MAX_ATOM_NUM,  # maximum number of ligand atoms
        max_add_atom_num=MAX_ADD_ATOM_NUM,  # maximum number of ligand atoms to add 
        seed=SEED,
        tmp_dir="/trash/"
    ):
        self.data_dir = data_dir
        self.pocket_data_dir = f"{data_dir}/*.pdb"
        self.save_dir = save_dir
        if self.save_dir[-1] != "/":
            self.save_dir += "/"
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)

        self.max_atom_num = max_atom_num
        self.max_add_atom_num = max_add_atom_num

        self.pocket_data_fns = sorted(glob.glob(self.pocket_data_dir))
        self.num_data = len(self.pocket_data_fns)
        self.keys = [os.path.basename(s).split(".")[0] for s in self.pocket_data_fns]

        self._processed_keys = []
        self._tmp_dir = tmp_dir

    def __len__(
        self,
    ):
        return self.num_data

    def _filter_pocket(self, pocket_mol):
        if pocket_mol is None:
            # print("pocket_mol is None")
            return False
        if not len(pocket_mol.GetConformers()) == 1:
            # print("None or more than one ligand conformer")
            return False
        if not pocket_mol.GetNumAtoms() <= 3000:  # TODO
            return False
        return True

    def _get_one_hot_vector(self, item, item_list, use_unk=True):
        if item not in item_list:
            if use_unk:
                ind = -1
            else:
                print(f"Item not in the list: {item}")
                exit()
        else:
            ind = item_list.index(item)

        if use_unk:
            return list(np.eye(len(item_list) + 1)[ind])
        else:
            return list(np.eye(len(item_list))[ind])

    def _get_pocket_atom_features(self, pocket_mol, pocket_str):
        bio_atom_list = [a for a in pocket_str.get_atoms() if a.element != "H"]
        assert len(bio_atom_list) == pocket_mol.GetNumAtoms(), "ERROR"
        atom_feature_list = []
        for i, atom in enumerate(pocket_mol.GetAtoms()):
            feature = (
                self._get_one_hot_vector(atom.GetSymbol(), ATOM_TYPES)
                + self._get_one_hot_vector(atom.GetDegree(), DEGREES)
                + self._get_one_hot_vector(atom.GetHybridization(), HYBRIDIZATIONS)
                + self._get_one_hot_vector(atom.GetFormalCharge(), FORMALCHARGES)
                + [int(atom.GetIsAromatic())]
                + self._get_one_hot_vector(
                    bio_atom_list[i].get_parent().get_resname(), AA_TYPES
                )
            )
            atom_feature_list.append(feature)
        return np.array(atom_feature_list)

    def _get_complex_interaction_info_with_heuristics(
        self, pocket_mol
    ):
        def get_hydrophobic_atom_indices(mol) -> np.ndarray:
            hydro_indice = []
            natoms = mol.GetNumAtoms()
            for atom_idx in range(natoms):
                atom = mol.GetAtomWithIdx(atom_idx)
                symbol = atom.GetSymbol()
                if symbol.upper() in HYDROPHOBICS:
                    hydro_indice += [atom_idx]
                elif symbol.upper() in ["C"]:
                    neighbors = [x.GetSymbol() for x in atom.GetNeighbors()]
                    neighbors_wo_c = list(set(neighbors) - set(["C"]))
                    if len(neighbors_wo_c) == 0:
                        hydro_indice += [atom_idx]
            hydro_indice = np.array(hydro_indice)
            return hydro_indice

        def get_aromatic_atom_indices(mol) -> np.ndarray:
            aromatic_indice = []
            natoms = mol.GetNumAtoms()
            for atom_idx in range(natoms):
                atom = mol.GetAtomWithIdx(atom_idx)
                if atom.GetIsAromatic():
                    aromatic_indice += [atom_idx]
            aromatic_indice = np.array(aromatic_indice)
            return aromatic_indice

        def get_hbd_atom_indices(mol, smarts_list=HBOND_DONOR_SMARTS) -> np.ndarray:
            hbd_indice = []
            for smarts in smarts_list:
                smarts = Chem.MolFromSmarts(smarts)
                hbd_indice += [idx[0] for idx in mol.GetSubstructMatches(smarts)]
            hbd_indice = np.array(hbd_indice)
            return hbd_indice

        def get_hba_atom_indices(mol, smarts_list=HBOND_ACCEPTOR_SMARTS) -> np.ndarray:
            hba_indice = []
            for smarts in smarts_list:
                smarts = Chem.MolFromSmarts(smarts)
                hba_indice += [idx[0] for idx in mol.GetSubstructMatches(smarts)]
            hba_indice = np.array(hba_indice)
            return hba_indice

        def get_anion_atom_indices(mol, smarts_list=SALT_ANION_SMARTS) -> np.ndarray:
            anion_indice = []
            for smarts in smarts_list:
                smarts = Chem.MolFromSmarts(smarts)
                for indices in mol.GetSubstructMatches(smarts):
                    for idx in indices:
                        atom = mol.GetAtomWithIdx(idx)
                        if atom.GetSymbol().upper() != "C":
                            anion_indice += [idx]
            anion_indice = np.array(anion_indice)
            return anion_indice

        def get_cation_atom_indices(mol, smarts_list=SALT_CATION_SMARTS) -> np.ndarray:
            cation_indice = []
            for smarts in smarts_list:
                smarts = Chem.MolFromSmarts(smarts)
                for indices in mol.GetSubstructMatches(smarts):
                    for idx in indices:
                        atom = mol.GetAtomWithIdx(idx)
                        if atom.GetSymbol().upper() != "C":
                            cation_indice += [idx]
            cation_indice = np.array(cation_indice)
            return cation_indice

        anion_indice = get_anion_atom_indices(pocket_mol)
        cation_indice = get_cation_atom_indices(pocket_mol)
        hbd_indice = get_hbd_atom_indices(pocket_mol)
        hba_indice = get_hba_atom_indices(pocket_mol)
        hydro_indice = get_hydrophobic_atom_indices(pocket_mol)
        aromatic_indice = get_aromatic_atom_indices(pocket_mol)
        return (
            anion_indice,
            cation_indice,
            hbd_indice,
            hba_indice,
            hydro_indice,
            aromatic_indice,
            None,
        )

    def _get_pocket_interaction_matrix(self, ligand_n, pocket_n, info):
        anion, cation, hbd, hba, hydro, pipi, _ = info
        pocket_intr_vectors = []
        for i in range(pocket_n):
            if i + ligand_n in pipi:
                vec = self._get_one_hot_vector("pipi", INTERACTION_TYPES)
            elif i + ligand_n in anion:
                vec = self._get_one_hot_vector("anion", INTERACTION_TYPES)
            elif i + ligand_n in cation:
                vec = self._get_one_hot_vector("cation", INTERACTION_TYPES)
            elif i + ligand_n in hbd:
                vec = self._get_one_hot_vector("hbd", INTERACTION_TYPES)
            elif i + ligand_n in hba:
                vec = self._get_one_hot_vector("hba", INTERACTION_TYPES)
            elif i + ligand_n in hydro:
                vec = self._get_one_hot_vector("hydro", INTERACTION_TYPES)
            else:
                vec = self._get_one_hot_vector("none", INTERACTION_TYPES)
            pocket_intr_vectors.append(vec)
        pocket_intr_mat = np.stack(pocket_intr_vectors, axis=0)
        #if mask is not None:
        #    pocket_intr_mat = pocket_intr_mat * mask.reshape(-1, 1)
        return pocket_intr_mat

    def _unlink_files(self, *files):
        for path in files:
            os.unlink(path)

    def _processor(self, pocket_fn, pocket_center=[0.0, 0.0, 0.0]):
        """
        Main part of the data processing
        """
        try:
            # Read both ligand and pocket file, assert both file is valid
            pocket_mol = Chem.MolFromPDBFile(pocket_fn)
            parser = PDBParser()
            pocket_str = parser.get_structure("pocket", pocket_fn)
            assert pocket_mol is not None  # "pocket mol is None" # RDKit Mol object
            assert (
                pocket_str is not None
            )  # "pocket str is None" # BioPython Structure object

            pocket_mol = Chem.RemoveHs(pocket_mol)
            assert self._filter_pocket(pocket_mol)  # , pocket_fn.split('/')[-1]

        except Exception as e:
            print(traceback.format_exc())
            return

        pocket_n = pocket_mol.GetNumAtoms()
        
        #interaction_info = self._get_complex_interaction_info(complex_fn)
        interaction_info2 = self._get_complex_interaction_info_with_heuristics(pocket_mol) # interactable
        #pocket_cond = self._get_pocket_interaction_matrix(
        #    ligand_n, pocket_n, interaction_info
        #)
        pocket_cond2 = self._get_pocket_interaction_matrix(
            0, pocket_n, interaction_info2
        ) # interactable

        # Get conformer of each molecule
        pocket_coord = pocket_mol.GetConformer(0).GetPositions()

        # Get pocket atom feature
        pocket_type = self._get_pocket_atom_features(pocket_mol, pocket_str)

        # Let pocket center as origin (0, 0, 0)
        center_of_mass = np.array(pocket_center)
        pocket_coord = pocket_coord - center_of_mass

        return (
            None,
            None,
            pocket_type,
            pocket_coord,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            pocket_cond2,
            center_of_mass
        )

    def run(self, idx):
        data = self._processor(self.pocket_data_fns[idx])
        if data is None:
            print(self.keys[idx], flush=True)
            return
        with open(self.save_dir + self.keys[idx], "wb") as w:
            pickle.dump(data, w)
        return


if __name__ == "__main__":

    pass