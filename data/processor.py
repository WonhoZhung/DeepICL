import os
import sys
import random
import glob
import tempfile
import traceback
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from scipy.spatial import distance_matrix
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.PDBIO import Select
from plip.structure.preparation import PDBComplex

import pickle


DATA_DIR = "/home/wonho/work/data/PDBbind_v2020/total-set/"
SAVE_DIR = "/home/wonho/work/DL/gschnet/DeepSLIP/data/train/"

ATOM_TYPES = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br']
AA_TYPES = ["GLY", "ALA", "VAL", "LEU", "ILE", "PHE", "PRO", "MET", "TRP", \
            "SER", "THR", "TYR", "CYS", "ARG", "HIS", "LYS", "ASN", "ASP", \
            "GLN", "GLU"]
INTERACTION_TYPES = ["pipi", "salt", "hbond", "hydro"]
DEGREES = [0, 1, 2, 3, 4]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2
]
FORMALCHARGES = [-2, -1, 0, 1, 2, 3]
SYMBOL_TO_MASS = {
        'C': 12.,
        'N': 14.,
        'O': 16.,
        'F': 19.,
        'P': 31.,
        'S': 32.1,
        'Cl': 35.5,
        'Br': 80.,
}
MAX_ATOM_NUM = 50
MAX_ADD_ATOM_NUM = 30
SEED = 0

FORMAT = ["sdf", "mol2", "pdb", "xyz"]


def write_xyz(types, coords, msg="", fn=None, is_onehot=True):
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

def xyz_to_sdf(xyz_fn, sdf_fn):
    os.system(f"obabel {xyz_fn} -O {sdf_fn}")

def read_file(filename):
    extension = filename.split('.')[-1]
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
        #print("Wrong file format...")
        return
    if mol is None:
        #print("No mol from file...")
        return
    return mol

def get_properties(
        mol,
        key="ligand",
        pdb_id=None
        ):
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

    # 7. Ring counts
    ring = mol.GetRingInfo().NumRings()

    properties.update(
        {
            'mw': np.array([mw]),
            'tpsa': np.array([tpsa]),
            'logp': np.array([logp]),
            'sasa': np.array([sasa]),
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

def calc_bound_sasa(ligand_mol, pocket_mol):
    # TODO
    pass


class PDBbindDataProcessor():

    def __init__(
            self,
            data_dir=DATA_DIR,
            save_dir=SAVE_DIR,
            max_atom_num=MAX_ATOM_NUM, # maximum number of ligand atoms
            max_add_atom_num=MAX_ADD_ATOM_NUM, # maximum number of ligand atoms to add from scaffold
            seed=SEED,
            use_whole_protein=False,
            predefined_scaffold=None
            ):

        self.data_dir = data_dir
        self.ligand_data_dir = f"{data_dir}/????/????_ligand.sdf"
        self.pocket_data_dir = f"{data_dir}/????/????_protein.pdb"
        self.save_dir = save_dir
        self.use_whole_protein = use_whole_protein
        self.predefined_scaffold = predefined_scaffold
        if self.save_dir[-1] !=  '/':
            self.save_dir += '/'
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)

        self.max_atom_num = max_atom_num
        self.max_add_atom_num = max_add_atom_num

        self.ligand_data_fns = sorted(glob.glob(self.ligand_data_dir))
        self.pocket_data_fns = sorted(glob.glob(self.pocket_data_dir))
        assert len(self.ligand_data_fns) == len(self.pocket_data_fns), \
                "Different ligand and pocket number"
        self.num_data = len(self.ligand_data_fns)
        self.keys = [s.split('/')[-2] for s in self.ligand_data_fns]

        self._processed_keys = []


    def __len__(
            self,
            ):
        return self.num_data

    def _filter_ligand(
            self,
            ligand_mol
            ):
        if ligand_mol is None:
            #print("ligand_mol is None")
            return False
        if not all([(a.GetSymbol() in ATOM_TYPES) for a in ligand_mol.GetAtoms()]):
            #print("Invalid atom type in ligand")
            return False
        if not ligand_mol.GetNumAtoms() <= self.max_atom_num:
            #print("Exceed max ligand atom num")
            return False
        if not len(ligand_mol.GetConformers()) == 1:
            #print("None or more than one ligand conformer")
            return False
        return True

    def _filter_pocket(
            self,
            pocket_mol
            ):
        if pocket_mol is None:
            #print("pocket_mol is None")
            return False
        if not len(pocket_mol.GetConformers()) == 1:
            #print("None or more than one ligand conformer")
            return False
        return True

    def _find_correct_index_match(
            self, 
            ref_coord, 
            scaff_coord, 
            index_matches, 
            eps=1e-6
            ):
        for im in index_matches:
            idx = np.asarray(im, dtype=np.compat.long)
            matched_coord = ref_coord[idx]
            rmsd = np.sqrt(np.sum(np.power(matched_coord - scaff_coord, 2)))
            if rmsd < eps: return idx
        return

    def _find_full_occupied_scaffold_atoms(
            self,
            scaff_mol
            ):
        full_occupied = []
        for idx, atom in enumerate(scaff_mol.GetAtoms()):
            atom_symbol = atom.GetSymbol()
            # symbol must be in ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br']
            if atom_symbol == 'F':
                full_occupied.append(idx)
            elif atom_symbol in ['C', 'N', 'O']:
                if atom.GetNumExplicitHs() + atom.GetNumImplicitHs() == 0:
                    full_occupied.append(idx)
            elif atom_symbol in ['P', 'S', 'Cl', 'Br']:
                continue
        full_occupied = np.asarray(full_occupied, dtype=np.compat.long)
        return full_occupied

    def _get_scaffold_from_index(
            self,
            ligand_mol,
            indices
            ):
        from rdkit.Geometry.rdGeometry import Point3D
        new_mol = Chem.RWMol(Chem.Mol())
        new_conf = Chem.Conformer(len(indices))
        atom_map = {}
        for idx in indices:
            atom = ligand_mol.GetAtomWithIdx(idx)
            atom_map[idx] = new_mol.AddAtom(atom)
            atom_pos = Point3D(*ligand_mol.GetConformer(0).GetPositions()[idx])
            new_conf.SetAtomPosition(atom_map[idx], atom_pos)

        indices = set(indices)
        for idx in indices:
            a = ligand_mol.GetAtomWithIdx(idx)
            for b in a.GetNeighbors():
                if b.GetIdx() not in indices: continue
                bond = ligand_mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx())
                bt = bond.GetBondType()
                if a.GetIdx() < b.GetIdx(): 
                    print(a.GetIdx(), b.GetIdx(), bt)
                    new_mol.AddBond(atom_map[a.GetIdx()], atom_map[b.GetIdx()], bt)

        scaff_mol = new_mol.GetMol()
        conf = scaff_mol.AddConformer(new_conf, assignId=True)
        return scaff_mol

    def _get_one_hot_vector(
            self,
            item,
            item_list,
            use_unk=True
            ):
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
    
    def _get_ligand_atom_features(self, ligand_mol):
        atom_feature_list = []
        for atom in ligand_mol.GetAtoms():
            feature = self._get_one_hot_vector(atom.GetSymbol(), ATOM_TYPES)
            atom_feature_list.append(feature)
        return np.array(atom_feature_list)

    def _get_pocket_atom_features(self, pocket_mol, pocket_str):
        bio_atom_list = [a for a in pocket_str.get_atoms() if a.element != 'H']
        assert len(bio_atom_list) == pocket_mol.GetNumAtoms(), "ERROR"
        atom_feature_list = []
        for i, atom in enumerate(pocket_mol.GetAtoms()):
            feature = self._get_one_hot_vector(atom.GetSymbol(), ATOM_TYPES) + \
                      self._get_one_hot_vector(atom.GetDegree(), DEGREES) + \
                      self._get_one_hot_vector(atom.GetHybridization(), \
                      HYBRIDIZATIONS) + \
                      self._get_one_hot_vector(atom.GetFormalCharge(), \
                      FORMALCHARGES) + \
                      [int(atom.GetIsAromatic())] + \
                      self._get_one_hot_vector(bio_atom_list[i].get_parent().get_resname(), \
                      AA_TYPES)
            atom_feature_list.append(feature)
        return np.array(atom_feature_list)

    def _join_complex(
            self,
            ligand_fn, 
            pocket_fn, 
            complex_fn=None
            ):
        if complex_fn is None:
            fd, complex_fn = \
                tempfile.mkstemp(suffix=".pdb", prefix="WH_tmp_com_", dir="/home/wonho/trash")
        command = f"obabel {ligand_fn} {pocket_fn} -O {complex_fn} -j -d 2> /dev/null"
        os.system(command)
        with open(complex_fn, 'r') as f:
            lines = f.readlines()
        num_ligand_atom = Chem.SDMolSupplier(ligand_fn)[0].GetNumAtoms()
        new_lines = []
        for i, line in enumerate(lines):
            if i > 1 and i < num_ligand_atom + 2:
                new_line = line[:17] + "LIG" + line[20:25] + "1 " + line[27:] # enforce lig_resname as LIG
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        
        with open(complex_fn, 'w') as f:
            f.writelines(new_lines)
        complex_mol = Chem.MolFromPDBFile(complex_fn)
        return complex_mol, complex_fn

    def _extract_binding_pocket(
            self,
            ligand_mol, 
            protein_pdb, 
            cutoff=5., 
            base_dir="/home/wonho/trash",
            use_whole_protein=False
            ):
        parser = PDBParser()
        if not os.path.exists(protein_pdb) : 
            return
        structure = parser.get_structure("protein", protein_pdb)
        ligand_positions = ligand_mol.GetConformer().GetPositions()

        class NonHeteroSelect(Select):
            def accept_residue(self, residue):
                if residue.get_resname() == "HOH":
                    return 0
                if residue.get_id()[0] != " ":
                    return 0
                else:
                    return 1

        class DistSelect(Select):
            def accept_residue(self, residue):
                if residue.get_resname() == "HOH":
                    return 0
                if residue.get_id()[0] != " ":
                    return 0
                residue_positions = np.array([np.array(list(atom.get_vector())) \
                    for atom in residue.get_atoms() if 'H' not in atom.get_id()])
                min_dis = np.min(distance_matrix(residue_positions, ligand_positions))
                if min_dis < cutoff:
                    return 1
                else:
                    return 0

        io = PDBIO()
        io.set_structure(structure)

        fd, path = tempfile.mkstemp(suffix=".pdb", prefix="WH_tmp_poc_", dir=base_dir)
        if use_whole_protein:
            io.save(path, NonHeteroSelect())
        else:
            io.save(path, DistSelect())
        m2 = Chem.MolFromPDBFile(path)
        structure2 = parser.get_structure("pocket", path)
        os.close(fd)
        return m2, structure2, path

    def _get_complex_interaction_info(
            self,
            complex_fn, 
            get_ligand_idx=False
            ):
        my_mol = PDBComplex()
        my_mol.load_pdb(complex_fn)
        ligs = [":".join([x.hetid, x.chain, str(x.position)]) for x in my_mol.ligands \
                if x.hetid == "LIG"]
        if len(ligs) == 0:
            return
        my_mol.analyze()
        my_interactions = my_mol.interaction_sets[ligs[0]]

        saltbridges = my_interactions.saltbridge_lneg + my_interactions.saltbridge_pneg
        hbonds = my_interactions.hbonds_ldon + my_interactions.hbonds_pdon
        hydrophobics = my_interactions.hydrophobic_contacts
        pipistacks = my_interactions.pistacking

        # 1. salt-bridges
        sb_l_indices = []
        sb_p_indices = []
        for sb in saltbridges:
            if sb.protispos:
                sb_l_indices += [x - 1 for x in sb.negative.atoms_orig_idx]
                sb_p_indices += [x - 1 for x in sb.positive.atoms_orig_idx]
            else:
                sb_l_indices += [x - 1 for x in sb.positive.atoms_orig_idx]
                sb_p_indices += [x - 1 for x in sb.negative.atoms_orig_idx]

        # 2. hydrogen bonds
        hb_l_indices = []
        hb_p_indices = []
        for hbond in hbonds:
            if hbond.protisdon:
                hb_l_indices += [hbond.a_orig_idx - 1]
                hb_p_indices += [hbond.d_orig_idx - 1]
            else:
                hb_l_indices += [hbond.d_orig_idx - 1]
                hb_p_indices += [hbond.a_orig_idx - 1]
        
        # 3. Hydrophobic interactions
        hp_l_indices = []
        hp_p_indices = []
        for hp in hydrophobics:
            hp_l_indices += [hp.ligatom_orig_idx - 1]
            hp_p_indices += [hp.bsatom_orig_idx - 1]
        
        # 4. Pi-Pi Stackings
        pi_l_indices = []
        pi_p_indices = []
        for pi in pipistacks:
            pi_l_indices += [x - 1 for x in pi.ligandring.atoms_orig_idx]
            pi_p_indices += [x - 1 for x in pi.proteinring.atoms_orig_idx]

        sb_l_indices = list(set(sb_l_indices))
        sb_p_indices = list(set(sb_p_indices))
        hb_l_indices = list(set(hb_l_indices))
        hb_p_indices = list(set(hb_p_indices))
        hp_l_indices = list(set(hp_l_indices))
        hp_p_indices = list(set(hp_p_indices))
        pi_l_indices = list(set(pi_l_indices))
        pi_p_indices = list(set(pi_p_indices))

        if get_ligand_idx:
            return sb_p_indices, hb_p_indices, hp_p_indices, pi_p_indices, \
                   sb_l_indices, hb_l_indices, hp_l_indices, pi_l_indices
        return sb_p_indices, hb_p_indices, hp_p_indices, pi_p_indices

    def _get_pocket_interaction_matrix(
            self,
            ligand_n,
            pocket_n,
            indices,
            ):
        salt, hbond, hydro, pipi = indices[:4]
        pocket_intr_vectors = []
        for i in range(pocket_n):
            if i + ligand_n in pipi:
                vec = self._get_one_hot_vector("pipi", INTERACTION_TYPES)
            elif i + ligand_n in salt:
                vec = self._get_one_hot_vector("salt", INTERACTION_TYPES)
            elif i + ligand_n in hbond:
                vec = self._get_one_hot_vector("hbond", INTERACTION_TYPES)
            elif i + ligand_n in hydro:
                vec = self._get_one_hot_vector("hydro", INTERACTION_TYPES)
            else:
                vec = self._get_one_hot_vector("none", INTERACTION_TYPES)
            pocket_intr_vectors.append(vec)
        pocket_intr_mat = np.stack(pocket_intr_vectors, axis=0)
        return pocket_intr_mat

    def _unlink_files(self, *files):
        for path in files:
            os.unlink(path)

    def _processor(
            self,
            ligand_fn,
            pocket_fn
            ):
        """
        Main part of the data processing
        """
        
        try:
            # Read both ligand and pocket file, assert both file is valid
            ligand_mol = read_file(ligand_fn)
            assert ligand_mol is not None #, "ligand mol is None"
            pocket_mol, pocket_str, pocket_fn_2 = \
                    self._extract_binding_pocket(ligand_mol, pocket_fn, \
                            use_whole_protein=self.use_whole_protein)
            assert pocket_mol is not None #, "pocket mol is None" # RDKit Mol object
            assert pocket_str is not None #, "pocket str is None" # BioPython Structure object
        
            ligand_mol = Chem.RemoveHs(ligand_mol)
            pocket_mol = Chem.RemoveHs(pocket_mol)
            
            assert self._filter_ligand(ligand_mol) #, ligand_fn.split('/')[-1]
            assert self._filter_pocket(pocket_mol) #, pocket_fn.split('/')[-1]

            complex_mol, complex_fn = \
                    self._join_complex(ligand_fn, pocket_fn_2)

            assert complex_mol is not None
            
        except Exception as e:
            print(traceback.format_exc())
            return

        ligand_n, pocket_n, complex_n = \
            ligand_mol.GetNumAtoms(), pocket_mol.GetNumAtoms(), complex_mol.GetNumAtoms()
        interaction_info = self._get_complex_interaction_info(complex_fn, False)
        pocket_cond = self._get_pocket_interaction_matrix(ligand_n, pocket_n, interaction_info)

        #self._unlink_files(pocket_fn_2, complex_fn)
            
        # Get conformer of each molecule
        ligand_coord = ligand_mol.GetConformer(0).GetPositions()
        pocket_coord = pocket_mol.GetConformer(0).GetPositions()
        
        # Get ligand and scaffold atom feature
        ligand_type = self._get_ligand_atom_features(ligand_mol)
        
        # Get pocket atom feature
        pocket_type = self._get_pocket_atom_features(pocket_mol, pocket_str)

        # Get adjacency matrices of ligand
        ligand_adj = AllChem.GetAdjacencyMatrix(ligand_mol)
        
        scaff_tag = True
        try:
            if self.predefined_scaffold is not None:
                scaff_ref = Chem.MolFromSmiles(self.predefined_scaffold)
                scaff_idx = ligand_mol.GetSubstructMatch(scaff_ref)
                scaff_mol = self._get_scaffold_from_index(ligand_mol, scaff_idx)
            else:
                scaff_mol = GetScaffoldForMol(ligand_mol) # Get Murcko-Scaffold
            assert scaff_mol is not None, "no scaffold mol"
            
            scaff_coord = scaff_mol.GetConformer(0).GetPositions()
            
            # Find index matching between ligand and scaffold
            index_matches = ligand_mol.GetSubstructMatches(scaff_mol)
            im = self._find_correct_index_match(ligand_coord, scaff_coord, index_matches)
            if im is None or len(im) == 0:
                #print("Failed to find the correct index matching")
                raise AssertionError

            # Check atoms to be added from scaffold to ligand
            other = []
            for i in range(ligand_mol.GetNumAtoms()):
                if i not in im: other.append(i)
            other = np.asarray(other, dtype=np.compat.long)
            if len(other) == 0:
                #print("No more atom needed to be added on scaffold")
                raise AssertionError
            elif len(other) > self.max_add_atom_num:
                #print("Too many atoms to be added on scaffold")
                raise AssertionError
        except:
            scaff_tag = False

        # Get COM of ligand and let COM as origin (0, 0, 0)
        mass = np.array([SYMBOL_TO_MASS[s.GetSymbol()] for s in ligand_mol.GetAtoms()])
        center_of_mass = np.average(ligand_coord, axis=0, weights=mass)
        ligand_coord = ligand_coord - center_of_mass
        pocket_coord = pocket_coord - center_of_mass

        if scaff_tag:

            scaff_coord = ligand_coord[im] # coordination of scaffold 
            other_coord = ligand_coord[other] # coordination of atoms which are not included in scaffold

            # Sort other atoms based on the distance with COM
            scaff_com_dist = \
                np.power(np.sum(np.power(scaff_coord, 2), axis=-1), 0.5)
            scaff_order = im[np.argsort(scaff_com_dist)]
            other_com_dist = \
                np.power(np.sum(np.power(other_coord, 2), axis=-1), 0.5)
            other_order = other[np.argsort(other_com_dist)]

            ligand_order = np.concatenate([scaff_order, other_order])
            ligand_n = ligand_mol.GetNumAtoms()
            scaff_n = scaff_mol.GetNumAtoms()

            # Make index [scaffold atoms, rest of atoms] so that ligand[:scaff_n] = scaff
            ligand_type = ligand_type[ligand_order]
            ligand_coord = ligand_coord[ligand_order]
            ligand_adj = ligand_adj[:, ligand_order][ligand_order, :]

            # Get properties
            pdb_id = ligand_fn.split('/')[-1][:4]
            #ligand_prop = get_properties(ligand_mol, "ligand", pdb_id)
            #scaff_prop = get_properties(scaff_mol, "scaff", pdb_id)
            
            ligand_smi = Chem.CanonSmiles(Chem.MolToSmiles(ligand_mol))
            scaff_smi = Chem.CanonSmiles(Chem.MolToSmiles(scaff_mol))

            return ligand_type, ligand_coord, pocket_type, pocket_coord, ligand_adj, \
                   None, None, ligand_n, scaff_n, ligand_smi, scaff_smi, \
                   pocket_cond

        else:

            # Sort other atoms based on the distance with COM
            com_dist = \
                np.power(np.sum(np.power(ligand_coord, 2), axis=-1), 0.5)
            ligand_order = np.argsort(com_dist)

            ligand_n = ligand_mol.GetNumAtoms()

            ligand_type = ligand_type[ligand_order]
            ligand_coord = ligand_coord[ligand_order]
            ligand_adj = ligand_adj[:, ligand_order][ligand_order, :]

            # Get properties
            pdb_id = ligand_fn.split('/')[-1][:4]
            #ligand_prop = get_properties(ligand_mol, "ligand", pdb_id)
            
            ligand_smi = Chem.CanonSmiles(Chem.MolToSmiles(ligand_mol))

            return ligand_type, ligand_coord, pocket_type, pocket_coord, ligand_adj, \
                   None, None, ligand_n, 0, ligand_smi, None, pocket_cond

    def run(
            self,
            idx
            ):
        #print(self.keys[idx], flush=True)
        data = self._processor(
                self.ligand_data_fns[idx], 
                self.pocket_data_fns[idx]
        )
        if data is None:
            print(self.keys[idx], flush=True)
            return
        with open(self.save_dir + self.keys[idx], 'wb') as w:
            pickle.dump(data, w)
        return


if __name__ == "__main__":
    import torch

    SAVE_DIR = "./data/"
    DATA_DIR = "/home/wonho/work/data/PDBbind_v2020/total-set/"

    processor = PDBbindDataProcessor(DATA_DIR, SAVE_DIR, use_whole_protein=False)
    idx = processor.keys.index("1dis")

    """
    ligand_fn = processor.ligand_data_fns[idx]
    pocket_fn = processor.pocket_data_fns[idx]

    ligand_mol = read_file(ligand_fn)
    pocket_mol, pocket_str, pocket_fn_2 = \
            processor._extract_binding_pocket(ligand_mol, pocket_fn, \
                    use_whole_protein=True)

    complex_mol, complex_fn = \
            processor._join_complex(ligand_fn, pocket_fn_2)

    print(complex_fn)

    ligand_n, pocket_n, complex_n = \
        ligand_mol.GetNumAtoms(), pocket_mol.GetNumAtoms(), complex_mol.GetNumAtoms()
    interaction_info = processor._get_complex_interaction_info(complex_fn, False)
    pocket_cond = processor._get_pocket_interaction_matrix(ligand_n, pocket_n, interaction_info)
    pocket_cond = torch.Tensor(pocket_cond)

    print(interaction_info)
    print(pocket_cond.shape)
    print(torch.nonzero(pocket_cond[:, :-1]))
    """

    processor.run(idx)
