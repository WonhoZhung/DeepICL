import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.nn import radius_graph
from torch_geometric.data import HeteroData, Batch

import rdkit
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.AllChem import GetAdjacencyMatrix, CalcNumRotatableBonds

import glob
import os
import pickle
import random
import traceback
import utils
import numpy as np
from copy import deepcopy
from itertools import combinations
from utils import ATOM_TYPES
from layers import SoftOneHot, HardOneHot


class DataProcessor(object):

    def __init__(
            self,
            args,
            mode='train'
            ):

        self.args = args

        self.atom_type = ATOM_TYPES
        self.num_token = 2
        self.num_dim = 3
        self.radial_cutoff = args.dist_one_hot_param1[1]
        
        self.o_Z = torch.zeros(1, args.num_ligand_atom_feature) # COM
        self.o_R = torch.zeros(1, self.num_dim)
        self.t_Z = torch.zeros(1, args.num_ligand_atom_feature) # termination
        self.t_Z[:, args.num_ligand_atom_feature - 1] = 1. # termination
        self.f_Z = torch.zeros(1, args.num_ligand_atom_feature) # focus
        self.f_Z[:, args.num_ligand_atom_feature - 1] = 1. # focus
        
        self.distance_expand1 = SoftOneHot(*args.dist_one_hot_param1, \
                gamma=args.gamma1)
        self.distance_expand2 = SoftOneHot(*args.dist_one_hot_param2, \
                gamma=args.gamma2, normalize=True)
        self.distance_dummy = torch.zeros(1, args.dist_one_hot_param2[-1])
        
        self.mode = mode

        self.use_scaffold = False

    def get_random_traj(
            self,
            l_dict,
            p_dict,
            p_knn_index,
            ):
        r"""
        """

        avail = l_dict["l_Z"].new_zeros((l_dict["l_n"],)).float()
        adj = l_dict["l_adj"]
        
        if self.use_scaffold:
            s_n = l_dict["s_n"]
            order = list(range(s_n)) # Index of node sequence
            for j in range(s_n):
                avail[j] = 1.
                adj[:, j] = 0
            i = s_n
            l_Z_init = torch.cat([self.o_Z, l_dict["l_Z"][:s_n]], 0)
            l_R_init = torch.cat([self.o_R, l_dict["l_R"][:s_n]], 0)
        else: 
            order = []
            i = 0
            l_Z_init = self.o_Z
            l_R_init = self.o_R

        # Initialize trajectory data
        data_list = []
        while i == 0 or torch.sum(avail) > 0:
            
            if  i == 0:
                f_R = self.o_R
                p_ind = torch.topk(torch.cdist(self.o_R, p_dict["p_R"]), \
                        k=self.args.k, dim=-1, largest=False)[1][0]

            else:
                now = torch.multinomial(avail, 1)[0] # Choose current index within available nodes
                cur = order[now] # Get node index of current index
                f_R = l_R_init[now+1:now+2] # Get coordinate of focus atom
                p_ind = p_knn_index[cur]

            # Add tokens
            l_Z_in = torch.cat([self.f_Z, l_Z_init], 0) 
            l_R_in = torch.cat([f_R, l_R_init], 0)

            # Get surrounding pocket atoms 
            p_Z_in = p_dict["p_Z"][p_ind] 
            p_R_in = p_dict["p_R"][p_ind]
            cond_in = p_dict["p_cond"][p_ind]

            # Make PyG HeteroData
            data = HeteroData()
            data.pocket_prop = cond_in
            data["ligand"].x = l_Z_in 
            data["pocket"].x = p_Z_in
            data["ligand"].pos = l_R_in
            data["pocket"].pos = p_R_in

            l2l_index = radius_graph(l_R_in, r=self.radial_cutoff) # TODO
            p2p_index = radius_graph(p_R_in, r=self.radial_cutoff) # TODO
            l_index = torch.LongTensor(list(range(i+self.num_token))).unsqueeze(1).repeat(1, self.args.k).view(-1) 
            p_index = torch.LongTensor(list(range(self.args.k))).unsqueeze(0).repeat(i+self.num_token, 1).view(-1) 

            data["ligand", "l2l", "ligand"].edge_index = l2l_index
            data["pocket", "p2p", "pocket"].edge_index = p2p_index
            data["pocket", "p2l", "ligand"].edge_index = torch.stack([p_index, l_index], dim=0)

            l2l_weight = torch.norm(data["ligand"].pos[data["l2l"].edge_index[0]] - \
                    data["ligand"].pos[data["l2l"].edge_index[1]], dim=-1)
            p2p_weight = torch.norm(data["pocket"].pos[data["p2p"].edge_index[0]] - \
                    data["pocket"].pos[data["p2p"].edge_index[1]], dim=-1)
            p2l_weight = torch.norm(data["pocket"].pos[p_index] - \
                    data["ligand"].pos[l_index], dim=-1)
            
            l2l_attr = self.distance_expand1(l2l_weight)[0]
            p2p_attr = self.distance_expand1(p2p_weight)[0]
            p2l_attr = self.distance_expand1(p2l_weight)[0]

            data["l2l"].edge_weight = l2l_weight
            data["p2p"].edge_weight = p2p_weight
            data["p2l"].edge_weight = p2l_weight

            data["l2l"].edge_attr = l2l_attr
            data["p2p"].edge_attr = p2p_attr
            data["p2l"].edge_attr = p2l_attr

            if i == 0:
                next = 0 # node closest to CM
            else:
                if torch.sum(adj[cur]) == 0: # No neighbors, dead end
                    avail[now] = 0.
                    data.type_output = self.t_Z # termination
                    data.dist_ll_output = self.distance_dummy.repeat(i+self.num_token, 1)
                    data.dist_lp_output = self.distance_dummy.repeat(self.args.k, 1)
                    data.mask = 0
                    data_list.append(data)
                    continue
            
                # Select next node from neighboring nodes
                next = torch.nonzero(adj[cur] > 0)[0][0] 
                # next = torch.multinomial(adj[cur], 1)[0] # select randomly from the neighboring nodes

            # Get true type and coordinate
            next_Z = l_dict["l_Z"][next:next+1] 
            next_R = l_dict["l_R"][next:next+1]
            
            next_ll_d = self.distance_expand2(torch.cdist(next_R, l_R_in))[0]
            next_lp_d = self.distance_expand2(torch.cdist(next_R, p_R_in))[0]

            data.type_output = next_Z
            data.dist_ll_output = next_ll_d
            data.dist_lp_output = next_lp_d
            data.mask = 1
            data_list.append(data)

            # Update order, adj, avail
            order += [int(next)]
            adj[:, next] = 0
            avail[i] = 1.0
            
            l_Z_init = torch.cat([l_Z_init, next_Z], 0) 
            l_R_init = torch.cat([l_R_init, next_R], 0)

            i += 1

        # Combine as a single-batch
        traj = Batch.from_data_list(data_list)

        return traj
        
    def get_input_from_data(
            self,
            data
            ):
        r"""
        """
        
        # Get data generated from data/pdbbind.py
        ligand_type, ligand_coord, pocket_type, pocket_coord, ligand_adj, \
        ligand_prop, scaff_prop, ligand_n, scaff_n, _, _, pocket_prop = data

        # Dimension
        dim = self.num_dim
        
        # Ligand info
        l_Z = torch.Tensor(ligand_type) # whole ligand
        l_R = torch.Tensor(ligand_coord)
        l_n = l_Z.shape[0]
        l_dict = {
                "l_Z": l_Z,
                "l_R": l_R,
                "l_n": l_n,
                "l_adj": torch.Tensor(ligand_adj)
        }

        # Pocket info
        p_Z = torch.Tensor(pocket_type) # pocket
        p_R = torch.Tensor(pocket_coord)
        p_n = p_Z.shape[0]
        pocket_prop = torch.Tensor(pocket_prop)
        p_dict = {
                "p_Z": p_Z,
                "p_R": p_R,
                "p_n": p_n,
                "p_cond": pocket_prop
        }

        if self.use_scaffold:
            l_dict["s_n"] = scaff_n

        # Edge indices
        p_knn_index = torch.topk(torch.cdist(l_R, p_R), k=self.args.k, dim=-1, \
                largest=False)[1]
        p_index = p_knn_index.view(-1)
        l_index = torch.LongTensor(list(range(l_n))).unsqueeze(-1).repeat(1, \
                self.args.k).view(-1)
        pl_index = torch.stack([p_index, l_index], 0)

        # Ligand, Scaffold Properties
        if ligand_prop is not None:
            for k, v in ligand_prop.items():
                ligand_prop[k] = torch.Tensor(v)
        if scaff_prop is not None:
            for k, v in scaff_prop.items():
                scaff_prop[k] = torch.Tensor(v)

        # Whole complex graph
        complex = HeteroData()
        complex["ligand"].x = l_Z
        complex["pocket"].x = p_Z
        complex["ligand"].pos = l_R
        complex["pocket"].pos = p_R

        complex["ligand", "l2l", "ligand"].edge_index = radius_graph(l_R, \
                r=self.radial_cutoff)
        complex["pocket", "p2p", "pocket"].edge_index = radius_graph(p_R, \
                r=self.radial_cutoff)
        complex["pocket", "p2l", "ligand"].edge_index = pl_index

        l2l_weight = torch.norm(complex["ligand"].pos[complex["l2l"].edge_index[0]] - \
                complex["ligand"].pos[complex["l2l"].edge_index[1]], dim=-1)
        p2p_weight = torch.norm(complex["pocket"].pos[complex["p2p"].edge_index[0]] - \
                complex["pocket"].pos[complex["p2p"].edge_index[1]], dim=-1)
        p2l_weight = torch.norm(complex["pocket"].pos[p_index] - \
                complex["ligand"].pos[l_index], dim=-1)

        l2l_attr = self.distance_expand1(l2l_weight)[0]
        p2p_attr = self.distance_expand1(p2p_weight)[0]
        p2l_attr = self.distance_expand1(p2l_weight)[0]
        
        complex["l2l"].edge_weight = l2l_weight
        complex["p2p"].edge_weight = p2p_weight
        complex["p2l"].edge_weight = p2l_weight

        complex["l2l"].edge_attr = l2l_attr
        complex["p2p"].edge_attr = p2p_attr
        complex["p2l"].edge_attr = p2l_attr

        complex.ligand_prop = ligand_prop
        complex.scaff_prop = scaff_prop
        complex.pocket_prop = pocket_prop 
        
        complex = Batch.from_data_list([complex])

        # Trajectory
        traj = self.get_random_traj(
                        l_dict, 
                        p_dict, 
                        p_knn_index, 
        )

        return complex, traj


class MoleculeDataset(Dataset):

    def __init__(
            self,
            args,
            mode='train',
            ):
        super().__init__()

        self.args = args

        self._mode_list = ['train', 'test', 'valid', 'generate']
        assert mode in self._mode_list, f"Wrong mode: {mode}"
        self.mode = mode

        self.data_dir = args.data_dir
        self.key_dir = args.key_dir

    def __len__(
            self
            ):
        raise NotImplementedError

    def __getitem__(
            self
            ):
        raise NotImplementedError



class PDBbindDataset(MoleculeDataset):

    def __init__(
            self, 
            args, 
            mode="train",
            ):
        super().__init__(args, mode)

        self.processor = DataProcessor(args, mode=self.mode)

        if not os.path.exists(self.key_dir+f"{mode}_keys.pkl"):
            print(f"No {mode}_keys.pkl exists!")
            exit()
        with open(self.key_dir+f"{mode}_keys.pkl", 'rb') as f: 
            self.key_list = pickle.load(f)

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, idx):
        
        key = self.key_list[idx]
        try: 
            with open(self.data_dir+key, 'rb') as f: 
                data = pickle.load(f)
            whole, traj = self.processor.get_input_from_data(data)
        except Exception as e: 
            print(traceback.format_exc())
            exit()

        data_dict = {
                "whole": whole,
                "traj": traj,
                "key": key
        }
        return data_dict


if __name__ == "__main__":
   
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.num_ligand_atom_feature = 9
    args.num_pocket_atom_feature = 51
    args.num_hidden_feature = 32
    args.num_dense_layers = 3
    args.num_layers = 4
    args.gamma1 = 5e1
    args.gamma2 = 1e1
    args.dist_one_hot_param1 = [0, 10, 25]
    args.dist_one_hot_param2 = [0, 15, 300]
    args.data_dir = "../data/data/"
    args.key_dir = "../data/tmp_single/keys/"
    args.k = 8
    args.conditional = False
    args.node_coeff = 0.5

    dataset = PDBbindDataset(args, mode="train")
    idx = dataset.key_list.index("10gs")

    data = dataset.__getitem__(idx)["whole"]
    
    from model import Embedding
    from layers import IAI_Layer

    emb = Embedding(args)
    lay = IAI_Layer(args)

    
    data["ligand"].h = emb.l_node_emb(data["ligand"].x)
    data["pocket"].h = emb.p_node_emb(data["pocket"].x)

    lay(data)
