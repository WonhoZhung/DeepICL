import torch
import torch.nn as nn

import os
import subprocess
import math
import copy
import numpy as np

from rdkit import Chem


ATOM_TYPES = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br']
ATOM_NUM = [6, 7, 8, 9, 15, 16, 17, 35]
ATOM_MASS = [12., 14., 16., 19., 31., 32.1, 35.5, 79.9]

NUM_LIGAND_ATOM_TYPES = len(ATOM_TYPES) + 1
NUM_POCKET_ATOM_TYPES = 51
NUM_INTERACTION_TYPES = 5

PERIODIC_TABLE = """                                                       
H,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,HE
LI,BE,1,1,1,1,1,1,1,1,1,1,B,C,N,O,F,NE
NA,MG,1,1,1,1,1,1,1,1,1,1,AL,SI,P,S,CL,AR
K,CA,SC,TI,V,CR,MN,FE,CO,NI,CU,ZN,GA,GE,AS,SE,BR,KR
RB,SR,Y,ZR,NB,MO,TC,RU,RH,PD,AG,CD,IN,SN,SB,TE,I,XE
CS,BA,LU,HF,TA,W,RE,OS,IR,PT,AU,HG,TL,PB,BI,PO,AT,RN
"""


# settings for using GPU
def find_free_port():
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def get_abs_path(path):
    return os.path.expanduser(path)

def dic_to_device(dic, device):
    for dic_key, dic_value in dic.items():
        if isinstance(dic_value, torch.Tensor):
            dic_value = dic_value.to(device)
            dic[dic_key] = dic_value
    
    return dic
        
def get_cuda_visible_devices(num_gpus: int) -> str:
    """Get available GPU IDs as a str (e.g., '0,1,2')"""
    max_num_gpus = 16
    idle_gpus = []
    
    if num_gpus:
        for i in range(max_num_gpus):
            cmd = ["nvidia-smi", "-i", str(i)]

            import sys
            major, minor = sys.version_info[0], sys.version_info[1]
            if major == 3 and minor > 6:
                proc = subprocess.run(cmd, capture_output=True, text=True)  # after python 3.7
            if major == 3 and minor <= 6:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, universal_newlines=True)  # for python 3.6
         
            if "No devices were found" in proc.stdout:
                break
         
            if "No running" in proc.stdout:
                idle_gpus.append(i)
         
            if len(idle_gpus) >= num_gpus:
                break
     
        if len(idle_gpus) < num_gpus:
            msg = "Avaliable GPUs are less than required!"
            msg += f" ({num_gpus} required, {len(idle_gpus)} available)"
            raise RuntimeError(msg)
     
        # Convert to a str to feed to os.environ.
        idle_gpus = ",".join(str(i) for i in idle_gpus[:num_gpus])
    
    else:
        idle_gpus = ""
    
    return idle_gpus

def stat_cuda(msg):
    print("--", msg)
    print("allocated: %dM, max allocated: %dM, cached: %dM, max cached: %dM" %
          (torch.cuda.memory_allocated() / 1024 / 1024,
           torch.cuda.max_memory_allocated() / 1024 / 1024,
           torch.cuda.memory_reserved() / 1024 / 1024,
           torch.cuda.max_memory_reserved() / 1024 / 1024))

def load_save_file(model, use_gpu, save_file=None):
    if save_file is None:
        return model
    else:
        if not use_gpu: # CPU settings
            save_file_dict = torch.load(save_file, map_location='cpu')
            new_save_file_dict = dict()
            for k in save_file_dict:
                new_key = k.replace("module.", "")
                new_save_file_dict[new_key] = save_file_dict[k]
            model.load_state_dict(new_save_file_dict, strict=False)
        else:
            save_file_dict = torch.load(save_file)
            new_save_file_dict = dict()
            for k in save_file_dict:
                new_key = k.replace("module.", "")
                new_save_file_dict[new_key] = save_file_dict[k]
            model.load_state_dict(new_save_file_dict, strict=False)
        return model


def initialize_model(model, use_gpu=True, load_save_file=False):
    if load_save_file:
        if not use_gpu: # CPU settings
            save_file_dict = torch.load(load_save_file, map_location='cpu')
            new_save_file_dict = dict()
            for k in save_file_dict:
                new_key = k.replace("module.", "")
                new_save_file_dict[new_key] = save_file_dict[k]
            model.load_state_dict(new_save_file_dict, strict=False)
        else:
            save_file_dict = torch.load(load_save_file)
            new_save_file_dict = dict()
            for k in save_file_dict:
                new_key = k.replace("module.", "")
                new_save_file_dict[new_key] = save_file_dict[k]
            model.load_state_dict(new_save_file_dict, strict=False)
    else:
        for param in model.parameters():
            if not param.requires_grad: 
                continue
            if param.dim() == 1: # bias
                #nn.init.normal_(param, 0, 1)
                nn.init.zeros_(param)
            else:
                #nn.init.xavier_normal_(param)
                nn.init.xavier_uniform_(param)

    return model

# write result file
def write_result(fn, true_dict, pred_dict):
    lines = []
    for k, v in true_dict.items():
        assert pred_dict.get(k) is not None
        lines.append(f"{k}\t{float(v):.3f}\t{float(pred_dict[k]):.3f}\n")

    with open(fn, 'w') as w: w.writelines(lines)
    return

# utility functions for post-processing generated data
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

def xyz_to_sdf(xyz_fn, sdf_fn):
    os.system(f"obabel {xyz_fn} -O {sdf_fn}")

def xyz_to_pdb(xyz_fn, pdb_fn):
    os.system(f"obabel {xyz_fn} -O {pdb_fn}")

def text_filling(text, char='#', num_char=80):
    if len(text) > num_char:
        return text
    elif len(text) % 2 == 0:
        right = left = (num_char - len(text)) // 2 - 1
        return char*left + f" {text} " + char*right
    else:
        right = (num_char - len(text)) // 2 - 1
        left = (num_char - len(text)) // 2
        return char*left + f" {text} " + char*right


if __name__ == '__main__':

    pass
