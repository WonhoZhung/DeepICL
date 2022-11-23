import argparse
import utils
from math import pi as PI


def train_args_parser():
    parser = argparse.ArgumentParser()                                              

    # DEFAULT SETTINGS
    parser.add_argument("--world_size", help="world size", type=int, default=4)        
    parser.add_argument("--distributed", help="distributed", action="store_true", \
            default=True)
    parser.add_argument("--autocast", help="autocast", action="store_true", \
            default=False)
    parser.add_argument("--num_workers", help="number of workers", type=int, \
            default=1)
    parser.add_argument("--batch_size", help="batch_size", type=int, default=1)   

    # DIRECTORY SETTINGS
    parser.add_argument("--save_dir", help="save directory", type=str)
    parser.add_argument("--data_dir", help="data directory", type=str)
    parser.add_argument("--key_dir", help="key directory", type=str, default=None)

    # DATASET SETTINGS
    parser.add_argument("--k", help="k for k-NN parameter", type=int)
    
    # MODEL SETTINGS
    parser.add_argument("--num_layers", help="num layers", type=int)
    parser.add_argument("--num_dense_layers", help="num dense layers", type=int)
    parser.add_argument("--num_ligand_atom_feature", help="ligand atom features", \
            type=int, default=utils.NUM_LIGAND_ATOM_TYPES)
    parser.add_argument("--num_pocket_atom_feature", help="pocket atom features", \
            type=int, default=utils.NUM_POCKET_ATOM_TYPES)
    parser.add_argument("--num_hidden_feature", help="num hidden features", \
            type=int)
    parser.add_argument("--gamma1", type=float, default=1e1)
    parser.add_argument("--gamma2", type=float, default=5e1)
    parser.add_argument("--dist_one_hot_param1", help="dist. one-hot param for representation", \
            type=int, nargs="+", default=[0, 15, 30])
    parser.add_argument("--dist_one_hot_param2", help="dist. one-hot param for next distance", \
            type=int, nargs="+", default=[0, 10, 200])
    parser.add_argument("--conditional", help="conditional", action="store_true")
    parser.add_argument("--num_cond_feature", help="num condition features", \
            type=int, default=utils.NUM_INTERACTION_TYPES)
    
    # TRAINING SETTINGS
    parser.add_argument("--num_epochs", help="num epochs", type=int)            
    parser.add_argument("--lr", help="lr", type=float, default=2e-3)
    parser.add_argument("--lr_decay", help="lr_decay", type=float, default=0.9)
    parser.add_argument("--lr_tolerance", help="lr_tolerance", type=int, default=5)
    parser.add_argument("--lr_min", help="lr_min", type=float, default=2e-4)
    parser.add_argument("--weight_decay", help="weight_decay", type=float, \
            default=0.0)
    parser.add_argument("--vae_loss_coeff", help="vae coeff for annealing", \
            type=float, nargs="+", default=[0.0, 1.0])            
    parser.add_argument("--vae_loss_beta", help="decaying coeff for vae loss annealing", \
            type=float, default=0.2)            
    parser.add_argument("--restart_file", help="restart_file", type=str, \
            default=None)
    parser.add_argument("--save_every", help="save every n epochs", type=int, \
            default=1)
    
    args = parser.parse_args()
    return args

def generate_args_parser():
    parser = argparse.ArgumentParser()                                              
    
    # DEFAULT SETTINGS
    parser.add_argument("--ngpu", help="ngpu", type=int, default=0)        
    parser.add_argument("--ncpu", help="ncpu", type=int, default=0)        
    parser.add_argument("-y", help="delete", action="store_true")        
    
    # DIRECTORY SETTINGS
    parser.add_argument("--data_dir", help="data directory", type=str)
    parser.add_argument("--key_dir", help="key directory", type=str)
    parser.add_argument("--result_dir", help="result directory", type=str)
    parser.add_argument("--restart_dir", help="restart model directory", type=str)
    
    # MODEL SETTINGS
    parser.add_argument("--num_layers", help="num layers", type=int)
    parser.add_argument("--num_dense_layers", help="num layers", type=int)
    parser.add_argument("--num_ligand_atom_feature", help="ligand atom features", \
            type=int, default=utils.NUM_LIGAND_ATOM_TYPES)
    parser.add_argument("--num_pocket_atom_feature", help="pocket atom features", \
            type=int, default=utils.NUM_POCKET_ATOM_TYPES)
    parser.add_argument("--num_hidden_feature", help="node hidden features", \
            type=int)
    parser.add_argument("--gamma1", type=float, default=1e1)
    parser.add_argument("--gamma2", type=float, default=5e1)
    parser.add_argument("--dist_one_hot_param1", help="dist. one-hot param for representation", \
            type=int, nargs="+", default=[0, 15, 30])
    parser.add_argument("--dist_one_hot_param2", help="dist. one-hot param for next distance", \
            type=int, nargs="+", default=[0, 10, 200])
    parser.add_argument("--use_scaffold", help="use_scaffold", action="store_true")
    parser.add_argument("--conditional", help="conditional", action="store_true")
    parser.add_argument("--num_cond_feature", help="num condition features", \
            type=int, default=utils.NUM_INTERACTION_TYPES)
    
    # GENERATING SETTINGS
    parser.add_argument("--k", help="k for k-NN parameter", type=int)
    parser.add_argument("--max_num_add_atom", type=int, default=30)
    parser.add_argument("--radial_limits", type=float, nargs='+', default=[0.9, 2.2])
    parser.add_argument("--num_sample", help="num samples", type=int, default=30)
    parser.add_argument("--add_noise", help="add noise", action="store_true")
    parser.add_argument("--pocket_coeff_max", type=float, default=100.0)
    parser.add_argument("--pocket_coeff_thr", type=float, default=3.0)
    parser.add_argument("--pocket_coeff_beta", type=float, default=0.44)
    parser.add_argument("--dropout", help="dropout parameter", type=float, \
            default=0.0)
    parser.add_argument("--temperature_factor1", type=float, default=0.0)
    parser.add_argument("--temperature_factor2", type=float, default=0.0)
    parser.add_argument("--translation_coeff", type=float, default=0.2)
    parser.add_argument("--rotation_coeff", type=float, default=PI/90)
    parser.add_argument("--verbose", help="verbal mode", action="store_true")
    
    args = parser.parse_args()
    return args
