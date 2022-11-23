import os
import sys
import glob
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from pdb2fasta import pdb2fasta
from cluster import cluster


parser = argparse.ArgumentParser()
parser.add_argument("--pdbbind_data_dir", type=str, \
        default="/home/wonho/work/data/PDBbind_v2020/total-set/")
parser.add_argument("--processed_data_dir", type=str, \
        default=f"/home/wonho/work/DL/gschnet/DeepSLIP/data/data/")
parser.add_argument("--key_dir", type=str, \
        default=f"/home/wonho/work/DL/gschnet/DeepSLIP/data/keys/")
parser.add_argument("--fasta_filename", type=str, \
        default=f"pdbbind_v2020_general_set_processed.fa")
parser.add_argument("--cluster_filename", type=str, \
        default="cluster_output")
parser.add_argument("--affinity_dict_filename", type=str, \
        default="./pdbbind_v2020_total_calc_affinity_dict.pkl")
parser.add_argument("--cutoff", type=float, default=0.6)
parser.add_argument("--num_iter", type=int, default=3)
parser.add_argument("--train_ratio", type=float, default=0.8)
parser.add_argument("--test_ratio", type=float, default=0.1)
parser.add_argument("--valid_ratio", type=float, default=0.1)
args = parser.parse_args()
print(args)

np.testing.assert_almost_equal(args.train_ratio + args.valid_ratio + args.test_ratio, 1.0)


targets = list(glob.glob(args.pdbbind_data_dir + "/????/????_protein.pdb"))
keys = os.listdir(args.processed_data_dir)
num_data = len(keys)
print(f"# Processed data : {num_data}")
targets = [fn for fn in targets if fn.split('/')[-2] in keys]

if not os.path.exists(args.fasta_filename):
    lines = []
    for target in tqdm(targets):
        fasta = pdb2fasta(target)
        lines += fasta
    with open(args.fasta_filename, 'w') as w:
        w.writelines('\n'.join(lines))

if args.cutoff > 0.7: 
    n = 5
elif args.cutoff > 0.6:
    n = 4
elif args.cutoff > 0.5:
    n = 3
elif args.cutoff > 0.4:
    n = 2
else:
    print("Invalid cutoff value")
    exit()

if not os.path.exists(f"{args.cluster_filename}_{args.cutoff}.clstr"):
    os.system(f"cd-hit -i {args.fasta_filename} -o {args.cluster_filename}_{args.cutoff} -c {args.cutoff} -n {n}")
cluster_dict = cluster(f"{args.cluster_filename}_{args.cutoff}.clstr", \
        iter=args.num_iter)
clusters = list(cluster_dict.values())
clusters = sorted(clusters, key=lambda x: len(x), reverse=True)
num_data = sum([len(cls) for cls in clusters])

###
#with open(args.affinity_dict_filename, 'rb') as f:
#    aff_dict = pickle.load(f)
#ligand_key = set(aff_dict["ligand"].keys())
#scaff_key = set(aff_dict["scaff"].keys())
#calc_keys = list(ligand_key & scaff_key)    
###

train_keys, test_keys, valid_keys = [], [], []
while len(train_keys) < num_data * args.train_ratio and len(clusters) > 0:
    clstr = list(clusters.pop(0))
    train_keys += clstr
while len(test_keys) < num_data * args.test_ratio and len(clusters) > 0:
    clstr = list(clusters.pop(0))
    test_keys += clstr
for clstr in clusters:
    valid_keys += list(clstr)

print("\n####################################################################\n")
print("BEFORE FILTERING...")
print(f"# Train data : {len(train_keys)}")
print(f"# Test data : {len(test_keys)}")
print(f"# Validation data : {len(valid_keys)}")
print(f"# Total: {len(train_keys) + len(valid_keys) + len(test_keys)}")

#train_keys = list(filter(lambda x: x in calc_keys, train_keys))
#test_keys = list(filter(lambda x: x in calc_keys, test_keys))
#valid_keys = list(filter(lambda x: x in calc_keys, valid_keys))

if not os.path.exists(args.key_dir):
    os.mkdir(args.key_dir)
with open(f"{args.key_dir}/train_keys.pkl", 'wb') as w: pickle.dump(train_keys, w)
with open(f"{args.key_dir}/test_keys.pkl", 'wb') as w: pickle.dump(test_keys, w)
with open(f"{args.key_dir}/valid_keys.pkl", 'wb') as w: pickle.dump(valid_keys, w)

print("\n####################################################################\n")
print("AFTER FILTERING...")
print(f"# Train data : {len(train_keys)}")
print(f"# Test data : {len(test_keys)}")
print(f"# Validation data : {len(valid_keys)}")
print(f"# Total: {len(train_keys) + len(valid_keys) + len(test_keys)}")

smiles = []
for key in train_keys:

    with open(args.processed_data_dir+"/"+key, 'rb') as f:
        data = pickle.load(f)
    smi = data[-2]
    smiles.append(key + '\t' + smi + '\n')

with open(args.key_dir+"/train_smiles.txt", 'w') as w: 
    w.writelines(smiles)

smiles = []
for key in test_keys:

    with open(args.processed_data_dir+"/"+key, 'rb') as f:
        data = pickle.load(f)
    smi = data[-2]
    smiles.append(key + '\t' + smi + '\n')

with open(args.key_dir+"/test_smiles.txt", 'w') as w: 
    w.writelines(smiles)

smiles = []
for key in valid_keys:

    with open(args.processed_data_dir+"/"+key, 'rb') as f:
        data = pickle.load(f)
    smi = data[-2]
    smiles.append(key + '\t' + smi + '\n')

with open(args.key_dir+"/valid_smiles.txt", 'w') as w: 
    w.writelines(smiles)

