import os
import pickle
import numpy as np
from processor import PDBbindDataProcessor
from tqdm import tqdm
    

generate_keys_dir = "./keys/test_keys.pkl"
with open(generate_keys_dir, 'rb') as f:
    generate_keys = pickle.load(f)


preprocessor = PDBbindDataProcessor(
        data_dir="/home/wonho/work/data/PDBbind_v2020/total-set/",
        save_dir="./generate_data_3/",
        use_whole_protein=True
)
for k in tqdm(generate_keys):
    index = preprocessor.keys.index(k)
    tag = preprocessor.run(index)
    if tag:
        print(f"{k} FAILS TO PRE-PROCESS")
        continue

#with open("generate_keys/test_keys.pkl", 'wb') as w:
#    pickle.dump(os.listdir("generate_data/"), w)

