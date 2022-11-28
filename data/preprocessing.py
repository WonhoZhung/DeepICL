from multiprocessing import Pool
from processor import PDBbindDataProcessor

import pickle
import os
import sys
import time
from tqdm import tqdm


# 1. Initialize data processor
data_dir = sys.argv[1]
save_dir = sys.argv[2]
print(f'save_dir: {save_dir}')
if len(os.listdir(save_dir)) > 0:
    token = input(f"Remove files in {save_dir}?: (y/n)")
    if token == "y":
        os.system(f"rm {save_dir}/*")
        time.sleep(5.0)
        print(len(os.listdir(save_dir)))
    elif token == "n":
        pass
    else:
        print("Wrong input:", token)
        exit()

preprocessor = PDBbindDataProcessor(
        data_dir=data_dir,
        save_dir=save_dir,
        max_atom_num=50,
        max_add_atom_num=30,
        use_whole_protein=False,
        predefined_scaffold=None
)
print("NUM DATA:", preprocessor.num_data)
time.sleep(2.0)


# 2. Run preprocessor
st = time.time()
pool = Pool(int(sys.argv[3]))
r = pool.map_async(preprocessor.run, list(range(preprocessor.num_data)))
r.wait()
pool.close()
pool.join()

# 3. Print the result
print("NUM PROCESSED DATA:", len(os.listdir(preprocessor.save_dir)))
print("PROCESSING TIME:", f"{time.time() - st:.1f}", "(s)")

