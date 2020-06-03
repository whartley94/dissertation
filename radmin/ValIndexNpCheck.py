import numpy as np
from pathlib import Path
import os

d = Path().resolve().parent
n_dir = './resources/ilsvrclin12_val_inds.npy'
n_rel = os.path.join(d,n_dir)
val_inds = np.load(n_rel)

for i in val_inds:
	print(i)