import os
from options.train_options import TrainOptions
import shlex
import numpy as np


slurm_log_dir = '/Users/Will/Documents/Uni/MscEdinburgh/Diss/slurm_logs'

name_arr = []
batch_size_arr = []
gpu_arr = []
time_epoch = []


for i in os.listdir(slurm_log_dir):
    if '.out' in i:
        f = open(os.path.join(slurm_log_dir, i), "r")
        success = False
        for j in f:
            if 'End of epoch' in j:
                success = True
        if success:
            f = open(os.path.join(slurm_log_dir, i), "r")
            # print('Slurm: ', i)
            for k in f:
                if 'Running provided' in k:
                    argString = str(k[42:])
                    opt = TrainOptions().gather_options2(shlex.split(argString))
                    # print(k[42:], end='')
                    # print(opt.batch_size)
                    batch_size_arr.append(opt.batch_size)
                    gpu_arr.append(opt.gpu_ids)
                    # name_arr.append(opt.name)
                # if 'gpu_ids' in k and not 'Running provided' in k and not 'auto_names' in k:
                #     print(k, end='')
                if 'End of epoch' in k:
                    words = []
                    for word in k.split():
                        if word.isdigit():
                            words.append(word)
                    assert len(words) == 3
                    # print(words)
                    time_epoch.append(words)
                if 'Experiment Name' in k:
                    name_arr.append(str(k[18:-1]))

time_epoch = np.asarray(time_epoch)

assert len(name_arr) == len(batch_size_arr)
print(batch_size_arr)
print(name_arr)
print(gpu_arr)
print(time_epoch)