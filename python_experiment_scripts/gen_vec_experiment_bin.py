#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os
import numpy as np
import itertools

# The home dir on the node's scratch disk
USER = os.getenv('USER')
# This may need changing to e.g. /disk/scratch_fast depending on the cluster
SCRATCH_DISK = '/disk/scratch'
SCRATCH_HOME = f'{SCRATCH_DISK}/{USER}'
sorted_path = f'{SCRATCH_HOME}/dataset'
cpoint_path = f'{SCRATCH_HOME}/checkpoints'



# base_call = f"python3 train.py -i {DATA_HOME}/input -o {DATA_HOME}/output"

base_call = f"python train.py --sample_p .125 --niter_decay 0 --display_id -1 " \
    f"--data_dir {sorted_path} --checkpoints_dir {cpoint_path} --phase train_partition " \
    f"--batch_size 58 --gpu_ids 0,1,2,3,4,5,6,7 --lr 0.000001 --weighted_mask --save_optimizer --bin_variation"
# base_call2 = f"python train.py --name Auto --sample_p .125 --niter 20 --niter_decay 0 --lr 0.00001 --display_id -1 " \
#     f"--data_dir {sorted_path} --checkpoints_dir {cpoint_path} --phase train --load_sg_model"

# data_call = f


# train.py --name siggraph_reg2 --sample_p .125 --niter 20 --niter_decay 0 --lr 0.00001 --load_model --phase train

# --gpu_ids 0,1,2,3 --batch_size 25 --phase train_small
# learning_rates = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
# weight_decays = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]

components = ['niter', 'epoch_count']
# niters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
#           12, 13, 14, 15, 16, 17, 18, 19, 20,
#           21, 22, 23, 24, 25, 26, 27, 28, 29,
#           38, 47]
niters = np.arange(1, 16, 1)
# epoch_counts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
epoch_counts = np.arange(0, 15, 1)
flattened_list  = list(zip(niters, epoch_counts))
portion_vec = ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
assert len(flattened_list) == len(portion_vec)
# print(flattened_list)
# batch_sizes = [32, 64, 98, 128]
# gpu_idss = ['0,1,2,3', '0,1,2,3,4,5,6,7']
# num_threads = [0, 4]
# phases = ['train_small', 'train']
# flattened_list = list(itertools.product(niters, epoch_counts))
# print(flattened_list)
# settings = [(lr, wd) for lr in learning_rates for wd in weight_decays]
# settings = [(bs, phase) for bs in batch_sizes for phase in phases]
# settings = [i for i in flattened_list]
# print(settings)

output_file = open("bin_experiments_vec.txt", "w")
output_file2 = open("bin_portions_vec.txt", "w")

for i in flattened_list:
    expt_call = f"{base_call} "
    for j in range(len(i)):
        expt_call += f"--" + components[j] + f" {i[j]} "
    if not i[1] == 0:
        expt_call += f"--load_model"
    else:
        expt_call += f"--load_sg_model"
    print(expt_call, file=output_file)

for i in portion_vec:
    print(i, file=output_file2)
#
# for i in flattened_list:
#     expt_call = f"{base_call2} "
#     for j in range(len(i)):
#         expt_call += f"--" + components[j] + f" {i[j]} "
#     print(expt_call, file=output_file)

# for batch_size, gpu_ids, phase in flattened_list:
#     expt_call = (
#         f"{base_call} "
#         f"--batch_size {batch_size} "
#         f"--phase {phase} "
#         f"--gpu_ids {gpu_ids} "
#     )
#     print(expt_call, file=output_file)

output_file.close()
