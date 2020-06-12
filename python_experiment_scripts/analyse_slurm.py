import os
from options.train_options import TrainOptions
import shlex
import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re

slurm_log_dir = '/Users/Will/Documents/Uni/MscEdinburgh/Diss/slurm_logs'
slurm_dirs = []
slurm_exp = '947669'
for i in os.listdir(slurm_log_dir):
    if '.out' in i and slurm_exp in i:
        slurm_dirs.append(i)

time_epoch_pre = []
count_listdir = 0

for i in slurm_dirs:
        count_listdir += 1
        f = open(os.path.join(slurm_log_dir, i), "r")

        for j in f:
            if 'End of epoch' in j:
                words = []
                for word in j.split():
                    if word.isdigit():
                        words.append(word)
                assert len(words) == 3
                time_epoch_pre.append(words)

time_epoch_pre = np.asarray(time_epoch_pre).astype(int)
max_epochs = np.max(time_epoch_pre, axis=0)[1]
time_epoch = np.empty((count_listdir, max_epochs, 3))
time_epoch[:] = np.nan

name_arr = []
batch_size_arr = []
gpu_arr = []
phase_arr = []
dict_arrs = []

slurm_num = 0
for i in slurm_dirs:
    f = open(os.path.join(slurm_log_dir, i), "r")
    epoch = 0
    dict_arr = []

    for k in f:

        if 'Running provided' in k:
            argString = str(k[42:])
            opt = TrainOptions().gather_options2(shlex.split(argString))

            batch_size_arr.append(opt.batch_size)
            gpu_arr.append(opt.gpu_ids)
            phase_arr.append(opt.phase)

        if 'End of epoch' in k:
            words = []
            for word in k.split():
                if word.isdigit():
                    words.append(word)
            assert len(words) == 3
            time_epoch[slurm_num, epoch, :] = words
            epoch += 1

        if 'Experiment Name' in k:
            name_arr.append(str(k[18:-1]))

        if '(epoch: ' in k:
            splits = k[1:].split(',')
            splits_dic = {}
            for r in splits[:-2]:
                res = re.split(': |\) ', r)
                for m in range(0, len(res), 2):
                    # print(m)
                    splits_dic[res[m]]  = res[m+1]
            dict_arr.append(splits_dic)

    slurm_num += 1
    dict_arrs.append(dict_arr)



total_dict_arrs = len(dict_arrs)
g_ce_arr = np.empty(total_dict_arrs)
g_ce_arr[:] = np.nan
num_dict_exp = np.empty(total_dict_arrs)
for i in range(total_dict_arrs):
    len_dicts = len(dict_arrs[i])
    num_dict_exp[i] = len_dicts
    if len_dicts-1 >= 0:
        if int(dict_arrs[i][len_dicts-1]['epoch']) == int(max_epochs)-1:
            g_ce_arr[i] = dict_arrs[i][len_dicts-1][' G_L1_reg']

    # print(dict_arrs[i][len_dicts])

# print(dict_arrs[12][300])

assert len(name_arr) == len(batch_size_arr)
assert len(name_arr) == len(time_epoch)

batch_size_arr = np.asarray(batch_size_arr)
gpu_arr = np.asarray(gpu_arr)
phase_arr = np.asarray(phase_arr)

time_epoch = np.asarray(time_epoch).astype(float)
gpu_arr = np.asarray([len(gpu_arr[i].split(',')) for i in range(len(gpu_arr))])
phase_num = [len(i) for i in phase_arr]
times = np.mean(time_epoch[:, :, 2], axis=1)

fails = [np.isnan(i) for i in times]
g_ce_arr[fails] = np.nan

trains = [i == 'train' for i in phase_arr]
train_smalls = [i == 'train_small' for i in phase_arr]

batch_size_arr_trains = batch_size_arr[trains]
g_ce_arr_trains = g_ce_arr[trains]
gpu_arr_trains = gpu_arr[trains]

bs10 = [i == 10 for i in batch_size_arr_trains]
bs25 = [i == 25 for i in batch_size_arr_trains]
bs32 = [i == 32 for i in batch_size_arr_trains]
bs64 = [i == 64 for i in batch_size_arr_trains]
bs128 = [i == 128 for i in batch_size_arr_trains]

print(g_ce_arr_trains[bs10])
print(gpu_arr_trains[bs10])

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)

new_x, new_y = zip(*sorted(zip(gpu_arr_trains[bs10], g_ce_arr_trains[bs10])))
ax.plot(new_x, new_y, label='Batch Size 10')
new_x, new_y = zip(*sorted(zip(gpu_arr_trains[bs25], g_ce_arr_trains[bs25])))
ax.plot(new_x, new_y, label='Batch Size 25')
new_x, new_y = zip(*sorted(zip(gpu_arr_trains[bs32], g_ce_arr_trains[bs32])))
ax.plot(new_x, new_y, label='Batch Size 32')
new_x, new_y = zip(*sorted(zip(gpu_arr_trains[bs64], g_ce_arr_trains[bs64])))
ax.plot(new_x, new_y, label='Batch Size 64')
new_x, new_y = zip(*sorted(zip(gpu_arr_trains[bs128], g_ce_arr_trains[bs128])))
ax.plot(new_x, new_y, label='Batch Size 128')
# ax.scatter(gpu_arr[bs25], g_ce_arr[bs25], label='Batch Size 25')
# ax.scatter(gpu_arr[bs32], g_ce_arr[bs32], label='Batch Size 32')
# ax.scatter(gpu_arr[bs64], g_ce_arr[bs64], label='Batch Size 64')
# ax.scatter(gpu_arr[bs128], g_ce_arr[bs128], label='Batch Size 128')
# ax.scatter(gpu_arr[train_smalls], times[train_smalls], label='Train Small')
ax.set_xlabel('Number of GPUs')
ax.set_ylabel('Loss After 5 Epochs')
ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.subplots_adjust(right=0.7)
plt.show()
# plt.savefig(self.mpl_name, bbox_inches="tight", dpi=400)
plt.close(fig)



# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(111)
# # ax.set_title(self.name + ' loss over time')
# ax.scatter(batch_size_arr[trains], times[trains], label='Train')
# # ax.scatter(batch_size_arr[train_smalls], times[train_smalls], label='Train Small')
# ax.set_xlabel('Batch Size')
# ax.set_ylabel('Mean Epoch Time Taken')
# ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
# plt.subplots_adjust(right=0.7)
# plt.show()
# # plt.savefig(self.mpl_name, bbox_inches="tight", dpi=400)
# plt.close(fig)
# #
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(batch_size_arr[trains], gpu_arr[trains], times[trains])
# ax.set_xlabel('Batch Size')
# ax.set_ylabel('Number of GPUs')
# ax.set_zlabel('Mean Time Taken (One Epoch)')
# # plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(batch_size_arr[train_smalls], gpu_arr[train_smalls], times[train_smalls])
# ax.set_xlabel('Batch Size')
# ax.set_ylabel('Number of GPUs')
# ax.set_zlabel('Mean Time Taken (One Epoch)')
# # plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(batch_size_arr[trains], gpu_arr[trains], g_ce_arr[trains])
# ax.set_xlabel('Batch Size')
# ax.set_ylabel('Number of GPUs')
# ax.set_zlabel('Loss After 5 Epochs')
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(batch_size_arr[train_smalls], gpu_arr[train_smalls], g_ce_arr[train_smalls])
# ax.set_xlabel('Batch Size')
# ax.set_ylabel('Number of GPUs')
# ax.set_zlabel('Loss After 5 Epochs')
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # normalized1 = (g_ce_arr[trains]-np.nanmin(g_ce_arr[trains]))/(np.nanmax(g_ce_arr[trains])-np.nanmin(g_ce_arr[trains]))
# # normalized2 = (times[trains]-np.nanmin(times[trains]))/(np.nanmax(times[trains])-np.nanmin(times[trains]))
#
# # normalized3 = 1 / (1.927 - g_ce_arr[trains])
# trains_or_smalls = trains
#
# normalized3 =  g_ce_arr[trains_or_smalls]/np.nanmax(g_ce_arr[trains_or_smalls])
# normalized4 = times[trains_or_smalls]/np.nanmax(times[trains_or_smalls])
#
# mulp = g_ce_arr[trains_or_smalls] * times[trains_or_smalls]
# mulp2 = normalized3 * normalized4
# # print(mulp)
# ax.scatter(batch_size_arr[trains_or_smalls], gpu_arr[trains_or_smalls], mulp2)
# ax.set_xlabel('Batch Size')
# ax.set_ylabel('Number of GPUs')
# ax.set_zlabel('\'Ratio\' Loss & Time Taken')
# plt.show()