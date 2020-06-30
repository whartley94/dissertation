import os
from options.train_options import TrainOptions
import shlex
import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re


def get_logs(slurm_exp):
    slurm_dirs = []
    for i in os.listdir(slurm_log_dir):
        stand = False
        for j in slurm_exp:
            if j in i:
                stand = True
        if '.out' in i and stand:
            slurm_dirs.append(i)
    # slurm_dirs.sort()
    slurm_dirs.sort(key=lambda f: int(re.sub('\D', '', f)))
    # print(slurm_dirs)
    assert len(slurm_dirs) > 0, 'Probs Mistyped Experiment Number'
    return slurm_dirs


def get_epoch_times_variable_length(slurm_dirs):
    time_epoch_pre = []
    for i in slurm_dirs:  # Loop over the relevant out files.
        file = open(os.path.join(slurm_log_dir, i), "r")
        for line in file:  # Loop over the lines of each out file.
            # Find .out files which some epochs finished, i.e containing 'End of epoch'.
            if 'End of epoch' in line:
                # Example: End of epoch 3 / 5 	 Time Taken: 1509 sec --- We want the numbers, 3,5 and 1509 in our array.
                words = []
                for word in line.split():
                    if word.isdigit():  # Get just the numbers we want form the line.
                        words.append(word)
                assert len(words) == 3
                time_epoch_pre.append(words)
    time_epoch_pre = np.asarray(time_epoch_pre).astype(int)
    return time_epoch_pre


def populate_data_arrays():
    name_arr, batch_size_arr, gpu_arr, phase_arr, dict_arrs = [], [], [], [], []

    for slurm_num, i in enumerate(slurm_dirs):  # Loop over .out files.
        file = open(os.path.join(slurm_log_dir, i), "r")
        epoch = 0
        dict_arr = []

        for line in file:

            if 'Running provided' in line:  # Gets the command used to run that particular .out file.
                arg_string = str(line[42:])
                # Parse the command args as if we were actually calling it, for the dictionary.
                opt = TrainOptions().gather_options2(shlex.split(arg_string))

                batch_size_arr.append(opt.batch_size)
                gpu_arr.append(opt.gpu_ids)
                phase_arr.append(opt.phase)

            if 'End of epoch' in line:  # Gets info from lines with the epoch number and how long it took.
                words = []
                for word in line.split():
                    if word.isdigit():
                        words.append(word)
                assert len(words) == 3
                # time_epoch[slurm_num, epoch, :] = words
                epoch += 1

            if 'Experiment Name' in line:  # Gets the experiment name.
                name_arr.append(str(line[18:-1]))

            if '(epoch: ' in line:  # This is getting the desired losses from rows in which it was printed.
                splits = line[1:].split(',')
                splits_dic = {}
                for r in splits[:-2]:
                    res = re.split(': |\) ', r)
                    for m in range(0, len(res), 2):
                        # print(m)
                        splits_dic[res[m]] = res[m + 1]
                dict_arr.append(splits_dic)

        dict_arrs.append(dict_arr)

    batch_size_arr = np.asarray(batch_size_arr)
    gpu_arr = np.asarray(gpu_arr)
    phase_arr = np.asarray(phase_arr)
    gpu_arr = np.asarray([len(gpu_arr[i].split(',')) for i in range(len(gpu_arr))])

    assert len(name_arr) == len(batch_size_arr)

    return name_arr, batch_size_arr, gpu_arr, phase_arr, dict_arrs


def get_final_losses_array(dict_arrs):
    total_dict_arrs = len(dict_arrs)
    g_ce_arr = np.empty(total_dict_arrs)
    g_ce_arr[:] = np.nan
    num_dict_exp = np.empty(total_dict_arrs)
    for i in range(total_dict_arrs):
        len_dicts = len(dict_arrs[i])
        num_dict_exp[i] = len_dicts
        if len_dicts - 1 >= 0:
            if int(dict_arrs[i][len_dicts - 1]['epoch']) == int(max_epochs) - 1:
                g_ce_arr[i] = dict_arrs[i][len_dicts - 1][' G_L1_reg']

    return g_ce_arr, num_dict_exp


def plot_batch_size_vs_time_taken():
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    # ax.set_title(self.name + ' loss over time')
    ax.scatter(batch_size_arr[trains], times[trains], label='Train')
    # ax.scatter(batch_size_arr[train_smalls], times[train_smalls], label='Train Small')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Mean Epoch Time Taken')
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.subplots_adjust(right=0.7)
    plt.show()
    # plt.savefig(self.mpl_name, bbox_inches="tight", dpi=400)
    plt.close(fig)


def zip_plot(gpu_arr_trains, g_ce_arr_trains, ax, bs, bs_num):
    new_x, new_y = zip(*sorted(zip(gpu_arr_trains[bs], g_ce_arr_trains[bs])))
    ax.plot(new_x, new_y, label='Batch Size ' + str(bs_num))


def loss_vs_gpu_plot():
    # Split line out foe each batch size used.
    batch_size_numbers = [10, 25, 32, 64, 128]
    batch_size_lines = []
    for count, bs in enumerate(batch_size_numbers):
        bs = [i == bs for i in batch_size_arr_trains]
        batch_size_lines.append(bs)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    # Plot each line (one per batch size).
    for line, bs in enumerate(batch_size_lines):
        # Zips and sorts so that the number of GPUs is in order.
        zip_plot(gpu_arr_trains, g_ce_arr_trains, ax, bs, batch_size_numbers[line])

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

def plot_loss_dict(times, losses, total_time, loss_dictionary, name, vert_shift=0):
    num_prints = len(loss_dictionary)


    if num_prints == 0:
        print('Nothing to show for ' + str(name) + '!')
    else:
        for i, line in enumerate(loss_dictionary):
            total_time += np.float(line[' time'])
            times.append(total_time)
            losses.append(line[' G_L1_reg'])

    return times, losses, total_time



def plot_batch_gpu_time_taken():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(batch_size_arr[trains], gpu_arr[trains], times[trains])
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Number of GPUs')
    ax.set_zlabel('Mean Time Taken (One Epoch)')
    plt.title('Train Phase')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(batch_size_arr[train_smalls], gpu_arr[train_smalls], times[train_smalls])
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Number of GPUs')
    ax.set_zlabel('Mean Time Taken (One Epoch)')
    plt.title('Train Small Phase')
    plt.show()


def plot_high_gpu_loss_graphs():
    times = []
    losses = []
    total_time = 0
    for i, dict in enumerate(dict_arrs):
        # print(name_arr[i])
        # if 'lsm0' in name_arr[i]:
            # print(i)
            # if i > 8 and i <= 17:
        # if i < 2:
        times, losses, total_time = plot_loss_dict(times, losses, total_time, dict, name_arr[i], vert_shift=0)
    # N = 1200
    # losses = np.convolve(losses, np.ones((N,))/N, mode='valid')
    times = np.asarray(times).astype(float)
    losses = np.asarray(losses).astype(float)
    plt.plot(times, losses)
    # print(len(times))
    # print(len(losses))
    # plt.title(name)
    # plt.show()
    plt.ylabel('Loss')
    plt.xlabel('Time')
    plt.show()


def plot_batch_gpu_ratio():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # normalized1 = (g_ce_arr[trains]-np.nanmin(g_ce_arr[trains]))/(np.nanmax(g_ce_arr[trains])-np.nanmin(g_ce_arr[trains]))
    # normalized2 = (times[trains]-np.nanmin(times[trains]))/(np.nanmax(times[trains])-np.nanmin(times[trains]))

    # normalized3 = 1 / (1.927 - g_ce_arr[trains])
    trains_or_smalls = trains

    normalized3 = g_ce_arr[trains_or_smalls] / np.nanmax(g_ce_arr[trains_or_smalls])
    normalized4 = times[trains_or_smalls] / np.nanmax(times[trains_or_smalls])

    mulp = g_ce_arr[trains_or_smalls] * times[trains_or_smalls]
    mulp2 = normalized3 * normalized4
    # print(mulp)
    ax.scatter(batch_size_arr[trains_or_smalls], gpu_arr[trains_or_smalls], mulp2)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Number of GPUs')
    ax.set_zlabel('\'Ratio\' Loss & Time Taken')
    plt.show()


def plot_batch_gpu_loss():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(batch_size_arr[trains], gpu_arr[trains], g_ce_arr[trains])
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Number of GPUs')
    ax.set_zlabel('Loss After 5 Epochs')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(batch_size_arr[train_smalls], gpu_arr[train_smalls], g_ce_arr[train_smalls])
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Number of GPUs')
    ax.set_zlabel('Loss After 5 Epochs')
    plt.show()


if __name__ == '__main__':

    slurm_log_dir = '/Users/Will/Documents/Uni/MscEdinburgh/Diss/slurm_logs'
    slurm_dirs = get_logs(slurm_exp=['955987','956170', '956271'])
    print(slurm_dirs)

    # Get the epoch times but without worrying about producing a numpy array with fixed shape.
    time_epoch_pre = get_epoch_times_variable_length(slurm_dirs)

    # max_epochs = np.max(time_epoch_pre, axis=0)[1]  # Get max number of epochs completed across any of the .out files.
    # # Initialise a numpy array of fixed shape.
    # time_epoch = np.empty((len(slurm_dirs), max_epochs, 3))
    # time_epoch[:] = np.nan
    time_epoch = np.asarray(time_epoch_pre).astype(float)
    # print(time_epoch)

    # Dict_arrs is a list of lists of dictionaries of the loss printouts for each experiment.
    name_arr, batch_size_arr, gpu_arr, phase_arr, dict_arrs = populate_data_arrays()
    # print(dict_arrs)
    # assert len(name_arr) == len(time_epoch)
    # g_ce_arr, num_dict_exp = get_final_losses_array(dict_arrs)  # Get final loss for each experiment

    # Prepare data for plotting
    phase_num = [len(i) for i in phase_arr]  # Convert phase ('train'/'train_small') to an integer for plotting.
    # print(time_epoch[:, 2])
    # times = np.mean(time_epoch[:, 2])  # Find the average time per epoch for each experiment.
    # fails = [np.isnan(i) for i in times]
    # g_ce_arr[fails] = np.nan  # Zero out any losses for epochs which didn't complete.
    trains = [i == 'train' for i in phase_arr]  # Indexes of .out files with 'train' experiments.
    train_smalls = [i == 'train_small' for i in phase_arr]  # Indexes of .out files with 'train_small' experiments.
    batch_size_arr_trains = batch_size_arr[trains]
    # g_ce_arr_trains = g_ce_arr[trains]
    # gpu_arr_trains = gpu_arr[trains]

# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------

    # DO PLOTTING:
    # Plot Loss Vs Number Of GPUs
    # loss_vs_gpu_plot()

    # Plot The Loss Graph For A Given Out File list of Dictionaries.
    plot_high_gpu_loss_graphs()

    # Plot graph for batch size vs time taken.
    # plot_batch_size_vs_time_taken()

    # 3D Plot graph for batch & GPUs vs time taken.
    # plot_batch_gpu_time_taken()

    # 3D Pot for batch & GPUs vs loss
    # plot_batch_gpu_loss()

    # 3D Pot for batch & GPUs vs ratio of loss and time taken
    # plot_batch_gpu_ratio()

