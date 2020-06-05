#!/bin/bash

# slurm info - more at https://slurm.schedmd.com/sbatch.html#lbAJ
echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"


# ===================
# Environment setup
# ===================

echo "Setting up bash enviroment"

# Make available all commands on $PATH as on headnode
source ~/.bashrc

# Make script bail out after first error
set -e


# Make your own folder on the node's scratch disk
# N.B. disk could be at /disk/scratch_big, or /disk/scratch_fast. Check
# yourself using an interactive session, or check the docs:
#     http://computing.help.inf.ed.ac.uk/cluster-computing
echo "Making Scratch Disk"
SCRATCH_DISK=/disk/scratch
SCRATCH_HOME=${SCRATCH_DISK}/${USER}
mkdir -p ${SCRATCH_HOME}
echo "Scratch Disk Created"

# Activate your conda environment
CONDA_ENV_NAME=tideep
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}
echo "Activated conda environment"




# =================================
# Move input data to scratch disk
# =================================
# Move data from a source location, probably on the distributed filesystem
# (DFS), to the scratch space on the selected node. Your code should read and
# write data on the scratch space attached directly to the compute node (i.e.
# not distributed), *not* the DFS. Writing/reading from the DFS is extremely
# slow because the data must stay consistent on *all* nodes. This constraint
# results in much network traffic and waiting time for you!
#
# This example assumes you have a folder containing all your input data on the
# DFS, and it copies all that data  file to the scratch space, and unzips it.
#
# For more guidelines about moving files between the distributed filesystem and
# the scratch space on the nodes, see:
#     http://computing.help.inf.ed.ac.uk/cluster-tips

echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"

# input data directory path on the DFS
src_path=/home/${USER}/datasets/SUN2012/Images

# input data directory path on the scratch disk of the node
dest_path=${SCRATCH_HOME}/datasets/SUN2012/Images
mkdir -p ${dest_path}  # make it if required

# Important notes about rsync:
# * the --compress option is going to compress the data before transfer to send
#   as a stream. THIS IS IMPORTANT - transferring many files is very very slow
# * the final slash at the end of ${src_path}/ is important if you want to send
#   its contents, rather than the directory itself. For example, without a
#   final slash here, we would create an extra directory at the destination:
#       ${SCRATCH_HOME}/project_name/data/input/input
# * for more about the (endless) rsync options, see the docs:
#       https://download.samba.org/pub/rsync/rsync.html

rsync --archive --update --compress --progress ${src_path}/ ${dest_path}
echo "Rsync Completed"

cpoint_path=${SCRATCH_HOME}/checkpoints
mkdir -p ${cpoint_path}  # make it if required

echo "Forming Symlink Datafiles:"
sorted_path=${SCRATCH_HOME}/dataset
echo "OG Dataset Dir: ${dest_path}"
echo "Sorted Dataset Dir: ${sorted_path}"
python -u make_sun12_dataset.py --in_path ${dest_path} --out_path ${sorted_path} > data_progress.out

echo "Starting python call"
python train.py --name FSunTest --sample_p 1.0 --niter 2 --niter_decay 0 --classification --phase train --gpu_ids 0,1 --display_id -1 --data_dir ${sorted_path} --checkpoints_dir ${cpoint_path} --print_freq 1 --display_freq 1 --invisible_network --save_npy --save_mpl
echo "Python ended"

# ======================================
# Move output data from scratch to DFS
# ======================================
# This presumes your command wrote data to some known directory. In this
# example, send it back to the DFS with rsync

echo "Moving output data back to DFS"
#src_path=${SCRATCH_HOME}/project_name/data/output
dest_path=/home/${USER}/git/dissertation/checkpoints
rsync --archive --update --compress --progress ${cpoint_path}/ ${dest_path}
echo "Rsync done"

# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
