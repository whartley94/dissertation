#!/bin/bash

# ====================
# Options for sbatch
# ====================
# Location for stdout log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --output=/home/%u/slurm_logs/slurm-%A_%a.out

# Location for stderr log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --error=/home/%u/slurm_logs/slurm-%A_%a.out

# Maximum number of nodes to use for the job
#SBATCH --nodes=1

# Generic resources to use - typically you'll want gpu:n to get n gpus
#SBATCH --gres=gpu:1

# Megabytes of RAM required. Check `cluster-status` for node configurations
#SBATCH --mem=14000

# Number of CPUs to use. Check `cluster-status` for node configurations
#SBATCH --cpus-per-task=4

# Maximum time for the job to run, format: days-hours:minutes:seconds
# #SBATCH --time=1-08:00:00

# Partition of the cluster to pick nodes from (check `sinfo`)
#SBATCH --partition=PGR-Standard

# Any nodes to exclude from selection
# #SBATCH --exclude=charles[05,12-18]

# Request a node
# #SBATCH --nodelist=damnii07


# =====================
# Logging information
# =====================
MODEL_NAME=wholeinetgt

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

echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"

# input data directory path on the DFS
src_path=/home/${USER}/datasets/INetData/FromHomeVal
#caffe_src=/home/${USER}/git/dissertation/pretrained_models/checkpoints/siggraph_caffemodel
model_src=/home/${USER}/git/dissertation/checkpoints/${MODEL_NAME}
resource_src=/home/${USER}/git/dissertation/resources
mkdir -p ${model_src}  # make it if required

# input data directory path on the scratch disk of the node
dest_path=${SCRATCH_HOME}/datasets/INetData/FromHomeVal
mkdir -p ${dest_path}  # make it if required

rsync --archive --update --compress --progress ${src_path}/ ${dest_path}
echo "Rsync Data Completed"

cpoint_path=${SCRATCH_HOME}/checkpoints/${MODEL_NAME}
cpoint_dir=${SCRATCH_HOME}/checkpoints/
mkdir -p ${cpoint_path}  # make it if required
#rsync --archive --update --compress --progress ${cpoint_path}/ ${model_src}
rsync --archive --update --compress --progress ${model_src}/ ${cpoint_path}
echo "Rsync Models Completed"

resource_path=${SCRATCH_HOME}/resources
mkdir -p ${resource_path}
rsync --archive --update --compress --progress ${resource_src}/ ${resource_path}
echo "Rsync Npz ValIndex Completed"

#code="${dest_path}/"
#for f in ${code}*.tar; do
#  d=`basename "$f" .tar`
#  dpath="${code}TrainFolders/$d"
#  echo "${dpath}"
#  if [[ ! -d "$dpath" ]]; then
#  	mkdir -p "${dpath}"
#  	tar --keep-newer-files -xf "$f" -C "${dpath}"
#  fi
#done

echo "Forming Symlink Datafiles:"
sorted_path=${SCRATCH_HOME}/dataset
echo "OG Dataset Dir: ${dest_path}"
echo "Sorted Dataset Dir: ${sorted_path}"
python make_ilsvrc_dataset_with_val.py --in_path ${dest_path} --out_path ${sorted_path} --resource_path ${resource_path}


# ==============================
# Finally, run the experiment!
# ==============================
echo "Starting python call"
python test_sweep.py --gpu_ids 0 --name ${MODEL_NAME} --data_dir ${sorted_path} --checkpoints_dir ${cpoint_dir} --resources_dir ${resource_path} --weighted_mask --resize_test
echo "Python ended"


# ======================================
# Move output data from scratch to DFS
# ======================================
# This presumes your command wrote data to some known directory. In this
# example, send it back to the DFS with rsync

echo "Moving output data back to DFS"
#src_path=${SCRATCH_HOME}/project_name/data/output
# dest_path=/home/${USER}/git/dissertation/checkpoints
rsync --archive --update --compress --progress ${cpoint_path}/ ${model_src}
echo "Rsync done"

echo "Removing Results From Scratch"
rm -rv ${cpoint_path}
echo "Remove done"


# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
