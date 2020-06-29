#!/bin/bash
# Author(s): James Owers (james.f.owers@gmail.com)
#
# Template for running an sbatch arrayjob with a file containing a list of
# commands to run. Copy this, remove the .template, and edit as you wish to
# fit your needs.
# 
# Assuming this file has been edited and renamed slurm_arrayjob.sh, here's an
# example usage:
# ```
# EXPT_FILE=experiments.txt  # <- this has a command to run on each line
# NR_EXPTS=`cat ${EXPT_FILE} | wc -l`
# MAX_PARALLEL_JOBS=12 
# sbatch --array=1-${NR_EXPTS}%${MAX_PARALLEL_JOBS} slurm_arrayjob.sh $EXPT_FILE
# ```
#
# or, equivalently and as intended, with provided `run_experiement`:
# ```
# run_experiment -b slurm_arrayjob.sh -e experiments.txt -m 12
# ```


# ====================
# Options for sbatch
# ====================
# FMI about options, see https://slurm.schedmd.com/sbatch.html
# N.B. options supplied on the command line will overwrite these set here

# *** To set any of these options, remove the first comment hash '# ' ***
# i.e. `# # SBATCH ...` -> `#SBATCH ...`

# Location for stdout log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --output=/home/%u/slurm_logs/slurm-%A_%a.out

# Location for stderr log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --error=/home/%u/slurm_logs/slurm-%A_%a.out

# Maximum number of nodes to use for the job
#SBATCH --nodes=1

# Generic resources to use - typically you'll want gpu:n to get n gpus
#SBATCH --gres=gpu:8

# Megabytes of RAM required. Check `cluster-status` for node configurations
#SBATCH --mem=14000

# Number of CPUs to use. Check `cluster-status` for node configurations
#SBATCH --cpus-per-task=4

# Maximum time for the job to run, format: days-hours:minutes:seconds
# #SBATCH --time=1-08:00:00

# Partition of the cluster to pick nodes from (check `sinfo`)
#SBATCH --partition=Teach-Standard,Teach-LongJobs,PGR-Standard

# Any nodes to exclude from selection
# #SBATCH --exclude=charles[05,12-18]

# Request a node
#SBATCH --nodelist=damnii07


# =====================
# Logging information
# =====================
MODEL_NAME=retraincaffeinettesttwopgr

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
src_path=/home/${USER}/datasets/INetData/FromHome
caffe_src=/home/${USER}/git/dissertation/pretrained_models/checkpoints/siggraph_caffemodel
model_src=/home/${USER}/git/dissertation/checkpoints/${MODEL_NAME}
mkdir -p ${model_src}  # make it if required

cp /home/${USER}/git/dissertation/portions_vec.txt /home/${USER}/slurm_logs/${SLURM_JOB_ID}_portions_vec.txt
cp /home/${USER}/git/dissertation/experiments_vec.txt /home/${USER}/slurm_logs/${SLURM_JOB_ID}_experiments_vec.txt

# input data directory path on the scratch disk of the node
dest_path=${SCRATCH_HOME}/datasets/INetData/FromHome
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

portions_text_file=$2
COMMAND="`sed \"${SLURM_ARRAY_TASK_ID}q;d\" ${portions_text_file}`"
COMMANDGRAB="${COMMAND}*"
rsync --archive --update --compress --progress --include ${COMMANDGRAB} --exclude '*' ${src_path}/ ${dest_path}
echo "Rsync Completed"


cpoint_path=${SCRATCH_HOME}/checkpoints/${MODEL_NAME}
mkdir -p ${cpoint_path}  # make it if required
caffe_path=${SCRATCH_HOME}/checkpoints/siggraph_caffemodel
mkdir -p ${caffe_path}  # make it if required
rsync --archive --update --compress --progress ${model_src}/ ${cpoint_path}
rsync --archive --update --compress --progress ${caffe_src}/ ${caffe_path}

code="${dest_path}/"
for f in ${code}*.tar; do
  d=`basename "$f" .tar`
  dpath="${code}TrainFolders/$d"
  echo "${dpath}"
  if [[ ! -d "$dpath" ]]; then
  	mkdir -p "${dpath}"
  	tar --keep-newer-files -xf "$f" -C "${dpath}"
  fi
done

echo "Forming Symlink Datafiles:"
sorted_path=${SCRATCH_HOME}/dataset
echo "OG Dataset Dir: ${dest_path}"
echo "Sorted Dataset Dir: ${sorted_path}"
python make_ilsvrc_dataset_set.py --in_path ${dest_path} --out_path ${sorted_path} --partition ${COMMAND}


# ==============================
# Finally, run the experiment!
# ==============================
# Read line number ${SLURM_ARRAY_TASK_ID} from the experiment file and run it
# ${SLURM_ARRAY_TASK_ID} is simply the number of the job within the array. If
# you execute `sbatch --array=1:100 ...` the jobs will get numbers 1 to 100
# inclusive.

experiment_text_file=$1
COMMAND="`sed \"${SLURM_ARRAY_TASK_ID}q;d\" ${experiment_text_file}`"
echo "Running provided command: ${COMMAND}"
eval "${COMMAND} --name ${MODEL_NAME}"
echo "Command ran successfully!"


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
