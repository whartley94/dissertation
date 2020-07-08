
import os
import sys
from util import util
import numpy as np
import argparse
import shutil

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--in_path', type=str, default='/data/big/dataset/ILSVRC2012')
parser.add_argument('--out_path', type=str, default='./dataset/ilsvrc2012/')
parser.add_argument('--partition', type=str, default='')
train_loc = 'TrainFolders'

opt = parser.parse_args()
orig_path = opt.in_path
print('Copying ILSVRC from...[%s]'%orig_path)


# Copy over part of training set (for initializer)
trn_small_path = os.path.join(opt.out_path,'train_small')
if os.path.isdir(trn_small_path):
	shutil.rmtree(trn_small_path)
util.mkdirs(trn_small_path)
train_subdirs = os.listdir(os.path.join(opt.in_path, train_loc))
train_subdirs_arr = np.asarray(train_subdirs)
is_tars = [not '.tar' in train_subdirs[i] for i in range(len(train_subdirs))]
train_subdirs = train_subdirs_arr[is_tars]
# print(train_subdirs)
for train_subdir in train_subdirs[:10]:
	os.symlink(os.path.join(opt.in_path,train_loc,train_subdir),os.path.join(trn_small_path,train_subdir))
print('Making small training set in...[%s]'%trn_small_path)

# # Copy over whole training set
# trn_path = os.path.join(opt.out_path, 'train')
# print(trn_path)
# if os.path.isdir(trn_path):
# 	os.unlink(trn_path)
# # 	os.rmdir(trn_path)
# util.mkdirs(opt.out_path)
# os.symlink(os.path.join(opt.in_path,train_loc),trn_path)
# print('Making training set in...[%s]'%trn_path)


# Copy over  training set
trn_path = os.path.join(opt.out_path, 'train')
if os.path.isdir(trn_path):
	try:
		os.unlink(trn_path)
		print('Unlink')
	except:
		print('Cant unlink')
	try:
		shutil.rmtree(trn_path)
		print('Rmtree')
	except:
		print('Cant rmtree')
util.mkdirs(trn_path)

# print(train_subdirs)
for train_subdir in train_subdirs:
	os.symlink(os.path.join(opt.in_path,train_loc,train_subdir),os.path.join(trn_path,train_subdir))
print('Making training set in...[%s]'%trn_path)



# Copy over training partition
trn_partition_path = os.path.join(opt.out_path, 'train_partition')
if os.path.isdir(trn_partition_path):
	shutil.rmtree(trn_partition_path)
util.mkdirs(trn_partition_path)
is_partition = [train_subdirs[i].startswith(opt.partition) for i in range(len(train_subdirs))]
partitions = train_subdirs[is_partition]
# print(partitions)
for partition in partitions:
	os.symlink(os.path.join(opt.in_path,train_loc,partition),os.path.join(trn_partition_path,partition))
print('Making partitioned training set in...[%s]'%trn_partition_path)


	# # Copy over subset of ILSVRC12 val set for colorization val set
	# val_path = os.path.join(opt.out_path,'val/imgs')
	# util.mkdirs(val_path)
	# print('Making validation set in...[%s]'%val_path)
	# for val_ind in range(1000):
	# 	os.system('ln -s %s/val/ILSVRC2012_val_%08d.JPEG %s/ILSVRC2012_val_%08d.JPEG'%(orig_path,val_ind+1,val_path,val_ind+1))
	# 	# os.system('cp %s/val/ILSVRC2012_val_%08d.JPEG %s/ILSVRC2012_val_%08d.JPEG'%(orig_path,val_ind+1,val_path,val_ind+1))
	#
	# # Copy over subset of ILSVRC12 val set for colorization test set
	# test_path = os.path.join(opt.out_path,'test/imgs')
	# util.mkdirs(test_path)
	# val_inds = np.load('./resources/ilsvrclin12_val_inds.npy')
	# print('Making test set in...[%s]'%test_path)
	# for val_ind in val_inds:
	# 	os.system('ln -s %s/val/ILSVRC2012_val_%08d.JPEG %s/ILSVRC2012_val_%08d.JPEG'%(orig_path,val_ind+1,test_path,val_ind+1))
	# 	# os.system('cp %s/val/ILSVRC2012_val_%08d.JPEG %s/ILSVRC2012_val_%08d.JPEG'%(orig_path,val_ind+1,test_path,val_ind+1))

print('Done')
