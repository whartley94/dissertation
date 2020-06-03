
import os
import sys
from util import util
import numpy as np
import argparse
import random
import shutil
import pickle

random.seed(10)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--in_path', type=str, default='/Users/Will/Documents/Uni/MscEdinburgh/Diss/datasets/SUN2012/Images/')
parser.add_argument('--out_path', type=str, default='./dataset/SUN2012/')

opt = parser.parse_args()
orig_path = opt.in_path
print('Copying Sun from...[%s]'%orig_path)

# if os.path.isdir(opt.out_path):
#     shutil.rmtree(opt.out_path)
exists = False
leng = 0
if os.path.isdir(opt.out_path):
	exists = True
	for (dirpath, dirnames, filenames) in os.walk(opt.out_path):
		leng+=len(filenames)
else:
	util.mkdirs(opt.out_path)
    # shutil.rmtree(opt.out_path)

if exists == False or (exists == True and leng == 0):

	if not os.path.isfile('resources/train_set.data'):
		print('Warning!: Train Pickle Was NOT FOUND!')

		all_files = []
		for (dirpath, dirnames, filenames) in os.walk(opt.in_path):
			for f in filenames:
				all_files.append(os.path.join(dirpath, f))
			# all_files.extend(filenames)
		all_files = [x for x in all_files if '.jpg' in x]
		random.shuffle(all_files)

		num_images = len(all_files)
		train_p = 0.8
		test_p = 0.1
		val_p = 1-(test_p+train_p)
		assert val_p >= 0

		num_train = int(num_images*train_p)
		num_test = int(num_images*test_p)
		num_val = int(num_images*val_p)

		train_set = all_files[:num_train]
		test_set = all_files[num_train:num_train+num_test]
		val_set = all_files[num_train+num_test:]

		assert len(train_set)+len(val_set)+len(test_set) == num_images

		in_chars = len(opt.in_path)
		train_tails = []
		test_tails = []
		val_tails = []
		for i in train_set:
			train_tails.append(i[in_chars:])
		for j in test_set:
			test_tails.append(j[in_chars:])
		for k in val_set:
			val_tails.append(k[in_chars:])


		with open('resources/train_set.data', 'wb') as filehandle:
			pickle.dump(train_tails, filehandle)
		with open('resources/test_set.data', 'wb') as filehandle2:
			pickle.dump(test_tails, filehandle2)
		with open('resources/val_set.data', 'wb') as filehandle3:
			pickle.dump(val_tails, filehandle3)

		# print(train_set)

	else:
		with open('resources/train_set.data', 'rb') as filehandle:
			train_tails = pickle.load(filehandle)
		with open('resources/test_set.data', 'rb') as filehandle2:
			test_tails = pickle.load(filehandle2)
		with open('resources/val_set.data', 'rb') as filehandle3:
			val_tails = pickle.load(filehandle3)


		num_train = len(train_tails)
		num_test = len(test_tails)
		num_val = len(val_tails)

		train_set = []
		test_set = []
		val_set = []
		for i in train_tails:
			train_set.append(opt.in_path + i)
		for j in test_tails:
			test_set.append(opt.in_path + j)
		for k in val_tails:
			val_set.append(opt.in_path + k)


	# Copy over part of training set (for initializer)
	trn_small_path = os.path.join(opt.out_path,'train_small/subdir')
	util.mkdirs(trn_small_path)
	for train_path in train_set[:int(num_train*.1)]:
		os.symlink(train_path, os.path.join(trn_small_path,os.path.basename(train_path)))
	print('Making small training set in...[%s]'%trn_small_path)


	# Copy over whole training set
	trn_path = os.path.join(opt.out_path,'train/subdir')
	util.mkdirs(opt.out_path)
	util.mkdirs(trn_path)
	for train_path in train_set:
		os.symlink(train_path, os.path.join(trn_path,os.path.basename(train_path)))
	print('Making training set in...[%s]'%trn_path)


	# Copy over subset of ILSVRC12 val set for colorization val set
	val_path = os.path.join(opt.out_path,'val/imgs')
	util.mkdirs(val_path)
	print('Making validation set in...[%s]'%val_path)
	for vl_path in val_set:
		os.symlink(vl_path, os.path.join(val_path,os.path.basename(vl_path)))
	print('Making training set in...[%s]'%val_path)

	# Copy over subset of ILSVRC12 val set for colorization test set
	test_path = os.path.join(opt.out_path,'test/imgs')
	util.mkdirs(test_path)
	print('Making test set in...[%s]'%test_path)
	for ts_path in test_set:
		os.symlink(ts_path, os.path.join(test_path,os.path.basename(ts_path)))
	print('Making training set in...[%s]'%test_path)

	print('Done')
else:
	print('Nothing to do')