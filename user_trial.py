
import os
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
import random

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable

from util import util
from IPython import embed
import numpy as np
import progressbar as pb
import shutil

import datetime as dt
import matplotlib.pyplot as plt
import cv2

def do_scan(j, path1, path2):
    if '.JPEG' in j:
        substrings = j.split('.')
        forestring = substrings[0]
        folder_string = ''
        for k in path2:
            if forestring in k and not '.JPEG' in k:
                folder_string = k
        ours_path = os.path.join(path1, folder_string, 'ours_fullres.png')
        gt_path = os.path.join(path1, j)

        ours = cv2.imread(ours_path)
        gt = cv2.imread(gt_path)
        ours = cv2.cvtColor(ours, cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        og_psnr = util.calculate_psnr_np(ours, gt)
        return og_psnr

rootdir = "/Users/Will/Documents/Uni/MscEdinburgh/Diss/bitsync/Images/"
dirs = os.listdir(rootdir)
dirs2 = []
for i in dirs:
    if not '.DS_Store' in i and not 'ExampleSet' in i:
        dirs2.append(i)

names = []
og_psnr_means = []
sp_psnr_means = []
both_psnr_means = []
all_og_psnrs = []
all_sp_psnrs = []
for i in dirs2:
    names.append(i)
    og_path = os.path.join(rootdir, i, 'OG')
    sp_path = os.path.join(rootdir, i, 'SP')
    og_subdirs = os.listdir(og_path)
    sp_subdirs = os.listdir(sp_path)
    og_psnrs = []
    sp_psnrs = []
    for j in og_subdirs:
        psnr = do_scan(j, og_path, og_subdirs)
        if psnr is not None:
            og_psnrs.append(psnr)
            all_og_psnrs.append(psnr)
    for j in sp_subdirs:
        psnr = do_scan(j, sp_path, sp_subdirs)
        if psnr is not None:
            sp_psnrs.append(psnr)
            all_sp_psnrs.append(psnr)

    og_psnr_mean = np.mean(og_psnrs)
    sp_psnr_mean = np.mean(sp_psnrs)
    og_psnr_means.append(og_psnr_mean)
    sp_psnr_means.append(sp_psnr_mean)
    both = np.mean([og_psnr_mean, sp_psnr_mean])
    both_psnr_means.append(both)

print('Names: ', names)
print('OG Scores: ', og_psnr_means)
print('SP Scores: ', sp_psnr_means)
print('Average User Scores: ', both_psnr_means)
print('Overall OG Mean: ', np.round(np.mean(og_psnr_means), 1))
print('Overall SP Mean: ', np.round(np.mean(sp_psnr_means), 1))
print('Overall OG Var: ', np.round(np.var(all_og_psnrs), 1))
print('Overall SP Var: ', np.round(np.var(all_sp_psnrs), 1))

both_sorted = np.sort(both_psnr_means)
names_sorted = np.argsort(both_psnr_means)
names = np.asarray(names)
names_sorted = names[names_sorted]
both_sorted = both_sorted[::-1]
names_sorted = names_sorted[::-1]

plt.bar(names_sorted, both_sorted)
plt.show()