
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
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
from matplotlib import cm
import cv2

def do_scan(j, path1, path2, threshold):
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
        # og_psnr = util.calculate_psnr_np(ours, gt)
        og_psnr = util.calculate_sim_thresh_np(ours, gt, threshold)
        return og_psnr

def get_values(threshold=20):
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
            psnr = do_scan(j, og_path, og_subdirs, threshold)
            if psnr is not None:
                og_psnrs.append(psnr)
                all_og_psnrs.append(psnr)
        for j in sp_subdirs:
            psnr = do_scan(j, sp_path, sp_subdirs, threshold)
            if psnr is not None:
                sp_psnrs.append(psnr)
                all_sp_psnrs.append(psnr)

        og_psnr_mean = np.mean(og_psnrs)
        sp_psnr_mean = np.mean(sp_psnrs)
        og_psnr_means.append(og_psnr_mean)
        sp_psnr_means.append(sp_psnr_mean)
        both = np.mean([og_psnr_mean, sp_psnr_mean])
        both_psnr_means.append(both)

    overall_og_mean = np.mean(og_psnr_means)
    overall_sp_mean = np.mean(sp_psnr_means)
    if print_ans:
        print('Names: ', names)
        print('OG Scores: ', og_psnr_means)
        print('SP Scores: ', sp_psnr_means)
        print('Average User Scores: ', both_psnr_means)
        print('Overall OG Mean: ', np.round(overall_og_mean, 3))
        print('Overall SP Mean: ', np.round(overall_sp_mean, 3))
        print('Overall OG Var: ', np.round(np.var(all_og_psnrs), 3))
        print('Overall SP Var: ', np.round(np.var(all_sp_psnrs), 3))

    both_sorted = np.sort(both_psnr_means)
    names_sorted = np.argsort(both_psnr_means)
    names = np.asarray(names)
    names_sorted = names[names_sorted]
    both_sorted = both_sorted[::-1]
    names_sorted = names_sorted[::-1]

    if plot_bar:
        plt.bar(names_sorted, both_sorted)
        plt.show()

    return overall_og_mean, overall_sp_mean


plot_bar = False
print_ans = False
run = False

rootdir = "/Users/Will/Documents/Uni/MscEdinburgh/Diss/bitsync/Images/"
dirs = os.listdir(rootdir)
dirs2 = []
for i in dirs:
    if not '.DS_Store' in i and not 'ExampleSet' in i:
        dirs2.append(i)

if run:
    thresholds = np.linspace(10, 80, 30)
    diffs = np.zeros(len(thresholds))
    actual1 = np.zeros(len(thresholds))
    actual2 = np.zeros(len(thresholds))
    for j, i in enumerate(thresholds):
        a, b = get_values(i)
        actual1[j] = a
        actual2[j] = b
        diffs[j] = a-b

    np.save('thresholds.npy', thresholds)
    np.save('diffs.npy', diffs)
    np.save('actual1.npy', actual1)
    np.save('actual2.npy', actual2)
else:
    thresholds = np.load('thresholds.npy')
    diffs = np.load('diffs.npy')
    actual1 = np.load('actual1.npy')
    actual2 = np.load('actual2.npy')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))

ax2.plot(thresholds, diffs, label='Observed Decline')
ax2.set_xlabel(r'\textbf{Threshold} $t$')
ax2.set_ylabel(r'\textbf{Difference} $D$')
ax2.plot([thresholds[0], thresholds[-1]], [diffs[0], diffs[-1]], label='Linear Decline')
ax2.legend()

ax1.plot(thresholds, actual1, label='Baseline')
ax1.plot(thresholds, actual2, label='Ours')
ax1.set_xlabel(r'\textbf{Threshold} $t$')
ax1.set_ylabel(r'\textbf{Portion Accepted}')
ax1.legend()
plt.tight_layout()
# plt.savefig('/Users/Will/Documents/Uni/MscEdinburgh/Diss/ReportFiles/UserStudy/comparison.png', dpi=600)
plt.show()