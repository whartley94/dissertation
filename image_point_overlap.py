
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

# rootdir = "/Users/Will/Documents/Uni/MscEdinburgh/Diss/ReportFiles/Leeky/HirPeps/"
rootdir = '/Users/Will/Documents/Uni/MscEdinburgh/Diss/bitsync/Images/Dan Radford/SP/'
dirs = os.listdir(rootdir)
dirs2 = []
for i in dirs:
    if not '.DS' in i and not '.JPEG' in i and not '.png' in i:
        dirs2.append(i)
print(dirs2)
for m in dirs2:
    img = cv2.imread(os.path.join(rootdir, m, "ours_fullres.png"))
    points = cv2.imread(os.path.join(rootdir, m, "input_ab.png"))


    # img = cv2.imread("/Users/Will/Documents/Uni/MscEdinburgh/Diss/myExampleDir/ILSVRC2012_val_00003016_with_dist_200802_16/ours_fullres.png")
    # points = cv2.imread("/Users/Will/Documents/Uni/MscEdinburgh/Diss/myExampleDir/ILSVRC2012_val_00003016_with_dist_200802_16/input_ab.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    points = cv2.cvtColor(points, cv2.COLOR_BGR2RGB)
    # print(points)
    # print(img.shape)
    # points[points==254] = 255
    points[points==118] = 255
    points = cv2.resize(points, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    # img = cv2.resize(img, (points.shape[1], points.shape[0]), interpolation=cv2.INTER_CUBIC)
    # img[points!=118] = points[points!=118]

    points2 = np.zeros(points.shape).astype(int)
    points2 += 255
    size = 3
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            if not np.array_equal(points[i,j,:], np.array([255, 255, 255]).astype(int)):
                points2[i-size:i+size,j-size:j+size,:] = points[i,j,:]

    points3 = np.zeros(points2.shape).astype(int)
    points3 += 255

    size2 = 1
    for i in range(1, points.shape[0]-1):
        for j in range(1, points.shape[1]-1):
            if not np.array_equal(points2[i+1,j+1,:], np.array([255, 255, 255]).astype(int)):
                if np.array_equal(points2[i,j,:], np.array([255, 255, 255]).astype(int)):
                    points3[i-size2:i+1, j-size2:j+1, :] = 0
            if not np.array_equal(points2[i+1,j,:], np.array([255, 255, 255]).astype(int)):
                if np.array_equal(points2[i,j,:], np.array([255, 255, 255]).astype(int)):
                    points3[i-size2:i+1, j, :] = 0
            if not np.array_equal(points2[i-1,j-1,:], np.array([255, 255, 255]).astype(int)):
                if np.array_equal(points2[i,j,:], np.array([255, 255, 255]).astype(int)):
                    points3[i:i+size2+1, j:j+size2+1, :] = 0
            if not np.array_equal(points2[i-1,j+1,:], np.array([255, 255, 255]).astype(int)):
                if np.array_equal(points2[i,j,:], np.array([255, 255, 255]).astype(int)):
                    points3[i:i+size2+1, j-size2:j+1, :] = 0
            if not np.array_equal(points2[i+1,j-1,:], np.array([255, 255, 255]).astype(int)):
                if np.array_equal(points2[i,j,:], np.array([255, 255, 255]).astype(int)):
                    points3[i-size2:i+1, j:j+size2+1, :] = 0
            #         if np.array_equal(points2[i, j + 1, :], np.array([255, 255, 255]).astype(int)):
            #             if np.array_equal(points2[i + 1, j, :], np.array([255, 255, 255]).astype(int)):
            #                 points3[i, j, :] = 0

    img[points2!=255] = points2[points2!=255]
    img[points3==0] = 255
    # plt.imshow(img)
    # plt.show()
    result_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(rootdir, m, 'Appended.png'), result_bgr)
