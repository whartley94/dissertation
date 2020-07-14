# python plot_sweep.py --gpu_ids -1 --name siggraph_retrained\
#  --data_dir /Users/Will/Documents/Uni/MscEdinburgh/Diss/colorization-pytorch/dataset/SUN2012/\
#  --checkpoints_dir /Users/Will/Documents/Uni/MscEdinburgh/Diss/checkpoints_from_pd/


import os
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html

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

if __name__ == '__main__':
    opt = TrainOptions().parse()
    opt.load_model = True
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.display_id = -1  # no visdom display
    opt.phase = 'test'
    # opt.dataroot = '/Users/Will/Documents/Uni/MscEdinburgh/Diss/colorization-pytorch/dataset/SUN2012/%s/' % opt.phase
    opt.dataroot = os.path.join(opt.data_dir, opt.phase)
    # opt.dataroot = './dataset/ilsvrc2012/%s/' % opt.phase
    opt.loadSize = 256
    opt.how_many = 1000
    opt.aspect_ratio = 1.0
    opt.sample_Ps = [6,]
    # opt.sample_Ps = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     # 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 5, 6, 7, 8, 9, ]
    opt.load_model = True

    # number of random points to assign
    num_points = np.round(10**np.arange(-.1, 2.8, .1))
    num_points[0] = 0
    num_points = np.unique(num_points.astype('int'))
    N = len(num_points)
    num_points_hack = 1. * num_points
    num_points_hack[0] = .4

    opt.data_dir = '/Users/Will/Documents/Uni/MscEdinburgh/Diss/colorization-pytorch/dataset/SUN2012/'
    opt.checkpoints_dir = '/Users/Will/Documents/Uni/MscEdinburgh/Diss/checkpoints_from_pd/'

    names = []
    strings = []
    colors = []
    # names.append('siggraph_retrained')
    # strings.append('%02d_%02d_%02d%02d' % (7, 9, 12, 33))
    # colors.append('b')
    # names.append('siggraph_caffemodel')
    # strings.append('%02d_%02d_%02d%02d' % (7, 9, 10, 9))
    # colors.append('g')
    names.append('wholeinet')
    strings.append('%02d_%02d_%02d%02d' % (7, 10, 14, 33))
    colors.append('c')
    names.append('wholeinet')
    strings.append('%02d_%02d_%02d%02d' % (7, 13, 15, 24))
    colors.append('y')




    for i in range(len(strings)):

        psnrs_mean = np.load('%s%s/psnrs_mean_%s.npy' % (opt.checkpoints_dir, names[i], strings[i]))
        psnrs_std = np.load('%s%s/psnrs_std_%s.npy' % (opt.checkpoints_dir, names[i],strings[i]))
        psnrs = np.load('%s%s/psnrs_%s.npy' % (opt.checkpoints_dir, names[i],strings[i]))
        print(psnrs[psnrs ==0])
        psnrmeans = ['%.2f' % psnr for psnr in psnrs_mean]
        print('PSNR Means: ', psnrmeans)

        plt.plot(num_points_hack, psnrs_mean, colors[i] +'o-', label=names[i])
        plt.plot(num_points_hack, psnrs_mean + psnrs_std, colors[i] + '--')
        plt.plot(num_points_hack, psnrs_mean - psnrs_std, colors[i] + '--')


    old_results = np.load('%s/psnrs_siggraph.npy' % opt.resources_dir)
    old_mean = np.mean(old_results, axis=0)
    old_std = np.std(old_results, axis=0) / np.sqrt(old_results.shape[0])
    oldmeans = ['%.2f' % psnr for psnr in old_mean]
    print('Old PSNR Means: ', oldmeans)



    plt.plot(num_points_hack, old_mean, 'ro-', label='siggraph17')
    plt.plot(num_points_hack, old_mean + old_std, 'r--')
    plt.plot(num_points_hack, old_mean - old_std, 'r--')

    plt.xscale('log')
    plt.xticks([.4,1,2,5,10,20,50,100,200,500],
        ['Auto','1','2','5','10','20','50','100','200','500'])
    plt.xlabel('Number of points')
    plt.ylabel('PSNR [db]')
    plt.legend(loc=0)
    plt.xlim((num_points_hack[0], num_points_hack[-1]))
    plt.show()
    # plt.savefig('%s%s/sweep_%s.png' % (opt.checkpoints_dir, opt.name, str_now))
