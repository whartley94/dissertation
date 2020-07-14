# python test_sweep.py --gpu_ids -1 --name siggraph_retrained --data_dir /Users/Will/Documents/Uni/MscEdinburgh/Diss/colorization-pytorch/dataset/SUN2012/ --checkpoints_dir /Users/Will/Documents/Uni/MscEdinburgh/Diss/checkpoints_from_pd/


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
    opt.load_model = True
    # if opt.plot_data_gen:
        # np.random.seed(5)
        # torch.manual_seed(5)
        # random.seed(5)

    # number of random points to assign
    num_points = np.round(10**np.arange(-.1, 2.8, .1))
    num_points[0] = 0
    num_points = np.unique(num_points.astype('int'))
    N = len(num_points)


    if not opt.load_sweep:
        # dataset = torchvision.datasets.ImageFolder(opt.dataroot,
        #                                            transform=transforms.Compose([
        #                                                transforms.Resize((opt.loadSize, opt.loadSize)),
        #                                                transforms.ToTensor()]))
        dataset = torchvision.datasets.ImageFolder(opt.dataroot,
                                                   transform=transforms.Compose([
                                                       transforms.RandomChoice(
                                                           [transforms.Resize(opt.loadSize, interpolation=1),
                                                            transforms.Resize(opt.loadSize, interpolation=2),
                                                            transforms.Resize(opt.loadSize, interpolation=3),
                                                            transforms.Resize((opt.loadSize, opt.loadSize),
                                                                              interpolation=1),
                                                            transforms.Resize((opt.loadSize, opt.loadSize),
                                                                              interpolation=2),
                                                            transforms.Resize((opt.loadSize, opt.loadSize),
                                                                              interpolation=3)]),
                                                       transforms.RandomChoice(
                                                           [transforms.RandomResizedCrop(opt.fineSize, interpolation=1),
                                                            transforms.RandomResizedCrop(opt.fineSize, interpolation=2),
                                                            transforms.RandomResizedCrop(opt.fineSize,
                                                                                         interpolation=3)]),
                                                       transforms.RandomHorizontalFlip(),
                                                       transforms.ToTensor()]))
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=not opt.serial_batches)

        model = create_model(opt)
        model.setup(opt)
        model.eval()

        time = dt.datetime.now()
        str_now = '%02d_%02d_%02d%02d' % (time.month, time.day, time.hour, time.minute)
        model_path = os.path.join(opt.checkpoints_dir, '%s/latest_net_G.pth' % opt.name)
        model_backup_path =  os.path.join(opt.checkpoints_dir, '%s/%s_net_G.pth' % (opt.name, str_now))
        # print('mp', model_path)
        # print('./checkpoints/%s/latest_net_G.pth' % opt.name)
        shutil.copyfile(model_path, model_backup_path)

        psnrs = np.zeros((opt.how_many, N))

        # if opt.weighted_mask:
            # opt.sample_Ps = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             # 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 5, 6, 7, 8, 9, ]

        bar = pb.ProgressBar(max_value=opt.how_many)
        for i, data_raw in enumerate(dataset_loader):
            if len(opt.gpu_ids) > 0:
                data_raw[0] = data_raw[0].cuda()
            data_raw[0] = util.crop_mult(data_raw[0], mult=8)

            for nn in range(N):
                # embed()
                data = util.get_colorization_data(data_raw, opt, ab_thresh=0., num_points=num_points[nn])


                model.set_input(data)
                model.test()
                visuals = model.get_current_visuals()

                real = util.tensor2im(visuals['real'])
                fake_reg = util.tensor2im(visuals['fake_reg'])
                if opt.plot_data_gen:
                    util.plot_data_results(data, real, fake_reg, opt)
                    print('nn', nn)

                psnrsz = util.calculate_psnr_np(real, fake_reg)
                if opt.plot_data_gen:
                    print(psnrsz)
                psnrs[i, nn] = psnrsz

            if i == opt.how_many - 1:
                break

            bar.update(i)

        # Save results
        psnrs_mean = np.mean(psnrs, axis=0)
        psnrs_std = np.std(psnrs, axis=0) / np.sqrt(opt.how_many)

        # save_cpoint_dir = os.path.join(opt.checkpoints_dir, '%s/psnrs_mean_%s' % (opt.name,str_now))
        np.save('%s%s/psnrs_mean_%s' % (opt.checkpoints_dir, opt.name,str_now), psnrs_mean)
        np.save('%s%s/psnrs_std_%s' % (opt.checkpoints_dir, opt.name,str_now), psnrs_std)
        np.save('%s%s/psnrs_%s' % (opt.checkpoints_dir, opt.name,str_now), psnrs)
        psnrmeans = ['%.2f' % psnr for psnr in psnrs_mean]
        print('PSNR Means: ', psnrmeans)

    else:
        str_now = '%02d_%02d_%02d%02d' % (7, 9, 12, 33)
        psnrs_mean = np.load('%s%s/psnrs_mean_%s.npy' % (opt.checkpoints_dir, opt.name, str_now))
        psnrs_std = np.load('%s%s/psnrs_std_%s.npy' % (opt.checkpoints_dir, opt.name,str_now))
        psnrs = np.load('%s%s/psnrs_%s.npy' % (opt.checkpoints_dir, opt.name,str_now))
        psnrmeans = ['%.2f' % psnr for psnr in psnrs_mean]
        print('PSNR Means: ', psnrmeans)


    old_results = np.load('%s/psnrs_siggraph.npy' % opt.resources_dir)
    old_mean = np.mean(old_results, axis=0)
    old_std = np.std(old_results, axis=0) / np.sqrt(old_results.shape[0])
    oldmeans = ['%.2f' % psnr for psnr in old_mean]
    print('Old PSNR Means: ', oldmeans)


    num_points_hack = 1. * num_points
    num_points_hack[0] = .4

    plt.plot(num_points_hack, psnrs_mean, 'bo-', label=str_now)
    plt.plot(num_points_hack, psnrs_mean + psnrs_std, 'b--')
    plt.plot(num_points_hack, psnrs_mean - psnrs_std, 'b--')
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
    plt.savefig('%s%s/sweep_%s.png' % (opt.checkpoints_dir, opt.name, str_now))
