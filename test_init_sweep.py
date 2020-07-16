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
import copy

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
    opt.weighted_mask = True
    # opt.load_model = True
    # # if opt.plot_data_gen:
    #     # np.random.seed(5)
    #     # torch.manual_seed(5)
    #     # random.seed(5)

    num_points = 10
    opt2 = copy.deepcopy(opt)
    opt2.weighted_mask = False
    opt2.spread_mask = False
    opt2.name = 'siggraph_retrained'



    if not opt.load_sweep:
        dataset = torchvision.datasets.ImageFolder(opt.dataroot,
                                                   transform=transforms.Compose([
                                                       transforms.Resize((opt.fineSize, opt.fineSize)),
                                                       transforms.ToTensor()]))
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=not opt.serial_batches)

        model = create_model(opt)
        model.setup(opt)
        model.eval()

        model2 = create_model(opt2)
        model2.setup(opt2)
        model2.eval()


        time = dt.datetime.now()
        str_now = '%02d_%02d_%02d%02d' % (time.month, time.day, time.hour, time.minute)
        model_path = os.path.join(opt.checkpoints_dir, '%s/latest_net_G.pth' % opt.name)
        model_backup_path =  os.path.join(opt.checkpoints_dir, '%s/%s_net_G.pth' % (opt.name, str_now))
        # print('mp', model_path)
        # print('./checkpoints/%s/latest_net_G.pth' % opt.name)
        # shutil.copyfile(model_path, model_backup_path)

        psnrs = np.zeros((opt.how_many, 2))

        # if opt.weighted_mask:
            # opt.sample_Ps = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             # 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 5, 6, 7, 8, 9, ]

        bar = pb.ProgressBar(max_value=opt.how_many)
        for i, data_raw in enumerate(dataset_loader):
            if len(opt.gpu_ids) > 0:
                data_raw[0] = data_raw[0].cuda()
            data_raw[0] = util.crop_mult(data_raw[0], mult=8)



            data = util.get_colorization_data(data_raw, opt, ab_thresh=0., num_points=num_points)
            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()

            real = util.tensor2im(visuals['real'])
            fake_reg = util.tensor2im(visuals['fake_reg'])
            if opt.plot_data_gen:
                util.plot_data_results(data, real, fake_reg, opt)

            psnrsz = util.calculate_psnr_np(real, fake_reg)
            if opt.plot_data_gen:
                print('GT: ', psnrsz)
            psnrs[i, 0] = psnrsz

            data['mask_B'][data['mask_B'] != -1] = 0.5
            data['mask_B'][data['mask_B'] == -1] = -0.5

            model2.set_input(data)
            model2.test()
            visuals2 = model2.get_current_visuals()
            plt.imshow(util.tensor2im(visuals2['fake_reg']))
            plt.show()

            # just_ab_as_rgb_smoothed = apply_smoothing(data['abRgb'][nn, :, :, :], opt)
            just_ab_as_rgb_smoothed = util.apply_smoothing(visuals2['fake_reg'], opt)
            ab_bins, ab_decoded = util.zhang_bins(just_ab_as_rgb_smoothed, opt)
            # labels = dbscan_encoded_indexed(ab_bins)
            labels, num_labels = util.bins_scimage_group_minimal(ab_bins)

            plt.imshow(labels)
            plt.show()

            # num_same_bin = len(labels[labels == unique_bins[0]])
            # total_size = data['A'].shape[2] * data['B'].shape[3]
            # weight1 = float(num_same_bin / (total_size))

            data['mask_B'][data['mask_B'] == -0.5] = -1
            active_points = np.asarray(np.where(data['mask_B'] == 0.5))

            for point in range(len(active_points[0])):
                ix = active_points[:, point]
                label = labels[ix[2], ix[3]]
                num_same_bin = len(labels[labels == label])
                total_size = data['A'].shape[2] * data['A'].shape[3]
                weight1 = float(num_same_bin / (total_size))
                data['mask_B'][ix[0], ix[1], ix[2], ix[3]] = weight1

            # print('LEN' ,len(np.where(data['mask_B'] == 0.5)[0]))

            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()

            real = util.tensor2im(visuals['real'])
            fake_reg = util.tensor2im(visuals['fake_reg'])
            if opt.plot_data_gen:
                util.plot_data_results(data, real, fake_reg, opt)

            psnrsz = util.calculate_psnr_np(real, fake_reg)
            if opt.plot_data_gen:
                print('CP: ', psnrsz)
            psnrs[i, 1] = psnrsz




            if i == opt.how_many - 1:
                break

            bar.update(i)

        # Save results
        psnrs_mean = np.mean(psnrs, axis=0)
        psnrs_std = np.std(psnrs, axis=0) / np.sqrt(opt.how_many)

        # save_cpoint_dir = os.path.join(opt.checkpoints_dir, '%s/psnrs_mean_%s' % (opt.name,str_now))
        np.save('%s%s/shifted_psnrs_mean_%s' % (opt.checkpoints_dir, opt.name,str_now), psnrs_mean)
        np.save('%s%s/shifted_psnrs_std_%s' % (opt.checkpoints_dir, opt.name,str_now), psnrs_std)
        np.save('%s%s/shifted_psnrs_%s' % (opt.checkpoints_dir, opt.name,str_now), psnrs)
        psnrmeans = ['%.2f' % psnr for psnr in psnrs_mean]
        print('PSNR Means: ', psnrmeans)

    else:
        str_now = '%02d_%02d_%02d%02d' % (7, 9, 12, 33)
        psnrs_mean = np.load('%s%s/shifted_psnrs_mean_%s.npy' % (opt.checkpoints_dir, opt.name, str_now))
        psnrs_std = np.load('%s%s/shifted_psnrs_std_%s.npy' % (opt.checkpoints_dir, opt.name,str_now))
        psnrs = np.load('%s%s/shifted_psnrs_%s.npy' % (opt.checkpoints_dir, opt.name,str_now))
        psnrmeans = ['%.2f' % psnr for psnr in psnrs_mean]
        print('PSNR Means: ', psnrmeans)



    num_points_hack = 1. * num_points
    # num_points_hack[0] = .4

    plt.plot(randomisers, psnrs_mean, 'bo-', label=str_now)
    plt.plot(randomisers, psnrs_mean + psnrs_std, 'b--')
    plt.plot(randomisers, psnrs_mean - psnrs_std, 'b--')

    # plt.xscale('log')
    # plt.xticks([.4,1,2,5,10,20,50,100,200,500],
    #     ['Auto','1','2','5','10','20','50','100','200','500'])
    plt.xlabel('Random Shift')
    plt.ylabel('PSNR [db]')
    plt.legend(loc=0)
    # plt.xlim((randomisers[0], randomisers[-1]))
    plt.savefig('%s%s/shift_sweep_%s.png' % (opt.checkpoints_dir, opt.name, str_now))
