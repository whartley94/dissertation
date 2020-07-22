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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


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
    opt.sample_Ps = [3,]
    opt.load_model = True
    if opt.plot_data_gen:
        np.random.seed(5)
        torch.manual_seed(5)
        random.seed(5)

    # # number of random points to assign
    # num_points = np.round(10**np.arange(-.1, 2.8, .1))
    # num_points[0] = 0
    # num_points = np.unique(num_points.astype('int'))
    # N = len(num_points)
    num_points = 1
    # weights = np.linspace(-4, 1, 20)
    weights = np.linspace(-1, 1, 20)

    weights_l = len(weights)
    hyp = np.sqrt(opt.fineSize ** 2 + opt.fineSize ** 2)
    opt.ops = np.linspace(0, hyp, 20)

    if not opt.load_sweep:
        dataset = torchvision.datasets.ImageFolder(opt.dataroot,
                                                   transform=transforms.Compose([
                                                       transforms.Resize((opt.fineSize, opt.fineSize)),
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

        threshes = 3
        portionss = np.zeros((opt.how_many, weights_l, threshes, len(opt.ops)-1))

        bar = pb.ProgressBar(max_value=opt.how_many)
        for i, data_raw in enumerate(dataset_loader):
            if len(opt.gpu_ids) > 0:
                data_raw[0] = data_raw[0].cuda()
            data_raw[0] = util.crop_mult(data_raw[0], mult=8)

            location = np.random.randint(10, opt.fineSize-10, 2)

            for nn in range(weights_l):
                # embed()
                # location = [50, 50]
                data = util.get_colorization_data(data_raw, opt, ab_thresh=0., num_points=0)
                col = torch.tensor([-0.8, -0.8], dtype=torch.float)
                data['mask_B'][0, 0, location[0], location[1]] = weights[nn]
                data['hint_B'][0, :, location[0], location[1]] = col
                # print(data['hint_B'][data['hint_B']!=0])
                # print(data['hint_B'])

                model.set_input(data)
                model.test()
                visuals = model.get_current_visuals()

                real = util.tensor2im(visuals['real'])
                fake_reg = util.tensor2im(visuals['fake_reg'])

                dist = np.zeros((opt.fineSize, opt.fineSize))
                for j in range(opt.fineSize):
                    for k in range(opt.fineSize):
                        dist[j, k] = np.sqrt(((j - location[0]) ** 2) + ((k - location[1]) ** 2))

                portionss[i, nn, 0, :] = util.integrate(location[0], location[1], model.fake_B_reg, col, dist, opt, thresh=0.82)
                portionss[i, nn, 1, :] = util.integrate(location[0], location[1], model.fake_B_reg, col, dist, opt)
                portionss[i, nn, 2, :] = util.integrate(location[0], location[1], model.fake_B_reg, col, dist, opt, thresh=0.74)

                if opt.plot_data_gen:
                    util.plot_data_results(data, real, fake_reg, opt)
                    print('nn', nn)

            if i == opt.how_many - 1:
                break

            bar.update(i)

        # save_cpoint_dir = os.path.join(opt.checkpoints_dir, '%s/psnrs_mean_%s' % (opt.name,str_now))
        np.save('%s%s/portionss_%s' % (opt.checkpoints_dir, opt.name,str_now), portionss)

    else:
        # str_now = '%02d_%02d_%02d%02d' % (7, 16, 12, 3)
        # str_now = '%02d_%02d_%02d%02d' % (7, 17, 11, 37)
        # str_now = '%02d_%02d_%02d%02d' % (7, 17, 12, 3) #This one was -4 to 1 weight - col 0.8, 0.8.
        str_now = '%02d_%02d_%02d%02d' % (7, 21, 16, 51) #This one was -1 to 1 weight - col 0.8, 0.8.

        portionss = np.load('%s%s/portionss_%s.npy' % (opt.checkpoints_dir, opt.name, str_now))

    # Save results
    # print(portionss.shape)
    which_one = 1
    mean = np.nanmean(portionss[:, :, which_one, :], axis=0)
    std = np.std(portionss[:, :, which_one, :], axis=0) / np.sqrt(opt.how_many)
    # print(mean.shape)
    # print(mean)

    # x = np.linspace(0.0, 1.0, 100)
    viridis = cm.get_cmap('viridis', mean.shape[0])
    # viridis = cm.get_cmap('cividis', mean.shape[0])
    # viridis = cm.get_cmap('gray', mean.shape[0]*1.2)


    for l in range(mean.shape[0]):
        plt.plot(opt.ops[1:], mean[l, :], label=str(weights[l]), c=viridis(l))
    # plt.plot(opt.ops, mean, 'bo-', label=str_now)
    # plt.plot(randomisers, psnrs_mean + psnrs_std, 'b--')
    # plt.plot(randomisers, psnrs_mean - psnrs_std, 'b--')
    plt.xlabel('Distance')
    plt.ylabel('Integral')
    plt.legend(loc=0)
    plt.plot()
    if opt.load_sweep:
        plt.show()
    # else:
    #     plt.savefig('%s%s/portionss_%s.png' % (opt.checkpoints_dir, opt.name, str_now))


    if opt.load_sweep:
        viridis = cm.get_cmap('viridis', mean.shape[1])
        for l in range(mean.shape[1]):
            # print(l)
            plt.plot(weights, mean[:, l], label=str(opt.ops[l]), c=viridis(l))
        plt.legend(loc=0)
        plt.xlabel('Weight Value')
        plt.ylabel('Integral')
        plt.show()

