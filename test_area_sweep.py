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
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
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
    opt.ops = np.linspace(0, hyp, 50)

    cols = [[0.8, 0.8], [-0.8, 0.8], [0.8, -0.8], [-0.8, -0.8]]
    # cols = [[-0.8, 0.8], [0.8, -0.8]]
    cols_l = len(cols)

    threshes = np.linspace(0.74, 1.4, 20)
    threshes_l = len(threshes)

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

        portionss = np.zeros((opt.how_many, weights_l, threshes_l, cols_l, len(opt.ops)-1))

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

                for col_i, colr in enumerate(cols):
                    col = torch.tensor(colr, dtype=torch.float)
                    # print(col)
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

                    for tt_i, tt in enumerate(threshes):
                        portionss[i, nn, tt_i, col_i, :] = util.integrate(location[0], location[1], model.fake_B_reg, col, dist, opt, thresh=tt)
                        # print(portionss[i, nn, tt_i, col_i, :])

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
        # str_now = '%02d_%02d_%02d%02d' % (7, 17, 12, 3) #This one was -4 to 1 weight - col 0.8, 0.8. -opt.ops = np.linspace(0, hyp, 20)
        # str_now = '%02d_%02d_%02d%02d' % (7, 21, 16, 51) #This one was -1 to 1 weight - col 0.8, 0.8.  - opt.ops = np.linspace(0, hyp, 20)
        # str_now = '%02d_%02d_%02d%02d' % (7, 22, 10, 11) #This one was -1 to 1 weight - col 0.8, 0.8.  - opt.ops = np.linspace(0, hyp, 50)
        str_now = '%02d_%02d_%02d%02d' % (7, 22, 14, 5)

        portionss = np.load('%s%s/portionss_%s.npy' % (opt.checkpoints_dir, opt.name, str_now))

    # Save results
    # print(portionss.shape)
    which_thresh = 1
    which_col = 0
    # mean = np.nanmean(portionss[:, :, which_thresh, :], axis=0)
    # std = np.std(portionss[:, :, which_thresh, :], axis=0) / np.sqrt(opt.how_many)
    mean = np.nanmean(portionss[:, :, which_thresh, which_col, :], axis=0)
    std = np.std(portionss[:, :, which_thresh, which_col, :], axis=0) / np.sqrt(opt.how_many)
    # print(mean.shape)
    # print(mean)

    # x = np.linspace(0.0, 1.0, 100)
    viridis = cm.get_cmap('viridis', mean.shape[0])
    # viridis = cm.get_cmap('cividis', mean.shape[0])
    # viridis = cm.get_cmap('gray', mean.shape[0]*1.2)

    plt.rc('legend', fontsize=10)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    for l in range(mean.shape[0]):
        ax.plot(opt.ops[1:], mean[l, :], label=str(np.round(weights[l], 2)), c=viridis(l))
    # plt.plot(opt.ops, mean, 'bo-', label=str_now)
    # plt.plot(randomisers, psnrs_mean + psnrs_std, 'b--')
    # plt.plot(randomisers, psnrs_mean - psnrs_std, 'b--')
    ax.set_xlabel(r'\textbf{Radial Distance} (Pixels)')
    ax.set_ylabel(r'\textbf{Integral}')
    ax.legend(ncol=4)
    plt.plot()
    plt.savefig('%s%s/portionss_%s%s.png' % (opt.checkpoints_dir, opt.name, str_now, which_col), dpi=700)
    if opt.load_sweep:
        plt.show()
    # else:



    # if opt.load_sweep:
    #     viridis = cm.get_cmap('viridis', mean.shape[1])
    #     for l in range(mean.shape[1]):
    #         # print(l)
    #         plt.plot(weights, mean[:, l], label=str(opt.ops[l]), c=viridis(l))
    #     plt.legend(loc=0)
    #     plt.xlabel('Weight Value')
    #     plt.ylabel('Integral')
    #     plt.show()

