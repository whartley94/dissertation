
import os
from options.train_options import TrainOptions
from models import create_model
from util import html
from PIL import Image
import matplotlib.pyplot as plt
import copy
import torch
import torchvision
import torchvision.transforms as transforms
from util import util
import numpy as np
from mpl_toolkits.mplot3d import Axes3D



if __name__ == '__main__':
    np.random.seed(1)
    images_to_scan = 2  # Number of images to process
    root = '/Users/Will/Documents/Uni/MscEdinburgh/Diss/colorization-pytorch/'

    opt = TrainOptions().parse()
    opt.load_model = True
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.display_id = -1  # no visdom display
    opt.phase = 'val'

    opt.dataroot = os.path.join(opt.data_dir, opt.phase, "")
    print('Data Path: ', opt.dataroot)

    opt.serial_batches = True
    opt.aspect_ratio = 1.
    opt.loadSize = 250

    dataset = torchvision.datasets.ImageFolder(opt.dataroot,
                                               transform=transforms.Compose([
                                                   transforms.Resize((opt.loadSize, opt.loadSize)),
                                                   transforms.ToTensor()]))
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=not opt.serial_batches)

    for i, data_raw in enumerate(dataset_loader):
        if len(opt.gpu_ids) > 0 :
            data_raw[0] = data_raw[0].cuda()
        data_raw[0] = util.crop_mult(data_raw[0], mult=8)

        fig, ax = plt.subplots(figsize=(5, 5))
        fx_image = util.tensor2im(data_raw[0])
        c = ax.imshow(fx_image)
        plt.show()
        plt.close()

        lab_img = util.rgb2lab(data_raw[0], opt)
        just_ab = torch.zeros_like(lab_img)
        just_ab[:, 1:, :, :] = lab_img[:, 1:, :, :]
        rgb_img = util.lab2rgb(just_ab, opt)

        fig, ax = plt.subplots(figsize=(5, 5))
        fx_image = util.tensor2im(rgb_img)
        c = ax.imshow(fx_image)
        plt.show()
        plt.close()

        fig = plt.figure()
        jab = util.tensor2im(data_raw[0])
        # print(jab.shape)
        # print(jab[:, :, 1])

        flat1 = jab[:, :, 0].flatten()
        flat2 = jab[:, :, 1].flatten()
        flat3 = jab[:, :, 2].flatten()

        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(flat1, flat2, flat3)
        # # ax.set_xlabel('Batch Size')
        # # ax.set_ylabel('Number of GPUs')
        # # ax.set_zlabel('Loss After 5 Epochs')
        # plt.show()

        # print(type(flat2))
        # print(gg.shape)

        h = plt.hist(flat1, bins=100)
        ss1 = np.zeros(len(flat1))
        for j, s in enumerate(flat1):
            armin = np.argmin(np.abs(h[1] - s))-1
            ss1[j] = h[0][armin]
        plt.close()

        h = plt.hist(flat2, bins=100)
        ss2 = np.zeros(len(flat2))
        for j, s in enumerate(flat2):
            armin = np.argmin(np.abs(h[1] - s))-1
            ss2[j] = h[0][armin]
        plt.close()

        h = plt.hist(flat3, bins=100)
        ss3 = np.zeros(len(flat3))
        for j, s in enumerate(flat3):
            armin = np.argmin(np.abs(h[1] - s))-1
            ss3[j] = h[0][armin]
        plt.close()

        ss1 = ss1/np.max(ss1)
        ss2 = ss2 / np.max(ss2)
        ss3 = ss3 / np.max(ss3)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(flat1, flat2, flat3, s=ss1*ss2*ss3)
        # ax.set_xlabel('Batch Size')
        # ax.set_ylabel('Number of GPUs')
        # ax.set_zlabel('Loss After 5 Epochs')
        plt.show()


        # np.round()

        # fig = plt.figure(figsize=(10, 5))
        # ax = fig.add_subplot(111)
        # ax.scatter(jab[:, :, 1].flatten(), jab[:, :, 2].flatten())
        # # ax.scatter(batch_size_arr[train_smalls], times[train_smalls], label='Train Small')
        # # ax.set_xlabel('Batch Size')
        # # ax.set_ylabel('Mean Epoch Time Taken')
        # # ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        # # plt.subplots_adjust(right=0.7)
        # plt.show()
        # # plt.savefig(self.mpl_name, bbox_inches="tight", dpi=400)
        # plt.close(fig)
        # #
