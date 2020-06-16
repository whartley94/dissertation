
import os
from options.train_options import TrainOptions
from models import create_model
from util import html
from PIL import Image
import matplotlib.pyplot as plt
import copy
import cv2 as cv
import torch
import torchvision
import torchvision.transforms as transforms
from util import util
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import MeanShift
import numpy as np
import matplotlib.pyplot as plt
# Though the following import is not directly being used, it is required
# for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D

import sklearn
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import datasets


def cluster_plot():
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

def k_means_col():
    # print(my_X)
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(my_X)
    # print(len(kmeans.labels_))
    centers = [kmeans.cluster_centers_[i] for i in kmeans.labels_]
    centers = np.asarray(centers).astype(int)
    # print(centers.shape)
    labels_mx = kmeans.labels_.reshape(rgb_image_array_smoothed[:, :, 0].shape)
    centers = centers.reshape(rgb_image_array_smoothed.shape)
    # print(centers)
    # print(labels_mx.shape)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # fx_image = util.tensor2im(rgb_img)
    c = ax1.imshow(centers)
    ax2.imshow(rgb_image_array_smoothed)
    plt.show()
    plt.close()


def k_means_ab(ab_X, just_ab_image_array, image):
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(ab_X)
    # print(len(kmeans.labels_))
    centers = [kmeans.cluster_centers_[i] for i in kmeans.labels_]
    centers = np.asarray(centers).astype(int)
    # print(centers.shape)
    labels_mx = kmeans.labels_.reshape(just_ab_image_array[:, :, 0].shape)
    centers = centers.reshape(just_ab_image_array.shape)
    # print(centers)
    # print(labels_mx.shape)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    # fx_image = util.tensor2im(rgb_img)
    c = ax1.imshow(labels_mx)
    ax2.imshow(rgb_image_array_smoothed)
    ax3.imshow(image)
    plt.show()

def k_means_ab_indexed(ab_X1, ab_X2, just_ab_image_array):
    n_clusters = 15

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(ab_X1)
    # centers = [kmeans.cluster_centers_[i] for i in kmeans.labels_]
    # centers = np.asarray(centers).astype(int)
    labels_mx = kmeans.labels_.reshape(just_ab_image_array[:, :, 0].shape)
    # centers = centers.reshape(just_ab_image_array.shape)

    kmeans2 = KMeans(n_clusters=n_clusters, random_state=0).fit(ab_X2)
    # centers = [kmeans2.cluster_centers_[i] for i in kmeans2.labels_]
    # centers = np.asarray(centers).astype(int)
    labels_mx2 = kmeans2.labels_.reshape(just_ab_image_array[:, :, 0].shape)
    # centers2 = centers.reshape(ab_X_indexed.shape)


    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(14, 5))
    # fx_image = util.tensor2im(rgb_img)
    c = ax1.imshow(labels_mx)
    # ax2.imshow(centers2[:, :, 2:])
    ax2.imshow(labels_mx2)
    ax3.imshow(rgb_image_array_orig)
    ax4.imshow(just_ab_image_array)
    plt.show()

def k_means_vs_psnr(ab_X1, just_ab_image_array, l_tensor):
    print('Go')
    n_clusterss = 25
    psnrs = []
    ks = []
    for n_clusters in range(1, n_clusterss):

        # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(ab_X1)
        # # centers = [kmeans.cluster_centers_[i] for i in kmeans.labels_]
        # # centers = np.asarray(centers).astype(int)
        # labels_mx = kmeans.labels_.reshape(just_ab_image_array[:, :, 0].shape)
        # # centers = centers.reshape(just_ab_image_array.shape)

        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(ab_X1)
        centers = [kmeans.cluster_centers_[i] for i in kmeans.labels_]
        centers = np.asarray(centers)
        # centers = centers[:, 2:]
        labels_mx2 = kmeans.labels_.reshape(just_ab_image_array[:, :, 0].shape)
        centers = centers.reshape(just_ab_image_array[:, :, :].shape)
        center_lab = np.zeros((just_ab_image_array[:, :, :].shape[0], just_ab_image_array[:, :, :].shape[1], 3))
        center_lab[:, :, 1] = centers[:, :, 0]
        center_lab[:, :, 2] = centers[:, :, 1]
        center_lab[:, :, 0] = l_tensor[0, 0, :, :]

        center_lab_convert = util.im2tensor(center_lab, 'unknown')
        center_lab_convert = util.lab2rgb(center_lab_convert, opt)
        center_lab_img = util.tensor2im(center_lab_convert)

        psnrs.append(util.calculate_psnr_np(util.tensor2im(center_lab_convert), rgb_image_array_orig))
        ks.append(n_clusters)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(14, 5))
        c = ax1.imshow(labels_mx2)
        ax2.imshow(center_lab_img)
        ax3.imshow(rgb_image_array_orig)
        # ax4.imshow(just_ab_image_array)
        ax4.plot(ks, psnrs)
        plt.show()

def cluster_ab_indexed(ab_X, ab_X_indexed_flat, ab_X_indexed, just_ab_image_array):
    print('Scaler')
    # scaler = StandardScaler()
    ab_X = sklearn.preprocessing.scale(ab_X, axis=0)
    ab_X_indexed_flat = sklearn.preprocessing.scale(ab_X_indexed_flat, axis=0)
    # ab_X_indexed_flat[:, 0:2]
    # scaler.fit(ab_X)
    # ab_X = scaler.transform(ab_X)
    print('Scaler Done')
    print(np.var(ab_X, axis=0))
    print(ab_X.shape)
    print('Starting')
    eps = .05
    min_samples = 5
    means = DBSCAN(eps=eps, min_samples=min_samples).fit(ab_X)
    print('Done')
    # centers = [means.core_sample_indices_[i] for i in means.labels_]
    # centers = np.asarray(centers).astype(int)
    labels_mx = means.labels_.reshape(just_ab_image_array[:, :, 0].shape)
    # centers = centers.reshape(just_ab_image_array.shape)

    print('Starting')
    # eps = .12
    # min_samples = 40
    means2 = DBSCAN(eps=eps, min_samples=min_samples).fit(ab_X_indexed_flat)
    # centers = [means2.cluster_centers_[i] for i in means2.labels_]
    # centers = np.asarray(centers).astype(int)
    labels_mx2 = means2.labels_.reshape(just_ab_image_array[:, :, 0].shape)
    # centers2 = centers.reshape(ab_X_indexed.shape)
    print('Done')

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(14, 5))
    # fx_image = util.tensor2im(rgb_img)
    c = ax1.imshow(labels_mx)
    # ax2.imshow(centers2[:, :, 2:])
    ax2.imshow(labels_mx2)
    ax3.imshow(rgb_image_array_orig)
    ax4.imshow(just_ab_image_array)
    plt.show()

def to_X(ljab):
    flat1 = ljab[:, :, 0].flatten()
    flat2 = ljab[:, :, 1].flatten()
    flat3 = ljab[:, :, 2].flatten()
    my_X = np.zeros((len(flat1), 3))
    my_X[:, 0] = flat1
    my_X[:, 1] = flat2
    my_X[:, 2] = flat3
    return my_X


def plot_raw_data(data):
    fig, ax = plt.subplots(figsize=(5, 5))
    fx_image = util.tensor2im(data)
    c = ax.imshow(fx_image)
    plt.show()
    plt.close()

def lab_box_finder(just_ab_image_array,rgb_image_array):
    image_shape = just_ab_image_array.shape[:2]
    N, C, H, W = data_raw[0].shape
    h = np.random.randint(H)
    w = np.random.randint(W)

    diffs = []
    means = []
    center_pixel = just_ab_image_array[h:h + 1, w:w + 1, :][0][0]
    # print('Cp', center_pixel)
    biggle = 30
    h1_biggle = 0
    h2_biggle = 0
    w1_biggle = 0
    w2_biggle = 0
    check_biggle = True
    for it in range(0, 60, 1):
        h1 = np.clip(h - it, 0, just_ab_image_array.shape[1])
        h2 = np.clip(h + 1 + it, 0, just_ab_image_array.shape[1])
        w1 = np.clip(w - it, 0, just_ab_image_array.shape[0])
        w2 = np.clip(w + 1 + it, 0, just_ab_image_array.shape[0])
        pixels = just_ab_image_array[h1:h2, w1:w2, :]
        abs_pixels = np.abs(pixels - center_pixel)
        means.append(np.mean(abs_pixels))
        diff = np.linalg.norm(abs_pixels)
        diffs.append(diff / (it + 1))
        if check_biggle:
            if diff >= 40:
                biggle = it
                check_biggle = False
                h1_biggle = h1
                h2_biggle = h2
                w1_biggle = w1
                w2_biggle = w2
        # pixels2 = rgb_image_array[w-1:w+2, h-1:h+2, :]
        # variances.append(np.var(np.var(pixels, axis=0),axis=0))
        # variances.append(np.var(pixels))

    pixels_done = just_ab_image_array[h - biggle:h + 1 + biggle, w - biggle:w + 1 + biggle, :]
    pixels = copy.deepcopy(just_ab_image_array)
    pixels = util.draw_square_twos(pixels, h1_biggle, h2_biggle, w1_biggle, w2_biggle, 0, opt)
    pixels = pixels[h1:h2, w1:w2, :]

    pixels_rgb = copy.deepcopy(rgb_image_array_orig)
    pixels_rgb = util.draw_square_twos(pixels_rgb, h1_biggle, h2_biggle, w1_biggle, w2_biggle, 0, opt)
    pixels_rgb = pixels_rgb[h1:h2, w1:w2, :]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 6))
    fx_image = util.tensor2im(rgb_image_array_orig)
    c = ax1.imshow(fx_image)
    ax1.scatter(w, h)
    if pixels_done.shape[0] > 0 and pixels_done.shape[1] > 0:
        ax2.imshow(pixels_done)
    ax3.imshow(pixels)
    ax4.imshow(pixels_rgb)
    plt.show()
    plt.close()

def rgb_boxfinder(rgb_image_array):
    image_shape = rgb_image_array.shape[:2]
    N, C, H, W = data_raw[0].shape
    h = np.random.randint(H)
    w = np.random.randint(W)

    diffs = []
    means = []
    center_pixel = rgb_image_array[h:h + 1, w:w + 1, :][0][0]
    # print('Cp', center_pixel)
    biggle = 30
    h1_biggle = 0
    h2_biggle = 0
    w1_biggle = 0
    w2_biggle = 0
    check_biggle = True
    for it in range(0, 60, 1):
        h1 = np.clip(h - it, 0, rgb_image_array.shape[1])
        h2 = np.clip(h + 1 + it, 0, rgb_image_array.shape[1])
        w1 = np.clip(w - it, 0, rgb_image_array.shape[0])
        w2 = np.clip(w + 1 + it, 0, rgb_image_array.shape[0])
        pixels = rgb_image_array[h1:h2, w1:w2, :]
        abs_pixels = np.abs(pixels - center_pixel)
        means.append(np.mean(abs_pixels))
        diff = np.linalg.norm(abs_pixels)
        diffs.append(diff / (it + 1))
        if check_biggle:
            if diff >= 40:
                biggle = it
                check_biggle = False
                h1_biggle = h1
                h2_biggle = h2
                w1_biggle = w1
                w2_biggle = w2
        # pixels2 = rgb_image_array[w-1:w+2, h-1:h+2, :]
        # variances.append(np.var(np.var(pixels, axis=0),axis=0))
        # variances.append(np.var(pixels))

    pixels_done = rgb_image_array[h - biggle:h + 1 + biggle, w - biggle:w + 1 + biggle, :]
    pixels = copy.deepcopy(rgb_image_array)
    pixels = util.draw_square_twos(pixels, h1_biggle, h2_biggle, w1_biggle, w2_biggle, 0, opt)
    pixels = pixels[h1:h2, w1:w2, :]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 6))
    fx_image = util.tensor2im(rgb_image_array_orig)
    c = ax1.imshow(fx_image)
    ax1.scatter(w, h)
    ax2.imshow(pixels_done)
    ax3.imshow(pixels)
    plt.show()
    plt.close()

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

        # plot_raw_data(data_raw[0])

        #  Convert to Lab, take ab, convert back to RGB
        lab_tensor = util.rgb2lab(data_raw[0], opt)
        just_ab = torch.zeros_like(lab_tensor)
        just_ab[:, 1:, :, :] = lab_tensor[:, 1:, :, :]
        just_ab_asrgb = util.lab2rgb(just_ab, opt)
        # util.tensor2im(just_ab_asrgb)
        img = util.tensor2im(just_ab_asrgb)
        # plt.imshow(img)
        # plt.show()

        # Kernel smoothing
        kernel = np.ones((5, 5), np.float32) / 25

        rgb_image_array_orig = util.tensor2im(data_raw[0])
        rgb_image_array_smoothed = np.asarray(cv.filter2D(rgb_image_array_orig, -1, kernel)).astype(int)

        just_ab_asrgb_array_orig = util.tensor2im(just_ab_asrgb)
        just_ab_asrgb_array_smoothed = np.asarray(cv.filter2D(just_ab_asrgb_array_orig, -1, kernel)).astype(int)
        # plt.imshow(just_ab_asrgb_array_smoothed)
        # plt.show()


        just_ab_smoothed_asab_tensor = util.im2tensor(just_ab_asrgb_array_smoothed)
        # test = util.tensor2im(just_ab_smoothed_asab_tensor)
        # plt.imshow(test)
        # plt.show()
        just_ab_smoothed_asab = util.rgb2lab(just_ab_smoothed_asab_tensor, opt)
        # print(just_ab_smoothed_asab.shape)
        just_ab_smoothed_asab = just_ab_smoothed_asab[0, 1:, :, :]
        # just_ab_smoothed_asab = util.lab2rgb(just_ab_smoothed_asab, opt)
        # just_ab_smoothed_asab = util.tensor2im(just_ab_smoothed_asab)
        # plt.imshow(just_ab_smoothed_asab)
        # plt.show()



        #
        # lab_img = util.tensor2im(just_ab_asrgb)
        # lab_img_smoothed = np.asarray(cv.filter2D(lab_img, -1, kernel)).astype(int)
        # lab_tensor = util.rgb2lab(data_raw[0], opt)
        # just_ab_smoothed = lab_img_smoothed[:, :, 1:]
        # just_ab_smooth_flat = just_ab_smoothed.reshape(just_ab_smoothed.shape[0]*just_ab_smoothed.shape[1],
        #                                    just_ab_smoothed.shape[2])

        # Reshaping & Scaling & Concatenating with indexed
        # Generate & Rescale Indexes
        indexes = np.arange(just_ab_asrgb_array_smoothed.shape[0], just_ab_asrgb_array_smoothed.shape[1])
        indexes = np.mgrid[0:just_ab_asrgb_array_smoothed.shape[0], 0:just_ab_asrgb_array_smoothed.shape[1]].transpose()
        indexes = indexes/30

        # Reshape standard ab in rgb form
        ab_asrgb = just_ab_asrgb_array_smoothed.reshape(just_ab_asrgb_array_smoothed.shape[0] * just_ab_asrgb_array_smoothed.shape[1],
                                                        just_ab_asrgb_array_smoothed.shape[2])

        # Concat standard ab in rgb form with indexes
        ab_asrgb_indexed = np.concatenate((indexes, just_ab_asrgb_array_smoothed), axis=2)
        ab_asrgb_indexed_flat = ab_asrgb_indexed.reshape(ab_asrgb_indexed.shape[0] * ab_asrgb_indexed.shape[1],
                                                         ab_asrgb_indexed.shape[2])

        # Concat just ab in ab form with indexes

        # print(just_ab_smoothed_asab.shape)
        just_ab_smoothed_asab = np.asarray(np.transpose(just_ab_smoothed_asab, (1, 2, 0)))
        # print(just_ab_smoothed_asab.shape)
        just_ab_smoothed_asab_flat = just_ab_smoothed_asab.reshape(just_ab_smoothed_asab.shape[0] *
                                                                   just_ab_smoothed_asab.shape[1],
                                                                   just_ab_smoothed_asab.shape[2])
        # just_ab_smoothed_indexed = np.concatenate((indexes, just_ab_smoothed_asab), axis=2)
        # just_ab_smoothed_indexed_flat = just_ab_smoothed_indexed.reshape(just_ab_smoothed_indexed.shape[0] *
        #                                                          just_ab_smoothed_indexed.shape[1],
        #                                                          just_ab_smoothed_indexed.shape[2])



        # k_means_col()
        # k_means_ab(ab_X, just_ab_image_array, just_ab_image_array)
        # k_means_ab_indexed(just_ab_smooth_flat, ab_X_indexed_flat, just_ab_image_array)
        k_means_vs_psnr(just_ab_smoothed_asab_flat, just_ab_smoothed_asab, lab_tensor)
        # cluster_ab_indexed(just_ab_smooth_flat, ab_X_indexed_flat, ab_X_indexed, just_ab_image_array)

        # lab_box_finder(just_ab_image_array,rgb_image_array)

        # plot_raw_data(data_raw[0])





