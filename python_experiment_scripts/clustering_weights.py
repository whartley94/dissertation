
import os
from options.train_options import TrainOptions
from models import create_model
from util import html
from PIL import Image
import matplotlib.pyplot as plt
from skimage import measure
import matplotlib
import copy
import time
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


def k_means_vs_psnr(just_ab_smoothed_asab, l_tensor):
    # just_ab_smoothed_asab_flat, just_ab_smoothed_asab, lab_tensor
    just_ab_image_array = np.asarray(np.transpose(just_ab_smoothed_asab, (1, 2, 0)))
    # print(just_ab_smoothed_asab.shape)
    ab_X1 = just_ab_image_array.reshape(just_ab_image_array.shape[0] *
                                                               just_ab_image_array.shape[1],
                                                               just_ab_image_array.shape[2])
    # print('Go')
    n_clusterss = 25
    psnrs = []
    ks = []
    times = []

    for n_clusters in range(1, n_clusterss):

        # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(ab_X1)
        # # centers = [kmeans.cluster_centers_[i] for i in kmeans.labels_]
        # # centers = np.asarray(centers).astype(int)
        # labels_mx = kmeans.labels_.reshape(just_ab_image_array[:, :, 0].shape)
        # # centers = centers.reshape(just_ab_image_array.shape)

        t0 = time.time()
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(ab_X1)
        t1 = time.time()
        times.append(t1 - t0)

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

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(14, 5))
        c = ax1.imshow(labels_mx2)
        ax2.imshow(center_lab_img)
        ax3.imshow(rgb_image_array_orig)
        # ax4.imshow(just_ab_image_array)
        ax4.plot(ks, psnrs)
        ax5.plot(ks, times)
        ax1.set_title('K Means with ' + str(n_clusters) + ' Clusters')
        ax2.set_title('Concat')
        ax3.set_title('Original Image')
        ax4.set_title('PSNR vs Num Clusters')
        ax4.set_xlabel('Num Cluster')
        ax4.set_ylabel('PSNR')
        ax5.set_title('Time Taken vs Num Clusters')
        ax5.set_xlabel('Num Cluster')
        ax5.set_ylabel('Time Taken')
        plt.tight_layout()
        if n_clusters < n_clusterss - 1:
            plt.show(block=False)
            plt.pause(.1)

        else:
            plt.show()
        plt.savefig('k_kmeans_results/' +str(n_clusters) + '_clusters.png')
        plt.close()


def zhang_bins(just_ab_smoothed_asab, l_tensor, i, vis=True):
    # print('Go')
    n_clusterss = 25
    psnrs = []
    ks = []
    times = []
    # print(just_ab_smoothed_asab.shape)
    just_ab_smoothed_asab = torch.reshape(just_ab_smoothed_asab, (1, 2, 250, 250))
    encoded_ab = util.encode_ab_ind(just_ab_smoothed_asab, opt)
    # encoded_ab = encoded_ab
    # encoded_ab_flat = np.asarray(encoded_ab).reshape(250*250, 1)
    encoded_img = encoded_ab.reshape(250, 250)
    decoded_ab = util.my_decode_ind_ab(encoded_ab, opt)

    center_lab = np.zeros((250, 250, 3))
    center_lab[:, :, 1] = decoded_ab[0, 0, :, :]
    center_lab[:, :, 2] = decoded_ab[0, 1, :, :]
    center_lab[:, :, 0] = l_tensor[0, 0, :, :]
    center_lab_convert = util.im2tensor(center_lab, 'unknown')
    center_lab_convert = util.lab2rgb(center_lab_convert, opt)
    center_lab_img = util.tensor2im(center_lab_convert)

    center_lab = np.zeros((250, 250, 3))
    center_lab[:, :, 1] = decoded_ab[0, 0, :, :]
    center_lab[:, :, 2] = decoded_ab[0, 1, :, :]
    # center_lab[:, :, 0] = l_tensor[0, 0, :, :]
    center_lab_convert = util.im2tensor(center_lab, 'unknown')
    center_lab_convert = util.lab2rgb(center_lab_convert, opt)
    center_lab_img2 = util.tensor2im(center_lab_convert)

    center_lab = np.zeros((250, 250, 3))
    # center_lab[:, :, 1] = decoded_ab[0, 0, :, :]
    # center_lab[:, :, 2] = decoded_ab[0, 1, :, :]
    center_lab[:, :, :] = np.transpose(l_tensor[0], (1, 2, 0))
    center_lab_convert = util.im2tensor(center_lab, 'unknown')
    center_lab_convert = util.lab2rgb(center_lab_convert, opt)
    center_lab_img3 = util.tensor2im(center_lab_convert)

    if vis:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(14, 5))
        ax1.imshow(encoded_img)
        ax2.imshow(center_lab_img2)
        ax3.imshow(center_lab_img)
        ax4.imshow(center_lab_img3)
        ax1.set_title('Colour Bin Indexes')
        ax2.set_title('Colour Bins Coloured')
        ax3.set_title('Colour Bins with Original L')
        ax4.set_title('Original Image')

        plt.show()
        plt.tight_layout()
        # plt.savefig('bin_results/example_' + str(i) + '.png')
        plt.close()

    return encoded_ab, center_lab_img2, center_lab_img


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


def zhang_bin_box_finder(binned_tensor, rgb_image_array, ab_bins_coloured):
    image_shape = rgb_image_array.shape[:2]
    H, W, C = rgb_image_array.shape
    h = np.random.randint(H)
    w = np.random.randint(W)
    binned_image = np.asarray(binned_tensor[0, 0, :, :])
    mean_threshold = 0
    center_pixel = int(binned_tensor[0, 0, h:h + 1, w:w + 1])
    print('Cp', center_pixel)
    h1_biggle = 0
    h2_biggle = 0
    w1_biggle = 0
    w2_biggle = 0
    left = True
    right = True
    up = True
    down = True
    h1 = np.clip(h, 0, H)
    h2 = np.clip(h+1, 0, H)
    w1 = np.clip(w, 0, W)
    w2 = np.clip(w+1, 0, W)
    # print(h1, h2, w1, w2)
    max_it = np.max([H, W])
    for it in range(0, max_it, 1):
        h1_old = h1
        h2_old = h2
        w1_old = w1
        w2_old = w2
        if left:
            h1_trial = np.clip(h - it, 0, H)
            pixels = np.asarray(binned_tensor[0, 0, h1_trial:h2_old, w1_old:w2_old]).astype(int)
            abs_pixels = np.abs(pixels - center_pixel)
            mean = np.mean(abs_pixels)
            if mean > mean_threshold:
                left = False
                h1_biggle = h1_trial
            else:
                h1 = h1_trial
        if right:
            h2_trial = np.clip(h + 1 + it, 0, H)
            pixels = np.asarray(binned_tensor[0, 0, h1_old:h2_trial, w1_old:w2_old]).astype(int)
            abs_pixels = np.abs(pixels - center_pixel)
            mean = np.mean(abs_pixels)
            if mean > mean_threshold:
                right = False
                h2_biggle = h2_trial
            else:
                h2 = h2_trial
        if up:
            w1_trial = np.clip(w - it, 0, W)
            pixels = np.asarray(binned_tensor[0, 0, h1_old:h2_old, w1_trial:w2_old]).astype(int)
            abs_pixels = np.abs(pixels - center_pixel)
            mean = np.mean(abs_pixels)
            if mean > mean_threshold:
                up = False
                w1_biggle = w1_trial
            else:
                w1 = w1_trial
        if down:
            w2_trial = np.clip(w + 1 + it, 0, W)
            pixels = np.asarray(binned_tensor[0, 0, h1_old:h2_old, w1_old:w2_trial]).astype(int)
            abs_pixels = np.abs(pixels - center_pixel)
            mean = np.mean(abs_pixels)
            if mean > mean_threshold:
                down = False
                w2_biggle = w2_trial
            else:
                w2 = w2_trial

        if not left and not right and not up and not down:
            break

        if it == max_it-1:
            # if h1_biggle == 0:
            #     h1_biggle = it
            if h2_biggle == 0:
                h2_biggle = it
            # if w1_biggle == 0:
            #     w1_biggle = it
            if w2_biggle == 0:
                w2_biggle = it
            # print(h1_biggle,h2_biggle,w1_biggle,w2_biggle)

        # pixels = np.asarray(binned_tensor[0, 0, h1:h2, w1:w2]).astype(int)
        # abs_pixels = np.abs(pixels - center_pixel)
        # mean = np.mean(abs_pixels)
        # means.append(mean)
        #
        # if check_biggle:
        #     if mean > 0:
        #         biggle = it
        #         check_biggle = False
        #         h1_biggle = h1
        #         h2_biggle = h2
        #         w1_biggle = w1
        #         w2_biggle = w2
        # pixels2 = rgb_image_array[w-1:w+2, h-1:h+2, :]
        # variances.append(np.var(np.var(pixels, axis=0),axis=0))
        # variances.append(np.var(pixels))

    # print(biggle)
    pixels_done = rgb_image_array[h1_biggle:h2_biggle, w1_biggle:w2_biggle, :]
    pixels = copy.deepcopy(rgb_image_array)
    pixels = util.draw_square_twos(pixels, h1_biggle, h2_biggle, w1_biggle, w2_biggle, 1, opt)
    # pixels = pixels[h1:h2, w1:w2, :]
    #
    # pixels_rgb = copy.deepcopy(rgb_image_array_orig)
    # pixels_rgb = util.draw_square_twos(pixels_rgb, h1_biggle, h2_biggle, w1_biggle, w2_biggle, 0, opt)
    # pixels_rgb = pixels_rgb[h1:h2, w1:w2, :]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 6))
    fx_image = util.tensor2im(rgb_image_array_orig)
    c = ax1.imshow(fx_image)
    ax1.scatter(w, h)
    ax2.imshow(ab_bins_coloured)
    ax3.imshow(pixels_done)
    ax4.imshow(pixels)
    ax1.set_title('Point Selected')
    ax2.set_title('Binned AB channels')
    ax3.set_title('Pixels Selected (RGB)')
    ax4.set_title('Final Bounding Box')
    plt.show()
    plt.close()


def mask_off(binned_array, current_h, current_w, center_pixel, mask):
    current_h = np.clip(current_h, 0, mask.shape[0]-1)
    current_w = np.clip(current_w, 0, mask.shape[1]-1)
    diff = binned_array[current_h, current_w] - center_pixel
    if diff == 0:
        neighbour_mask = mask[current_h - 1:current_h + 2, current_w - 1:current_w + 2]
        sum_neighbor = np.sum(neighbour_mask)
        if sum_neighbor > 0:
            mask[current_h, current_w] = 1


def zhang_bin_area_spiraler(binned_tensor, rgb_image_array, ab_bins_coloured):
    image_shape = rgb_image_array.shape[:2]
    H, W, C = rgb_image_array.shape
    indexes = np.mgrid[0:H, 0:W]

    global h
    h = np.random.randint(H)
    global w
    w = np.random.randint(W)
    binned_image = np.asarray(binned_tensor[0, 0, :, :])
    mean_threshold = 0
    center_pixel = int(binned_tensor[0, 0, h:h + 1, w:w + 1])
    print('Cp', center_pixel)

    global mask
    mask = np.zeros(binned_image.shape)
    mask[h, w] = 1
    binned_array = binned_image.astype(int)

    go = True
    move = 2
    current_h = h
    current_w = w

    while go:
        mask_old = copy.deepcopy(mask)
        # Move To Next Ring
        current_h -= 1
        current_w -= 1
        mask_off(binned_array, current_h, current_w, center_pixel, mask)

        # Right
        for m in range(move):
            current_w += 1
            mask_off(binned_array, current_h, current_w, center_pixel, mask)
        # Down
        for m in range(move):
            current_h += 1
            mask_off(binned_array, current_h, current_w, center_pixel, mask)
        # Left
        for m in range(move):
            current_w -= 1
            mask_off(binned_array, current_h, current_w, center_pixel, mask)
        # Up
        for m in range(move):
            current_h -= 1
            mask_off(binned_array, current_h, current_w, center_pixel, mask)

        move += 2
        if np.array_equal(mask_old, mask):
            plot_spiral()
            break


def plot_spiral():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 6))
    fx_image = util.tensor2im(rgb_image_array_orig)
    c = ax1.imshow(fx_image)
    ax1.scatter(w, h)
    ax2.imshow(ab_bins_coloured)
    ax3.imshow(mask)
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


def points_in_circle_np(radius, x0=0, y0=0, ):
    x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
    y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
    x, y = np.where((x_[:, np.newaxis] - x0) ** 2 + (y_ - y0) ** 2 <= radius ** 2)
    # x, y = np.where((np.hypot((x_-x0)[:,np.newaxis], y_-y0)<= radius)) # alternative implementation
    for x, y in zip(x_[x], y_[y]):
        yield x, y

def my_averagge_cluster_dbscan(lab_orig, means):
    lab_orig_flat = np.asarray(lab_orig[0])
    lab_orig_flat = lab_orig_flat.reshape(3, 250*250)
    lab_orig_flat = np.transpose(lab_orig_flat, (1, 0))
    # lab_orig_flat = np.asarray(lab_orig[0].reshape(250*250, 3))
    n_clusters_ = len(set(means.labels_))
    clusters = [lab_orig_flat[means.labels_ == m] for m in range(n_clusters_)]
    cluster_means = [np.mean(cluster, axis=0) for cluster in clusters]
    # print(means.labels_)
    cluster_means = np.asarray(cluster_means)
    # print(len(cluster_means))
    # print(n_clusters_)
    flat_out = [cluster_means[ind+1] for ind in means.labels_]
    # print(flat_out)
    # print(flat_out)
    # print('Flat', flat_out)

    flat_out = np.asarray(flat_out)
    # print(flat_out.shape)
    out = flat_out.reshape(250, 250, 3)
    out = np.transpose(out, (2, 0, 1))
    # print(out.shape)
    # print(lab_orig.shape)
    # out[:, :, 0] = np.asarray(lab_orig[0,0,:,:])
    # out[:, :, 1] = np.asarray(lab_orig[0,1,:,:])
    # out[:, :, 2] = np.asarray(lab_orig[0,2,:,:])
    # out[:, :, 1:] = 0
    # out = lab_orig
    out_think = torch.zeros_like(lab_orig)
    out_think[0, 0, :, :] = torch.tensor(np.asarray(lab_orig[0,0,:,:]))
    out_think[0, 1:, :, :] = torch.tensor(out[1:, :, :])
    # out_tensor = util.im2tensor(out)
    out_rgb = util.lab2rgb(out_think, opt)
    out_im = util.tensor2im(out_rgb)
    return out_im


def dbscan_encoded_indexed(encoded, encoded_coloured, rgb_image_array_orig, lab_orig, vis=True):
    encoded_img = np.asarray(encoded[0, :, :, :])
    encoded_flat = encoded_img.flatten()
    # uniques = np.unique(encoded_flat)
    # for unique in uniques:
    #     rand = np.random.randint(-100, 100)
    #     print(rand)
    #     encoded_img[encoded_img==unique] += rand
    # print(encoded_img)
    indexes = np.mgrid[0:encoded_img.shape[1], 0:encoded_img.shape[2]]
    # indexes = indexes/30
    both = np.concatenate((encoded_img, indexes), axis=0)
    both = np.transpose(both, (1, 2, 0))
    both_flat = both.reshape(both.shape[0]* both.shape[1], both.shape[2])

    # print('Scaler')
    # scaler = StandardScaler()
    both_scaled = sklearn.preprocessing.scale(both_flat, axis=0)
    scale_rescale = 20
    both_scaled[:, 1] = both_scaled[:, 1] * scale_rescale
    both_scaled[:, 2] = both_scaled[:, 2] * scale_rescale
    both_scaled[:, 0] = both_scaled[:, 0] * 6
    # ab_X_indexed_flat = sklearn.preprocessing.scale(ab_X_indexed_flat, axis=0)
    # ab_X_indexed_flat[:, 0:2]
    # scaler.fit(ab_X)
    # ab_X = scaler.transform(ab_X)
    # print('Scaler Done')

    # print('Starting')
    eps = .5
    min_samples = 5
    means = DBSCAN(eps=eps, min_samples=min_samples).fit(both_scaled)
    # print('Done')
    # centers = [means.core_sample_indices_[i] for i in means.labels_]
    # centers = np.asarray(centers).astype(int)
    labels_mx = means.labels_.reshape(both[:,:,0].shape)

    out_im = my_averagge_cluster_dbscan(lab_orig, means)


    if vis:
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(14, 5))
        # fx_image = util.tensor2im(rgb_img)
        cmap = matplotlib.colors.ListedColormap(np.random.rand(256, 3))
        c = ax1.imshow(labels_mx, cmap=cmap)
        # ax2.imshow(centers2[:, :, 2:])
        # ax2.imshow(labels_mx2)
        # ax2.imshow(labels_mx2, cmap=cmap)
        ax2.imshow(util.tensor2im(rgb_image_array_orig))
        ax3.imshow(encoded_img[0])
        ax4.imshow(encoded_coloured)
        ax5.imshow(out_im)
        plt.show()

    return labels_mx


def bins_scimage_group(encoded, encoded_coloured, rgb_image_array_orig, lab_orig, concat_lab_bins, vis=True):
    encoded_np = np.asarray(encoded[0, 0, :, :]).astype(int)
    img_labeled, n_labeled = measure.label(encoded_np, connectivity=1, return_num=True)
    t = [len(np.unique(encoded_coloured[img_labeled==i], axis=0)) for i in range(1, n_labeled)]
    # t = [encoded_coloured[img_labeled==i] for i in range(n_labeled)]
    # print(np.unique(t))
    a = np.asarray(lab_orig[0, 1, :, :])
    b = np.asarray(lab_orig[0, 2, :, :])
    avga = [np.mean(a[img_labeled == label]) for label in range(1, n_labeled)]
    avga = np.asarray(avga)
    avgb = [np.mean(b[img_labeled == label]) for label in range(1, n_labeled)]

    flat_labels = img_labeled.flatten()

    avg_flat_a = [avga[(ix-2)] for ix in flat_labels]
    avg_flat_a = np.asarray(avg_flat_a)
    avg_flat_b = [avgb[(ix-2)] for ix in flat_labels]
    avg_flat_b = np.asarray(avg_flat_b)

    avg_a = avg_flat_a.reshape(250,250)
    avg_b = avg_flat_b.reshape(250,250)

    img = torch.zeros_like(lab_orig)
    img[0, 0, :, :] = lab_orig[0, 0, :, :]
    img[0, 1, :, :] = torch.tensor(avg_a)
    img[0, 2, :, :] = torch.tensor(avg_b)
    rgb_img = util.lab2rgb(img, opt)
    rgb_img = util.tensor2im(rgb_img)


    if vis:
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(14, 5))
        # fx_image = util.tensor2im(rgb_img)
        cmap = matplotlib.colors.ListedColormap(np.random.rand(256, 3))
        c = ax1.imshow(img_labeled, cmap=cmap)
        # ax2.imshow(centers2[:, :, 2:])
        # ax2.imshow(labels_mx2)
        # ax2.imshow(labels_mx2, cmap=cmap)
        ax2.imshow(util.tensor2im(rgb_image_array_orig))
        ax3.imshow(concat_lab_bins)
        ax4.imshow(rgb_img)
        ax5.imshow(encoded_coloured)
        # ax5.imshow(out_im)
        ax1.set_title('Clustered')
        ax2.set_title('Original Image')
        ax3.set_title('Concatenate Binned ab Colours')
        ax4.set_title('Concatenate Cluster Average ab')
        ax5.set_title('Binned Colours')
        plt.tight_layout()
        plt.show()


def ab_scimage_group(encoded, encoded_coloured, rgb_image_array_orig, lab_orig, concat_lab_bins, vis=True):
    np_ab = np.asarray(lab_orig[0, 1:, :, :])
    # lab_orig[0, 1:, :, :]
    A = 2 * opt.ab_max
    data_q = np_ab[0,:,:]*A + np_ab[1,:,:]
    img_labeled, n_labeled = measure.label(data_q, connectivity=2, return_num=True)
    print(img_labeled.shape)
    # t = [len(np.unique(encoded_coloured[img_labeled==i], axis=0)) for i in range(1, n_labeled)]
    # # t = [encoded_coloured[img_labeled==i] for i in range(n_labeled)]
    # # print(np.unique(t))
    # a = np.asarray(lab_orig[0, 1, :, :])
    # b = np.asarray(lab_orig[0, 2, :, :])
    # avga = [np.mean(a[img_labeled == label]) for label in range(1, n_labeled)]
    # avga = np.asarray(avga)
    # avgb = [np.mean(b[img_labeled == label]) for label in range(1, n_labeled)]
    #
    # flat_labels = img_labeled.flatten()
    #
    # avg_flat_a = [avga[(ix-2)] for ix in flat_labels]
    # avg_flat_a = np.asarray(avg_flat_a)
    # avg_flat_b = [avgb[(ix-2)] for ix in flat_labels]
    # avg_flat_b = np.asarray(avg_flat_b)
    #
    # avg_a = avg_flat_a.reshape(250,250)
    # avg_b = avg_flat_b.reshape(250,250)
    #
    # img = torch.zeros_like(lab_orig)
    # img[0, 0, :, :] = lab_orig[0, 0, :, :]
    # img[0, 1, :, :] = torch.tensor(avg_a)
    # img[0, 2, :, :] = torch.tensor(avg_b)
    # rgb_img = util.lab2rgb(img, opt)
    # rgb_img = util.tensor2im(rgb_img)


    if vis:
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(14, 5))
        # fx_image = util.tensor2im(rgb_img)
        cmap = matplotlib.colors.ListedColormap(np.random.rand(256, 3))
        c = ax1.imshow(img_labeled, cmap=cmap)
        # ax2.imshow(centers2[:, :, 2:])
        # ax2.imshow(labels_mx2)
        # ax2.imshow(labels_mx2, cmap=cmap)
        ax2.imshow(util.tensor2im(rgb_image_array_orig))
        ax3.imshow(concat_lab_bins)
        # ax4.imshow(rgb_img)
        ax5.imshow(encoded_coloured)
        # ax5.imshow(out_im)
        ax1.set_title('Clustered')
        ax2.set_title('Original Image')
        ax3.set_title('Concatenate Binned ab Colours')
        ax4.set_title('Concatenate Cluster Average ab')
        ax5.set_title('Binned Colours')
        plt.tight_layout()
        plt.show()

def extract_lab_channels(data, opt):
    #  Convert to Lab, take ab, convert back to RGB
    full_lab_tensor = util.rgb2lab(data, opt)
    just_ab = torch.zeros_like(full_lab_tensor)
    just_ab[:, 1:, :, :] = full_lab_tensor[:, 1:, :, :]
    just_ab_asrgb = util.lab2rgb(just_ab, opt)
    return just_ab, just_ab_asrgb, full_lab_tensor


def apply_smoothing(just_ab_asrgb):
    # Kernel smoothing
    kernel = np.ones((5, 5), np.float32) / 25
    just_ab_asrgb_im = util.tensor2im(just_ab_asrgb)
    just_ab_asrgb_im_smoothed = np.asarray(cv.filter2D(just_ab_asrgb_im, -1, kernel)).astype(int)
    just_ab_smoothed_asab_tensor = util.im2tensor(just_ab_asrgb_im_smoothed)
    just_ab_smoothed_asab = util.rgb2lab(just_ab_smoothed_asab_tensor, opt)
    just_ab_smoothed_asab = just_ab_smoothed_asab[0, 1:, :, :]

    return just_ab_smoothed_asab


def try_smoothing_just_lab():
    plt.imshow(just_ab_asrgb_array_smoothed)
    plt.show()

    plt.imshow(just_ab_asrgb_array_orig)
    plt.show()

    just_ab_2 = np.asarray(just_ab[0, 1:, :, :])
    print(just_ab_2.shape)
    print(just_ab_2)
    g = np.asarray(cv.filter2D(just_ab_2*1000, -1, kernel)).astype(int)
    print(g)
    print(g.shape)
    g = torch.tensor(g/1000)
    print(g.shape)
    just_ab_smoothed_ting = torch.zeros_like(lab_tensor)
    print(just_ab_smoothed_ting.shape)
    just_ab_smoothed_ting[0, 1:, :, :] = g
    rgbit = util.lab2rgb(just_ab_smoothed_ting, opt)
    # plt.imshow(util.tensor2im(rgbit))
    # plt.show()


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
        # print('Start', i)
        just_ab, just_ab_asrgb, full_lab_tensor = extract_lab_channels(data_raw[0], opt)
        # print(just_ab_asrgb.shape)
        just_ab_smoothed_asab = apply_smoothing(just_ab_asrgb)

        ab_bins, ab_bins_coloured, concat_lab_bins = zhang_bins(just_ab_smoothed_asab, full_lab_tensor, opt, False)
        # zhang_bin_box_finder(ab_bins, rgb_image_array_orig, ab_bins_coloured)
        # zhang_bin_area_spiraler(ab_bins, rgb_image_array_orig, ab_bins_coloured)

        # labels_mx = dbscan_encoded_indexed(ab_bins, ab_bins_coloured, data_raw[0], full_lab_tensor, True)
        labels_mx_2 = bins_scimage_group(ab_bins, ab_bins_coloured, data_raw[0], full_lab_tensor, concat_lab_bins)
        # labels_mx_3 = ab_scimage_group(ab_bins, ab_bins_coloured, data_raw[0], full_lab_tensor, concat_lab_bins)



        # h = 55
        # w = 89
        # bin = labels_mx[h, w]
        # shared = len(labels_mx[labels_mx == bin])
        # portion = shared/labels_mx.size
        # print('Stop', i)



        # break





