
import os
from options.train_options import TrainOptions
from models import create_model
from util import html
from PIL import Image
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import copy
import torch
import torchvision
import torchvision.transforms as transforms
from util import util
import numpy as np


if __name__ == '__main__':
    np.random.seed(1)
    # images_to_scan = 40  # Number of images to process
    images_to_scan = 4
    n = 8  # Number of points to sample
    P = 6  # Point sizes
    init_points = 10  # Number of points to use for 'before' image
    root = '/Users/Will/Documents/Uni/MscEdinburgh/Diss/colorization-pytorch/'
    name = root + 'resources/its' + str(images_to_scan) +\
             'n' + str(n) + 'p' + str(P) + 'ip' + str(init_points)
    npname = name + '.npy'

    if not os.path.isfile(npname) or True:

        show_images = False
        show_effect_mx = True
        save_figs = True
        to_visualize = ['gray', 'hint', 'hint_ab', 'fake_entr', 'real', 'fake_reg', 'real_ab', 'fake_ab_reg', ]
        to_display = 'fake_reg'
        which_channel = 'fake_reg'  # Which channel to scan for colour symmetry checks?

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

        dataset = torchvision.datasets.ImageFolder(opt.dataroot,
                                                   transform=transforms.Compose([
                                                       transforms.Resize((opt.loadSize, opt.loadSize)),
                                                       transforms.ToTensor()]))
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=not opt.serial_batches)

        model = create_model(opt)
        model.setup(opt)  # Loads up model named, in the checkpoints dir pointed to
        model.eval()

        all_e_diff = []

        gg = []

        for i, data_raw in enumerate(dataset_loader):
            if len(opt.gpu_ids) > 0 :
                data_raw[0] = data_raw[0].cuda()
            data_raw[0] = util.crop_mult(data_raw[0], mult=8)

            # Initialise data with some random number of hints.
            data = util.get_colorization_data(data_raw, opt, ab_thresh=0., num_points=init_points)

            ca = np.random.randint(-opt.ab_norm, opt.ab_norm)
            cb = np.random.randint(-opt.ab_norm, opt.ab_norm)
            point_a_col = [ca, cb]  # Do this fixed for now, possibly sample it randomly too.?
            # print('A Col Assigned: ', np.asarray(point_a_col).astype(float)/opt.ab_norm)

            lab_colours = np.zeros((n, 3))
            rgb_colours = np.zeros((n, 3))
            locations = np.zeros((n, 2)).astype(int)
            effect_mx = np.zeros((n, n))

            N, C, H, W = data['B'].shape

            model.set_input(data)
            model.test(True)  # True means that losses will be computed
            visuals = util.get_subset_dict(model.get_current_visuals(), to_visualize)

            for loc in range(len(locations)):
                h = np.random.randint(H - P + 1)
                w = np.random.randint(W - P + 1)
                locations[loc, 0] = h
                locations[loc, 1] = w
                point = visuals[which_channel][:, :, h:h + P, w:w + P]
                # print('point', point)
                lab_colours[loc] = util.mean_pixel(point, opt, True)
                rgb_colours[loc] = util.mean_pixel(point, opt, False)
            # print('Locations: ', locations)
            # print('Lab Colours:', lab_colours*opt.ab_norm)
            # print('Rgb Colours:', rgb_colours)

            # Scan over the points making each the source
            for j in range(n):
                data_n = copy.deepcopy(data)  # Need to deep copy the data otherwise we just add more and more colour
                data_n = util.add_color_patch(data_n, opt, P, [locations[j, 0], locations[j, 1]], point_a_col)
                model.set_input(data_n)
                model.test(True)  # True means that losses will be computed
                visuals_n = util.get_subset_dict(model.get_current_visuals(), to_visualize)

                # lab_colours = np.zeros((n, 3))
                rgb_colours_after = np.zeros((n, 3))

                # For each source, scan the other points and measure the change
                for loc in range(n):
                    h = locations[loc, 0]
                    w = locations[loc, 1]
                    point = visuals_n[which_channel][:, :, h:h + P, w:w + P]
                    # print('point', point)
                    lab_colours_after = util.mean_pixel(point, opt, True)
                    rgb_colours_after[loc] = util.mean_pixel(point, opt, False)
                    effect_mx[j, loc] = np.linalg.norm(rgb_colours[loc] - rgb_colours_after[loc])
                    # effect_mx[j, loc] = rgb_colours_after[loc][0]

                if show_images:
                    for k in to_visualize:
                        if k in to_display:
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                            fig.suptitle(k)

                            real_im = util.tensor2im(visuals[k])
                            real_im = util.draw_fill_square(real_im, locations[j, 0], locations[j, 1], P,
                                                            rgb_colours[j], 'White')
                            for l in range(n):
                                if l is not j:
                                    real_im = util.draw_fill_square(real_im, locations[l, 0], locations[l, 1], P,
                                                                    rgb_colours[l], 'Black')
                            image_pil = Image.fromarray(real_im)




                            ax1.imshow(image_pil)

                            real_im = util.tensor2im(visuals_n[k])
                            real_im = util.draw_fill_square(real_im, locations[j, 0], locations[j, 1], P,
                                                            rgb_colours_after[j], 'White')
                            for l in range(n):
                                if l is not j:
                                    real_im = util.draw_fill_square(real_im, locations[l, 0], locations[l, 1], P,
                                                                    rgb_colours_after[l], 'Black')
                            image_pil = Image.fromarray(real_im)
                            ax2.imshow(image_pil)

                            if i <= 2 and save_figs:
                                figname = root + 'python_experiment_scripts/symmetry_results/' + 'image' + str(i) + 'point' + str(j) + '.png'
                                plt.savefig(figname)
                            else:
                                plt.show()
                            plt.close()

                # ----------------
                # HERES THE PLOT OF JUST THE IMAGE WITH SQUARES HILIGHTING WHERE
                # -------------------
                save_image_with_squares = False
                if save_image_with_squares:
                    for k in to_visualize:
                        if k in to_display:
                            real_im = util.tensor2im(visuals[k])
                            real_im = util.draw_fill_square(real_im, locations[j, 0], locations[j, 1], P,
                                                            rgb_colours[j], 'Black')
                            for l in range(n):
                                if l is not j:
                                    real_im = util.draw_fill_square(real_im, locations[l, 0], locations[l, 1], P,
                                                                    rgb_colours[l], 'Black')
                            image_pil = Image.fromarray(real_im)

                            plt.imshow(image_pil)
                            # plt.savefig('/Users/Will/Documents/Uni/MscEdinburgh/Diss/InformaticsMScDissertationLatex/squares.pgf')
                            # plt.savefig('/Users/Will/Documents/Uni/MscEdinburgh/Diss/InformaticsMScDissertationLatex/squares.png', dpi=400)
                            plt.show()

                            real_im = util.tensor2im(visuals_n[k])
                            real_im = util.draw_fill_square(real_im, locations[j, 0], locations[j, 1], P,
                                                            rgb_colours_after[j], 'White')
                            for l in range(n):
                                if l is not j:
                                    real_im = util.draw_fill_square(real_im, locations[l, 0], locations[l, 1], P,
                                                                    rgb_colours_after[l], 'Black')
                            image_pil = Image.fromarray(real_im)
                            plt.imshow(image_pil)

                            if i <= 2 and save_figs:
                                figname = root + 'python_experiment_scripts/symmetry_results/' + 'image' + str(
                                    i) + 'point' + str(j) + '.png'
                                plt.savefig(
                                    '/Users/Will/Documents/Uni/MscEdinburgh/Diss/InformaticsMScDissertationLatex/squares' + str(
                                    i) +'.png', dpi=600)
                                plt.savefig(figname)
                            else:
                                plt.show()
                            plt.close()

            if show_effect_mx:
                # print(effect_mx)
                fig, ax = plt.subplots(figsize=(5, 5))
                # fx_image = Image.fromarray(effect_mx)
                for l in range(effect_mx.shape[0]):
                    effect_mx[l, l] = 0
                c = ax.imshow(effect_mx)
                cbar = fig.colorbar(c)
                cbar.set_label(r'\textbf{Relative Colour Change} $(D)$', rotation=270, labelpad=22, fontsize=16)
                # plt.tight_layout()
                if i <= 2:
                    # plt.savefig('/Users/Will/Documents/Uni/MscEdinburgh/Diss/InformaticsMScDissertationLatex/effmx' +str(i) +'.pgf')
                    plt.savefig(
                        '/Users/Will/Documents/Uni/MscEdinburgh/Diss/InformaticsMScDissertationLatex/effmx' + str(
                            i) + '.png', dpi=400)
                if i <= 2 and save_figs:
                    figname = root + 'python_experiment_scripts/symmetry_results/' + 'image' + str(i) + 'M' + '.png'
                    plt.savefig(figname)
                else:
                    plt.show()
                plt.close()

                # print(effect_mx)
                fig, ax = plt.subplots(figsize=(5, 5))
                # fx_image = Image.fromarray(effect_mx)
                c = ax.imshow(effect_mx - effect_mx.T)
                cbar = fig.colorbar(c)
                if i <= 2 and save_figs:
                    figname = root + 'python_experiment_scripts/symmetry_results/' + 'image' + str(i) + 'M-Mt' + '.png'
                    plt.savefig(figname)
                else:
                    plt.show()
                plt.close()

            for hor in range(effect_mx.shape[0]):
                for ver in range(effect_mx.shape[1]):
                    if hor != ver:
                        all_e_diff.append(effect_mx[hor, ver] - effect_mx[ver, hor])

            print('Progress: ', int((i/images_to_scan)*100), '%')
            if i >= images_to_scan:
                break

        np.save(npname, all_e_diff)

    else:
        print('Loading from npy')
        all_e_diff = np.load(npname)

    # lendip = len(all_e_diff)/35
    # gendip = int((lendip**2)/50)
    # print('Bins ', gendip)
    print(len(all_e_diff))

    plt.figure(figsize=(3.5,2.6))
    plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    plt.hist(all_e_diff, bins=85)
    plt.xlabel(r'\textbf{Error} $(E)$')
    # plt.xlabel('Error E')
    plt.ylabel(r'\textbf{Frequency}')
    plt.tight_layout()
    # plt.savefig('/Users/Will/Documents/Uni/MscEdinburgh/Diss/InformaticsMScDissertationLatex/histogram.pgf')

    plt.show()
