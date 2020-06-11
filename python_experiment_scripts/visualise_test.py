
import os
from options.train_options import TrainOptions
from models import create_model
from util.visualizer import save_images
from util import html
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import copy

import string
import torch
import torchvision
import torchvision.transforms as transforms

from util import util
from IPython import embed
import numpy as np


if __name__ == '__main__':
    sample_ps = [1., .125, .03125]
    to_visualize = ['gray', 'hint', 'hint_ab', 'fake_entr', 'real', 'fake_reg', 'real_ab', 'fake_ab_reg', ]
    S = len(sample_ps)

    opt = TrainOptions().parse()
    opt.load_model = True
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.display_id = -1  # no visdom display
    opt.phase = 'val'


    # opt.dataroot = './dataset/ilsvrc2012/%s/' % opt.phase
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
    model.setup(opt) #Loads up model named, in the checkpoints dir pointed to
    model.eval()

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    # statistics
    psnrs = np.zeros((opt.how_many, S))
    entrs = np.zeros((opt.how_many, S))

    for i, data_raw in enumerate(dataset_loader):
        if len(opt.gpu_ids) > 0 :
            data_raw[0] = data_raw[0].cuda()
        data_raw[0] = util.crop_mult(data_raw[0], mult=8)

        # with no points
        # for (pp, sample_p) in enumerate(sample_ps):
        #     img_path = [str.replace('%08d_%.3f' % (i, sample_p), '.', 'p')]
        # data = util.get_colorization_data(data_raw, opt, ab_thresh=0., p=.03125)
        data = util.get_colorization_data(data_raw, opt, ab_thresh=0., num_points=0)


        # point_a_loc = [60, 60]
        point_a_col = [-50, 50] #Do this fixed for now, possibly sample it randomly too.?
        print('A Col Assigned: ', np.asarray(point_a_col).astype(float)/opt.ab_norm)

        n = 3
        lab_colours = np.zeros((n, 3))
        rgb_colours = np.zeros((n, 3))
        locations = np.zeros((n, 2)).astype(int)
        effect_mx = np.zeros((n,n))

        np.random.seed(0)
        P = 5
        N, C, H, W = data['B'].shape

        model.set_input(data)
        model.test(True)  # True means that losses will be computed
        visuals = util.get_subset_dict(model.get_current_visuals(), to_visualize)

        for loc in range(len(locations)):
            h = np.random.randint(H - P + 1)
            w = np.random.randint(W - P + 1)
            locations[loc, 0] = h
            locations[loc, 1] = w
            point = visuals['fake_reg'][:, :, h:h + P, w:w + P]
            lab_colours[loc] = util.mean_pixel(point, opt, True)
            rgb_colours[loc] = util.mean_pixel(point, opt, False)
        # print('Locations: ', locations)
        # print('Lab Colours:', lab_colours*opt.ab_norm)
        # print('Rgb Colours:', rgb_colours)

        # for k in to_visualize:
        #     print(k)
        #     real_im = util.tensor2im(visuals[k])
        #
        #     for l in range(n):
        #         real_im = util.draw_fill_square(real_im, locations[l, 0], locations[l, 1], P, rgb_colours[l])
        #
        #     image_pil = Image.fromarray(real_im)
        #     imgplot = plt.imshow(image_pil)
        #     plt.show()
        #     plt.close()

        for j in range(n):
            data_n = copy.deepcopy(data)
            data_n = util.add_color_patch(data_n, opt, P, [locations[j, 0], locations[j, 1]], point_a_col)

            model.set_input(data_n)
            model.test(True)  # True means that losses will be computed
            # print(data['B'][:,0,0,0])
            visuals_n = util.get_subset_dict(model.get_current_visuals(), to_visualize)

            # lab_colours = np.zeros((n, 3))
            rgb_colours_after = np.zeros((n, 3))

            for loc in range(n):
                h = locations[loc, 0]
                w = locations[loc, 1]
                point = visuals_n['fake_reg'][:, :, h:h + P, w:w + P]
                lab_colours_after = util.mean_pixel(point, opt, True)
                rgb_colours_after[loc] = util.mean_pixel(point, opt, False)
                effect_mx[j, loc]  = np.linalg.norm(rgb_colours[loc] - rgb_colours_after[loc])

            print(effect_mx)

            for k in to_visualize:
                print(k)
                real_im = util.tensor2im(visuals_n[k])

                # real_im = util.draw_fill_square(real_im, locations[j, 0], locations[j, 1], P, point_a_col)
                for l in range(n):
                    real_im = util.draw_square(real_im, locations[l, 0], locations[l, 1], P, rgb_colours_after[l])
                image_pil = Image.fromarray(real_im)
                imgplot = plt.imshow(image_pil)
                plt.show()
                plt.close()

        print(effect_mx)
        break


        #
        #     # N, C, H, W = data['B'].shape
        #     # print(H, W)
        #
        #     point_a = visuals['fake_reg'][:, :, ha:ha+P, wa:wa+P]
        #     mean_a = util.mean_pixel_lab(point_a, opt)
        #
        #     point_b = visuals['fake_reg'][:, :, hb:hb + P, wb:wb + P]
        #     mean_b = util.mean_pixel_lab(point_b, opt)
        #
        #     point_c = visuals['fake_reg'][:, :, hc:hc + P, wc:wc + P]
        #     mean_c = util.mean_pixel_lab(point_c, opt)
        #
        #     # print(visuals['fake_reg'][:,0,0,0])
        #     #
        #     mean_as.append(mean_a)
        #     mean_bs.append(mean_b)
        #     mean_cs.append(mean_c)

        #
        # mean_as = np.asarray(mean_as)
        # mean_bs = np.asarray(mean_bs)
        # mean_cs = np.asarray(mean_cs)
        # print(mean_as)
        # print(mean_bs)
        # print(mean_cs)
        # # break
        #
        # # to_visualize = ['gray', 'hint', 'hint_ab', 'fake_entr', 'real', 'fake_reg', 'real_ab', 'fake_ab_reg', ]
        # boxes = np.zeros(util.tensor2im(visuals['real']).shape)
        # boxes[hb:hb + P, wb:wb + P, :] = 200
        # boxes = util.tensor2im(boxes)
        #
        # for k in to_visualize:
        #     print(k)
        #     real_im = util.tensor2im(visuals[k])
        #     # real_im[hb: hb + P, wb: wb + P,:] = np.max(real_im)
        #     # print(np.max(real_im))
        #     real_im = util.draw_square(real_im, ha, wa, P, mean_as[1])
        #     print(mean_as)
        #     real_im = util.draw_square(real_im, hb, wb, P, mean_bs[1])
        #     real_im = util.draw_square(real_im, hc, wc, P, mean_cs[1])
        #
        #     image_pil = Image.fromarray(real_im)
        #     imgplot = plt.imshow(image_pil)
        #     # imgplotb = plt.imshow(boxes)
        #     plt.show()
        #     plt.close()
        #     # break
        # # break
