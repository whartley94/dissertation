import time
from options.train_options import TrainOptions
from models import create_model
from util.visualizer import Visualizer

import torch
import torchvision
import torchvision.transforms as transforms
import os

from util import util

if __name__ == '__main__':
    print('Python Starting Script')
    opt = TrainOptions().parse()

    # opt.dataroot = './dataset/SUN2012/%s/' % opt.phase
    opt.dataroot = os.path.join(opt.data_dir, opt.phase, "")
    print('Data Path: ', opt.dataroot)
    # opt.dataroot = '/home/s1843503/datasets/INetData/Torr/Tiny/%s/' % opt.phase


    if opt.name == 'Auto':
        name = 'Exp_'
        for i in opt.auto_names:
            app = ''
            for j in i.split('_'):
                app += j[0]
            if i == 'gpu_ids':
                bpp = str(len(vars(opt)[i])) + '_'
            else:
                if isinstance(vars(opt)[i], int):
                    bpp = str(int(vars(opt)[i])) + '_'
                else:
                    bpp = ''
                    for k in str(vars(opt)[i]).upper().split('_'):
                        bpp+= k[0]
                    bpp += '_'
            name += app + bpp
        name = name[:-1]

        if opt.load_model:
            opt.name = name
        else:
            if os.path.isdir(str(opt.checkpoints_dir) + '/' + name):
                num = 0
                while os.path.isdir(str(opt.checkpoints_dir) + '/' + name + '_It' + str(num)):
                    num += 1
                # print(str(opt.checkpoints_dir) + name + '_It' + str(num))
                opt.name = name + '_it' + str(num)
            else:
                opt.name = name
    print('Experiment Name: ', opt.name)


    dataset = torchvision.datasets.ImageFolder(opt.dataroot,
                                               transform=transforms.Compose([
                                                   transforms.RandomChoice([transforms.Resize(opt.loadSize, interpolation=1),
                                                                            transforms.Resize(opt.loadSize, interpolation=2),
                                                                            transforms.Resize(opt.loadSize, interpolation=3),
                                                                            transforms.Resize((opt.loadSize, opt.loadSize), interpolation=1),
                                                                            transforms.Resize((opt.loadSize, opt.loadSize), interpolation=2),
                                                                            transforms.Resize((opt.loadSize, opt.loadSize), interpolation=3)]),
                                                   transforms.RandomChoice([transforms.RandomResizedCrop(opt.fineSize, interpolation=1),
                                                                            transforms.RandomResizedCrop(opt.fineSize, interpolation=2),
                                                                            transforms.RandomResizedCrop(opt.fineSize, interpolation=3)]),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor()]))
                                                   # transforms.RandomChoice([transforms.ColorJitter(brightness=.05, contrast=.05, saturation=.05, hue=.05),
                                                   #                          transforms.ColorJitter(brightness=0, contrast=0, saturation=.05, hue=.1),
                                                   #                          transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0), ]),
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True,
                                                 num_workers=int(opt.num_threads))

    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    model.print_networks(not opt.invisible_network)

    visualizer = Visualizer(opt)
    total_steps = 0

    print('GPUs Available', torch.cuda.device_count())
    print('Checkpoint Location: ', opt.checkpoints_dir)

    go_time = time.time()
    max_time = .5 * 60 * 60
    max_time = 100
    time_since_go_time = time.time() - go_time
    print('TimeSinceGo', time_since_go_time)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay):
        print('StartingEpoch')
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        time_since_go_time = time.time() - go_time
        if time_since_go_time > max_time:
            print('Beaking Epoch For Max Time')
            break

        # for i, data in enumerate(dataset):
        for i, data_raw in enumerate(dataset_loader):
            time_since_go_time = time.time() - go_time
            # print('TimeSince Go Time', time_since_go_time)
            if time_since_go_time > max_time:
                print('Breaking Data For Max Time')
                break
            # print('I', i)
            # print('Data_raw ', data_raw)
            # print('Data_raw 0', data_raw[0])
            if len(opt.gpu_ids) > 0:
                data_raw[0] = data_raw[0].cuda()
            data = util.get_colorization_data(data_raw, opt, p=opt.sample_p)
            if opt.plot_data_gen:
                util.plot_data(data, opt)
            if(data is None):
                print('DataIsNone')
                break
                # continue

            iter_start_time = time.time()
            # print('Total Steps', total_steps)
            if total_steps % opt.print_freq == 0:
                # time to load data
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                # time to do forward&backward
                # t = (time.time() - iter_start_time) / opt.batch_size
                t = time.time() - iter_start_time
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    # embed()
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses, True,
                                                   opt.save_npy, opt.save_mpl)
                else:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses, False,
                                                   opt.save_npy, opt.save_mpl)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

    print('Times Up or done.. Save Model!')
    print(time_since_go_time)
    model.save_networks('latest')
    print('Model Saved')

