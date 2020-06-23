from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
from collections import OrderedDict
import cv2 as cv
from sklearn.cluster import DBSCAN
import sklearn
import matplotlib
from IPython import embed
import matplotlib.pyplot as plt

# Shortcut for getting mean lab value of an area, for visualise_test
def mean_pixel(point_a, opt, lab=True):
    if lab:
        point_a = rgb2lab(point_a, opt)
    point_a = point_a.cpu().float().numpy()
    point_a = np.clip(point_a, 0, 1)
    mean_a = np.mean(np.mean(point_a, axis=3), axis=2)[0]
    # if not lab:
    #     print(point_a)
    #     tr = np.transpose(point_a[0])
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 5))
    #     ax1.imshow(tr)
    #     tr2 = np.zeros(tr.shape)
    #     tr2 += mean_a
    #     ax2.imshow(tr2)
    #     ax2.set_title(mean_a)
    #     plt.show()
    #     plt.close()
    return mean_a


def draw_square(real_im, hb, wb, P, col):
    col = col*255
    for g in np.arange(hb, hb + P):
        real_im[g, wb, :] = col
        real_im[g, wb + P - 1 , :] = col
    for g in np.arange(wb, wb + P):
        real_im[hb, g, :] = col
        real_im[hb + P - 1, g, :] = col
    return real_im


def draw_square_twos(real_im, h1, h2, w1, w2, col, opt):
    col = col*255
    half_width = int((w2 - w1) / 2)
    half_height = int((h2 - h1) / 2)
    col2 = np.mean(real_im[h2-half_height, w2-half_width, :])
    if col2 > 127.5:
        col = 0
    if col2 <= 127.5:
        col = 255
    if h1 < 0:
        h1=0
    if h2 >= opt.loadSize:
        h2 = h2-1
    if w1 < 0:
        w1 = 0
    if w2 >= opt.loadSize:
        w2 = w2 - 1
    for g in np.arange(h1, h2+1):
        real_im[g, w1, :] = col
        real_im[g, w2, :] = col
    for g in np.arange(w1, w2+1):
        real_im[h1, g, :] = col
        real_im[h2, g, :] = col
    return real_im




def draw_fill_square(real_im, hb, wb, P, col, boarder=''):
    col = col*255
    real_im[hb:hb+P, wb:wb+P, :] = col
    if boarder == 'Black':
        real_im = draw_square(real_im, hb-1, wb-1, P+1, 255)
    if boarder == 'White':
        real_im = draw_square(real_im, hb - 1, wb - 1, P + 1, 3)
    return real_im

# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = np.clip((np.transpose(image_numpy, (1, 2, 0)) ),0, 1) * 255.0
    return image_numpy.astype(imtype)

# Converts a an image array (numpy) to tebsor
def im2tensor(input_image, type='rgb'):
    if isinstance(input_image, torch.Tensor):
        return input_image
    else:
        numpy_init = np.zeros((1, input_image.shape[2], input_image.shape[0], input_image.shape[1]))
        image_numpy = np.transpose(input_image, (2, 0, 1))
        # if type == 'unknown':
            # print(image_numpy)
        if type == 'rgb':
            image_numpy = np.clip(image_numpy, 0, 255)/255
        numpy_init[0, :, :, :] = image_numpy
        # image_tensor = torch.from_numpy(numpy_init)
        image_tensor = torch.tensor(numpy_init, dtype=torch.float32)
        # print(image_tensor.shape)
        # print(type(image_tensor))
        # print(image_tensor)
        return image_tensor
    # image_numpy = image_tensor[0].cpu().float().numpy()
    # if image_numpy.shape[0] == 1:
    #     image_numpy = np.tile(image_numpy, (3, 1, 1))
    # image_numpy = np.clip((np.transpose(image_numpy, (1, 2, 0)) ),0, 1) * 255.0
    # return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_subset_dict(in_dict,keys):
    if(len(keys)):
        subset = OrderedDict()
        for key in keys:
            subset[key] = in_dict[key]
    else:
        subset = in_dict
    return subset



# Color conversion code
def rgb2xyz(rgb): # rgb from [0,1]
    # xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
        # [0.212671, 0.715160, 0.072169],
        # [0.019334, 0.119193, 0.950227]])

    mask = (rgb > .04045).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (((rgb+.055)/1.055)**2.4)*mask + rgb/12.92*(1-mask)

    x = .412453*rgb[:,0,:,:]+.357580*rgb[:,1,:,:]+.180423*rgb[:,2,:,:]
    y = .212671*rgb[:,0,:,:]+.715160*rgb[:,1,:,:]+.072169*rgb[:,2,:,:]
    z = .019334*rgb[:,0,:,:]+.119193*rgb[:,1,:,:]+.950227*rgb[:,2,:,:]
    out = torch.cat((x[:,None,:,:],y[:,None,:,:],z[:,None,:,:]),dim=1)

    # if(torch.sum(torch.isnan(out))>0):
        # print('rgb2xyz')
        # embed()
    return out

def xyz2rgb(xyz):
    # array([[ 3.24048134, -1.53715152, -0.49853633],
    #        [-0.96925495,  1.87599   ,  0.04155593],
    #        [ 0.05564664, -0.20404134,  1.05731107]])

    r = 3.24048134*xyz[:,0,:,:]-1.53715152*xyz[:,1,:,:]-0.49853633*xyz[:,2,:,:]
    g = -0.96925495*xyz[:,0,:,:]+1.87599*xyz[:,1,:,:]+.04155593*xyz[:,2,:,:]
    b = .05564664*xyz[:,0,:,:]-.20404134*xyz[:,1,:,:]+1.05731107*xyz[:,2,:,:]

    rgb = torch.cat((r[:,None,:,:],g[:,None,:,:],b[:,None,:,:]),dim=1)
    rgb = torch.max(rgb,torch.zeros_like(rgb)) # sometimes reaches a small negative number, which causes NaNs

    mask = (rgb > .0031308).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (1.055*(rgb**(1./2.4)) - 0.055)*mask + 12.92*rgb*(1-mask)

    # if(torch.sum(torch.isnan(rgb))>0):
        # print('xyz2rgb')
        # embed()
    return rgb

def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    if(xyz.is_cuda):
        sc = sc.cuda()

    xyz_scale = xyz/sc

    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if(xyz_scale.is_cuda):
        mask = mask.cuda()

    xyz_int = xyz_scale**(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)

    L = 116.*xyz_int[:,1,:,:]-16.
    a = 500.*(xyz_int[:,0,:,:]-xyz_int[:,1,:,:])
    b = 200.*(xyz_int[:,1,:,:]-xyz_int[:,2,:,:])
    out = torch.cat((L[:,None,:,:],a[:,None,:,:],b[:,None,:,:]),dim=1)

    # if(torch.sum(torch.isnan(out))>0):
        # print('xyz2lab')
        # embed()

    return out

def lab2xyz(lab):
    y_int = (lab[:,0,:,:]+16.)/116.
    x_int = (lab[:,1,:,:]/500.) + y_int
    z_int = y_int - (lab[:,2,:,:]/200.)
    if z_int.is_cuda:
        z_int = torch.max(torch.Tensor((0,)).cuda(), z_int)
    else:
        z_int = torch.max(torch.Tensor((0,)), z_int)

    out = torch.cat((x_int[:,None,:,:],y_int[:,None,:,:],z_int[:,None,:,:]),dim=1)
    mask = (out > .2068966).type(torch.FloatTensor)
    if(out.is_cuda):
        mask = mask.cuda()

    out = (out**3.)*mask + (out - 16./116.)/7.787*(1-mask)

    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    sc = sc.to(out.device)

    out = out*sc

    # if(torch.sum(torch.isnan(out))>0):
        # print('lab2xyz')
        # embed()

    return out

def rgb2lab(rgb, opt):
    lab = xyz2lab(rgb2xyz(rgb))
    l_rs = (lab[:,[0],:,:]-opt.l_cent)/opt.l_norm
    ab_rs = lab[:,1:,:,:]/opt.ab_norm
    out = torch.cat((l_rs,ab_rs),dim=1)
    # if(torch.sum(torch.isnan(out))>0):
        # print('rgb2lab')
        # embed()
    return out

def lab2rgb(lab_rs, opt):
    l = lab_rs[:,[0],:,:]*opt.l_norm + opt.l_cent
    ab = lab_rs[:,1:,:,:]*opt.ab_norm
    lab = torch.cat((l,ab),dim=1)
    out = xyz2rgb(lab2xyz(lab))
    # if(torch.sum(torch.isnan(out))>0):
        # print('lab2rgb')
        # embed()
    return out

def get_colorization_data(data_raw, opt, ab_thresh=5., p=.125, num_points=None):
    data = {}

    data_lab = rgb2lab(data_raw[0], opt)
    data['A'] = data_lab[:,[0,],:,:]
    data['B'] = data_lab[:,1:,:,:]

    if(ab_thresh > 0): # mask out grayscale images
        thresh = 1.*ab_thresh/opt.ab_norm
        mask = torch.sum(torch.abs(torch.max(torch.max(data['B'],dim=3)[0],dim=2)[0]-torch.min(torch.min(data['B'],dim=3)[0],dim=2)[0]),dim=1) >= thresh
        data['A'] = data['A'][mask,:,:,:]
        data['B'] = data['B'][mask,:,:,:]
        # print('Removed %i points'%torch.sum(mask==0).numpy())
        if(torch.sum(mask)==0):
            return None

    return add_color_patches_rand_gt(data, opt, p=p, num_points=num_points)

def add_color_patches_rand_gt(data,opt,p=.125,num_points=None,use_avg=True,samp='normal'):
# Add random color points sampled from ground truth based on:
#   Number of points
#   - if num_points is 0, then sample from geometric distribution, drawn from probability p
#   - if num_points > 0, then sample that number of points
#   Location of points
#   - if samp is 'normal', draw from N(0.5, 0.25) of image
#   - otherwise, draw from U[0, 1] of image
    N,C,H,W = data['B'].shape

    data['hint_B'] = torch.zeros_like(data['B'])
    data['mask_B'] = torch.zeros_like(data['A'])

    for nn in range(N):
        pp = 0
        cont_cond = True
        while(cont_cond):
            if(num_points is None): # draw from geometric
                # embed()
                cont_cond = np.random.rand() < (1-p)
            else: # add certain number of points
                cont_cond = pp < num_points
            if(not cont_cond): # skip out of loop if condition not met
                continue

            P = np.random.choice(opt.sample_Ps) # patch size

            # sample location
            if(samp=='normal'): # geometric distribution
                h = int(np.clip(np.random.normal( (H-P+1)/2., (H-P+1)/4.), 0, H-P))
                w = int(np.clip(np.random.normal( (W-P+1)/2., (W-P+1)/4.), 0, W-P))
            else: # uniform distribution
                h = np.random.randint(H-P+1)
                w = np.random.randint(W-P+1)

            # add color point
            if(use_avg):
                # embed()
                data['hint_B'][nn,:,h:h+P,w:w+P] = torch.mean(torch.mean(data['B'][nn,:,h:h+P,w:w+P],dim=2,keepdim=True),dim=1,keepdim=True).view(1,C,1,1)
            else:
                data['hint_B'][nn,:,h:h+P,w:w+P] = data['B'][nn,:,h:h+P,w:w+P]

            data['mask_B'][nn,:,h:h+P,w:w+P] = 1

            # increment counter
            pp+=1

    data['mask_B']-=opt.mask_cent

    return data

def add_sized_colour_patches(data,opt,p=.125,num_points=None,use_avg=True,samp='normal'):
# Add random color points sampled from ground truth based on:
#   Number of points
#   - if num_points is 0, then sample from geometric distribution, drawn from probability p
#   - if num_points > 0, then sample that number of points
#   Location of points
#   - if samp is 'normal', draw from N(0.5, 0.25) of image
#   - otherwise, draw from U[0, 1] of image
    N,C,H,W = data['B'].shape

    data['hint_B'] = torch.zeros_like(data['B'])
    data['mask_B'] = torch.zeros_like(data['A'])

    for nn in range(N):
        pp = 0
        cont_cond = True
        while(cont_cond):
            if(num_points is None): # draw from geometric
                # embed()
                cont_cond = np.random.rand() < (1-p)
            else: # add certain number of points
                cont_cond = pp < num_points
            if(not cont_cond): # skip out of loop if condition not met
                continue

            P = np.random.choice(opt.sample_Ps) # patch size

            # sample location
            if(samp=='normal'): # geometric distribution
                h = int(np.clip(np.random.normal( (H-P+1)/2., (H-P+1)/4.), 0, H-P))
                w = int(np.clip(np.random.normal( (W-P+1)/2., (W-P+1)/4.), 0, W-P))
            else: # uniform distribution
                h = np.random.randint(H-P+1)
                w = np.random.randint(W-P+1)

            # add color point
            if(use_avg):
                # embed()
                data['hint_B'][nn,:,h:h+P,w:w+P] = torch.mean(torch.mean(data['B'][nn,:,h:h+P,w:w+P],dim=2,keepdim=True),dim=1,keepdim=True).view(1,C,1,1)
            else:
                data['hint_B'][nn,:,h:h+P,w:w+P] = data['B'][nn,:,h:h+P,w:w+P]

            data['mask_B'][nn,:,h:h+P,w:w+P] = 1

            # increment counter
            pp+=1

    data['mask_B']-=opt.mask_cent

    return data


def add_color_patch(data, opt, P=1, hw=[128,128], ab=[0,0]):
    # Add a color patch at (h,w) with color (a,b)
    data['hint_B'][:,0,hw[0]:hw[0]+P,hw[1]:hw[1]+P] = 1.*ab[0]/opt.ab_norm
    data['hint_B'][:,1,hw[0]:hw[0]+P,hw[1]:hw[1]+P] = 1.*ab[1]/opt.ab_norm
    data['mask_B'][:,:,hw[0]:hw[0]+P,hw[1]:hw[1]+P] = 1-opt.mask_cent

    return data


def add_color_patch_old(data,opt,P=1,hw=[128,128],ab=[0,0]):
    # Add a color patch at (h,w) with color (a,b)
    data[:,0,hw[0]:hw[0]+P,hw[1]:hw[1]+P] = 1.*ab[0]/opt.ab_norm
    data[:,1,hw[0]:hw[0]+P,hw[1]:hw[1]+P] = 1.*ab[1]/opt.ab_norm
    mask[:,:,hw[0]:hw[0]+P,hw[1]:hw[1]+P] = 1-opt.mask_cent

    return (data,mask)


def crop_mult(data,mult=16,HWmax=[800,1200]):
    # crop image to a multiple
    H,W = data.shape[2:]
    Hnew = int(min(H/mult*mult,HWmax[0]))
    Wnew = int(min(W/mult*mult,HWmax[1]))
    h = (H-Hnew)/2
    w = (W-Wnew)/2
    h = int(h)
    w = int(w)

    return data[:,:,h:h+Hnew,w:w+Wnew]

def encode_ab_ind(data_ab, opt):
    # Encode ab value into an index
    # INPUTS
    #   data_ab   Nx2xHxW \in [-1,1]
    # OUTPUTS
    #   data_q    Nx1xHxW \in [0,Q)

    data_ab_rs = torch.round((data_ab*opt.ab_norm + opt.ab_max)/opt.ab_quant) # normalized bin number
    data_q = data_ab_rs[:,[0],:,:]*opt.A + data_ab_rs[:,[1],:,:]
    return data_q

def decode_ind_ab(data_q, opt):
    # Decode index into ab value
    # INPUTS
    #   data_q      Nx1xHxW \in [0,Q)
    # OUTPUTS
    #   data_ab     Nx2xHxW \in [-1,1]
    #
    data_a = data_q/opt.A
    data_b = data_q - data_a*opt.A
    # assert isinstance(opt.A, (int, float))
    # data_b = np.mod(data_q, opt.A)
    # data_a = (data_q-data_b)/opt.A

    data_ab = torch.cat((data_a, data_b), dim=1)

    if data_q.is_cuda:
        type_out = torch.cuda.FloatTensor
    else:
        type_out = torch.FloatTensor
    data_ab = ((data_ab.type(type_out)*opt.ab_quant) - opt.ab_max)/opt.ab_norm

    return data_ab


def my_decode_ind_ab(data_q, opt):
    # Decode index into ab value
    # INPUTS
    #   data_q      Nx1xHxW \in [0,Q)
    # OUTPUTS
    #   data_ab     Nx2xHxW \in [-1,1]
    #
    # data_a = data_q/opt.A
    # data_b = data_q - data_a*opt.A
    assert isinstance(opt.A, (int, float))
    # data_b = np.mod(data_q, opt.A)
    data_b = torch.fmod(data_q, opt.A)
    data_a = (data_q-data_b)/opt.A

    data_ab = torch.cat((data_a, data_b), dim=1)

    if data_q.is_cuda:
        type_out = torch.cuda.FloatTensor
    else:
        type_out = torch.FloatTensor
    data_ab = ((data_ab.type(type_out)*opt.ab_quant) - opt.ab_max)/opt.ab_norm

    return data_ab

def decode_max_ab(data_ab_quant, opt):
    # Decode probability distribution by using bin with highest probability
    # INPUTS
    #   data_ab_quant   NxQxHxW \in [0,1]
    # OUTPUTS
    #   data_ab         Nx2xHxW \in [-1,1]

    data_q = torch.argmax(data_ab_quant,dim=1)[:,None,:,:]
    return decode_ind_ab(data_q, opt)

def decode_mean(data_ab_quant, opt):
    # Decode probability distribution by taking mean over all bins
    # INPUTS
    #   data_ab_quant   NxQxHxW \in [0,1]
    # OUTPUTS
    #   data_ab_inf     Nx2xHxW \in [-1,1]

    (N,Q,H,W) = data_ab_quant.shape
    a_range = torch.arange(-opt.ab_max, opt.ab_max+opt.ab_quant, step=opt.ab_quant).to(data_ab_quant.device)[None,:,None,None]
    a_range = a_range.type(data_ab_quant.type())

    # reshape to AB space
    data_ab_quant = data_ab_quant.view((N,int(opt.A),int(opt.A),H,W))
    data_a_total = torch.sum(data_ab_quant,dim=2)
    data_b_total = torch.sum(data_ab_quant,dim=1)

    # matrix multiply
    data_a_inf = torch.sum(data_a_total * a_range,dim=1,keepdim=True)
    data_b_inf = torch.sum(data_b_total * a_range,dim=1,keepdim=True)

    data_ab_inf = torch.cat((data_a_inf,data_b_inf),dim=1)/opt.ab_norm

    return data_ab_inf

def calculate_psnr_np(img1, img2):
    import numpy as np
    SE_map = (1.*img1-img2)**2
    cur_MSE = np.mean(SE_map)
    return 20*np.log10(255./np.sqrt(cur_MSE))

def calculate_psnr_torch(img1, img2):
    SE_map = (1.*img1-img2)**2
    cur_MSE = torch.mean(SE_map)
    return 20*torch.log10(1./torch.sqrt(cur_MSE))


def extract_lab_channels(data, opt):
    #  Convert to Lab, take ab, convert back to RGB
    full_lab_tensor = rgb2lab(data, opt)
    just_ab = torch.zeros_like(full_lab_tensor)
    just_ab[:, 1:, :, :] = full_lab_tensor[:, 1:, :, :]
    just_ab_asrgb = lab2rgb(just_ab, opt)
    return just_ab, just_ab_asrgb, full_lab_tensor


def apply_smoothing(just_ab_asrgb, opt):
    # Kernel smoothing
    kernel = np.ones((5, 5), np.float32) / 25
    just_ab_asrgb_im = tensor2im(just_ab_asrgb)
    just_ab_asrgb_im_smoothed = np.asarray(cv.filter2D(just_ab_asrgb_im, -1, kernel)).astype(int)
    just_ab_smoothed_asab_tensor = im2tensor(just_ab_asrgb_im_smoothed)
    just_ab_smoothed_asab = rgb2lab(just_ab_smoothed_asab_tensor, opt)
    just_ab_smoothed_asab = just_ab_smoothed_asab[0, 1:, :, :]

    return just_ab_smoothed_asab


def zhang_bins(just_ab_smoothed_asab, l_tensor, opt, vis=True):
    # print('Go')
    n_clusterss = 25
    psnrs = []
    ks = []
    times = []
    just_ab_smoothed_asab = torch.reshape(just_ab_smoothed_asab, (1, 2, 250, 250))
    encoded_ab = encode_ab_ind(just_ab_smoothed_asab, opt)
    # encoded_ab = encoded_ab
    # encoded_ab_flat = np.asarray(encoded_ab).reshape(250*250, 1)
    encoded_img = encoded_ab.reshape(250, 250)
    decoded_ab = my_decode_ind_ab(encoded_ab, opt)

    center_lab = np.zeros((250, 250, 3))
    center_lab[:, :, 1] = decoded_ab[0, 0, :, :]
    center_lab[:, :, 2] = decoded_ab[0, 1, :, :]
    center_lab[:, :, 0] = l_tensor[0, 0, :, :]
    center_lab_convert = im2tensor(center_lab, 'unknown')
    center_lab_convert = lab2rgb(center_lab_convert, opt)
    center_lab_img = tensor2im(center_lab_convert)

    center_lab = np.zeros((250, 250, 3))
    center_lab[:, :, 1] = decoded_ab[0, 0, :, :]
    center_lab[:, :, 2] = decoded_ab[0, 1, :, :]
    # center_lab[:, :, 0] = l_tensor[0, 0, :, :]
    center_lab_convert = im2tensor(center_lab, 'unknown')
    center_lab_convert = lab2rgb(center_lab_convert, opt)
    center_lab_img2 = tensor2im(center_lab_convert)

    center_lab = np.zeros((250, 250, 3))
    # center_lab[:, :, 1] = decoded_ab[0, 0, :, :]
    # center_lab[:, :, 2] = decoded_ab[0, 1, :, :]
    center_lab[:, :, :] = np.transpose(l_tensor[0], (1, 2, 0))
    center_lab_convert = im2tensor(center_lab, 'unknown')
    center_lab_convert = lab2rgb(center_lab_convert, opt)
    center_lab_img3 = tensor2im(center_lab_convert)

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

    return encoded_ab, center_lab_img2


def dbscan_encoded_indexed(encoded, encoded_coloured, rgb_image_array_orig, vis=True):
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
    # centers = centers.reshape(just_ab_image_array.shape)
    #
    # # print('Starting')
    # eps = .5
    # min_samples = 5
    # means2 = DBSCAN(eps=eps, min_samples=min_samples).fit(both_scaled)
    # # centers = [means2.cluster_centers_[i] for i in means2.labels_]
    # # centers = np.asarray(centers).astype(int)
    # labels_mx2 = means2.labels_.reshape(both[:, :, 0].shape)
    # # centers2 = centers.reshape(ab_X_indexed.shape)
    # # print('Done')


    if vis:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(14, 5))
        # fx_image = util.tensor2im(rgb_img)
        cmap = matplotlib.colors.ListedColormap(np.random.rand(256, 3))
        c = ax1.imshow(labels_mx, cmap=cmap)
        # ax2.imshow(centers2[:, :, 2:])
        # ax2.imshow(labels_mx2)
        # ax2.imshow(labels_mx2, cmap=cmap)
        ax2.imshow(tensor2im(rgb_image_array_orig))
        ax3.imshow(encoded_img[0])
        ax4.imshow(encoded_coloured)
        plt.show()

    return labels_mx