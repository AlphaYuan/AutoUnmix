import numpy as np
import time
import argparse
from skimage import io
import scipy.io as sio
from scipy.optimize import minimize
import os
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from utils import MyDataSet, show_tif
import pytorch_msssim
import torch.nn.functional as F
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='simu4_group3', help="name of the dataset")
    parser.add_argument("--data_path", type=str, default='data_simu4_group3/test/mixed', help="path of the dataset")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--img_size", type=int, default=512, help="size of image")
    parser.add_argument("--mat_path", type=str, default='pic/channel_tagBPF_group3.mat', help="path of the mat") 
    opt = parser.parse_args()
    print(opt)
    return opt

opt = parse_args()

img_size = opt.img_size
data_path = opt.data_path
mat_path = opt.mat_path
cuda = False

def unmix4(data, H):
    b = data
    fun = lambda x: np.linalg.norm(np.dot(H, x) - b)
    sol = minimize(fun, np.asarray([0.5, 0.5, 0.5, 0.5]), method='SLSQP')
    # print(sol)
    return np.array([sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3]])

def unmix(data, H):
    b = data
    fun = lambda x: np.linalg.norm(np.dot(H, x) - b)
    sol = minimize(fun, np.asarray([0.5, 0.5, 0.5]), method='SLSQP')
    # print(sol['x'])
    return np.array([sol['x'][0], sol['x'][1], sol['x'][2]])


def lu(data, H, chann):
    data = np.reshape(np.array(data), (-1, chann))

    def perform_spectral_unmixing(_data):
        res = np.zeros(_data.shape)
        for idx in range(len(_data)):
            # print(idx)
            sys.stdout.write('\r%d' % (idx))
            if chann == 3:
                res[idx, :] = unmix(data[idx], H)
            elif chann == 4:
                res[idx, :] = unmix4(data[idx], H)
        return res

    lu_result = perform_spectral_unmixing(data)
    print('computation finished')
    return np.reshape(lu_result, (img_size, img_size, chann))


# img_shape = (4, 512, 512)

# data_path = 'data_real4/test/lu/mixed'
# data_path = 'data_3_bpae/test/real_1126/mixed'
target_path = data_path.replace('mixed', 'target')
transform = transforms.Compose([
    # transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
])
train_set = MyDataSet(data_path, target_path, transform, augmentation=False)
# x0 = train_set[0][0]
# y0 = train_set[0][1]
# test_data = []
# test_target = []
# data_files, target_files = os.listdir(data_path), os.listdir(target_path)
# data_files.sort(key=lambda x: int(x[4:-4]))
# for file in data_files:
#     filename = os.path.join(data_path, file)
#     img = io.imread(filename)
#     img = Image.fromarray(img)
#     img = img.resize((img_size, img_size))
#     test_data.append(np.array(img))
#
# target_files.sort(key=lambda x: int(x[4:-4]))
# for file in target_files:
#     filename = os.path.join(target_path, file)
#     img = io.imread(filename)
#     img = Image.fromarray(img)
#     img = img.resize((img_size, img_size))
#     test_target.append(np.array(img))
transform1 = transforms.Compose([
    transforms.ToTensor(),
])

# mat_path = 'pic/channel_bpae_3.mat'
channel_intensities = sio.loadmat(mat_path)['channel_intensities']
print(channel_intensities)
loss_list = [[], [], []]
for i, (x0, y0) in enumerate(train_set):
    # if i == 1:
    #     break
    # print(x0.shape, y0.shape)
    if opt.channels == 3:
        x0 = x0[[0,2,3],:,:]
        y0 = y0[[0,2,3],:,:]
    x = x0.permute(1,2,0)
    y = y0.permute(1,2,0)

    print(x.shape)
    # st = time.time()

    res = lu(x, channel_intensities.T, opt.channels)

    # ed = time.time()

    # print('time: {:.5f} s'.format(ed - st))

    res = res.astype('float32')
    np.save('images/simu4_group2/lu/%s_%s_%d.npy' % (opt.name, str(i), img_size), res)

    # x = x0
    # y = y0
    # y_hat = torch.Tensor(res)
    # y_hat = y_hat / y_hat.max()
    # diff = y_hat - y

    y_hat = transform1(res)
    # y_hat = torch.Tensor(res)
    y_hat = y_hat / y_hat.max()
    # image_grid = show_tif(x.squeeze(), y.squeeze(), y_hat.squeeze(), None, name='images/simu4_group3/lu/simu_gp3_0_%d.png' % (img_size), cuda=cuda)
    # diff = y_hat - y0
    print(x.shape, y.shape, y_hat.shape, res.shape)
    mseloss = F.mse_loss(y_hat, y0)
    # print(mseloss)
    l1loss = F.l1_loss(y_hat, y0)
    # print(l1loss)
    # y_hat1 = transform1(y_hat)
    # y01 = transform1(y0)
    ssimloss = pytorch_msssim.msssim(y_hat.unsqueeze(0), y0.unsqueeze(0))
    print(mseloss, l1loss, ssimloss)
    loss_list[0].append(mseloss)
    loss_list[1].append(l1loss)
    loss_list[2].append(ssimloss)
    # show_tif(x0, y0, y_hat, name='images/3_bpae/lu/%s_%s' % (opt.name, str(i)))
    # x_expand = torch.cat((x[:,:, 0], x[:,:, 1], x[:,:,2], x[:,:,3]), 1)
    # y_expand = torch.cat((y[:,:, 0], y[:,:, 1], y[:,:, 2], y[:,:, 3]), 1)
    # y_hat_expand = torch.cat((y_hat[:,:, 0], y_hat[:,:, 1], y_hat[:,:, 2], y_hat[:,:, 3]), 1)
    # diff_expand = torch.cat((diff[:,:, 0], diff[:,:, 1], diff[:,:, 2], diff[:,:, 3]), 1)

    # x_expand = torch.cat((x[:,:, 0], x[:,:, 1], x[:,:,2]), 1)
    # y_expand = torch.cat((y[:,:, 0], y[:,:, 1], y[:,:, 2]), 1)
    # y_hat_expand = torch.cat((y_hat[:,:, 0], y_hat[:,:, 1], y_hat[:,:, 2]), 1)
    # diff_expand = torch.cat((diff[:,:, 0], diff[:,:, 1], diff[:,:, 2]), 1)
    # # print(x_expand.shape)
    # image_grid = torch.cat((x_expand, torch.ones((10,img_size*3)) * 255, y_expand, torch.ones((10,img_size*3)) * 255, y_hat_expand), 0)
    # save_image(diff_expand, "test_diff-%s.png" % (opt.name), normalize=False)
    # save_image(image_grid, "test-%s.png" % (opt.name), normalize=False)
    # np.save('img_unmixed-%s.npy' % (opt.name), y_hat)
    # print()
with open('simu4_group2_ch2_lu.txt', 'w') as f:
    for i in loss_list:
        f.write(str(i))
        f.write('\n')