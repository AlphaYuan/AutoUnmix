import os

import random
# import matplotlib.pyplot as plt
import torch
import numpy as np
import math
from torch.autograd import Variable
from PIL import Image
from skimage import io
from torch.utils.data import Dataset
from torchvision.utils import save_image

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.startswith("Conv"):
        # torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        

class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


class MyDataSet(Dataset):
    def __init__(self, data_path, target_path, transform=None, augmentation=False, crop_size=(96, 96), num=None, test_batch4=False, avg_num=4):
        self.tif_stream_data, self.tif_stream_target = [], []
        data_files, target_files = os.listdir(data_path), os.listdir(target_path)
        data_files.sort(key=lambda x: int(x[4:-4]))
        # print('datafile num: ', len(data_files[:num]))
        for file in data_files[:num]:
            filename = os.path.join(data_path, file)
            img = io.imread(filename)
            # print(img.shape)
            if img.shape[0] < 10:
                img = img.transpose(1, 2, 0)
            # print(type(img), img.shape, img.max(), img.min())
            img = Image.fromarray(img)
            if transform:
                img = transform(img)
            self.tif_stream_data.append(img)

        target_files.sort(key=lambda x: int(x[4:-4]))
        for file in target_files[:num]:
            filename = os.path.join(target_path, file)
            img = io.imread(filename)
            if img.shape[0] < 10:
                img = img.transpose(1, 2, 0)
            img = Image.fromarray(img)
            if transform:
                img = transform(img)
            self.tif_stream_target.append(img)
        # y = self.tif_stream_target[0]
        # y_expand = torch.cat((y[0, :, :], y[1, :, :], y[2, :, :], y[3, :, :]), 1)
        # save_image(y_expand, "y_expand.png", normalize=False)
        if augmentation:
            if test_batch4:
                self.tif_stream_data, self.tif_stream_target = image_crop(self.tif_stream_data, self.tif_stream_target, crop_size)
            else:
                self.tif_stream_data, self.tif_stream_target = augment(self.tif_stream_data, self.tif_stream_target, avg=avg_num, tgt_size=crop_size)
        # print('dataset size: ', len(self.tif_stream_data))
        assert len(self.tif_stream_data) == len(self.tif_stream_data)

    def __len__(self):
        return len(self.tif_stream_data)

    def __getitem__(self, item):
        data, target = self.tif_stream_data, self.tif_stream_target
        return data[item], target[item]

def augment(data, target, avg=5, tgt_size=(256,256)):
    data_augmented, target_augmented = [], []
    assert len(data) == len(target)
    _, h, w = data[0].shape
    # _img = torch.zeros((data.shape[0], tgt_size[0], tgt_size[1]))
    # avg = nums / len(data)
    for i in range(len(data)):
        for j in range(avg + 1):
            r1, r2 = random.randint(0, h - tgt_size[0]), random.randint(0, w - tgt_size[1])
            data_augmented.append(data[i][:, r1:r1 + tgt_size[0], r2:r2 + tgt_size[1]])
            target_augmented.append(target[i][:, r1:r1 + tgt_size[0], r2:r2 + tgt_size[1]])
    return data_augmented, target_augmented

def image_crop(data, target, tgt_size):
    data_augmented, target_augmented = [], []
    assert len(data) == len(target)
    (h, w) = tgt_size
    zero_img = torch.zeros((data[0].shape[0], h, w))
    for i in range(len(data)):
        x, y = data[i], target[i]
        j = 0
        while j < x.shape[1]:
            k = 0
            while k < x.shape[2]:
                # x_tmp, y_tmp = zero_img, zero_img
                if j + h <= x.shape[1] and k + w <= x.shape[2]:
                    x_tmp = x[:, j:j + h, k:k + w]
                    y_tmp = y[:, j:j + h, k:k + w]
                elif k + w <= x.shape[2]:
                    x_tmp, y_tmp = zero_img, zero_img
                    x_tmp[:, :x.shape[1] - j, :w] = x[:, j:, k:k + w]
                    y_tmp[:, :x.shape[1] - j, :w] = y[:, j:, k:k + w]
                elif j + h <= x.shape[1]:
                    x_tmp, y_tmp = zero_img, zero_img
                    x_tmp[:, :h, :x.shape[2] - k] = x[:, j:j + h, k:]
                    y_tmp[:, :h, :x.shape[2] - k] = y[:, j:j + h, k:]
                data_augmented.append(x_tmp)
                target_augmented.append(y_tmp)
                k = k + w
            j = j + h
    # for i in range(4):
    #     show_tif(data_augmented[i], target_augmented[i], name=str(i))
    return data_augmented, target_augmented


def col_cat(img, cuda=False):
    if img is None:
        return torch.ones(1)

    white = torch.ones((img.shape[1], 10)) * 255
    if cuda:
        white = white.cuda()
    out_img = img[0, :, :]
    for i in range(1, img.shape[0]):
        out_img = torch.cat((out_img, white, img[i, :, :]), 1)
    return out_img


def show_tif(x, y, y_hat=None, x_hat=None, name='a', cuda=False, save_y=False, save_x=False):
    x1 = col_cat(x, cuda)
    y1 = col_cat(y, cuda)
    image_grid = x1
    white = torch.ones((10, x1.shape[1])) * 255
    if cuda:
        white = white.cuda()
    if x_hat is not None:
        image_grid = torch.cat((image_grid, white, col_cat(x_hat, cuda)), 0)
    image_grid = torch.cat((image_grid, white, y1), 0)
    if y_hat is not None:
        image_grid = torch.cat((image_grid, white, col_cat(y_hat, cuda)), 0)
    # plt.figure()
    # plt.imshow(image_grid.cpu())
    # plt.show()
    if save_y:
        np.save("%s_yhat.npy" % (name), y_hat.permute([1,2,0]).detach().cpu().numpy())
    if save_x:
        np.save("%s_xhat.npy" % (name), x_hat.permute([1,2,0]).detach().cpu().numpy())
    save_image(image_grid, "%s.png" % (name), normalize=False)
    return image_grid


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
