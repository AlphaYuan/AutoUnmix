import argparse
from ctypes import util
from ctypes.wintypes import PFILETIME
import datetime
import os
import sys
import time
import itertools

import numpy as np
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchsummary import summary
from skimage.metrics import peak_signal_noise_ratio

import model_new
import pytorch_msssim
from utils import MyDataSet, show_tif, L1_Charbonnier_loss
from mutual_info import MutualInformation

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs of training")
    parser.add_argument("--test_epoch", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="synthetic", help="name of the dataset")
    parser.add_argument("--data_path", type=str, default="data/train/mixed", help="path of the dataset")
    parser.add_argument("--target_path", type=str, default="data/train/target", help="path of the target set")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.000001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    # parser.add_argument("--img_height", type=int, default=512, help="size of image height")
    # parser.add_argument("--img_width", type=int, default=512, help="size of image width")
    parser.add_argument("--re_size", type=int, default=512, help="size of image")
    parser.add_argument("--crop_size", type=int, default=256, help="size of cropped image")
    parser.add_argument("--channels", type=int, default=4, help="number of image channels")
    # parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
    parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
    parser.add_argument("--loss", type=str, default="MSELoss", help="loss function")
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument("--model", default="ResNet_old", type=str, help="model for training")
    parser.add_argument("--version", default=0, type=int, help="version")
    parser.add_argument("--no_augmentation", action='store_false', help="image augmentation")
    parser.add_argument("--num", type=int, default=1, help="num of dataset")
    parser.add_argument("--no_amp", action='store_false', help="no image augmentation")
    opt = parser.parse_args()
    print(opt)
    return opt

opt = parse_args()
torch.cuda.set_device(6)

os.makedirs('images/%s/version_%s/ft' % (opt.dataset_name, opt.version), exist_ok=True)

cuda = torch.cuda.is_available()
input_shape = (opt.channels, opt.re_size, opt.re_size)

use_amp = opt.no_amp
if use_amp:
    try:
        from apex import amp
    except:
        print("Apex is not installed. Use PyTorch DDP")

transform = transforms.Compose([
    # transforms.Resize(opt.img_height),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
])

# train_set = MyDataSet(opt.data_path, opt.target_path, transform, augmentation=True, crop_size=(opt.crop_size, opt.crop_size))
test_set = MyDataSet(opt.data_path.replace('train', 'test'), opt.target_path.replace('train', 'test'), transform, opt.no_augmentation, (opt.crop_size, opt.crop_size), num=opt.num, avg_num=7)
# train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
# train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, num_workers=opt.n_cpu, pin_memory=True,
#                           sampler=train_sampler)
# train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, num_workers=opt.n_cpu, pin_memory=True,)
test_loader = DataLoader(dataset=test_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu, )
print("testset size: {}, testloader size: {}".format(len(test_set), len(test_loader)))

# if opt.model == 'ResNet':
#     enc = model.ResNet(input_shape, opt.n_residual_blocks)
#     dec = model.ResNet(input_shape, opt.n_residual_blocks)
# elif opt.model == 'ResNet_old':
#     enc = model.ResNet_old(input_shape, opt.n_residual_blocks)
#     dec = model.ResNet_old(input_shape, opt.n_residual_blocks)
# elif opt.model == 'CNN':
#     enc = model.Encoder()
#     dec = model.Decoder()
# elif opt.model == 'UNet':
#     enc = model.UNet(input_shape[0], input_shape[0])
#     # dec = model.ResNet(input_shape, opt.n_residual_blocks)
#     dec = model.UNet(input_shape[0], input_shape[0])
#     # enc = model.ResNet(input_shape, opt.n_residual_blocks)
# elif opt.model == 'UwUNet':
#     enc = model.UNet(input_shape[0], input_shape[0])
#     # enc = model.ResNet(input_shape, opt.n_residual_blocks)
#     dec = model.UwUNet(input_shape[0], input_shape[0])
# else:
#     enc = model.GeneratorResNet(input_shape, opt.n_residual_blocks)
#     dec = model.GeneratorResNet(input_shape, opt.n_residual_blocks)
enc = model_new.UNet(input_shape[0], input_shape[0])
dec = model_new.unmixingNet(input_shape[0], input_shape[0])

criterion_mse = nn.MSELoss()
criterion_l1 = nn.L1Loss()
criterion_ssim = pytorch_msssim.MSSSIM()
# criterion_charbonnier = L1_Charbonnier_loss()
criterion_MI = MutualInformation(device='cuda' if cuda else 'cpu')

if cuda:
    enc = enc.cuda()
    dec = dec.cuda()
    criterion_mse.cuda()
    criterion_l1.cuda()
    criterion_ssim.cuda()
    # criterion_MI.cuda()


epoch = ''
if opt.test_epoch > 0:
    epoch = '_' + str(opt.test_epoch)
elif opt.test_epoch == -1:
    epoch = '_ft'
print("saved_models/%s/enc_ft.pth" % (opt.dataset_name))
pretrained_enc = torch.load("saved_models/%s/version_%s/enc%s.pth" % (opt.dataset_name, opt.version, epoch), map_location='cpu')
print("saved_models/%s/dec_ft.pth" % (opt.dataset_name))
pretrained_dec = torch.load("saved_models/%s/version_%s/dec%s.pth" % (opt.dataset_name, opt.version, epoch), map_location='cpu')

enc.load_state_dict({k.replace('module.', ''): v for k, v in pretrained_enc.items()})
dec.load_state_dict({k.replace('module.', ''): v for k, v in pretrained_dec.items()})

print(summary(enc, (opt.channels, opt.re_size, opt.re_size), device='cuda' if cuda else 'cpu'))
print(summary(dec, (opt.channels, opt.re_size, opt.re_size), device='cuda' if cuda else 'cpu'))


optimizer = torch.optim.Adam(itertools.chain(dec.parameters(), enc.parameters()), lr=opt.lr)
if use_amp:
    [enc, dec], optimizer = amp.initialize([enc, dec], optimizer, opt_level='O1')

# for name, param in dec.named_parameters():
#     print(name)

print('Dec params disabled grad:')
for name, param in dec.named_parameters():
    # if name.split('.')[0][:] not in ['final', 'up7']: # ResNet: int(name.split('.')[1]) < 9:
    # # if not name.split('.')[0].startswith('spec_final'):
    if not (name.split('.')[0].startswith('final_conv') or name.split('.')[0].startswith('mi_conv')):
        print('\t', name)
        param.requires_grad = False
    # param.requires_grad = False

print('Enc params disabled grad:')
for name, param in enc.named_parameters():
    # if name.split('.')[0][:] not in ['final', 'up7']: # ResNet: int(name.split('.')[1]) < 9:
    if not name.split('.')[0].startswith('final'):
        print('\t', name)
        param.requires_grad = False
    # param.requires_grad = False

# print('params required grad:')
# for name, param in dec.named_parameters():
#     if param.requires_grad == True:
#         print('\t', name)

# optimizer = torch.optim.Adam(itertools.chain(dec.parameters(), enc.parameters()), lr=opt.lr)
# # [enc, dec], optimizer = amp.initialize([enc, dec], optimizer, opt_level='O1')

prev_time = time.time()
for epoch in range(opt.epochs):
    for i, (x, y) in enumerate(test_loader):
        if cuda:
            x = x.cuda()
            y = y.cuda()

        enc.train()
        dec.train()
        optimizer.zero_grad()

        # print(x.shape, y.shape)
        y_hat = dec(x)
        loss_mi = 0
        # for chann1 in range(0, input_shape[0]):
        #     for chann2 in range(chann1 + 1, input_shape[0]):
        #         # print(criterion_MI(y_hat[:, chann1: chann1+1, :, :], y_hat[:, chann2: chann2+1, :, :]))
        #         loss_mi += torch.mean(criterion_MI(y_hat[:, chann1: chann1+1, :, :], y_hat[:, chann2: chann2+1, :, :]))
        # # loss_ssim1 = 1 - criterion_ssim(y_hat, y)
        # loss_xy = loss_ssim1 + criterion_mse(y_hat, y) * 10
        x_hat = enc(y_hat)

        # y_hat_hat = dec(x_hat)
        # y_hat_hat = y
        # loss_ssim1 = 1 - criterion_ssim(y_hat_hat, y_hat)
        # loss_xy = loss_ssim1 + criterion_mse(y_hat_hat, y_hat) * 10 + criterion_l1(y_hat_hat, y_hat)
        # loss_ssim2 = 1 - criterion_ssim(x_hat, x)
        # loss_yx = loss_ssim2 + criterion_mse(x_hat, x) * 10 + criterion_l1(x_hat, x)

        # loss = loss_yx + loss_xy

        # loss = criterion_charbonnier(x_hat, x)
        loss_mse_xy = criterion_mse(x_hat, x)
        loss = loss_mse_xy# + loss_mi / 100
        # print(loss, type(loss), loss_mse_xy, type(loss_mse_xy))
        if use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # loss.backward()

        optimizer.step()

        batches_done = epoch * len(test_loader) + i
        batches_left = opt.epochs * len(test_loader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev = time.time()

        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch, %d/%d] [Loss: %f] [Loss_xy: %f, Loss MI: %f] ETA: %s"
            % (
                epoch,
                opt.epochs,
                i,
                len(test_loader),
                loss.item(),
                # loss_l1.item(),
                loss.item(),
                # loss_ssim2.item(),
                loss.item(),
                # loss_yx.item(),
                time_left,
            )
        )

        # writer.add_scalar('%s_loss/loss' % (opt.dataset_name), loss, epoch)
        # writer.add_scalar('%s_loss/loss_xy' % (opt.dataset_name), loss_xy, epoch)
        # writer.add_scalar('%s_loss/loss_yx' % (opt.dataset_name), loss_yx, epoch)
        if i % (len(test_set) / 2) == 0:
            (x, y) = next(iter(test_loader))
            enc.eval()
            dec.eval()
            if cuda:
                x = x.cuda()
                y = y.cuda()
            y_hat = dec(x)
            x_hat = enc(y_hat)
            if opt.batch_size == 4:
                def convert4to1(data):
                    data_row1 = torch.cat((data[0,:,:,:], data[1,:,:,:]), 2)
                    data_row2 = torch.cat((data[2,:,:,:], data[3,:,:,:]), 2)
                    # print(data_row1.shape)
                    return torch.cat((data_row1, data_row2), 1)

                x = convert4to1(x).unsqueeze(0)
                y = convert4to1(y).unsqueeze(0)
                y_hat = convert4to1(y_hat).unsqueeze(0)
                x_hat = convert4to1(x_hat).unsqueeze(0)

            show_tif(x.squeeze(), y.squeeze(), y_hat.squeeze(), x_hat.squeeze(), name='images/%s/version_%s/ft/%s_ft' % (opt.dataset_name, opt.version, batches_done), cuda=cuda)
    if epoch % 5 == 0 and opt.local_rank == 0:
        torch.save(enc.state_dict(), "saved_models/%s/version_%s/enc_%s_ft.pth" % (opt.dataset_name, opt.version, epoch))
        torch.save(dec.state_dict(), "saved_models/%s/version_%s/dec_%s_ft.pth" % (opt.dataset_name, opt.version, epoch))


if opt.local_rank == 0:
    torch.save(enc.state_dict(), "saved_models/%s/version_%s/enc_ft.pth" % (opt.dataset_name, opt.version))
    torch.save(dec.state_dict(), "saved_models/%s/version_%s/dec_ft.pth" % (opt.dataset_name, opt.version))
