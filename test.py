import argparse
import os
import sys
import time

import numpy as np
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio

import model
import model_new
import pytorch_msssim
from utils import MyDataSet, show_tif

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--test_epoch", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="synthetic", help="name of the dataset")
    parser.add_argument("--data_path", type=str, default="data/train/mixed", help="path of the dataset")
    parser.add_argument("--target_path", type=str, default="data/train/target", help="path of the target set")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
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
    parser.add_argument("--model", default="ResNet", type=str, help="model for training")
    parser.add_argument("--version", default=0, type=int, help="version")
    parser.add_argument("--augmentation", action='store_false', help="image augmentation")
    opt = parser.parse_args()
    print(opt)
    return opt

opt = parse_args()
torch.cuda.set_device(0)

os.makedirs('images/%s/version_%s/test' % (opt.dataset_name, opt.version), exist_ok=True)

cuda = torch.cuda.is_available()
input_shape = (opt.channels, opt.re_size, opt.re_size)

if cuda:
    try:
        from apex import amp
    except:
        print("Apex is not installed. Use PyTorch DDP")

transform = transforms.Compose([
    # transforms.Resize(opt.re_size),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
])

# train_set = MyDataSet(opt.data_path, opt.target_path, transform, augmentation=True, crop_size=(opt.crop_size, opt.crop_size))
test_set = MyDataSet(opt.data_path.replace('train', 'test'), opt.target_path.replace('train', 'test'), transform, opt.augmentation, (opt.crop_size, opt.crop_size), test_batch4=True)
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
#     # enc = model.ResNet(input_shape, opt.n_residual_blocks)
#     dec = model.UNet(input_shape[0], input_shape[0])
#     # dec = model.ResNet(input_shape, opt.n_residual_blocks)
# elif opt.model == 'UwUNet':
#     # enc = model.UwUNet(input_shape[0], input_shape[0])
#     # enc = model.ResNet(input_shape, opt.n_residual_blocks)
#     enc = model.UNet(input_shape[0], input_shape[0])
#     dec = model.UwUNet(input_shape[0], input_shape[0])
# elif opt.model == 'ConvNeXt':
#     enc = model_convnext.ConvNeXt(in_chans=input_shape[0], out_chans=input_shape[0], )
#     dec = model_convnext.ConvNeXt(in_chans=input_shape[0], out_chans=input_shape[0], )
# else:
#     enc = model.GeneratorResNet(input_shape, opt.n_residual_blocks)
#     dec = model.GeneratorResNet(input_shape, opt.n_residual_blocks)
## myNet
enc = model_new.UNet(input_shape[0], input_shape[0])
dec = model_new.unmixingNet(input_shape[0], input_shape[0])

criterion = nn.MSELoss()
criterion1 = nn.L1Loss()
ssim1 = pytorch_msssim.MSSSIM()

if cuda:
    enc = enc.cuda()
    dec = dec.cuda()
    criterion.cuda()
    criterion1.cuda()

epoch = ''
if opt.epochs == opt.test_epoch:
    epoch
elif opt.test_epoch >= 0:
    epoch = '_' + str(opt.test_epoch)
elif opt.test_epoch < 0:
    if opt.test_epoch == -1:
        epoch = '_ft'
    else:
        epoch = '_' + str(abs(opt.test_epoch)) + '_ft'
print("saved_models/%s/enc%s.pth" % (opt.dataset_name, epoch))
pretrained_enc = torch.load("saved_models/%s/version_%s/enc%s.pth" % (opt.dataset_name, opt.version, epoch), map_location='cpu')
print("saved_models/%s/dec%s.pth" % (opt.dataset_name, epoch))
pretrained_dec = torch.load("saved_models/%s/version_%s/dec%s.pth" % (opt.dataset_name, opt.version, epoch), map_location='cpu')

enc.load_state_dict({k.replace('module.', ''): v for k, v in pretrained_enc.items()})
dec.load_state_dict({k.replace('module.', ''): v for k, v in pretrained_dec.items()})

dec = amp.initialize(dec, opt_level='O0')

enc.eval()
dec.eval()

print("start test: ")
time_used = []
avg_loss = [[], [], [], []]
avg_loss_x = [[], [], [], []]
for i, (x, y) in enumerate(test_loader):
    
    x = x[:,[0,2,3],:,:]
    y = y[:,[0,2,3],:,:]
    x = x[:,:opt.channels,:,:]
    y = y[:,:opt.channels,:,:]
    # print(x.shape, y.shape)
    st = time.time()
    if cuda:
        x = x.cuda()
        y = y.cuda()
    
    # prev = time.time()
    with torch.no_grad():
        y_hat = dec(x)
        x_hat = enc(y)
    # time_used.append(time.time()-prev)
    # print(torch.median(x), torch.median(y), torch.median(y_hat), torch.median(x_hat))
    # y_hat = (y_hat - y_hat.min()) / (y_hat.max() - y_hat.min())
    y_hat[y_hat > 1.] = 1.
    x_hat[x_hat > 1.] = 1.
    # print(x.min(), y.max(), y.min(), y.max(), y_hat.min(), y_hat.max())
    ed = time.time()
    time_used.append(ed - st)
    ssim_loss = ssim1(y_hat, y)
    ssim_loss_yx = ssim1(x_hat, x)
    # print(y_hat.cpu().numpy().dtype, y_hat.cpu().numpy().dtype.type)
    sys.stdout.write('\ninput %04d takes time of %.8f, MSELoss: %f, L1Loss: %f, SSIM: %f, PSNR: %f' % (
                        i + 1, ed - st, criterion(y_hat, y).item(), criterion1(y_hat, y).item(), ssim_loss, peak_signal_noise_ratio(y_hat.cpu().numpy(), y.cpu().numpy())))
    sys.stdout.write('\ninput %04d takes time of %.8f, MSELoss: %f, L1Loss: %f, SSIM: %f, PSNR: %f (y->x)' % (
                        i + 1, ed - st, criterion(x_hat, x).item(), criterion1(x_hat, x).item(), ssim_loss_yx, peak_signal_noise_ratio(x_hat.cpu().numpy(), x.cpu().numpy())))
    # sys.stdout.write('\nAverage MSELoss: %f, L1Loss: %f, SSIM: %f' % (avg_loss[0]/(i+1), avg_loss[1]/(i+1), avg_loss[2]/(i+1)))

    if opt.batch_size == 4:
        def convert4to1(data):
            # print(data.shape)
            data_row1 = torch.cat((data[0,:,:,:], data[1,:,:,:]), 2)
            data_row2 = torch.cat((data[2,:,:,:], data[3,:,:,:]), 2)
            # print(data_row1.shape)
            return torch.cat((data_row1, data_row2), 1)

        x = convert4to1(x).unsqueeze(0)
        y = convert4to1(y).unsqueeze(0)
        y_hat = convert4to1(y_hat).unsqueeze(0)
        x_hat = convert4to1(x_hat).unsqueeze(0)
    diff = torch.abs(y_hat - y)
    avg_loss[0].append(criterion(y_hat, y).item())
    avg_loss[1].append(criterion1(y_hat, y).item())
    avg_loss[2].append(ssim_loss.item())
    avg_loss[3].append(peak_signal_noise_ratio(y_hat.cpu().numpy(), y.cpu().numpy()))
    avg_loss_x[0].append(criterion(x_hat, x).item())
    avg_loss_x[1].append(criterion1(x_hat, x).item())
    avg_loss_x[2].append(ssim_loss_yx.item())
    avg_loss_x[3].append(peak_signal_noise_ratio(x_hat.cpu().numpy(), x.cpu().numpy()))
    if opt.test_epoch == -1:
        show_tif(x.squeeze(), y.squeeze(), y_hat.squeeze(), x_hat.squeeze(),
                       name='images/%s/version_%s/test/ft_%s' % (opt.dataset_name, opt.version, i), cuda=cuda, save_y=True)
        y_hat_expand = torch.cat((y_hat[:, 0, :, :], y_hat[:, 1, :, :], y_hat[:, 2, :, :]), 2)
        save_image(y_hat_expand, "images/%s/version_%s/test/ft_yhat_expand_%s.png" % (opt.dataset_name, opt.version, i), normalize=False)
    # elif i % 5 == 0:
    else:
        image_grid = show_tif(x.squeeze(), y.squeeze(), y_hat.squeeze(), x_hat.squeeze(),
                       name='images/%s/version_%s/test/%s' % (opt.dataset_name, opt.version, i), cuda=cuda, save_y=True, save_x=True)

# avg_loss[0] /= len(test_loader)
# avg_loss[1] /= len(test_loader)
# avg_loss[2] /= len(test_loader)
print('\nTime: ', np.mean(time_used))
with open(os.path.join('result', '%s_v%s_epoch_%s.txt' % (opt.dataset_name, opt.version, str(opt.test_epoch))), 'w') as f:
    f.write(str(opt))
    f.write('\nAverage MSELoss: %f, L1Loss: %f, SSIM: %f, PSNR: %f' % (np.mean(avg_loss[0]), np.mean(avg_loss[1]), np.mean(avg_loss[2]), np.mean(avg_loss[3])))
    f.write('\nStd: %f, %f, %f, %f' % (np.std(avg_loss[0]), np.std(avg_loss[1]), np.std(avg_loss[2]), np.std(avg_loss[3])))
    f.write('\nAverage MSELoss: %f, L1Loss: %f, SSIM: %f, PSNR: %f (y->x)\n' % (np.mean(avg_loss_x[0]), np.mean(avg_loss_x[1]), np.mean(avg_loss_x[2]), np.mean(avg_loss_x[3])))
    for i in avg_loss:
        f.write(str(i))
        f.write('\n')
    for i in avg_loss_x:
        f.write(str(i))
        f.write('\n')   
sys.stdout.write('\nAverage MSELoss: %f, L1Loss: %f, SSIM: %f, PSNR: %f' % (np.mean(avg_loss[0]), np.mean(avg_loss[1]), np.mean(avg_loss[2]), np.mean(avg_loss[3])))
sys.stdout.write('\nAverage MSELoss: %f, L1Loss: %f, SSIM: %f, PSNR: %f (y->x)' % (np.mean(avg_loss_x[0]), np.mean(avg_loss_x[1]), np.mean(avg_loss_x[2]), np.mean(avg_loss_x[3])))
