import argparse
import datetime
import itertools
import os
import sys
import time
import datetime

import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchsummary import summary

import model
import model_new
import pytorch_msssim
from utils import L1_Charbonnier_loss, LambdaLR, MyDataSet, show_tif, weights_init_normal

print('logs/' + "{0:%Y-%m-%dT%H-%M-%s/}".format(datetime.datetime.now()))
writer = SummaryWriter('logs/' + "{0:%Y-%m-%dT%H-%M-%s/}".format(datetime.datetime.now()))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
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
    parser.add_argument("--re_size", type=int, default=256, help="size of image")
    parser.add_argument("--crop_size", type=int, default=256, help="size of cropped image")
    parser.add_argument("--channels", type=int, default=4, help="number of image channels")
    # parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
    parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
    parser.add_argument("--loss", type=str, default="MSELoss", help="loss function")
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument("--model", default="ResNet", type=str, help="model for training")
    parser.add_argument("--version", default=0, type=int, help="version")
    parser.add_argument("--no_augmentation", action='store_false', help="no image augmentation")
    parser.add_argument("--no_amp", action='store_false', help="no amp")
    parser.add_argument("--no_dist", action='store_false', help="no distribution")
    opt = parser.parse_args()
    print(opt)
    return opt


opt = parse_args()
os.makedirs("images/%s/version_%s" % (opt.dataset_name, opt.version), exist_ok=True)
os.makedirs("saved_models/%s/version_%s" % (opt.dataset_name, opt.version), exist_ok=True)
input_shape = (opt.channels, opt.re_size, opt.re_size)

cuda = torch.cuda.is_available()
torch.cuda.set_device(0)
use_amp = opt.no_amp
use_dist = opt.no_dist
if use_amp:
    scaler = torch.cuda.amp.GradScaler()

if use_dist:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(opt.local_rank)

transform = transforms.Compose([
    # transforms.Resize(opt.re_size),
    transforms.ToTensor(),
])

train_set = MyDataSet(opt.data_path, opt.target_path, transform, augmentation=opt.no_augmentation, crop_size=(opt.crop_size, opt.crop_size))
test_set = MyDataSet(opt.data_path.replace('train', 'test'), opt.target_path.replace('train', 'test'), transform, augmentation=opt.no_augmentation, crop_size=(opt.crop_size, opt.crop_size))
if use_dist:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, num_workers=opt.n_cpu, pin_memory=True,
                              sampler=train_sampler)
else:
    train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, num_workers=opt.n_cpu, pin_memory=True,)
test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=True, num_workers=opt.n_cpu, )
print("trainset size: {}, testset size: {}".format(len(train_loader), len(test_loader)))


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
#     dec = model.ResNet(input_shape, opt.n_residual_blocks)
#     # dec = model.UNet(input_shape[0], input_shape[0])
#     # enc = model.ResNet(input_shape, opt.n_residual_blocks)
# elif opt.model == 'UwUNet':
#     # enc = model.UwUNet(input_shape[0], input_shape[0])
#     enc = model.UNet(input_shape[0], input_shape[0])
#     dec = model.UwUNet(input_shape[0], input_shape[0])
# elif opt.model == 'ConvNeXt':
#     enc = model_convnext.ConvNeXt(in_chans=input_shape[0], out_chans=input_shape[0], )
#     dec = model_convnext.ConvNeXt(in_chans=input_shape[0], out_chans=input_shape[0], )
# print(summary(enc, (opt.channels, opt.re_size, opt.re_size), device='cuda' if cuda else 'cpu'))
# print(summary(dec, (opt.channels, opt.re_size, opt.re_size), device='cuda' if cuda else 'cpu'))

### myNet
enc = model_new.UNet(input_shape[0], input_shape[0])
dec = model_new.unmixingNet(input_shape[0], input_shape[0])


optimizer = torch.optim.Adam(itertools.chain(dec.parameters(), enc.parameters()), lr=opt.lr)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(opt.epochs, 0, opt.epochs // 2).step)
# lr_scheduler = CosineLRScheduler(optimizer, opt.lr, 1e-6, opt.epochs, warmup_t=20,)

criterion_mse = nn.MSELoss()
criterion_l1 = nn.L1Loss()
criterion_msssim = pytorch_msssim.SSIM()
# criterion_charbonnier = L1_Charbonnier_loss()
# if opt.loss == "L1Loss":
#     criterion = nn.L1Loss()
# criterion_msssim_l1 = pytorch_msssim.MS_SSIM_L1_LOSS()

fake_img = torch.randn(1, opt.channels, opt.re_size, opt.re_size)

if cuda:
    enc = enc.cuda()
    dec = dec.cuda()
    criterion_mse.cuda()
    # criterion_charbonnier.cuda()
    criterion_l1.cuda()
    criterion_msssim.cuda()
    # criterion_msssim_l1.cuda()

    fake_img = fake_img.cuda()
    writer.add_graph(enc, fake_img)
    writer.add_graph(dec, fake_img)

    if use_dist:
        enc = DistributedDataParallel(enc)
        dec = DistributedDataParallel(dec)

if opt.epoch == 0:
    enc.apply(weights_init_normal)
    dec.apply(weights_init_normal)

if opt.local_rank == 0:
    print(summary(enc, (opt.channels, opt.re_size, opt.re_size), device='cuda' if cuda else 'cpu'))
    print(summary(dec, (opt.channels, opt.re_size, opt.re_size), device='cuda' if cuda else 'cpu'))

# alpha = 0.84

prev = time.time()
for epoch in range(1, opt.epochs + 1):
    for i, (data, target) in enumerate(train_loader):
        if cuda:
            data = data.cuda()
            target = target.cuda()

        enc.train()
        dec.train()
        optimizer.zero_grad()
        # print(data.shape, target.shape)
        with torch.cuda.amp.autocast():
            y_hat = dec(data)
            x_hat = enc(y_hat)
        # print(data.shape, y_hat.shape, target.shape)
            loss_ssim1 = -torch.log(1 + criterion_msssim(y_hat, target)) / 1000
        # loss_xy = loss_ssim1 + criterion_mse(y_hat, target) * 10 + criterion_l1(y_hat, target)
        # loss_xy = criterion_charbonnier(y_hat, target)
            loss_xy = criterion_mse(y_hat, target)# + loss_ssim1
        # loss_xy = alpha * loss_ssim1 + (1 - alpha) * criterion_l1(y_hat, target)
        # loss_xy = criterion_msssim_l1(y_hat.cuda(0), target.cuda(0))
        # loss_xy = criterion_l1(y_hat, target)
        # loss_l1 = F.l1_loss(y_hat, target)
        # x_hat = enc(y_hat)
        # if opt.loss == 'MSE':
            loss_ssim2 = -torch.log(1 + criterion_msssim(x_hat, data)) / 1000
        # loss_yx = loss_ssim2 + criterion_mse(x_hat, data) * 10 + criterion_l1(x_hat, data)
        # else:
        # loss_yx = criterion_charbonnier(x_hat, data)
            loss_yx = criterion_mse(x_hat, data)# + loss_ssim2
        # loss_yx = alpha * loss_ssim2 + (1 - alpha) * criterion_l1(x_hat, data)
        # loss_yx = criterion_msssim_l1(x_hat.cuda(0), data.cuda(0))
        # loss_yx = criterion_l1(x_hat, data)

            loss = loss_xy * 0.5 + loss_yx * 0.5

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            optimizer.step()

        batches_done = epoch * len(train_loader) + i
        batches_left = opt.epochs * len(train_loader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev))
        prev = time.time()

        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch, %d/%d] [Loss: %f] [Loss xy: %f, loss yx: %f, loss_ssim: %f] ETA: %s"
            % (
                epoch,
                opt.epochs,
                i,
                len(train_loader),
                loss.item(),
                # loss_l1.item(),
                loss_xy.item(),
                loss_yx.item(),
                loss_ssim1.item(),
                time_left,
            )
        )

        writer.add_scalar('%s_loss/loss' % (opt.dataset_name), loss, epoch)
        writer.add_scalar('%s_loss/loss_xy' % (opt.dataset_name), loss_xy, epoch)
        writer.add_scalar('%s_loss/loss_yx' % (opt.dataset_name), loss_yx, epoch)
        # writer.add_scalar('%s_loss/loss_l1' % (opt.dataset_name), loss_l1, epoch)

        if i % (len(train_set)) == 0:
            (x, y) = next(iter(test_loader))
            enc.eval()
            dec.eval()
            if cuda:
                x = x.cuda()
                y = y.cuda()
            y_hat = dec(x)
            x_hat = enc(y)

            image_grid = show_tif(x.squeeze(), y.squeeze(), y_hat.squeeze(), x_hat.squeeze(), name='images/%s/version_%s/%s' % (opt.dataset_name, opt.version, batches_done), cuda=cuda)
            
    
    lr_scheduler.step()
    if epoch % 20 == 0 and opt.local_rank == 0:
        torch.save(enc.state_dict(), "saved_models/%s/version_%s/enc_%s.pth" % (opt.dataset_name, opt.version, epoch))
        torch.save(dec.state_dict(), "saved_models/%s/version_%s/dec_%s.pth" % (opt.dataset_name, opt.version, epoch))

if opt.local_rank == 0:
    torch.save(enc.state_dict(), "saved_models/%s/version_%s/enc.pth" % (opt.dataset_name, opt.version))
    torch.save(dec.state_dict(), "saved_models/%s/version_%s/dec.pth" % (opt.dataset_name, opt.version))
