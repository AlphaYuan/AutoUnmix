import torch
import torch.nn as nn

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # print('unetdown: ', x.shape)
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, normalize=False, dropout=0.5)
        # self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        # self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 256)
        self.up5 = UNetUp(512, 128)
        self.up6 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            # nn.Tanh(),
            nn.ReLU(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        # print('unet down1: ', x.shape, self.down1)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        # d8 = self.down8(d7)
        # u1 = self.up1(d8, d7)
        u1 = self.up1(d7, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)

        return self.final(u6)


class unmixingNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, inter_channels=8):
        super(unmixingNet, self).__init__()

        self.input_channels = input_channels
        num = 0
        mutual_info_conv = []
        for chann1 in range(input_channels):
            for chann2 in range(chann1+1, input_channels):
                mutual_info_conv.append(nn.Conv2d(2, inter_channels, 3, 1, 1, bias=False))
                # mutual_info_conv.append(nn.LeakyReLU(0.2))
                num += 1
        # print(num)
        self.mi_conv = nn.ModuleList(mutual_info_conv)
        self.unet = UNet(inter_channels, inter_channels)
        self.final_conv = self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(inter_channels * num, output_channels, 4, 2, padding=1),
            # nn.Tanh(),
            nn.ReLU(),
        )

    def forward(self, x):
        num = 0
        mi_conv_unet = None
        for chann1 in range(self.input_channels):
            for chann2 in range(chann1+1, self.input_channels):
                mi_res = self.mi_conv[num](torch.cat([x[:, chann1:chann1+1, :, :], x[:, chann2:chann2+1, :, :]], axis=1))
                # print(chann1, chann2, mi_res.shape)
                mi_res = self.unet(mi_res)
                if num == 0:
                    mi_conv_unet = mi_res
                else:
                    mi_conv_unet = torch.cat((mi_conv_unet, mi_res), axis=1)
                num += 1
        final_res = self.final_conv(mi_conv_unet)
        return  final_res
