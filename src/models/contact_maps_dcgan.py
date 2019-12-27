import torch.nn as nn
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, nz, new_arch=True, res=16):
        super(Generator, self).__init__()
        if res == 256:
            self.layers = nn.Sequential(
                nn.ConvTranspose2d(nz, 1024, 4, stride=4, padding=0), # 4x4
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1), # 8x8
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1), # 16x16
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), # 32x32
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 64x64
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # 128x128
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1), # 256x256
            )
        elif res == 128:
            self.layers = nn.Sequential(
                nn.ConvTranspose2d(nz, 512, 4, stride=4, padding=0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(256, 128, 4, stride=4, padding=0),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1)
            )
        elif res == 128 and new_arch:
            self.layers = nn.Sequential(
                nn.ConvTranspose2d(nz, 512, 4, stride=4, padding=0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(512, 512, 5, stride=1, padding=2),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(64, 64, 5, stride=1, padding=2),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1)
            )
        elif res == 64:
            self.layers = nn.Sequential(
                nn.ConvTranspose2d(nz, 512, 4, stride=1, padding=0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1)
            )
        elif res == 64 and new_arch:
            self.layers = nn.Sequential(
                nn.ConvTranspose2d(nz, 512, 4, stride=1, padding=0), # 4x4
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1), # 4x4, 64 - 4 / 1 + 1
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), # 8x8
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1), # 8x8
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 16x16
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1), # 16x16
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # 32x32
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1), # 32x32
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1) # 64x64
            )
        else:
            self.layers = nn.Sequential(
                nn.ConvTranspose2d(nz, 512, 4, stride=1, padding=0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(64, 1, 3, stride=1, padding=1)
            )

    def forward(self, z):
        # expects 4-d N, C, H ,W
        gz = self.layers(z)
        return gz

# rep the prob that x from the data rather than p_g
# outputs a single scalar.
class Discriminator(nn.Module):
    def __init__(self, din, dout, new_arch=True, res=16):
        super(Discriminator, self).__init__()
        if res == 256:
            self.layers = nn.Sequential(
                nn.Conv2d(din, 64, 4, stride=2, padding=1), # 128x128
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(64, 128, 4, stride=2, padding=1), # 64x64
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(128, 256, 4, stride=2, padding=1), # 32x32
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(256, 512, 4, stride=2, padding=1), # 16x16
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(512, 512, 4, stride=2, padding=1), # 8x8
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(512, 1024, 4, stride=2, padding=1), # 4x4
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(1024, dout, 4, stride=1, padding=0), # 1x1
                nn.Sigmoid()
            )
        elif res == 128:
            self.layers = nn.Sequential(
                nn.Conv2d(din, 64, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(128, 256, 4, stride=4, padding=0),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(256, 512, 4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(512, dout, 4, stride=1, padding=0),
                nn.Sigmoid()
            )
        elif res == 128 and new_arch:
            self.layers = nn.Sequential(
                nn.Conv2d(din, 64, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(64, 64, 5, stride=1, padding=2),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(128, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(256, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(256, 512, 4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(512, 512, 5, stride=1, padding=2),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(512, dout, 4, stride=1, padding=0),
                nn.Sigmoid()
            )
        elif res == 64:
            self.layers = nn.Sequential(
                nn.Conv2d(din, 64, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(128, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(256, 512, 4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(512, dout, 4, stride=1, padding=0),
                nn.Sigmoid()
            )
        elif res == 64 and new_arch:
            self.layers = nn.Sequential(
                nn.Conv2d(din, 64, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(128, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(256, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(256, 512, 4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(512, 512, 3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(512, dout, 4, stride=1, padding=0),
                nn.Sigmoid()
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(din, 64, 3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(128, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(256, 512, 3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Conv2d(512, dout, 4, stride=1, padding=0),
                nn.Sigmoid()
            )

    def forward(self, x):
        # expects 4-D - N x C x H x W
        return self.layers(x)

class CmDcgan(nn.Module):
    def __init__(self, input_dim, z_dim):
        super(CmDcgan, self).__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.loss = nn.BCELoss()
