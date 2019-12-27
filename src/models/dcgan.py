import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz, nc, ngf=64):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), # state size. (ngf*8) x 4 x 4
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False), # state size. (ngf*4) x 8 x 8
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False), # state size. (ngf*2) x 16 x 16
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False), # state size. (ngf) x 32 x 32
            nn.Tanh() # state size. (nc) x 64 x 64
        )

    def forward(self, imgs):
        result = self.generator(imgs)
        return result

class Discriminator(nn.Module):
    def __init__(self, nc, ndf=64):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), # input is (nc) x 64 x 64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), # state size. (ndf) x 32 x 32
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), # state size. (ndf*2) x 16 x 16
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), # state size. (ndf*4) x 8 x 8
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False), # state size. (ndf*8) x 4 x 4
            nn.Sigmoid()
        )

    def forward(self, imgs):
        result = self.discriminator(imgs)
        return result

class Dcgan(nn.Module):
    def __init__(self, input_dim, z_dim):
        super(Dcgan, self).__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.loss = nn.BCELoss()