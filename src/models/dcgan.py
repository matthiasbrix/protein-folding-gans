import torch.nn as nn
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, nz, res=16):
        super(Generator, self).__init__()
        if res == 128:
            self.layers = nn.Sequential(
                nn.ConvTranspose2d(nz, 512, 4, stride=4, padding=0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1), # TODO: added....
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
        self.a = nn.ConvTranspose2d(nz, 512, 4, stride=4, padding=0)
        self.b = nn.BatchNorm2d(512)
        self.c = nn.LeakyReLU(0.2)
        self.d = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.e = nn.BatchNorm2d(256)
        self.f = nn.LeakyReLU(0.2)
        self.her = nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1) # TODO: added....
        self.g = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.h = nn.BatchNorm2d(128)
        self.i = nn.LeakyReLU(0.2)
        self.j = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.k = nn.BatchNorm2d(64)
        self.l = nn.LeakyReLU(0.2)
        self.m = nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1)

    def forward(self, z):
        ''''# expects 4-d N, C, H ,W
        print(z.shape) # 0
        asd = self.a(z)
        print(asd.shape) # 1
        asd = self.b(asd)
        asd = self.c(asd)
        asd = self.d(asd)
        print(asd.shape) # 2
        asd = self.e(asd)
        asd = self.f(asd)
        asd = self.her(asd)
        print(asd.shape) # 3
        asd = self.e(asd)
        asd = self.f(asd)
        asd = self.g(asd)
        print(asd.shape) # 4
        asd = self.h(asd)
        asd = self.i(asd)
        asd = self.j(asd) # 5 
        print(asd.shape)
        asd = self.k(asd)
        asd = self.l(asd)
        gz = self.m(asd) # 6
        print(gz.shape)
        #exit(1)'''
        gz = self.layers(z)
        return gz

# rep the prob that x from the data rather than p_g
# outputs a single scalar.
class Discriminator(nn.Module):
    def __init__(self, din, dout, res=16):
        super(Discriminator, self).__init__()
        if res == 128:
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
                nn.Conv2d(256, 256, 4, stride=2, padding=1),
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
        self.a = nn.Conv2d(din, 64, 4, stride=2, padding=1)
        self.b = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.c = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.mis = nn.Conv2d(256, 256, 4, stride=2, padding=1) # TODO: added....
        self.d = nn.Conv2d(256, 512, 4, stride=2, padding=1)
        self.e = nn.Conv2d(512, dout, 4, stride=1, padding=0)
        self.f = nn.BatchNorm2d(128)
        self.g = nn.LeakyReLU(0.2)
        self.h = nn.Dropout(0.1)
        self.i = nn.Sigmoid()

    def forward(self, x):
        '''print("GENERATOR")
        print(x.shape)
        x = self.a(x)
        print(x.shape)
        x = self.b(x)
        print(x.shape)
        # TODO: insert here...
        x = self.c(x)
        print(x.shape)
        x = self.mis(x)
        print("mis", x.shape)
        x = self.d(x)
        print(x.shape)
        x = self.e(x)
        #print(x.shape)
        # TODO: Try also other dimensions on this one...
        # expects 4-D - N x C x H x W
        #exit(1)'''
        return self.layers(x)

class Dcgan(nn.Module):
    def __init__(self, input_dim, z_dim):
        super(Dcgan, self).__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.loss = nn.BCELoss()

    def kl_divergence_z(self, z):
        mean = torch.mean(z)
        variance = torch.var(z) # \sigma^2
        kl_divergence = 1/2 * torch.sum(1 + variance.log() - mean.pow(2) - variance)
        return kl_divergence