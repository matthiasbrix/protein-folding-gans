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
                nn.ConvTranspose2d(512, 512, 5, stride=1, padding=2), # TODO: added here
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
                nn.ConvTranspose2d(64, 64, 5, stride=1, padding=2), # TODO: added here...
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

    def forward(self, z):
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

    def forward(self, x):
        out = self.layers(x)
        return out

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