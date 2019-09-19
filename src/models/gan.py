import torch.nn as nn
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# learns to distinguish samples from the true distribution and the model distr.
# want to learn the generator distribution p_g over data x
class Generator(nn.Module):
    def __init__(self, din, dout, img_dims):
        super(Generator, self).__init__()
        self.img_dims = img_dims
        self.layers = nn.Sequential(
            nn.Linear(din, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, dout),
            nn.Tanh()
        )

    def forward(self, z):
        gz = self.layers(z)
        gz = gz.view(gz.size(0), *self.img_dims)
        return gz

# rep the prob that x from the data rather than p_g
# outputs a single scalar.
class Discriminator(nn.Module):
    def __init__(self, din, dout):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(din, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, dout),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)

class Gan(nn.Module):
    def __init__(self, input_dim, z_dim):
        super(Gan, self).__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.loss = nn.BCELoss()