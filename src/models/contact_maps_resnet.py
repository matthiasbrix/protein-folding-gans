import torch.nn as nn

from functools import partial

"""
Inspired by https://github.com/FrancescoSaverioZuppichini/ResNet
"""

DISCRIMINATOR = False

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size

class ConvTranspose2dAuto(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size

conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)
convt3x3 = partial(ConvTranspose2dAuto, kernel_size=3, bias=False)

# The residual block takes an input with in_channels, 
# applies some blocks of convolutional layers
# to reduce it to out_channels and sum it up to the original input
# If their sizes mismatch, then the input goes into an identity
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.blocks = nn.Identity()
        self.activate = nn.LeakyReLU(0.2)
        self.shortcut = nn.Identity()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x) # F(x)
        x += residual # x
        x = self.activate(x) # sigma(F(x) + x)
        x = self.dropout(x) if DISCRIMINATOR else x # added just for disc
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

class GeneratorResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, downsampling=1, expansion=1, convt=convt3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        # Is actually upsamlping not downsampling!
        self.downsampling, self.expansion = downsampling, expansion
        self.convt = convt
        # same convolution on the shortcuts
        self.shortcut = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels, self.expanded_channels,\
                    kernel_size=1, stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)
        ) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

class DiscriminatorResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, downsampling=1, expansion=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.downsampling, self.expansion = downsampling, expansion
        self.conv = conv
        # same convolution on the shortcuts
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False)
        ) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

class GeneratorResNetBasicBlock(GeneratorResNetResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            self.convt_bn(self.in_channels, self.out_channels, bias=False, stride=self.downsampling),
            self.activate,
            self.convt_bn(self.out_channels, self.expanded_channels, bias=False),
        )

    def convt_bn(self, in_channels, out_channels, *args, **kwargs):
        return nn.Sequential(
            self.convt(in_channels, out_channels, *args, **kwargs),
            nn.BatchNorm2d(out_channels)
        )

class DiscriminatorResNetBasicBlock(DiscriminatorResNetResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            self.convolution(self.in_channels, self.out_channels, bias=False, stride=self.downsampling),
            self.activate,
            nn.Dropout(0.2),
            self.convolution(self.out_channels, self.expanded_channels, bias=False)
        )

    def convolution(self, in_channels, out_channels, *args, **kwargs):
        return self.conv(in_channels, out_channels, *args, **kwargs)

class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block, n=1, *args, **kwargs):
        super().__init__()
        # We perform up/downsampling directly by convolutional layers that have a stride of 2.
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion,
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

class ResGenNet(nn.Module):
    def __init__(self, z_dim, n):
        super(ResGenNet, self).__init__()
        self.first_gen_layer = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 1024, 3, stride=2),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2)
        )
        self.gen_layers = nn.Sequential(
            ResNetLayer(1024, 512, GeneratorResNetBasicBlock, n=n),
            ResNetLayer(512, 256, GeneratorResNetBasicBlock, n=n),
            ResNetLayer(256, 128, GeneratorResNetBasicBlock, n=n),
            ResNetLayer(128, 64, GeneratorResNetBasicBlock, n=n),
            ResNetLayer(64, 32, GeneratorResNetBasicBlock, n=n),
            ResNetLayer(32, 16, GeneratorResNetBasicBlock, n=n),
            ResNetLayer(16, 1, GeneratorResNetBasicBlock, n=n)
        ) # 7 gen layers
        self.last_gen_layer = nn.Sequential(
            nn.Conv2d(1, 1, 2, 1, padding=0) # reduce by 1 pixel because we get to 257x257
        )
    
    def forward(self, z):
        gz = self.first_gen_layer(z)
        gz = self.gen_layers(gz)
        gz = self.last_gen_layer(gz)
        return gz

class ResDiscNet(nn.Module):
    def __init__(self, n):
        super(ResDiscNet, self).__init__()
        # discriminator first and last layer
        self.first_disc_layer = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        )
        self.disc_layers = nn.Sequential(
            ResNetLayer(1, 16, DiscriminatorResNetBasicBlock, n=n),
            ResNetLayer(16, 32, DiscriminatorResNetBasicBlock, n=n),
            ResNetLayer(32, 64, DiscriminatorResNetBasicBlock, n=n),
            ResNetLayer(64, 128, DiscriminatorResNetBasicBlock, n=n),
            ResNetLayer(128, 256, DiscriminatorResNetBasicBlock, n=n),
            ResNetLayer(256, 512, DiscriminatorResNetBasicBlock, n=n),
            ResNetLayer(512, 1024, DiscriminatorResNetBasicBlock, n=n)
        )
        self.last_disc_layer = nn.Sequential(
            nn.Conv2d(1024, 1, 1, stride=2, padding=0), # reduce by 1 pixel.
            nn.Sigmoid()
        )

    def forward(self, gz):
        d = self.disc_layers(gz) # 2 x 2
        d = self.last_disc_layer(d) # 1 x 1
        return d

class CmResNet(nn.Module):
    def __init__(self, input_dim, z_dim):
        super(CmResNet, self).__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.loss = nn.BCELoss()