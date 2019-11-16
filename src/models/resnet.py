import torch.nn as nn

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size

# The residual block takes an input with in_channels, applies some blocks of convolutional layers
# to reduce it to out_channels and sum it up to the original input
# If their sizes mismatch, then the input goes into an identity
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


if __name__ == "__main__":
    #conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)
    #conv = conv3x3(in_channels=32, out_channels=64)
    ResidualBlock(32, 64)

    dummy = torch.ones((1, 1, 1, 1))
    block = ResidualBlock(1, 64)
    res = block(dummy)
    print(res)