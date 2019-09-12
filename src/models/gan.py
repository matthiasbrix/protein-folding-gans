import torch.nn as nn
import torch

# learns to distinguish samples from the true distribution and the model distr.
# want to learn the generator distribution p_g over data x
def Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        pass

    def forward(self, z):
        pass

# rep the prob that x from the data rather than p_g
# outputs a single scalar.
def Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
    def forward(self, x):
        pass

class Gan(nn.Module):
    def __init__(self, input_dim, z_dim):
        super(Gan, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

    # objective is: min_G max_D (D, G) = E_x [log D(x)] + E_z [log(1-D(G(z)))]
    def loss_function():
        pass

    def forward(x):
        # TODO: gen noise for G
        gen = self.generator(z)
        disc = self.discriminator(x)
        # TODO: z_space?
        pass
        

    