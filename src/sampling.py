import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def gan_sampling(decoder, num_samples, z_dim):
    with torch.no_grad():
        sample = torch.randn((num_samples, z_dim))
        decoded = decoder(sample)
        return decoded