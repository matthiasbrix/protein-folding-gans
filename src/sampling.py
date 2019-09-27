import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def gan_sampling(generator, z_dim, num_samples):
    z = torch.randn(num_samples, z_dim).to(DEVICE)
    gz = generator(z)
    return gz
    
def dcgan_sampling(generator, z_dim, img_dims, num_samples):
    z = torch.randn(num_samples, z_dim, *img_dims).to(DEVICE)
    gz = generator(z)
    return gz