import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def gan_sampling(generator, z_dim, num_samples):
    z = torch.randn(num_samples, z_dim).to(DEVICE)
    gz = generator(z)
    return gz
    
def dcgan_sampling(generator, z_dim, num_samples):
    z = torch.randn(num_samples, z_dim, 1, 1).to(DEVICE)
    gz = generator(z)
    return gz

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = torch.randn((n_samples, latent_dim, 1, 1))
    return x_input

def interpolate_points(p1, p2, n_steps=10):
	# interpolate ratios between the points
    ratios = torch.linspace(0, 1, steps=n_steps)
	# linear interpolate vectors
    vectors = torch.zeros((n_steps, 100, 1, 1))
    for idx, ratio in enumerate(ratios):
        v = (1.0 - ratio) * p1 + ratio * p2
        vectors[idx] = v
    return vectors

