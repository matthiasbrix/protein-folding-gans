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

def find_closest_gt(generator, z_dim, test_loader):
    # generate some contact maps
    samples = dcgan_sampling(generator, z_dim, 25).detach().numpy()
    # so each idx represents a generated map, and value is its ground truth closest map
    min_losses = [(0, 0.0, None, None) for _ in range(len(samples))]
    gt_maps = []
    loss_func = torch.nn.MSELoss(reduction="sum")
    # on each contact map, iterate over the data set.
    for idx, generated_contact_map in enumerate(samples):
        losses = []
        # compute the MSE between generated and each ground truth
        for _, contact_map in enumerate(test_loader):
            if idx == 0:
                gt_maps.append(contact_map)
            if contact_map.shape[1] == 16:
                contact_map /= 10
            elif contact_map.shape[1] == 64:
                contact_map /= 100
            elif contact_map.shape[1] == 128:
                contact_map /= 100
            loss = loss_func(torch.FloatTensor(generated_contact_map), contact_map)
            losses.append(loss)
        # assess which has the lowest MSE! by finding argmin of losses
        losses = torch.FloatTensor(losses)
        min_loss_idx = torch.argmin(losses).item()
        # check number of gt_maps is equal to len losses
        assert len(gt_maps) == len(losses)
        assert len(test_loader) == len(gt_maps)
        #print(len(gt_maps), len(losses), min_loss_idx, generated_contact_map.shape)
        min_losses[idx] = (min_loss_idx, losses[min_loss_idx], generated_contact_map, gt_maps[min_loss_idx])
    return min_losses