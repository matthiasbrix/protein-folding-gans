import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATASETS = {
    "mnist": "MNIST",
    "proteins": "Proteins"
}

def _xticks(ls, ticks_rate):
    labels = np.arange(1, len(ls)+2, (len(ls)//ticks_rate))
    labels[1:] -= 1
    labels[-1] = len(ls)
    return labels.astype(int)

def plot_losses(solver, g_losses, d_losses):
    if len(g_losses) == 1 or len(d_losses) == 1:
        print("No plots, just 1 epoch. Need > 1.")
        return
    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, len(g_losses)+1), g_losses, label="G loss")
    plt.plot(np.arange(1, len(d_losses)+1), d_losses, label="D loss")
    ticks_rate = 4 if len(g_losses) >= 4 else len(g_losses)
    plt.xticks(_xticks(g_losses, ticks_rate))
    plt.title("Loss on data set {}, dim(z)={}".format(DATASETS[solver.data_loader.dataset.lower()], solver.model.z_dim))
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), fancybox=True, shadow=True, ncol=5)
    plt.subplots_adjust(left=0.2, right=0.85, top=0.9, bottom=0.25)
    plt.grid(True, linestyle='-.')
    if solver.data_loader.directories.make_dirs:
        plt.savefig(solver.data_loader.directories.result_dir + "/" + "plot_losses_" +\
            solver.data_loader.dataset + "_z=" + str(solver.model.z_dim) + ".png")

def plot_z_samples(gz, save_fig=False, result_dir=None, dataset=None):
    grid_img = torchvision.utils.make_grid(gz, nrow=10)
    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.imshow(grid_img.permute(1, 2, 0).detach().numpy())
    if save_fig:
        torchvision.utils.save_image(grid_img, result_dir +\
            "/plot_z_samples" + dataset + ".png")

# https://stackoverflow.com/questions/53255432/saving-a-grid-of-heterogenous-images-in-python
# useful/needed for contact maps
def contact_map_grid(ims, rows=None, cols=None, fill=True, showax=False, file_name=None, show=False):
    if rows is None != cols is None:
        raise ValueError("Set either both rows and cols or neither.")

    if rows is None:
        rows = len(ims)
        cols = 1

    gridspec_kw = {'wspace': 0, 'hspace': 0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw)

    if fill:
        bleed = 0
        fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), ims):
        ax.imshow(im[0]) # assuming C, H, W
        if not showax:
            ax.set_axis_off()

    if file_name and show:
        plt.show()
        fig.savefig(file_name, bbox_inches="tight", pad_inches=0.0, transparent=False)
    elif file_name:
        fig.savefig(file_name, bbox_inches="tight", pad_inches=0.0, transparent=False)
    else:
        plt.show()
    plt.close()
