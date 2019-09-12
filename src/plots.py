import numpy as np
import torch
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATASETS = {
    "mnist": "MNIST"
}

def _xticks(ls, ticks_rate):
    labels = np.arange(1, len(ls)+2, (len(ls)//ticks_rate))
    labels[1:] -= 1
    labels[-1] = len(ls)
    return labels.astype(int)

# Plotting train and test losses
def plot_losses(solver, train_loss_history, test_loss_history):
    if len(train_loss_history) == 1 or len(test_loss_history) == 1:
        print("No plots, just 1 epoch. Need > 1.")
        return
    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, len(train_loss_history)+1), train_loss_history, label="Train")
    plt.plot(np.arange(1, len(test_loss_history)+1), test_loss_history, label="Test")
    ticks_rate = 4 if len(train_loss_history) >= 4 else len(train_loss_history)
    plt.xticks(_xticks(train_loss_history, ticks_rate))
    plt.title("Loss on data set {}, dim(z)={}".format(DATASETS[solver.data_loader.dataset.lower()], solver.model.z_dim))
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), fancybox=True, shadow=True, ncol=5)
    plt.subplots_adjust(left=0.2, right=0.85, top=0.9, bottom=0.25)
    plt.grid(True, linestyle='-.')
    if solver.data_loader.directories.make_dirs:
        plt.savefig(solver.data_loader.directories.result_dir + "/" + "plot_losses_" +\
            solver.data_loader.dataset + "_z=" + str(solver.model.z_dim) + ".png")