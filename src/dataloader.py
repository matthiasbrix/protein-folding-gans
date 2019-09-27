import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import h5py
# from openprotein.util import contruct_dataloader_from_disk

def contruct_dataloader_from_disk(filename, minibatch_size, drop_last=False):
    return torch.utils.data.DataLoader(H5PytorchDataset(filename), batch_size=minibatch_size,
                                       shuffle=True, collate_fn=H5PytorchDataset.merge_samples_to_minibatch,\
                                        drop_last=drop_last)

def calc_pairwise_distances(chain_a, chain_b, use_gpu):
    distance_matrix = torch.Tensor(chain_a.size()[0], chain_b.size()[0]).type(torch.float)
    # add small epsilon to avoid boundary issues
    epsilon = 10 ** (-4) * torch.ones(chain_a.size(0), chain_b.size(0))
    if use_gpu:
        distance_matrix = distance_matrix.cuda()
        epsilon = epsilon.cuda()

    for i, row in enumerate(chain_a.split(1)):
        distance_matrix[i] = torch.sum((row.expand_as(chain_b) - chain_b) ** 2, 1).view(1, -1)

    return torch.sqrt(distance_matrix + epsilon)

class H5PytorchDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        super(H5PytorchDataset, self).__init__()

        self.h5pyfile = h5py.File(filename, 'r')
        self.num_proteins, self.max_sequence_len = self.h5pyfile['primary'].shape

    def __getitem__(self, index):
        mask = torch.Tensor(self.h5pyfile['mask'][index,:]).type(dtype=torch.uint8)
        prim = torch.masked_select(torch.Tensor(self.h5pyfile['primary'][index,:]).type(dtype=torch.long), mask)
        tertiary = torch.Tensor(self.h5pyfile['tertiary'][index][:int(mask.sum())]) # max length x 9
        return  prim, tertiary, mask

    def __len__(self):
        return self.num_proteins

    def merge_samples_to_minibatch(samples):
        samples_list = []
        for s in samples:
            samples_list.append(s)
        # sort according to length of aa sequence
        samples_list.sort(key=lambda x: len(x[0]), reverse=True)
        return zip(*samples_list)

class DataLoader():
    def __init__(self, directories, batch_size, dataset, training_file=None, validation_file=None, residue_fragments=128, atom=None):
        self.directories = directories
        self.data = None
        self.n_classes = None
        self.c = None
        self.h = None
        self.w = None
        self.img_dims = None
        self.batch_size = batch_size
        self.dataset = dataset

        self.root = directories.data_dir_prefix+dataset

        if dataset.lower() == "mnist":
            self.n_classes = 10
            self.img_dims = (self.c, self.h, self.w) = (1, 28, 28)
            train_set = datasets.MNIST(root=self.root, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ), train=True, download=True)
            test_set = datasets.MNIST(root=self.root, transform=transforms.ToTensor(), train=False, download=True)
            self._set_data_loader(train_set, test_set)
        elif dataset.lower() == "proteins":
            # Takes a matrix N x 3
            self.train_loader = contruct_dataloader_from_disk(training_file, batch_size, drop_last=True)
            self.validation_loader = contruct_dataloader_from_disk(validation_file, batch_size, drop_last=True)
            self.img_dims = (residue_fragments, residue_fragments)
            self.residue_fragments = residue_fragments
            self.atom = atom
            self.num_val_batches = len(self.validation_loader)
            self.num_val_samples = self.validation_loader.dataset.__len__() # self.num_val_batches*self.batch_size
        else:
            raise ValueError("DATASET N/A!")

        self.with_labels = dataset not in ["proteins"]
        self.input_dim = np.prod(self.img_dims)
        self.dataset = dataset

        self.num_train_batches = len(self.train_loader)
        self.num_train_samples = self.train_loader.dataset.__len__() # self.num_train_batches*self.batch_size

        print(self.img_dims, self.input_dim, len(self.train_loader), self.num_train_batches, self.num_train_samples, self.num_val_batches, self.num_val_samples)

    def _set_data_loader(self, train_set, test_set):
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set,\
            batch_size=self.batch_size, drop_last=True, shuffle=True)
        self.validation_loader = torch.utils.data.DataLoader(dataset=test_set,\
            batch_size=self.batch_size, drop_last=True, shuffle=True)

    def max_sequence_length(self):
        return self.train_loader.dataset.max_sequence_len

    def get_new_test_data_loader(self):
        if self.dataset.lower() == "mnist":
            test_set = datasets.MNIST(root=self.root, train=False, transform=transforms.ToTensor(), download=True)
        return torch.utils.data.DataLoader(dataset=test_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
