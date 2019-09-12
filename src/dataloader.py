import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np

class DataLoader():
    def __init__(self, directories, batch_size, dataset):
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
            train_set = datasets.MNIST(root=self.root, train=True, download=True)
            test_set = datasets.MNIST(root=self.root, train=False, download=True)
        else:
            raise ValueError("DATASET N/A!")

        self.input_dim = np.prod(self.img_dims)
        self.with_labels = True
        self.dataset = dataset

        self._set_data_loader(train_set, test_set)
        self.num_train_batches = len(self.train_loader)
        self.num_test_batches = len(self.test_loader)
        # could also call len(self.train_loader.dataset) but is more flexible this way
        self.num_train_samples = self.num_train_batches*self.batch_size
        self.num_test_samples = self.num_test_batches*self.batch_size

    def _set_data_loader(self, train_set, test_set):
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set,\
            batch_size=self.batch_size, drop_last=True, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set,\
            batch_size=self.batch_size, drop_last=True, shuffle=True)

    def get_new_test_data_loader(self, sampler=None):
        if self.dataset.lower() == "mnist":
            test_set = datasets.MNIST(root=self.root, train=False, transform=transforms.ToTensor(), download=True)
        batch_size = 1 if sampler is not None else self.batch_size
        shuffle = sampler is None
        drop_last = sampler is None
        return torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=shuffle, sampler=sampler, drop_last=drop_last)
