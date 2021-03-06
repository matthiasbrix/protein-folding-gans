import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np

from datasets import H5PytorchDataset, ContactMapDataset

class DataLoader():
    def __init__(self, directories, batch_size, dataset, training_file=None,\
        residue_fragments=128, atom=None, padding="pwd_pad"):
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
        self.padding = padding

        if dataset.lower() == "mnist":
            self.n_classes = 10
            self.img_dims = (self.c, self.h, self.w) = (1, 28, 28)
            train_set = datasets.MNIST(root=self.root, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ), train=True, download=True)
            test_set = datasets.MNIST(root=self.root, transform=transforms.ToTensor(), train=False, download=True)
            self._set_data_loader(train_set, test_set)
        elif dataset.lower() == "celeba":
            self.img_dims = (self.c, self.h, self.w) = (3, 64, 64)
            train_set = datasets.ImageFolder(root=self.root,
                        transform=transforms.Compose([
                            transforms.Resize(self.img_dims[1]),
                            transforms.CenterCrop(self.img_dims[1]),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),\
                                (0.5, 0.5, 0.5)),
                        ]))
            self._set_data_loader(train_set)
        elif dataset.lower() == "proteins":
            self.img_dims = (residue_fragments, residue_fragments)
            self.residue_fragments = residue_fragments
            self.atom = atom
            self._set_data_loader(training_file)
            self.training_file = training_file
        else:
            raise ValueError("DATASET N/A!")

        self.with_labels = dataset not in ["proteins"]
        self.input_dim = np.prod(self.img_dims)

        self.num_train_batches = len(self.train_loader)
        self.num_train_samples = self.train_loader.dataset.__len__()

        print("img dims: {0}, input dim: {1}, num batches: {2}, num train samples: {3}".format(self.img_dims,\
            self.input_dim, len(self.train_loader), self.num_train_samples))
        
        stats = "num_batches: {}\n"\
                "num_train_samples: {}\n".format(self.num_train_batches, self.num_train_samples)
        
        if dataset.lower() == "proteins":
            with open(training_file+"_"+str(self.residue_fragments)+"_stats.txt", "w") as contact_map_stats:
                stats = "num_batches: {}\n"\
                    "num_train_samples: {}".format(self.num_train_batches, self.num_train_samples)
                contact_map_stats.write(stats)

    def _set_data_loader(self, train_set, test_set=None):
        if self.dataset.lower() == "mnist":
            self.train_loader = torch.utils.data.DataLoader(dataset=train_set,\
                batch_size=self.batch_size, drop_last=True, shuffle=True)
            self.test_loader = torch.utils.data.DataLoader(dataset=test_set,\
                batch_size=self.batch_size, drop_last=True, shuffle=True)
        elif self.dataset.lower() == "celeba":
            self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size,\
                shuffle=True, drop_last=True)
        elif self.dataset.lower() == "proteins":
            self.train_loader = self._construct_dataloader_from_disk(train_set, self.batch_size,\
                self.residue_fragments, atom=self.atom, drop_last=False, padding=self.padding)

    def _construct_dataloader_from_disk(self, file_name, batch_size, num_residue_fragments,\
                                        mode="contact_map", atom=None, drop_last=False, padding="pwd_pad",\
                                        test_pdb=False):
        if mode == "protein":
            return torch.utils.data.DataLoader(H5PytorchDataset(file_name), batch_size=batch_size, shuffle=True,\
                                        collate_fn=H5PytorchDataset.merge_samples_to_minibatch,\
                                        drop_last=drop_last)
        elif mode == "contact_map":
            return torch.utils.data.DataLoader(ContactMapDataset(file_name, num_residue_fragments, atom, padding,\
                                        test_pdb=test_pdb), batch_size=batch_size, shuffle=True, drop_last=drop_last)

    def get_new_test_data_loader(self, testing_file=None, batch_size=None, padding="pwd_pad", test_pdb=False):
        if self.dataset.lower() == "mnist":
            test_set = datasets.MNIST(root=self.root, train=False, transform=transforms.ToTensor(), download=True)
            return torch.utils.data.DataLoader(dataset=test_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
        elif self.dataset.lower() == "proteins":
            batch_size = batch_size if batch_size else self.batch_size
            return self._construct_dataloader_from_disk(testing_file, batch_size,\
                self.residue_fragments, atom=self.atom, drop_last=False, padding=padding, test_pdb=test_pdb)
