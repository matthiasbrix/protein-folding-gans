import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import h5py
import contact_maps

class ContactMapDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, batch_size, num_residue_fragments, atom):
        super(ContactMapDataset, self).__init__()
        # property that reads in the contact maps from the given file name (and residue length)
        # in format batch_size x residue_length x residue_length
        self.contact_maps = contact_maps.get_contact_maps(file_name, num_residue_fragments, atom) #.create_contact_maps(file_name, num_residue_fragments, atom)

    def __getitem__(self, index):
        return self.contact_maps[index]

    def __len__(self):
        return len(self.contact_maps)

class H5PytorchDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        super(H5PytorchDataset, self).__init__()

        self.h5pyfile = h5py.File(filename, 'r')
        self.num_proteins, self.max_sequence_len = self.h5pyfile['primary'].shape
        #print("HER", self.num_proteins, self.max_sequence_len, self.h5pyfile['tertiary'][0].shape, self.h5pyfile['tertiary'][1].shape, self.h5pyfile['mask'][0])

    def __getitem__(self, index):
        mask = torch.Tensor(self.h5pyfile['mask'][index,:]).type(dtype=torch.uint8)
        prim = torch.masked_select(torch.Tensor(self.h5pyfile['primary'][index,:]).type(dtype=torch.long), mask)
        tertiary = torch.Tensor(self.h5pyfile['tertiary'][index][:int(mask.sum())]) # max length x 9
        return prim, tertiary, mask

    def __len__(self):
        return self.num_proteins

    def merge_samples_to_minibatch(self, samples):
        samples_list = []
        for s in samples:
            samples_list.append(s)
        # sort according to length of aa sequence
        samples_list.sort(key=lambda x: len(x[0]), reverse=True)
        return zip(*samples_list)

# TODO: merge with set data loader
def construct_dataloader_from_disk(file_name, batch_size, num_residue_fragments, mode="contact_map", atom=None, drop_last=False):
    if mode == "protein":
        return torch.utils.data.DataLoader(H5PytorchDataset(file_name), batch_size=batch_size, shuffle=True,\
                                       collate_fn=H5PytorchDataset.merge_samples_to_minibatch,\
                                       drop_last=drop_last)
    elif mode == "contact_map":
        return torch.utils.data.DataLoader(ContactMapDataset(file_name, batch_size, num_residue_fragments, atom),\
                                    batch_size=batch_size, shuffle=True,\
                                    drop_last=drop_last)

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
            self.img_dims = (residue_fragments, residue_fragments)
            self.residue_fragments = residue_fragments
            # Takes a matrix N x 3
            self.train_loader = construct_dataloader_from_disk(training_file, batch_size, self.residue_fragments, atom=atom, drop_last=True)
            self.validation_loader = construct_dataloader_from_disk(validation_file, batch_size, self.residue_fragments, atom=atom, drop_last=True)
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

    def get_new_test_data_loader(self):
        if self.dataset.lower() == "mnist":
            test_set = datasets.MNIST(root=self.root, train=False, transform=transforms.ToTensor(), download=True)
        return torch.utils.data.DataLoader(dataset=test_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
