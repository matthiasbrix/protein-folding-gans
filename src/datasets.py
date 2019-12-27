import torch
import h5py
import contact_maps

class ContactMapDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, num_residue_fragments, atom, padding, test_pdb):
        super(ContactMapDataset, self).__init__()
        # property that reads in the contact maps from the given file name (and residue length)
        # in format batch_size x residue_length x residue_length
        self.contact_maps = contact_maps.get_contact_maps(file_name, num_residue_fragments, atom, padding, test_pdb)

    def __getitem__(self, index):
        return self.contact_maps[index]

    def __len__(self):
        return len(self.contact_maps)

class H5PytorchDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        super(H5PytorchDataset, self).__init__()

        self.h5pyfile = h5py.File(filename, 'r')
        self.num_proteins, self.max_sequence_len = self.h5pyfile['primary'].shape

    def __getitem__(self, index):
        mask = torch.Tensor(self.h5pyfile['mask'][index, :]).type(dtype=torch.uint8)
        prim = torch.masked_select(torch.Tensor(self.h5pyfile['primary'][index, :]).type(dtype=torch.long), mask)
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