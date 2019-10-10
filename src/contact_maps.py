import h5py
import torch
import numpy as np
import os.path

# TODO: need to filter out the longer data than needed so that we can have nicer contact map structures
# TODO: calc internally of fragments, for each protein
# TODO: take each fragment out and then create a contact matrix for it, so get like (num_proteins*) X fragment_length X fragment_length
# 1 chain = 1 protein
# TODO: vi skal for hvert protein take fragmenter ud og for hver fragment laver vi en contact matrice..., dvs. er længden 16 så får vi 16 x 16
# 1 chains -> 76 residues/amino acids -> 76 x 76 contact map
# (original_aa_string, actual_coords_list, mask) = train_batch
# self.max_num_fragments = self.solver.data_loader.max_sequence_length()//self.solver.data_loader.residue_fragments

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

# expecting residue_length x 3 x 3
def atom_filter(residues, atom):
    if atom == "nitrogen":
        return residues[:, 0]
    elif atom == "calpha":
        return residues[:, 1]
    elif atom == "cprime":
        return residues[:, 2]
    else:
        return residues

def pad_fragment(fragment, fragment_length):
    tmp = np.zeros((fragment_length, 3))
    tmp[:fragment.shape[0]] = fragment
    return torch.Tensor(tmp).type(torch.float) # dim is then fragment_length x 3

def compute_contact_maps(residues, num_fragments_extract, fragment_length):
    contact_maps = torch.zeros((num_fragments_extract, fragment_length, fragment_length))
    for fragment_id in range(num_fragments_extract):
        start = fragment_id*fragment_length
        end = (fragment_id+1)*fragment_length
        # extracting fragment
        fragment = residues[start:end]
        # in case we extract remaining, that is < residue_fragments size, we pad
        if fragment.shape[0] < fragment_length:
            pad_fragment(fragment, fragment_length)
        # compute matrix on the fragments
        contact_maps[fragment_id] = calc_pairwise_distances(fragment, fragment, torch.cuda.is_available())
    return contact_maps

def create_contact_maps(file_name, fragment_length, atom):
    h5pyfile = h5py.File(file_name, 'r')
    num_proteins, max_sequence_len = h5pyfile['primary'].shape
    all_contact_maps = []
    broken_prots = 0
    for protein_idx in range(num_proteins): # have structure num_proteins x vairable length (amino acids) x 9
        mask = torch.Tensor(h5pyfile['mask'][protein_idx, :]).type(dtype=torch.uint8)
        #prim = torch.masked_select(torch.Tensor(h5pyfile['primary'][protein_idx, :]).type(dtype=torch.long), mask)
        tertiary = torch.Tensor(h5pyfile['tertiary'][protein_idx][:int(mask.sum())])
        # some of the tertiaries have just NaN entires as their coordinates (even though we mask
        if torch.isnan(tertiary).any().item():
            broken_prots += 1
            continue
        protein_length = len(tertiary)
        num_fragments_extract = protein_length//fragment_length # non-overlapping fragments
        residues = np.reshape(tertiary, ((protein_length, 3, 3)))
        # filter here residues by specific atoms
        residues = atom_filter(residues, atom)
        contact_maps = compute_contact_maps(residues, num_fragments_extract, fragment_length)

        all_contact_maps.append(contact_maps)
    print("{0} broken proteins read out of {1}!".format(broken_prots, num_proteins))
    contact_maps = torch.cat(all_contact_maps)
    # serialize contact maps and cache them
    dump_contact_maps(contact_maps, file_name)
    return contact_maps

# parses the contact maps to binary files for caching
def dump_contact_maps(contact_maps, file_name):
    tmp = file_name.split("/")
    fn = "/".join(tmp[:-1])+"/"+tmp[-1].split(".")[0]
    cache_file = fn+"_contact_maps.dat"
    print("Caching the contact maps to {}".format(cache_file))
    torch.save(contact_maps, cache_file)

# 1. get from file in data/proteins/contact_map_xx.dat (xx is 50 or so, so density)
# 2. if not existing, call create_contact_map
def get_contact_maps(file_name, fragment_length, atom):
    tmp = file_name.split("/")
    fn = "/".join(tmp[:-1])+"/"+tmp[-1].split(".")[0]
    cache_file = fn+"_contact_maps.dat"
    if os.path.isfile(cache_file):
        print("Reading cache file in {}".format(cache_file))
        return torch.load(cache_file)
    else:
        print("Creating the contact maps for {} as no cache was found!".format(file_name))
        return create_contact_maps(file_name, fragment_length, atom)
