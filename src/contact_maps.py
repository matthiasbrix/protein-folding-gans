import h5py
import torch
import numpy as np
import os.path

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
    tmp = torch.zeros((fragment_length, 3))
    tmp[:fragment.shape[0]] = fragment
    return torch.Tensor(tmp).type(torch.float) # dim is then fragment_length x 3

# Pad the pairwise distance matrix
def pad_pwd(pwd, fragment_length):
    tmp = torch.zeros((fragment_length, fragment_length))
    tmp[:pwd.shape[0], :pwd.shape[1]] = pwd
    return torch.Tensor(tmp).type(torch.float)

def compute_contact_maps(residues, num_fragments_extract, fragment_length, padding):
    contact_maps = torch.zeros((num_fragments_extract, fragment_length, fragment_length))
    for fragment_id in range(num_fragments_extract):
        start = fragment_id*fragment_length
        end = (fragment_id+1)*fragment_length
        # extracting fragment
        fragment = residues[start:end]
        if padding == "fragment_pad":
            if fragment.shape[0] < fragment_length:
                fragment = pad_fragment(fragment, fragment_length)
                # compute matrix on the fragments
            contact_maps[fragment_id] = calc_pairwise_distances(fragment, fragment, torch.cuda.is_available())
        if padding == "pwd_pad":
            pwd = calc_pairwise_distances(fragment, fragment, torch.cuda.is_available())
            # in case we extract remaining, that is < residue_fragments size, we pad
            if fragment.shape[0] < fragment_length:
                pwd = pad_pwd(pwd, fragment_length)
            contact_maps[fragment_id] = pwd
        if padding == "no_pad":
            if fragment.shape[0] < fragment_length:
                contact_maps = np.delete(contact_maps, fragment_id, 0)
            else:
                pwd = calc_pairwise_distances(fragment, fragment, torch.cuda.is_available())
                contact_maps[fragment_id] = pwd
    return contact_maps

def create_contact_maps(file_name, fragment_length, atom, padding):
    h5pyfile = h5py.File(file_name, 'r')
    num_proteins, _ = h5pyfile['primary'].shape # _ is max_sequence_len
    all_contact_maps = []
    broken_prots = 0
    for protein_idx in range(num_proteins): # have structure num_proteins x vairable length (amino acids) x 9
        mask = torch.Tensor(h5pyfile['mask'][protein_idx]).type(dtype=torch.uint8)
        #prim = torch.masked_select(torch.Tensor(h5pyfile['primary'][protein_idx, :]).type(dtype=torch.long), mask)
        tertiary = torch.Tensor(h5pyfile['tertiary'][protein_idx][:int(mask.sum())])
        # some of the tertiaries have just NaN entires as their coordinates (even though we mask
        if torch.isnan(tertiary).any().item():
            broken_prots += 1
            continue
        protein_length = len(tertiary)
        residues = np.reshape(tertiary, ((protein_length, 3, 3)))
        # filter here residues by specific atoms
        residues = atom_filter(residues, atom)
        num_fragments_extract = max(protein_length, fragment_length)//fragment_length # non-overlapping fragments
        #print(residues.shape, num_fragments_extract, fragment_length)
        contact_maps = compute_contact_maps(residues, num_fragments_extract, fragment_length, padding)
        all_contact_maps.append(contact_maps)
    print("{0} broken proteins read out of {1}!".format(broken_prots, num_proteins))
    contact_maps = torch.cat(all_contact_maps)
    # serialize contact maps and cache them
    dump_contact_maps(contact_maps, file_name, fragment_length, padding)
    return contact_maps

def get_cache_file(file_name, fragment_length, padding):
    return file_name+"_"+str(fragment_length)+"_"+padding+"_contact_maps.dat"

# serializes the contact maps to binary files for caching
def dump_contact_maps(contact_maps, file_name, fragment_length, padding):
    tmp = file_name.split("/")
    fn = "/".join(tmp[:-1])+"/"+tmp[-1].split(".")[0]
    cache_file = get_cache_file(fn, fragment_length, padding)
    print("Caching the contact maps to {}".format(cache_file))
    torch.save(contact_maps, cache_file)

# 1. get from file in data/proteins/contact_map_xx.dat (xx is 50 or so, so density)
# 2. if not existing, call create_contact_map
def get_contact_maps(file_name, fragment_length=None, atom=None, padding="pwd_pad"):
    tmp = file_name.split("/")
    fn = "/".join(tmp[:-1])+"/"+tmp[-1].split(".")[0]
    cache_file = get_cache_file(fn, fragment_length, padding)
    if os.path.isfile(cache_file):
        print("Reading cache file in {}".format(cache_file))
        return torch.load(cache_file)
    else:
        print("Creating the contact maps for {} as no cache was found!".format(file_name))
        return create_contact_maps(file_name, fragment_length, atom, padding)
