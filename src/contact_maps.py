import h5py
import torch
import numpy as np
import os.path

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

def pad_fragment_residue(fragment_residue, fragment_length):
    tmp = np.zeros((fragment_length, 3))
    tmp[:fragment_residue.shape[0]] = fragment_residue
    return torch.Tensor(tmp).type(torch.float) # dim is then fragment_length x 3

# TODO: need to filter out the longer data than needed so that we can have nicer contact map structures
# TODO: calc internally of fragments, for each protein
# TODO: take each fragment out and then create a contact matrix for it, so get like (num_proteins*) X fragment_length X fragment_length
# 1 chain = 1 protein
# 1 chains -> 76 residues/amino acids -> 76 x 76 contact map

def compute_contact_map(residues, num_fragments_extract, fragment_length):
    contact_map = torch.zeros((num_fragments_extract, fragment_length, fragment_length))
    for fragment_id_A in range(num_fragments_extract):
        start_A = fragment_id_A*fragment_length
        end_A = (fragment_id_A+1)*fragment_length
        # extracting fragment
        fragment_A = residues[start_A:end_A]
        print("start, end", start_A, end_A, num_fragments_extract)
        #print(num_proteins, max_sequence_len)
        for fragment_id_B in range(fragment_id_A+1, num_fragments_extract):
            start_B = fragment_id_B*fragment_length
            end_B = (fragment_id_B+1)*fragment_length
            fragment_B = residues[start_B:end_B]
            # in case we extract remaining, that is < residue_fragments size, we pad
            if fragment_B.shape[0] < fragment_length:
                pad_fragment_residue(fragment_B, fragment_length)
            print("HER", fragment_id_A, fragment_id_B)
            res = calc_pairwise_distances(fragment_A, fragment_B, torch.cuda.is_available())
            contact_map[fragment_id_A] = res
    print("CONTACT MAP")
    exit(1)
    return contact_map

# TODO: vi skal for hvert protein take fragmenter ud og for hver fragment laver vi en contact matrice..., dvs. er længden 16 så får vi 16 x 16
def create_contact_maps(file_name, fragment_length, atom):
    h5pyfile = h5py.File(file_name, 'r')
    num_proteins, max_sequence_len = h5pyfile['primary'].shape
    print(num_proteins, max_sequence_len)
    contact_maps = []
    for protein_idx in range(num_proteins): # have structure num_proteins x vairable length (amino acids) x 9
        mask = torch.Tensor(h5pyfile['mask'][protein_idx, :]).type(dtype=torch.uint8)
        #prim = torch.masked_select(torch.Tensor(h5pyfile['primary'][protein_idx, :]).type(dtype=torch.long), mask)
        tertiary = torch.Tensor(h5pyfile['tertiary'][protein_idx][:int(mask.sum())])
        protein_length = len(tertiary)
        num_fragments_extract = protein_length//fragment_length # non-overlapping fragments
        residues = np.reshape(tertiary, ((protein_length, 3, 3)))
        # filter here residues by specific atoms
        residues = atom_filter(residues, atom)
        contact_map = compute_contact_map(residues, num_fragments_extract, fragment_length)
        print(residues.shape, protein_idx, num_fragments_extract, protein_length, fragment_length)
        contact_maps.append(contact_map)
        print(contact_map)
    exit(1)
    # save contact maps to binary
    dump_contact_maps(contact_maps, file_name)
    return contact_maps

# parses the contact maps to json files in order to cache them
def dump_contact_maps(contact_maps, file_name):
    file_name = os.path.splitext(file_name)[0]
    cache_file = file_name+"_contact_maps.dat"
    contact_maps.detach().numpy().dump(cache_file)

# 1. get from file in data/proteins/contact_map_xx.dat (xx is 50 or so, so density)
# 2. if not existing, call create_contact_map
def get_contact_maps(file_name, fragment_length, atom):
    cache_file = os.path.splitext(file_name+"_contact_maps")[0]+".dat"
    if os.path.isfile(cache_file):
        return torch.FloatTensor(np.load(cache_file))
    else:
        return create_contact_maps(file_name, fragment_length, atom)
