# This file is part of the OpenProtein project.
#
# @author Jeppe Hallgren
#
# For license information, please see the LICENSE file in the root directory.

import math
import torch
import torch.utils.data
import Bio.PDB
import numpy as np
import PeptideBuilder
import pnerf.pnerf as pnerf

AA_ID_DICT = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9,
              'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17,
              'V': 18, 'W': 19,'Y': 20}

# only for the dcgan
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)

def calculate_dihedral_angles_over_minibatch(atomic_coords_padded, batch_sizes, use_gpu):
    angles = []
    atomic_coords = atomic_coords_padded.transpose(0,1)
    for idx, _ in enumerate(batch_sizes):
        angles.append(calculate_dihedral_angles(atomic_coords[idx][:batch_sizes[idx]], use_gpu))
    return torch.nn.utils.rnn.pad_packed_sequence(
            torch.nn.utils.rnn.pack_sequence(angles))

def protein_id_to_str(protein_id_list):
    _aa_dict_inverse = {v: k for k, v in AA_ID_DICT.items()}
    aa_list = []
    for a in protein_id_list:
        aa_symbol = _aa_dict_inverse[int(a)]
        aa_list.append(aa_symbol)
    return aa_list

def calculate_dihedral_angles(atomic_coords, use_gpu):

    assert int(atomic_coords.shape[1]) == 9
    atomic_coords = atomic_coords.contiguous().view(-1,3)

    zero_tensor = torch.tensor(0.0)
    if use_gpu:
        zero_tensor = zero_tensor.cuda()

    dihedral_list = [zero_tensor,zero_tensor]
    dihedral_list.extend(compute_dihedral_list(atomic_coords))
    dihedral_list.append(zero_tensor)
    angles = torch.tensor(dihedral_list).view(-1,3)
    return angles

def compute_dihedral_list(atomic_coords):
    # atomic_coords is -1 x 3
    ba = atomic_coords[1:] - atomic_coords[:-1]
    ba /= ba.norm(dim=1).unsqueeze(1)
    ba_neg = -1 * ba

    n1_vec = torch.cross(ba[:-2], ba_neg[1:-1], dim=1)
    n2_vec = torch.cross(ba_neg[1:-1], ba[2:], dim=1)
    n1_vec /= n1_vec.norm(dim=1).unsqueeze(1)
    n2_vec /= n2_vec.norm(dim=1).unsqueeze(1)

    m1_vec = torch.cross(n1_vec, ba_neg[1:-1], dim=1)

    x = torch.sum(n1_vec*n2_vec,dim=1)
    y = torch.sum(m1_vec*n2_vec,dim=1)

    return torch.atan2(y,x)

def get_structure_from_angles(aa_list_encoded, angles):
    aa_list = protein_id_to_str(aa_list_encoded)
    omega_list = angles[1:,0]
    phi_list = angles[1:,1]
    psi_list = angles[:-1,2]
    assert len(aa_list) == len(phi_list)+1 == len(psi_list)+1 == len(omega_list)+1
    structure = PeptideBuilder.make_structure(aa_list,
                                              list(map(lambda x: math.degrees(x), phi_list)),
                                              list(map(lambda x: math.degrees(x), psi_list)),
                                              list(map(lambda x: math.degrees(x), omega_list)))
    return structure

def write_to_pdb(structure, prot_id):
    out = Bio.PDB.PDBIO()
    out.set_structure(structure)
    out.save("output/protein_" + str(prot_id) + ".pdb")

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

def calc_drmsd(chain_a, chain_b, use_gpu=False):
    assert len(chain_a) == len(chain_b)
    distance_matrix_a = calc_pairwise_distances(chain_a, chain_a, use_gpu)
    distance_matrix_b = calc_pairwise_distances(chain_b, chain_b, use_gpu)
    return torch.norm(distance_matrix_a - distance_matrix_b, 2) \
            / math.sqrt((len(chain_a) * (len(chain_a) - 1)))

# method for translating a point cloud to its center of mass
def transpose_atoms_to_center_of_mass(x):
    # calculate com by summing x, y and z respectively
    # and dividing by the number of points
    centerOfMass = np.matrix([[x[0, :].sum() / x.shape[1]],
                    [x[1, :].sum() / x.shape[1]],
                    [x[2, :].sum() / x.shape[1]]])
    # translate points to com and return
    return x - centerOfMass

def structure_to_backbone_atoms(structure):
    predicted_coords = []
    for res in structure.get_residues():
        predicted_coords.append(torch.Tensor(res["N"].get_coord()))
        predicted_coords.append(torch.Tensor(res["CA"].get_coord()))
        predicted_coords.append(torch.Tensor(res["C"].get_coord()))
    return torch.stack(predicted_coords).view(-1,9)

def get_backbone_positions_from_angular_prediction(angular_emissions, batch_sizes, use_gpu):
    # angular_emissions -1 x minibatch size x 3 (omega, phi, psi)
    points = pnerf.dihedral_to_point(angular_emissions, use_gpu)
    coordinates = pnerf.point_to_coordinate(points, use_gpu) / 100 # devide by 100 to angstrom unit
    return coordinates.transpose(0,1).contiguous().view(len(batch_sizes),-1,9).transpose(0,1), batch_sizes

def encode_primary_string(primary):
    return list([AA_ID_DICT[aa] for aa in primary])
