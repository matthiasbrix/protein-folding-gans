# This file is part of the OpenProtein project.
#
# @author Jeppe Hallgren
#
# For license information, please see the LICENSE file in the root directory.

import glob
import os.path
import os
import platform
import numpy as np
import h5py
from util import AA_ID_DICT, calculate_dihedral_angles, protein_id_to_str, get_structure_from_angles,\
    structure_to_backbone_atoms, write_to_pdb, calculate_dihedral_angles_over_minibatch,\
    get_backbone_positions_from_angular_prediction, encode_primary_string
import torch

MAX_SEQUENCE_LENGTH = 2000

def process_raw_data(use_gpu, force_pre_processing_overwrite=True, prefix="", data_root_folder="../data/", testing=False):
    print("Starting pre-processing of raw data...")
    test_folder = "testing_files/" if testing else ""
    input_files = glob.glob(data_root_folder+"raw/"+test_folder+"*")
    input_files = list(filter(lambda x: not x.endswith("testing_files"), input_files)) if not testing else input_files
    print("files", input_files)
    input_files_filtered = filter_input_files(input_files)
    for file_path in input_files_filtered:
        if platform.system() is 'Windows':
            filename = file_path.split('\\')[-1]
        else:
            filename = file_path.split('/')[-1]
        folder_prefix = str(MAX_SEQUENCE_LENGTH)+prefix
        os.makedirs(data_root_folder+"/preprocessed/"+folder_prefix, exist_ok=True)
        preprocessed_file_name = data_root_folder+"preprocessed/"+folder_prefix+"/"+filename+".hdf5"

        # check if we should remove the any previously processed files
        if os.path.isfile(preprocessed_file_name):
            print("Preprocessed file for " + filename + " already exists.")
            if force_pre_processing_overwrite:
                print("force_pre_processing_overwrite flag set to True, overwriting old file...")
                os.remove(preprocessed_file_name)
            else:
                print("Skipping pre-processing for this file...")

        if not os.path.isfile(preprocessed_file_name):
            process_file(filename, preprocessed_file_name, use_gpu, testing)
            if testing:
                break
    print("Completed pre-processing.")

def process_file(input_file, output_file, use_gpu, testing):
    print("Processing raw data file", input_file)

    # create output file
    f = h5py.File(output_file, 'w')
    current_buffer_size = 1
    current_buffer_allocation = 0
    dset1 = f.create_dataset('primary',(current_buffer_size, MAX_SEQUENCE_LENGTH), maxshape=(None, MAX_SEQUENCE_LENGTH), dtype='int32')
    dset2 = f.create_dataset('tertiary',(current_buffer_size, MAX_SEQUENCE_LENGTH,9), maxshape=(None, MAX_SEQUENCE_LENGTH, 9), dtype='float')
    dset3 = f.create_dataset('mask', (current_buffer_size, MAX_SEQUENCE_LENGTH), maxshape=(None, MAX_SEQUENCE_LENGTH), dtype='uint8')

    if testing:
        input_files = glob.glob("../data/proteins/raw/testing_files/*")
    else:
        input_files = ["../data/proteins/raw/" + input_file]
    
    index = 0
    print(output_file, testing, len(input_files), input_files)
    for _ in range(len(input_files)):
        #input_file_pointer = open("../data/proteins/raw/" + input_file, "r")
        input_file_pointer = open(input_files[index], "r")
        while True:
            # while there's more proteins to process
            next_protein = read_protein_from_file(input_file_pointer)
            if next_protein is None:
                break

            sequence_length = len(next_protein['primary'])

            if sequence_length > MAX_SEQUENCE_LENGTH:
                print("Dropping protein as length too long:", sequence_length, MAX_SEQUENCE_LENGTH)
                continue

            if current_buffer_allocation >= current_buffer_size:
                current_buffer_size = current_buffer_size + 1
                dset1.resize((current_buffer_size, MAX_SEQUENCE_LENGTH))
                dset2.resize((current_buffer_size, MAX_SEQUENCE_LENGTH, 9))
                dset3.resize((current_buffer_size, MAX_SEQUENCE_LENGTH))

            primary_padded = np.zeros(MAX_SEQUENCE_LENGTH)
            tertiary_padded = np.zeros((9, MAX_SEQUENCE_LENGTH))
            mask_padded = np.zeros(MAX_SEQUENCE_LENGTH)

            # masking and padding here happens so that the stored dataset is of the same size. 
            # when the data is loaded in this padding is removed again. 
            primary_padded[:sequence_length] = next_protein['primary']
            t_transposed = np.ravel(np.array(next_protein['tertiary']).T)
            t_reshaped = np.reshape(t_transposed, (sequence_length,9)).T

            tertiary_padded[:,:sequence_length] = t_reshaped
            mask_padded[:sequence_length] = next_protein['mask']

            mask = torch.Tensor(mask_padded).type(dtype=torch.uint8)
            
            prim = torch.masked_select(torch.Tensor(primary_padded).type(dtype=torch.long), mask)
            pos = torch.masked_select(torch.Tensor(tertiary_padded), mask).view(9, -1).transpose(0, 1).unsqueeze(1) / 100

            if use_gpu:
                pos = pos.cuda()

            angles, batch_sizes = calculate_dihedral_angles_over_minibatch(pos, [len(prim)], use_gpu=use_gpu)

            tertiary, _ = get_backbone_positions_from_angular_prediction(angles, batch_sizes, use_gpu=use_gpu)
            tertiary = tertiary.squeeze(1)

            primary_padded = np.zeros(MAX_SEQUENCE_LENGTH)
            tertiary_padded = np.zeros((MAX_SEQUENCE_LENGTH, 9))

            length_after_mask_removed = len(prim)

            primary_padded[:length_after_mask_removed] = prim.data.cpu().numpy()
            tertiary_padded[:length_after_mask_removed, :] = tertiary.data.cpu().numpy()
            mask_padded = np.zeros(MAX_SEQUENCE_LENGTH)
            mask_padded[:length_after_mask_removed] = np.ones(length_after_mask_removed)

            dset1[current_buffer_allocation] = primary_padded
            dset2[current_buffer_allocation] = tertiary_padded
            dset3[current_buffer_allocation] = mask_padded
            current_buffer_allocation += 1

        # added:
        index += 1

    print("Wrote output to", current_buffer_allocation, "proteins to", output_file)

def filter_input_files(input_files):
    disallowed_file_endings = (".gitignore", ".DS_Store")
    return list(filter(lambda x: not x.endswith(disallowed_file_endings), input_files))

def read_protein_from_file(file_pointer):

        dict_ = {}
        _dssp_dict = {'L': 0, 'H': 1, 'B': 2, 'E': 3, 'G': 4, 'I': 5, 'T': 6, 'S': 7}
        _mask_dict = {'-': 0, '+': 1}

        while True:
            next_line = file_pointer.readline()
            if next_line == '[ID]\n':
                id_ = file_pointer.readline()[:-1]
                dict_.update({'id': id_})
            elif next_line == '[PRIMARY]\n':
                primary = encode_primary_string(file_pointer.readline()[:-1])
                dict_.update({'primary': primary})
            elif next_line == '[EVOLUTIONARY]\n':
                evolutionary = []
                for residue in range(21): evolutionary.append(
                    [float(step) for step in file_pointer.readline().split()])
                dict_.update({'evolutionary': evolutionary})
            elif next_line == '[SECONDARY]\n':
                secondary = list([_dssp_dict[dssp] for dssp in file_pointer.readline()[:-1]])
                dict_.update({'secondary': secondary})
            elif next_line == '[TERTIARY]\n':
                tertiary = []
                # 3 dimension
                for axis in range(3): tertiary.append(
                    [float(coord) for coord in file_pointer.readline().split()])
                dict_.update({'tertiary': tertiary})
            elif next_line == '[MASK]\n':
                mask = list([_mask_dict[aa] for aa in file_pointer.readline()[:-1]])
                dict_.update({'mask': mask})
            elif next_line == '\n':
                return dict_
            elif next_line == '':
                return None