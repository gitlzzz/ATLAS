import numpy as np


def model_res_dict_to_arr(res_dict):
    res_model_list = []
    for _, res in res_dict.items():
        res_model_list.append(res)
    res_model_list = np.array(res_model_list)
    return res_model_list


def get_model_forces_variance(forces_dict):
    forces_model_list = model_res_dict_to_arr(forces_dict)
    forces_var = forces_model_list.var(axis=0)

    return forces_var


def get_model_energies_variance(energies_dict):
    energies_model_list = model_res_dict_to_arr(energies_dict)
    energies_var = energies_model_list.var(axis=0)

    return energies_var


def get_model_forces_std(forces_dict):
    forces_model_list = model_res_dict_to_arr(forces_dict)
    forces_std = forces_model_list.std(axis=0)

    return forces_std


def get_model_energies_std(energies_dict):
    energies_model_list = model_res_dict_to_arr(energies_dict)
    energies_std = energies_model_list.std(axis=0)

    return energies_std


def select_dft_structures(struct_arr, frame_interval):
    """
    Select DFT structures using the interval given as an input of the workchain.

    Parameters
    ----------
    struct_arr : np.array
        Array containing all possible structures to compute.
    frame_interval : Int
        Integer representing the interval between structures to keep.

    Returns
    -------
    np.array
        Array containing only the selected structures.
    """
    slice_step = int(len(struct_arr) * frame_interval)

    if slice_step == 0:
        slice_step = int(len(struct_arr) / 2)

    selected_dft_structs_idxs = range(len(struct_arr))[::slice_step]
    selected_dft_structs = struct_arr[::slice_step]
    selected_high_error = np.nonzero(selected_dft_structs)[0]
    selected_high_error_idxs = np.array(selected_dft_structs_idxs)[selected_high_error]

    return selected_high_error_idxs


def compute_dft_energy_structure(struct):

    struct_type = identify_struct_type(struct)
    
    

def identify_struct_type(struct):
    ...
    