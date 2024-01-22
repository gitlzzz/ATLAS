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


def select_dft_structures(struct_list, frame_interval):
    """
    Select DFT structures using the interval given as an input of the workchain.

    Parameters
    ----------
    struct_list : np.array
        Array containing all possible structures to compute.
    frame_interval : Int
        Integer representing the interval between structures to keep.

    Returns
    -------
    np.array
        Array containing only the selected structures.
    """
    # TODO: Find a way of getting more separate frames.
    slice_step = int(len(struct_list) * frame_interval)

    if slice_step == 0:
        slice_step = int(len(struct_list)/2)

    selected_dft_structs = struct_list[:: slice_step]
    return selected_dft_structs
