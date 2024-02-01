import numpy as np
from MatDBForge.workflows import aiida_utils as mdb_aut


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


def get_dft_calc_builder(struct, row, calc_idx, group):
    struct_type = row["mdb_struct_type"]

    # Gathering row information
    (
        curr_structure,
        curr_material_name,
        curr_unique_id,
        curr_phase,
    ) = mdb_aut.gather_calc_data_from_row(row, curr_structure=struct)

    # TODO
    # HACK: Move this to a central json file in the CWD or data folder.
    kspacing_dict = {
        "alpha": 0.135088484104361,
        # "m1": 0.100530964914873,
        "beta-prime": 0.102415920507027,
        # "m2": 0.100530964914873,
        "gamma": 0.141371669411541,
        # "m3": 0.166504410640259,
        "epsilon": 0.105557513160617,
        "eta": 0.0993371597065093,
        # "m4": 0.0948760981384118,
        "delta": 0.0994491889005363,
    }

    queue_dict = {
        2: {
            "type": "sge",
            "node_cpus": 12,
            "code_string": "vasp-std-5.4.4@tekla2",
            "options_resources": {
                "parallel_env": "c12m48ib_mpi",
                "tot_num_mpiprocs": 12,
            },
            "multiple": 1,
        },
        5: {
            "type": "sge",
            "node_cpus": 12,
            "code_string": "vasp-std-5.4.4@tekla2",
            "options_resources": {
                "parallel_env": "c12m48ib_mpi",
                "tot_num_mpiprocs": 12,
            },
            "multiple": 1,
        },
        40: {
            "type": "sge",
            "node_cpus": 12,
            "code_string": "vasp-std-5.4.4@tekla2",
            "options_resources": {
                "parallel_env": "c12m48ib_mpi",
                "tot_num_mpiprocs": 12,
            },
            "multiple": 1,
        },
    }
    potential_family = "vasp-5.4-PBE-2023"
    potential_mapping = mdb_aut.generate_potential_mapping()

    builder = mdb_aut.submit_aiida_calculation(
        index=calc_idx,
        target_structure=struct,
        phase=curr_phase,
        material_name=curr_material_name,
        unique_id=curr_unique_id,
        kspacing_dict=kspacing_dict,
        calc_type=struct_type,
        queue_dict=queue_dict,
        potential_family=potential_family,
        potential_mapping=potential_mapping,
        return_builder=True,
        dry_run=False,
        incar_dict=None,
        group=group,
    )
    return builder


def identify_struct_type(struct):
    ...
