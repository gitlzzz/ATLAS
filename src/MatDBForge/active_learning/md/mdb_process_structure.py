#!/usr/bin/env python3
"""Script to run MACE MD simulations, descriptor generation and extrapolation checks.

This script is part of MatDBForge's active learning loop, and is used to combine
in one single calculation job the acquisition of the MD trajectory, the generation
of the descriptors followed by performing any extrapolation checks, filtering
the MD trajectory if necessary.
"""

import pathlib as pl
import tomllib
import warnings

import numpy as np
import torch
from ase.calculators.calculator import PropertyNotImplementedError
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.io.trajectory import TrajectoryReader, TrajectoryWriter
from shapely.geometry import Point, Polygon

import MatDBForge.active_learning.active_learning_utils as mdb_al_ut
import MatDBForge.active_learning.extrapolation.concave_hull as mdb_chull
from MatDBForge.active_learning.extrapolation import autoencoder as mdb_ae
from MatDBForge.core import code_utils as mdb_cud

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)


def check_traj_in_domain(
    concave_hull: np.ndarray, descriptor_dict: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_inside = []
    point_outside = []
    all_points_in_out = []

    # Check if the random points are inside the bounds of the
    # concave hull by checking if the points are inside the
    # polygon formed by the concave hull.
    polygon = Polygon(concave_hull)
    for uuid in descriptor_dict:
        descriptors = descriptor_dict[uuid]["latent_space"]

        for _, frame_desc in enumerate(descriptors):
            # c_p_in = []
            # c_p_out = []
            c_all_p = []
            for point in frame_desc:
                p = Point(point)
                if polygon.contains(p):
                    point_inside.append(point)
                    c_all_p.append(True)
                else:
                    point_outside.append(point)
                    c_all_p.append(False)

            # point_inside.append(np.array(c_p_in))
            # point_outside.append(np.array(c_p_out))
            all_points_in_out.append(np.array(c_all_p))

    point_inside = np.array(point_inside)
    point_outside = np.array(point_outside)
    all_points_in_out = np.array(all_points_in_out)
    return point_inside, point_outside, all_points_in_out


def gather_md_E_F_data(md_traj, res_folder, curr_temp):
    # Get the step, energy and forces and save into an array
    step_E_F_list = []
    for idx, frame in enumerate(md_traj):

        try:
            e_pot = frame.get_potential_energy()
        except PropertyNotImplementedError:
            e_pot = frame.calc.results["energy"]
        try:
            fmax = frame.get_forces().max()
        except PropertyNotImplementedError:
            fmax = frame.calc.results["forces"].max()
        step_E_F_list.append([idx, e_pot, fmax])

    step_E_F_arr = np.stack(step_E_F_list, axis=0)
    np.save(file=res_folder / f"step_E_F_arr_temp-{curr_temp}.npy", arr=step_E_F_arr)
    return step_E_F_arr


def limit_md_frames(md_traj, md_params):
    # Limit total number of frames
    total_num_frames = mdb_al_ut.get_total_num_frames(
        len_traj=len(md_traj),
        md_tstep_duration_ps=md_params["timestep_duration_ps"],
        frame_interval=md_params["al_keep_struct_every_n_ps"],
    )
    # Choose the number of frames evenly and create a mask for the arrays
    mask = np.linspace(0, len(md_traj) - 1, int(total_num_frames), dtype=int)

    # Apply mask
    return [md_traj[i] for i in mask], mask


def simple_extrapolation_check(curr_it_db_max, curr_it_db_min, descriptor_dict):
    mdb_cud.custom_print("Applying basic extrapolation check...", "info")
    for uuid, val in descriptor_dict.items():
        structure_uuid = uuid
        descr_list = val["descriptors"]

    for frame_idx, frame_descriptors in enumerate(descr_list):
        below_min = frame_descriptors < curr_it_db_min
        above_max = frame_descriptors > curr_it_db_max
        is_frame_extrapolating = np.any(np.logical_or(below_min, above_max))
        if is_frame_extrapolating:
            descriptor_dict[structure_uuid]["is_extrapolating"][frame_idx] = True


if __name__ == "__main__":

    # Load the rmse_arr.npy file and assign the values to the variables
    rmse_arr = np.load("rmse_arr.npy")
    e_rmse = rmse_arr[0]
    f_rmse = rmse_arr[1]

    # Out of domain frames
    ood_frames = []
    # In domain frames
    id_frames = []

    # Define results folder
    res_folder = pl.Path("./results")
    res_folder.mkdir(exist_ok=True)

    # Initialize the logger
    log_folder = pl.Path("./logs")
    log_folder.mkdir(exist_ok=True)
    logger = mdb_cud.init_logger(source="mdb", log_path=log_folder)

    # Initialize random seed
    rng_seed = np.random.randint(0, int(1e15))
    mdb_cud.custom_print(f"Using random seed: '{rng_seed}'")

    mdb_cud.custom_print("Starting process structure script...", 'info')

    # Read TOML file with settings
    with open("settings.toml", "rb") as f:
        settings = tomllib.load(f)

    # Parse settings
    md_params = settings.get("md", {}).get("parameters")
    T_list = md_params["temperature_list_K"]

    # Logging CUDA information
    if md_params.get("device") == "cuda":
        mdb_cud.custom_print(
            (
                f"CUDA INFO - available: {torch.cuda.is_available()}, "
                f"device_count: {torch.cuda.device_count()}, "
                f"current_device: {torch.cuda.current_device()}"
            ),
            "debug",
        )

    # Get the extrapolation type
    extrap_type = settings.get("extrapolation", {}).get(
        "check_extrapolation_type", "basic"
    )

    # Get the dimensionality reduction method settings
    dim_red_method = settings.get("descriptors", {}).get(
        "dimensionality_reduction_method", None
    )
    match dim_red_method:
        case "autoencoder":
            dim_red_settings = settings.get("descriptors", {}).get("autoencoder", {})
        case "pca":
            dim_red_settings = settings.get("descriptors", {}).get("pca", {})

    # Read the initial structure
    init_conf_orig = ase_read("curr_structure.xyz", format="extxyz")

    ## Running MD simulations for given temperatures
    for T_start in T_list:
        # Instantiating ase trajectory object
        init_conf = init_conf_orig.copy()
        traj_obj = TrajectoryWriter(
            res_folder / f"md_traj_final_temp-{T_start}.traj",
            mode="w",
            atoms=init_conf,
            properties=["energy", "forces", "REF_energy", "REF_forces"],
        )
        print()
        mdb_cud.custom_print(f"Running MD simulation for 'T={T_start} K'", "info")
        mdb_al_ut.run_mace_md_ase(
            # temperature_ramp=mdb_al_ut.md_apply_temperature_ramp,
            md_params=md_params,
            T_start=T_start,
            traj_obj=traj_obj,
            init_conf=init_conf,
        )
        mdb_cud.custom_print("MD simulation completed!", "done")

        # print('logger: ', logger[0].handlers)
        # print('logger: ', len(logger[0].handlers))

    # Read MD-generated trajectories for given temperatures
    traj_files = res_folder.glob("*final_temp-*.traj")

    for curr_traj in traj_files:
        mdb_cud.custom_print(f"Checking extrapolating frames for '{curr_traj}'", "info")
        extrapol_frames_idx = []

        curr_temp = str(curr_traj).split("temp-")[1].split(".traj")[0]
        print()

        # Read the trajectory
        md_traj = [frame for frame in TrajectoryReader(curr_traj)]
        orig_md_size = len(md_traj)

        # Add frame info and overwrite the trajectory
        for frame_idx, frame in enumerate(md_traj):
            frame.info["frame_idx"] = frame_idx
            frame.info["calc_type"] = "MACE_MD"
            frame.info["md_temperature"] = float(curr_temp)
            # frame.info['mdb_id'] = str(init_conf.info['mdb_id'])

        # Get the step, energy and forces and save into an array
        step_E_F_arr = gather_md_E_F_data(md_traj, res_folder, curr_temp)

        # TODO: Apply E/F extrapolation filters
        model_acc_multiplier = settings["active_learning"].get("model_acc_multiplier")

        e_error_threshold = model_acc_multiplier * e_rmse
        f_error_threshold = model_acc_multiplier * f_rmse
        maximum_value_e = 1000  # meV
        maximum_value_f = 1000  # meV

        # Checking if the energies are over the error threshold
        energies_stat = np.nanstd(step_E_F_arr[:, 1], axis=0, ddof=1)
        error_e_structures_sm = np.ma.make_mask(
            energies_stat >= e_error_threshold,
            shrink=False,
        )
        error_e_structures_bg = np.ma.make_mask(
            energies_stat < maximum_value_e,
            shrink=False,
        )
        # Any True value in this array is over the energy error threshold
        # and must be sent to calculate with DFT.
        error_e_structures = np.logical_and(
            error_e_structures_sm, error_e_structures_bg
        )
        forces_std = np.nanstd(step_E_F_arr[:, 2], axis=0, ddof=1)

        # Checking if the forces are over the error threshold
        err_f_struct_sm = np.ma.make_mask(
            forces_std >= f_error_threshold,
            shrink=False,
        )
        err_f_struct_bg = np.ma.make_mask(
            forces_std < maximum_value_f,
            shrink=False,
        )

        # Any True value in this array is over the force error threshold
        # and must be sent to calculate with DFT.
        error_f_structures = np.logical_and(err_f_struct_sm, err_f_struct_bg)

        # Adding extrapolating indices
        e_f_extrapol = []
        for idx in step_E_F_arr[error_e_structures][:, 0]:
            e_f_extrapol.append(int(idx[0]))
        for idx in step_E_F_arr[error_f_structures][:, 0]:
            e_f_extrapol.append(idx)
        extrapol_frames_idx.extend(set(e_f_extrapol))

        # Apply MD filters
        # TODO: We should remove the extrapolating frames from the trajectory
        md_filters = settings.get("md", {}).get("filters", [])
        print("md_filters: ", md_filters)
        if "layer_distance" in md_filters:
            max_dist = md_filters["layer_distance"]["max_layer_distance_ang"]
            for idx, frame in enumerate(md_traj):
                is_structure_wrong = mdb_al_ut.apply_layer_distance_filter(
                    struct=frame, max_layer_distance_ang=max_dist
                )
                if is_structure_wrong:
                    extrapol_frames_idx.append(idx)
        mdb_cud.custom_print(
            f"After layer distance filter: {len(extrapol_frames_idx)}", "debug"
        )

        if "check_atoms_no_neighbor" in md_filters:
            for idx, frame in enumerate(md_traj):
                is_structure_wrong = mdb_al_ut.apply_filter_no_neighbors(struct=frame)
                if is_structure_wrong:
                    extrapol_frames_idx.append(idx)
        mdb_cud.custom_print(
            f"After no neighbor filter: {len(extrapol_frames_idx)}", "debug"
        )

        # Limit total number of frames
        md_traj_short, mask = limit_md_frames(md_traj, md_params)

        ## Get the descriptors
        # Only MACE is supported for now
        mdb_cud.custom_print("Generating descriptors...", "info")
        match settings["descriptors"].get("descriptor_type", "mace"):
            case "soap":
                descriptor_dict, descriptor_arr = mdb_al_ut.generate_descriptors_soap(
                    database=md_traj_short,
                    descriptor_settings=settings["descriptors"],
                )
            case _:
                descriptor_dict, descriptor_arr = mdb_al_ut.generate_descriptors_mace(
                    model_path="curr_model.model",
                    database=md_traj_short,
                    descriptor_settings=settings["descriptors"],
                )

        # np.save(
        #     file=res_folder / f"all_descriptors_temp-{curr_temp}.npy",
        #     arr=descriptor_arr,
        # )

        # Add is_extrapolating list which contains boolean values showing
        # if the frame is extrapolating or not
        for structure_uuid in descriptor_dict:
            descriptor_dict[structure_uuid]["is_extrapolating"] = np.zeros(
                len(md_traj_short), dtype=bool
            )

        print()

        # Advanced extrapolation
        if extrap_type == "advanced":
            # Read the concave hull
            concave_hull = np.load("concave_hull.npy")

            mdb_cud.custom_print("Applying advanced extrapolation check...", "info")
            # Get latent space for the trajectory
            match dim_red_method:
                case "pca":
                    # latent_space = get_latent_space_pca(database=descriptor_dict)
                    raise NotImplementedError("PCA not implemented yet.")
                case _:
                    aut_t_params = settings["descriptors"]["autoencoder"].get(
                        "train_settings"
                    )

                    model = torch.load(aut_t_params.get("model_path"))
                    model.to(dtype=torch.float32)

                    descriptor_dict = mdb_ae.get_latent_space_autoencoder(
                        model=model,
                        descriptor_dict=descriptor_dict,
                    )

            point_inside, point_outside, all_points_in_out = check_traj_in_domain(
                concave_hull=concave_hull, descriptor_dict=descriptor_dict
            )

            mdb_chull.plot_concave_hull(
                concave_hull=concave_hull,
                point_inside=point_inside,
                point_outside=point_outside,
                filename=res_folder / f"concave_hull_temp-{curr_temp}.png",
            )

            for idx, frame in enumerate(all_points_in_out):
                if np.all(frame):
                    descriptor_dict[structure_uuid]["is_extrapolating"][idx] = False
                else:
                    descriptor_dict[structure_uuid]["is_extrapolating"][idx] = True
                    # extrapol_frames_idx.append(frame.info["frame_idx"])

        # Simple extrapolation check
        elif extrap_type == "basic":
            # Read the minimum and maximum values for each descriptor
            # for the entire database
            curr_it_db_max = np.load("curr_it_db_max.npy")
            curr_it_db_min = np.load("curr_it_db_min.npy")

            simple_extrapolation_check(curr_it_db_max, curr_it_db_min, descriptor_dict)

        for idx, is_extrapolating in enumerate(
            descriptor_dict[structure_uuid]["is_extrapolating"]
        ):
            if is_extrapolating:
                extrapol_frames_idx.append(md_traj_short[idx].info["frame_idx"])

        # Saving all the frames that are extrapolating
        extrapol_frames_idx = set(extrapol_frames_idx)
        extrapol_frames_final = [md_traj[i] for i in extrapol_frames_idx]
        mdb_cud.custom_print(
            f"Total count of extrapolating frames: {len(extrapol_frames_final)}", "info"
        )
        ase_write(
            res_folder / "extrapolating_frames.xyz",
            format="extxyz",
            images=extrapol_frames_final,
            append=True,
        )

    mdb_cud.custom_print("Structure processed!", "done")
