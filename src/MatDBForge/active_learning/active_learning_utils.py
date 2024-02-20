import ast
import json
from pathlib import Path

import numpy as np
import torch
from aiida.common.extendeddicts import AttributeDict
from aiida.engine import (
    calcfunction,
)
from aiida.orm import (
    Bool,
    Dict,
    Float,
    Int,
    List,
    SinglefileData,
    Str,
    StructureData,
    load_node,
)
from e3nn import o3
from e3nn.util import jit
from mace import data as mace_data
from mace import modules as mace_modules
from mace import tools as mace_tools
from mace.calculators import LAMMPS_MACE
from mace.tools.scripts_utils import (
    LRScheduler,
    create_error_table,
    get_dataset_from_xyz,
)
from MatDBForge.training import conversion as mdb_conv
from MatDBForge.workflows import aiida_utils as mdb_aut
from torch.optim.swa_utils import SWALR, AveragedModel
from torch_ema import ExponentialMovingAverage


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
            "node_cpus": 24,
            "code_string": "vasp-std-544-new@tekla2",
            "options_resources": {
                "parallel_env": "c24m128ib_mpi",
                "tot_num_mpiprocs": 24,
            },
            "multiple": 1,
        },
        5: {
            "type": "sge",
            "node_cpus": 24,
            "code_string": "vasp-std-544-new@tekla2",
            "options_resources": {
                "parallel_env": "c24m128ib_mpi",
                "tot_num_mpiprocs": 24,
            },
            "multiple": 1,
        },
        40: {
            "type": "sge",
            "node_cpus": 48,
            "code_string": "vasp-std-544-new@tekla2",
            "options_resources": {
                "parallel_env": "c48m256ib_mpi",
                "tot_num_mpiprocs": 48,
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


@calcfunction
def generate_placeholder_text():
    return Str("placeholder text")


@calcfunction
def prepare_output_dataframe(md_seed_results_df):
    md_seed_results_df.index = md_seed_results_df.index.map(str)
    training_df = Dict(md_seed_results_df.to_dict(orient="index"))
    return training_df


@calcfunction
def load_mace_settings_json(
    settings_path: str, train_data_path: str, curr_model: str, curr_iter: int
):
    if isinstance(curr_model, Str):
        settings_path = settings_path.value

    with open(settings_path, "r") as f:
        training_settings_dict = json.load(f)

    # Update training file path in mace train settings
    # to include the new database.
    if isinstance(train_data_path, Str):
        train_data_path: Path = Path(train_data_path.value)

    training_settings_dict["train_file"] = str(train_data_path.name)

    # Updating name to include model and iteration number
    curr_name = training_settings_dict["name"]

    if isinstance(curr_model, Str):
        curr_model = curr_model.value

    if isinstance(curr_iter, Int):
        curr_iter = curr_iter.value

    training_settings_dict["name"] = (
        str(curr_model) + "_" + curr_name + "_al-iteration_" + str(curr_iter)
    )

    return Dict(training_settings_dict)


@calcfunction
def run_mace_train_custom(mace_settings_dict):
    mace_settings_dict = mace_settings_dict.get_dict()

    mace_settings_dict["device"] = mace_settings_dict.get("device", "cuda")
    mace_settings_dict["seed"] = mace_settings_dict.get("seed", 123)
    mace_settings_dict["valid_file"] = mace_settings_dict.get("valid_file", None)
    mace_settings_dict["valid_fraction"] = mace_settings_dict.get(
        "valid_fraction", "0.1"
    )
    mace_settings_dict["test_file"] = mace_settings_dict.get("test_file", None)
    mace_settings_dict["energy_key"] = mace_settings_dict.get("energy_key", "energy")
    mace_settings_dict["forces_key"] = mace_settings_dict.get("forces_key", "forces")
    mace_settings_dict["stress_key"] = mace_settings_dict.get("stress_key", "stress")
    mace_settings_dict["virials_key"] = mace_settings_dict.get("virials_key", "virials")
    mace_settings_dict["dipole_key"] = mace_settings_dict.get("dipole_key", "dipole")
    mace_settings_dict["charges_key"] = mace_settings_dict.get("charges_key", "charges")
    mace_settings_dict["default_dtype"] = mace_settings_dict.get(
        "default_dtype", "float32"
    )
    mace_settings_dict["log_level"] = mace_settings_dict.get("log_level", "ERROR")
    mace_settings_dict["log_dir"] = mace_settings_dict.get("log_dir", "logs")
    mace_settings_dict["results_dir"] = mace_settings_dict.get("results_dir", "results")
    mace_settings_dict["model_dir"] = mace_settings_dict.get("model_dir", ".")
    mace_settings_dict["downloads_dir"] = mace_settings_dict.get(
        "downloads_dir", "downloads"
    )
    mace_settings_dict["checkpoints_dir"] = mace_settings_dict.get(
        "results_dir", "checkpoints"
    )

    tag = mace_tools.get_tag(
        name=mace_settings_dict["name"], seed=mace_settings_dict["seed"]
    )
    device = mace_tools.init_device(mace_settings_dict["device"])
    mace_tools.set_default_dtype(mace_settings_dict["default_dtype"])
    mace_tools.setup_logger(
        level=mace_settings_dict["log_level"],
        tag=tag,
        directory=mace_settings_dict["log_dir"],
    )

    try:
        config_type_weights = ast.literal_eval(
            mace_settings_dict["config_type_weights"]
        )
        assert isinstance(config_type_weights, dict)
    except Exception:  # pylint: disable=W0703
        # print(f"Config type weights not specified correctly ({e}), using Default")
        config_type_weights = {"Default": 1.0}

    # Data preparation
    col, atomic_energies_dict = get_dataset_from_xyz(
        train_path=mace_settings_dict["train_file"],
        valid_path=mace_settings_dict["valid_file"],
        valid_fraction=mace_settings_dict["valid_fraction"],
        config_type_weights=config_type_weights,
        test_path=mace_settings_dict["test_file"],
        seed=mace_settings_dict["seed"],
        energy_key=mace_settings_dict["energy_key"],
        forces_key=mace_settings_dict["forces_key"],
        stress_key=mace_settings_dict["stress_key"],
        virials_key=mace_settings_dict["virials_key"],
        dipole_key=mace_settings_dict["dipole_key"],
        charges_key=mace_settings_dict["charges_key"],
    )
    # tests_print = [nam + ": " + str(len(t_conf)) for nam, t_conf in col.tests]
    # print(
    #     f"Total number of configurations: train={len(col.train)},"
    #     f" valid={len(col.valid)}, "
    #     f"tests=[{', '.join(tests_print)}]"
    # )

    # Atomic number table
    # yapf: disable
    z_table = mace_tools.get_atomic_number_table_from_zs(
        z
        for configs in (col.train, col.valid)
        for config in configs
        for z in config.atomic_numbers
    )
    # yapf: enable
    # print(z_table)
    args_model = mace_settings_dict["model"]
    if args_model == "AtomicDipolesMACE":
        atomic_energies = None
        dipole_only = True
        compute_dipole = True
        compute_energy = False
        mace_settings_dict["compute_forces"] = False
        compute_virials = False
        mace_settings_dict["compute_stress"] = False
    else:
        dipole_only = False
        if args_model == "EnergyDipolesMACE":
            compute_dipole = True
            compute_energy = True
            mace_settings_dict["compute_forces"] = True
            compute_virials = False
            mace_settings_dict["compute_stress"] = False
        else:
            compute_energy = True
            compute_dipole = False
        if atomic_energies_dict is None or len(atomic_energies_dict) == 0:
            if mace_settings_dict.get("E0s") is not None:
                # print(
                #     "Atomic Energies not in training file, using command line argument E0s"
                # )
                if mace_settings_dict.get("E0s").lower() == "average":
                    # print(
                    #     "Computing average Atomic Energies using least squares regression"
                    # )
                    atomic_energies_dict = mace_data.compute_average_E0s(
                        col.train, z_table
                    )
                else:
                    try:
                        atomic_energies_dict = ast.literal_eval(
                            mace_settings_dict.get("E0s")
                        )
                        assert isinstance(atomic_energies_dict, dict)
                    except Exception as e:
                        raise RuntimeError(
                            f"E0s specified invalidly, error {e} occurred"
                        ) from e
            else:
                raise RuntimeError(
                    "E0s not found in training file and not specified in command line"
                )
        atomic_energies: np.ndarray = np.array(
            [atomic_energies_dict[z] for z in z_table.zs]
        )
        # print(f"Atomic energies: {atomic_energies.tolist()}")

    # Setting paramters for model creation. Using defaults from:
    # .../mace/tools/arg_parser.py
    mace_settings_dict["r_max"] = mace_settings_dict.get("r_max", 5.0)
    mace_settings_dict["batch_size"] = mace_settings_dict.get("batch_size", 10)
    mace_settings_dict["max_L"] = mace_settings_dict.get("max_L", None)
    mace_settings_dict["max_ell"] = mace_settings_dict.get("max_ell", 3)
    mace_settings_dict["num_channels"] = mace_settings_dict.get("num_channels", None)
    mace_settings_dict["num_radial_basis"] = mace_settings_dict.get(
        "num_radial_basis", 8
    )
    mace_settings_dict["num_cutoff_basis"] = mace_settings_dict.get(
        "num_cutoff_basis", 5
    )
    mace_settings_dict["error_table"] = mace_settings_dict.get(
        "error_table", "PerAtomRMSE"
    )
    mace_settings_dict["hidden_irreps"] = mace_settings_dict.get(
        "hidden_irreps", "128x0e + 128x1o"
    )
    mace_settings_dict["energy_weight"] = mace_settings_dict.get("energy_weight", 1.0)
    mace_settings_dict["swa_energy_weight"] = mace_settings_dict.get(
        "swa_energy_weight", 1000.0
    )
    mace_settings_dict["forces_weight"] = mace_settings_dict.get("forces_weight", 100.0)
    mace_settings_dict["swa_forces_weight"] = mace_settings_dict.get(
        "swa_forces_weight", 100.0
    )
    mace_settings_dict["weight_decay"] = mace_settings_dict.get("weight_decay", 5e-7)
    mace_settings_dict["virials_weight"] = mace_settings_dict.get("virials_weight", 1.0)
    mace_settings_dict["swa_virials_weight"] = mace_settings_dict.get(
        "swa_virials_weight", 10.0
    )
    mace_settings_dict["stress_weight"] = mace_settings_dict.get("stress_weight", 1.0)
    mace_settings_dict["swa_stress_weight"] = mace_settings_dict.get(
        "swa_stress_weight", 10.0
    )
    mace_settings_dict["huber_delta"] = mace_settings_dict.get("huber_delta", 0.01)
    mace_settings_dict["dipole_weight"] = mace_settings_dict.get("dipole_weight", 0.1)
    mace_settings_dict["compute_avg_num_neighbors"] = mace_settings_dict.get(
        "compute_avg_num_neighbors", True
    )
    mace_settings_dict["compute_stress"] = mace_settings_dict.get(
        "compute_stress", False
    )
    mace_settings_dict["compute_forces"] = mace_settings_dict.get(
        "compute_forces", True
    )
    mace_settings_dict["interaction"] = mace_settings_dict.get(
        "interaction", "RealAgnosticResidualInteractionBlock"
    )
    mace_settings_dict["num_interactions"] = mace_settings_dict.get(
        "num_interactions", 2
    )
    mace_settings_dict["interaction_first"] = mace_settings_dict.get(
        "interaction_first", "RealAgnosticResidualInteractionBlock"
    )
    mace_settings_dict["interaction"] = mace_settings_dict.get(
        "interaction", "RealAgnosticResidualInteractionBlock"
    )
    mace_settings_dict["correlation"] = mace_settings_dict.get("correlation", 3)
    mace_settings_dict["gate"] = mace_settings_dict.get("gate", "silu")
    mace_settings_dict["optimizer"] = mace_settings_dict.get("optimizer", "adam")
    mace_settings_dict["mlp_irreps"] = mace_settings_dict.get("mlp_irreps", "16x0e")
    mace_settings_dict["radial_mlp"] = mace_settings_dict.get(
        "radial_mlp", "[64, 64, 64]"
    )
    mace_settings_dict["radial_type"] = mace_settings_dict.get("radial_type", "bessel")
    mace_settings_dict["lr"] = mace_settings_dict.get("lr", 0.01)
    mace_settings_dict["swa_lr"] = mace_settings_dict.get("swa_lr", 1e-3)
    mace_settings_dict["amsgrad"] = mace_settings_dict.get("amsgrad", True)
    mace_settings_dict["lr_factor"] = mace_settings_dict.get("lr_factor", 0.8)
    mace_settings_dict["scheduler"] = mace_settings_dict.get(
        "scheduler", "ReduceLROnPlateau"
    )
    mace_settings_dict["scheduler_patience"] = mace_settings_dict.get(
        "scheduler_patience", 50
    )
    mace_settings_dict["start_swa"] = mace_settings_dict.get("start_swa", None)
    mace_settings_dict["ema"] = mace_settings_dict.get("ema", False)
    mace_settings_dict["ema_decay"] = mace_settings_dict.get("ema_decay", 0.99)
    mace_settings_dict["max_num_epochs"] = mace_settings_dict.get(
        "max_num_epochs", 2048
    )
    mace_settings_dict["patience"] = mace_settings_dict.get("patience", 2048)
    mace_settings_dict["eval_interval"] = mace_settings_dict.get("eval_interval", 2)
    mace_settings_dict["keep_checkpoints"] = mace_settings_dict.get(
        "keep_checkpoints", False
    )
    mace_settings_dict["restart_latest"] = mace_settings_dict.get(
        "restart_latest", False
    )

    mace_settings_dict["valid_batch_size"] = mace_settings_dict.get(
        "valid_batch_size", 10
    )
    mace_settings_dict["save_cpu"] = mace_settings_dict.get("save_cpu", False)
    mace_settings_dict["clip_grad"] = mace_settings_dict.get("clip_grad", 10.0)
    mace_settings_dict["model"] = mace_settings_dict.get("model", "MACE")
    mace_settings_dict["scaling"] = mace_settings_dict.get(
        "scaling", "rms_forces_scaling"
    )

    mace_settings_dict["wandb"] = mace_settings_dict.get("wandb", False)
    wandb_hypers = [
        "num_channels",
        "max_L",
        "correlation",
        "lr",
        "swa_lr",
        "weight_decay",
        "batch_size",
        "max_num_epochs",
        "start_swa",
        "energy_weight",
        "forces_weight",
    ]
    mace_settings_dict["wandb_log_hypers"] = mace_settings_dict.get(
        "wandb_log_hypers", wandb_hypers
    )

    # Converting to AttributeDict, which can be accessed similarly to args
    args = AttributeDict(mace_settings_dict)

    dataset = [
        mace_data.AtomicData.from_config(config, z_table=z_table, cutoff=args.r_max)
        for config in col.train
    ]

    train_loader = mace_tools.torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    valid_loader = mace_tools.torch_geometric.dataloader.DataLoader(
        dataset=[
            mace_data.AtomicData.from_config(config, z_table=z_table, cutoff=args.r_max)
            for config in col.valid
        ],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    args_loss = mace_settings_dict.get("loss", "weighted")
    loss_fn: torch.nn.Module
    if args_loss == "weighted":
        loss_fn = mace_modules.WeightedEnergyForcesLoss(
            energy_weight=args.energy_weight, forces_weight=args.forces_weight
        )
    elif args_loss == "forces_only":
        loss_fn = mace_modules.WeightedForcesLoss(forces_weight=args.forces_weight)
    elif args_loss == "virials":
        loss_fn = mace_modules.WeightedEnergyForcesVirialsLoss(
            energy_weight=args.energy_weight,
            forces_weight=args.forces_weight,
            virials_weight=args.virials_weight,
        )
    elif args_loss == "stress":
        loss_fn = mace_modules.WeightedEnergyForcesStressLoss(
            energy_weight=args.energy_weight,
            forces_weight=args.forces_weight,
            stress_weight=args.stress_weight,
        )
    elif args_loss == "huber":
        loss_fn = mace_modules.WeightedHuberEnergyForcesStressLoss(
            energy_weight=args.energy_weight,
            forces_weight=args.forces_weight,
            stress_weight=args.stress_weight,
            huber_delta=args.huber_delta,
        )
    elif args_loss == "dipole":
        assert (
            dipole_only is True
        ), "dipole loss can only be used with AtomicDipolesMACE model"
        loss_fn = mace_modules.DipoleSingleLoss(
            dipole_weight=args.dipole_weight,
        )
    elif args_loss == "energy_forces_dipole":
        assert dipole_only is False and compute_dipole is True
        loss_fn = mace_modules.WeightedEnergyForcesDipoleLoss(
            energy_weight=args.energy_weight,
            forces_weight=args.forces_weight,
            dipole_weight=args.dipole_weight,
        )
    else:
        # Unweighted Energy and Forces loss by default
        loss_fn = mace_modules.WeightedEnergyForcesLoss(
            energy_weight=1.0, forces_weight=1.0
        )
    # print("loss_fn: ", loss_fn)

    if args.compute_avg_num_neighbors:
        mace_settings_dict[
            "avg_num_neighbors"
        ] = mace_modules.compute_avg_num_neighbors(train_loader)
    # print(f"Average number of neighbors: {mace_settings_dict['avg_num_neighbors']}")

    # Selecting outputs
    compute_virials = False
    if args_loss in ("stress", "virials", "huber"):
        compute_virials = True
        args.compute_stress = True
        error_table = "PerAtomRMSEstressvirials"

    output_args = {
        "energy": compute_energy,
        "forces": args.compute_forces,
        "virials": compute_virials,
        "stress": args.compute_stress,
        "dipoles": compute_dipole,
    }
    # print(f"Selected the following outputs: {output_args}")

    # Build model
    # print("Building model")
    if args.num_channels is not None and args.max_L is not None:
        assert args.num_channels > 0, "num_channels must be positive integer"
        assert args.max_L >= 0, "max_L must be non-negative integer"
        args.hidden_irreps = o3.Irreps(
            (args.num_channels * o3.Irreps.spherical_harmonics(args.max_L))
            .sort()
            .irreps.simplify()
        )

    assert (
        len({irrep.mul for irrep in o3.Irreps(args.hidden_irreps)}) == 1
    ), "All channels must have the same dimension, use the num_channels and max_L keywords to specify the number of channels and the maximum L"

    # print(f"Hidden irreps: {args.hidden_irreps}")
    model_config = dict(
        r_max=args.r_max,
        num_bessel=args.num_radial_basis,
        num_polynomial_cutoff=args.num_cutoff_basis,
        max_ell=args.max_ell,
        interaction_cls=mace_modules.interaction_classes[args.interaction],
        num_interactions=args.num_interactions,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps(args.hidden_irreps),
        atomic_energies=atomic_energies,
        avg_num_neighbors=mace_settings_dict["avg_num_neighbors"],
        atomic_numbers=z_table.zs,
    )

    model: torch.nn.Module

    if args.model == "MACE":
        if args.scaling == "no_scaling":
            std = 1.0
            # print("No scaling selected")
        else:
            mean, std = mace_modules.scaling_classes[args.scaling](
                train_loader, atomic_energies
            )
        model = mace_modules.ScaleShiftMACE(
            **model_config,
            correlation=args.correlation,
            gate=mace_modules.gate_dict[args.gate],
            interaction_cls_first=mace_modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.mlp_irreps),
            atomic_inter_scale=std,
            atomic_inter_shift=0.0,
            radial_MLP=ast.literal_eval(args.radial_mlp),
            radial_type=args.radial_type,
        )
    elif model == "ScaleShiftMACE":
        mean, std = mace_modules.scaling_classes[args.scaling](
            train_loader, atomic_energies
        )
        model = mace_modules.ScaleShiftMACE(
            **model_config,
            correlation=args.correlation,
            gate=mace_modules.gate_dict[args.gate],
            interaction_cls_first=mace_modules.interaction_classes[
                args.interaction_first
            ],
            MLP_irreps=o3.Irreps(args.mlp_irreps),
            atomic_inter_scale=std,
            atomic_inter_shift=mean,
            radial_MLP=ast.literal_eval(args.radial_mlp),
            radial_type=args.radial_type,
        )
    elif model == "ScaleShiftBOTNet":
        mean, std = mace_modules.scaling_classes[args.scaling](
            train_loader, atomic_energies
        )
        model = mace_modules.ScaleShiftBOTNet(
            **model_config,
            gate=mace_modules.gate_dict[args.gate],
            interaction_cls_first=mace_modules.interaction_classes[
                args.interaction_first
            ],
            MLP_irreps=o3.Irreps(args.mlp_irreps),
            atomic_inter_scale=std,
            atomic_inter_shift=mean,
        )
    elif model == "BOTNet":
        model = mace_modules.BOTNet(
            **model_config,
            gate=mace_modules.gate_dict[args.gate],
            interaction_cls_first=mace_modules.interaction_classes[
                args.interaction_first
            ],
            MLP_irreps=o3.Irreps(args.mlp_irreps),
        )
    elif model == "AtomicDipolesMACE":
        # std_df = mace_modules.scaling_classes["rms_dipoles_scaling"](train_loader)
        assert args_loss == "dipole", "Use dipole loss with AtomicDipolesMACE model"
        assert (
            error_table == "DipoleRMSE"
        ), "Use error_table DipoleRMSE with AtomicDipolesMACE model"
        model = mace_modules.AtomicDipolesMACE(
            **model_config,
            correlation=args.correlation,
            gate=mace_modules.gate_dict[args.gate],
            interaction_cls_first=mace_modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.mlp_irreps),
            # dipole_scale=1,
            # dipole_shift=0,
        )
    elif model == "EnergyDipolesMACE":
        # std_df = mace_modules.scaling_classes["rms_dipoles_scaling"](train_loader)
        assert (
            args_loss == "energy_forces_dipole"
        ), "Use energy_forces_dipole loss with EnergyDipolesMACE model"
        assert (
            error_table == "EnergyDipoleRMSE"
        ), "Use error_table EnergyDipoleRMSE with AtomicDipolesMACE model"
        model = mace_modules.EnergyDipolesMACE(
            **model_config,
            correlation=args.correlation,
            gate=mace_modules.gate_dict[args.gate],
            interaction_cls_first=mace_modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.mlp_irreps),
        )
    else:
        raise RuntimeError(f"Unknown model: '{model}'")

    model.to(device)

    # Optimizer
    decay_interactions = {}
    no_decay_interactions = {}
    for name, param in model.interactions.named_parameters():
        if "linear.weight" in name or "skip_tp_full.weight" in name:
            decay_interactions[name] = param
        else:
            no_decay_interactions[name] = param

    param_options = dict(
        params=[
            {
                "name": "embedding",
                "params": model.node_embedding.parameters(),
                "weight_decay": 0.0,
            },
            {
                "name": "interactions_decay",
                "params": list(decay_interactions.values()),
                "weight_decay": args.weight_decay,
            },
            {
                "name": "interactions_no_decay",
                "params": list(no_decay_interactions.values()),
                "weight_decay": 0.0,
            },
            {
                "name": "products",
                "params": model.products.parameters(),
                "weight_decay": args.weight_decay,
            },
            {
                "name": "readouts",
                "params": model.readouts.parameters(),
                "weight_decay": 0.0,
            },
        ],
        lr=args.lr,
        amsgrad=args.amsgrad,
    )

    optimizer: torch.optim.Optimizer
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(**param_options)
    else:
        optimizer = torch.optim.Adam(**param_options)

    logger = mace_tools.MetricsLogger(directory=args.results_dir, tag=tag + "_train")

    lr_scheduler = LRScheduler(optimizer, args)

    swa = None
    swas = [False]
    if args.swa:
        assert dipole_only is False, "swa for dipole fitting not implemented"
        swas.append(True)
        if args.start_swa is None:
            args.start_swa = (
                args.max_num_epochs // 4 * 3
            )  # if not set start swa at 75% of training
        if args_loss == "forces_only":
            print("Can not select swa with forces only loss.")
        elif args_loss == "virials":
            loss_fn_energy = mace_modules.WeightedEnergyForcesVirialsLoss(
                energy_weight=args.swa_energy_weight,
                forces_weight=args.swa_forces_weight,
                virials_weight=args.swa_virials_weight,
            )
        elif args_loss == "stress":
            loss_fn_energy = mace_modules.WeightedEnergyForcesStressLoss(
                energy_weight=args.swa_energy_weight,
                forces_weight=args.swa_forces_weight,
                stress_weight=args.swa_stress_weight,
            )
        elif args_loss == "energy_forces_dipole":
            loss_fn_energy = mace_modules.WeightedEnergyForcesDipoleLoss(
                args.swa_energy_weight,
                forces_weight=args.swa_forces_weight,
                dipole_weight=args.swa_dipole_weight,
            )
            # print(
            #     f"Using stochastic weight averaging (after {args.start_swa} epochs) with energy weight : {args.swa_energy_weight}, forces weight : {args.swa_forces_weight}, dipole weight : {args.swa_dipole_weight} and learning rate : {args.swa_lr}"
            # )
        else:
            loss_fn_energy = mace_modules.WeightedEnergyForcesLoss(
                energy_weight=args.swa_energy_weight,
                forces_weight=args.swa_forces_weight,
            )
            # print(
            #     f"Using stochastic weight averaging (after {args.start_swa} epochs) with energy weight : {args.swa_energy_weight}, forces weight : {args.swa_forces_weight} and learning rate : {args.swa_lr}"
            # )
        swa = mace_tools.SWAContainer(
            model=AveragedModel(model),
            scheduler=SWALR(
                optimizer=optimizer,
                swa_lr=args.swa_lr,
                anneal_epochs=1,
                anneal_strategy="linear",
            ),
            start=args.start_swa,
            loss_fn=loss_fn_energy,
        )

    checkpoint_handler = mace_tools.CheckpointHandler(
        directory=args.checkpoints_dir,
        tag=tag,
        keep=args.keep_checkpoints,
        swa_start=args.start_swa,
    )

    start_epoch = 0
    if args.restart_latest:
        try:
            opt_start_epoch = checkpoint_handler.load_latest(
                state=mace_tools.CheckpointState(model, optimizer, lr_scheduler),
                swa=True,
                device=device,
            )
        except Exception:  # pylint: disable=W0703
            opt_start_epoch = checkpoint_handler.load_latest(
                state=mace_tools.CheckpointState(model, optimizer, lr_scheduler),
                swa=False,
                device=device,
            )
        if opt_start_epoch is not None:
            start_epoch = opt_start_epoch

    ema = None
    if args.ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)

    # print(model)
    # print(f"Number of parameters: {mace_tools.count_parameters(model)}")
    # print(f"Optimizer: {optimizer}")

    if args.wandb:
        # print("Using Weights and Biases for logging")
        import wandb

        wandb_config = {}
        # args_dict = vars(args)
        args_dict = dict(args)
        # print("args_dict: ", args_dict)
        args_dict_json = json.dumps(args_dict)
        for key in args.wandb_log_hypers:
            wandb_config[key] = args_dict[key]
        mace_tools.init_wandb(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=wandb_config,
        )
        wandb.run.summary["params"] = args_dict_json

    mace_tools.train(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpoint_handler=checkpoint_handler,
        eval_interval=args.eval_interval,
        start_epoch=start_epoch,
        max_num_epochs=args.max_num_epochs,
        logger=logger,
        patience=args.patience,
        output_args=output_args,
        device=device,
        swa=swa,
        ema=ema,
        max_grad_norm=args.clip_grad,
        log_errors=args.error_table,
        log_wandb=args.wandb,
    )

    # Evaluation on test datasets
    # print("Computing metrics for training, validation, and test sets")

    all_collections = [
        ("train", col.train),
        ("valid", col.valid),
    ] + col.tests

    for swa_eval in swas:
        # epoch = checkpoint_handler.load_latest(
        #     state=mace_tools.CheckpointState(model, optimizer, lr_scheduler),
        #     swa=swa_eval,
        #     device=device,
        # )
        model.to(device)
        # print(f"Loaded model from epoch {epoch}")

        for param in model.parameters():
            param.requires_grad = False
        table = create_error_table(
            table_type=args.error_table,
            all_collections=all_collections,
            z_table=z_table,
            r_max=args.r_max,
            valid_batch_size=args.valid_batch_size,
            model=model,
            loss_fn=loss_fn,
            output_args=output_args,
            log_wandb=args.wandb,
            device=device,
        )

        # print('table:', dir(type(table)))
        # HACK: Getting validation information from pretty table
        train_info = table[1].get_csv_string().split("\n")[1].split(",")
        rmse_e_valid = float(train_info[1])
        rmse_f_valid = float(train_info[2])

        # Save entire model
        if swa_eval:
            model_path = Path(args.checkpoints_dir) / (tag + "_swa.model")
        else:
            model_path = Path(args.checkpoints_dir) / (tag + ".model")
        # print(f"Saving model to {model_path}")
        if args.save_cpu:
            model = model.to("cpu")

        torch.save(model, model_path)

        if swa_eval:
            torch.save(model, Path(args.model_dir) / (args.name + "_swa.model"))
        else:
            torch.save(model, Path(args.model_dir) / (args.name + ".model"))

    return {
        "m_path": Str(model_path),
        "m_rmse_e": Float(rmse_e_valid),
        "m_rmse_f": Float(rmse_f_valid),
    }


@calcfunction
def create_mace_lammps_model(model_file):
    # print("model_file: ", model_file)
    # print("type(model_file): ", type(model_file))
    # print("dir(model_file): ", dir(model_file))

    # model_file_uuid = model_file.uuid
    # model_file.store()

    with model_file.as_path() as model_path:
        print("model_path: ", model_path)

        # Making path absolute
        # model_path: Path = Path(model_path.value).resolve()

        # Loading model
        model = torch.load(model_path)
        model = model.double().to("cpu")
        lammps_model = LAMMPS_MACE(model)
        lammps_model_compiled = jit.compile(lammps_model)

        # Creating new path
        new_model_path = str(model_path) + "-lammps.pt"

        # Saving LAMMPS model
        lammps_model_compiled.save(new_model_path)

        # TODO: Create SingleileData for lammps model?
        return SinglefileData(file=new_model_path)


def serialize_ase(curr_s):
    if not isinstance(curr_s, dict):
        curr_s = curr_s.todict()

    # curr_s = s.todict()
    curr_s["pbc"] = [bool(boo) for boo in curr_s["pbc"]]
    return curr_s


@calcfunction
def prepare_output_final_training_db(training_db_list):
    # Converting training_db to aiida types
    struct_list = []
    for ase_struct in training_db_list:
        # ase_struct: Atoms
        # aiida_struct = StructureData()

        # aiida_struct.set_ase(ase_struct)
        # struct_list.append(aiida_struct)
        # serial_struct = serialize_ase(ase_struct)
        struct_list.append(ase_struct)

    print("struct_list: ", struct_list)
    print("type struct_list: ", type(struct_list))

    return List(struct_list)


@calcfunction
def gather_dft_calcs(dft_calc_list):
    vasprun_list = []
    # Adding structures to the initial DB
    for finished_dft_calc in dft_calc_list:
        finished_dft_calc = load_node(finished_dft_calc)
        vasprun = mdb_conv._gather_mace_req_calc_data_from_node(finished_dft_calc)
        vasprun = vasprun.todict()
        vasprun["pbc"] = [bool(boo) for boo in vasprun["pbc"]]
        vasprun_list.append(vasprun)

    return_list = List([val for val in vasprun_list])
    # self.out("dft_calcs", List(self.ctx.vasprun_list))
    return return_list


@calcfunction
def remove_structs_from_seed_gen_db(seed_gen: List, delete_indices: list) -> List:
    """
    Remove specified structures from a seed generation database based on UUIDs.

    This function iterates over a list of UUIDs (delete_indices) and removes the
    corresponding structures from a seed generation database. The database is accessed
    and modified via the `seed_gen` object, which is converted to a list.
    Each element of the list is an ase.Atoms object with an unique identifier
    (`aiida_uuid`) in the info attribute.
    The function returns the modified list of structures after the specified ones
    have been removed.

    Parameters
    ----------
    seed_gen : List
        An object that contains the seed generation database.
    delete_indices : list
        A list of UUIDs (strings) identifying the structures to be removed
        from the seed generation database.

    Returns
    -------
    List
        An aiida List of the remaining structures in the seed generation database
        after the specified structures have been removed.
    """
    seed_gen_db = seed_gen.get_list()
    for curr_uuid in delete_indices:
        for del_idx, struct in enumerate(seed_gen_db):
            struct_uuid = struct["info"]["aiida_uuid"]
            if curr_uuid == struct_uuid:
                del seed_gen_db[del_idx]

    return List(seed_gen_db)


@calcfunction
def check_md_seed_agreement(return_list: list) -> Bool:
    """
    Check if all predictions agree for current seed.

    Parameters
    ----------
    return_list : list
        List containing all calculations for predictions where the
        models disagreed.

    Returns
    -------
    Bool
        True if all the predictions have agreed for the current MD seed
        on the current AL iteration. False if there is no agreement on
        on all structures.
    """
    if len(return_list) > 0:
        return Bool(False)
    else:
        return Bool(True)
