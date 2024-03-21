"""Run an active-learning procedure based on ML-MD using aiida."""

import pathlib as pl

from aiida import load_profile
from aiida.engine import run
from aiida.orm import Dict, Int
from aiida.plugins import WorkflowFactory

if __name__ == "__main__":
    data_path = pl.Path(
        # "/WAREHOUSE/projects/P2/nnp/nnp_tests/active_learning_devel/test_large_1",
        "/WAREHOUSE/projects/P2/nnp/nnp_tests/active_learning_devel"
    )

    init_db_path = data_path / "train_data_test.xyz"  # TESTING
    # init_db_path = data_path / "mace_training_data_20240123T145349.xyz"

    final_db_path = data_path / "results_db" / "final_data_test.xyz"
    mace_settings_path = data_path / "mace_settings.json"
    load_profile()

    # Getting builder for workchain
    al_calculation = WorkflowFactory("mdb-active-learning-base")
    builder = al_calculation.get_builder()

    # Setting mandatory inputs.
    # Input settings
    builder.active_learning.data_path = str(init_db_path)
    builder.active_learning.mace_settings_path = str(mace_settings_path)
    builder.active_learning.init_db_path = str(init_db_path)
    builder.active_learning.final_db_path = str(final_db_path)

    # General AL settings
    builder.active_learning.max_iterations = Int(10)

    # AL-MD settings
    # Size of the seed generating database in percentage of the total number of
    # available training structures
    # TESTING: Update seed size once debugging is done
    builder.active_learning.seed_size_frac = 0.12  # TESTING: 0.0010
    builder.active_learning.md_temperature_K = 300.0
    builder.active_learning.md_num_steps = 100  # TESTING: 33334
    builder.active_learning.md_timestep_duration_ps = 0.003
    builder.active_learning.commitee_num_models = 2  # 4
    builder.active_learning.chem_acc = 30  # meV? # TESTING: 10?
    builder.active_learning.chem_acc_multiplier = 10  # TESTING: 0.0001

    # Frames to keep for DFT
    builder.active_learning.al_keep_frame_interval_perc = 0.005  # TESTING: 0.01 # 0.005

    # Settings for the MACE NNP training.
    mace_train_settings = {
        "name": "nnp_training_test",
        "energy_key": "free_energy",
        "valid_fraction": 0.1,
        "config_type_weights": {"Default": 1.0},
        "weight_decay": 9.336844675542452e-07,
        "E0s": "average",
        "num_interactions": 2,
        "model": "MACE",
        "correlation": 3,
        "hidden_irreps": "16x0e + 16x1o",
        "lr": 0.005626773506534471,
        "r_max": 6.0,
        "max_ell": 3,
        "max_L": 2,
        "batch_size": 64,
        "max_num_epochs": 30,
        "swa": True,
        "ema": True,
        "ema_decay": 0.99,
        "amsgrad": True,
        "restart_latest": True,
        "device": "cuda",
        "default_dtype": "float32",
        "wandb": False,
    }

    # HACK: During debugging, run the calculation on 1 CPU and kill it
    # if it runs longer than 1800 seconds.
    # Settings for MACE-LAMMPS MD
    lammps_mace_settings = {
        "code": "mace-lammps-gpu@tekla2-updated-2024",
        "metadata": {
            "options": {
                "resources": {
                    "parallel_env": "c128m1024ib_mpi_32slots",
                    "tot_num_mpiprocs": 1,
                },
                "queue_name": "c128m1024ibgpu4.q",
                "max_memory_kb": 102400000,
                "max_wallclock_seconds": 117280000,
                "account": "",
                "qos": "",
                "withmpi": False,
                "custom_scheduler_commands": ("#$ -l gpu=1"),
            }
        }
    }
    builder.active_learning.lammps_mace = Dict(value=lammps_mace_settings)

    # MACE training settings
    mace_train_dict = {
        "code": "mace_run_train_gpu@tekla2-new-test",
        "metadata": {
            "options": {
                "resources": {
                    "parallel_env": "c128m1024ib_mpi_32slotsbis",
                    "tot_num_mpiprocs": 32,
                },
                "parser_name": "mace-training-parser",
                "queue_name": "c128m1024ibgpu4.q",
                "max_wallclock_seconds": 117280000,
                "max_memory_kb": 102400000,
                "account": "",
                "qos": "",
                "withmpi": True,
                "custom_scheduler_commands": ("#$ -l gpu=1"),
            },
        },
        "result_force_weight": 0.1,
        "train_settings": mace_train_settings,
    }
    builder.active_learning.mace_train = Dict(value=mace_train_dict)

    # TESTING: This should use aiida.engine.submit function once all debugging is done.
    node = run(builder)
    # print("node: ", node.pk)
