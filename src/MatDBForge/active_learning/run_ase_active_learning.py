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
    builder.active_learning.seed_size_frac = 0.02  # TESTING: 0.0010
    builder.active_learning.md_temperature_K = 300.0
    builder.active_learning.md_num_steps = 100  # TESTING: 33334
    builder.active_learning.md_timestep_duration_ps = 0.003
    builder.active_learning.commitee_num_models = 2  # 4

    # Frames to keep for DFT
    builder.active_learning.al_keep_frame_interval_perc = 0.005  # TESTING: 0.01 # 0.005

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
    }
    builder.active_learning.mace_train = Dict(value=mace_train_dict)

    # TESTING: This should use aiida.engine.submit function once all debugging is done.
    node = run(builder)
    # print("node: ", node.pk)
