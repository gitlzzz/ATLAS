"""Run an active-learning procedure based on ML-MD using aiida."""

import pathlib as pl

from aiida import load_profile
from aiida.engine import run, submit
from aiida.orm import Int, SinglefileData
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
    builder.active_learning.data_path = str(init_db_path)  # str(data_path)
    builder.active_learning.mace_settings_path = str(mace_settings_path)
    builder.active_learning.init_db_path = str(init_db_path)
    builder.active_learning.final_db_path = str(final_db_path)
    # builder.active_learning.mace_potential_names = mace_potential_names

    # MD settings
    md_steps = 100  # TESTING: 33334
    # TESTING: Update seed once debugging is done
    builder.active_learning.seed_size_frac = 0.1  # TESTING: 0.0010
    builder.active_learning.md_temperature_K = 300.0
    builder.active_learning.md_num_steps = md_steps
    builder.active_learning.md_timestep_duration_ps = 0.003
    builder.active_learning.commitee_num_models = 0  # 4
    # builder.active_learning.lammps_potential_path = str(potential_path)

    # AL settings
    builder.max_iterations = Int(100)
    # Frames to keep for DFT
    # TODO: Test this vlues
    builder.active_learning.al_keep_frame_interval_perc = 0.1  # TESTING: 0.01 # 0.005

    # TESTING: This should use aiida.engine.submit function once all debugging is done.
    node = run(builder)
    # print("node: ", node.pk)
