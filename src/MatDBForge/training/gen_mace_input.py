"""Generate inputs for the MACE potential."""
import pathlib as pl

import MatDBForge.core.utils as mdb_ut
import MatDBForge.training.conversion as mdb_train_cnv

LOG_PATH = "/tmp"
mdb_ut.init_logger(source=pl.Path(__file__).stem, log_path=LOG_PATH)

filter_dict = {
    "attributes.process_state": "finished",  # the process is finished AND
    "attributes.exit_status": 0,  # has exit_status == 0}})
    # "label": {"like": "%_sp"},
    # "label": {"!like": "%_perturb_%"},
}

# old: sp_surface_batch_20230828T231454

mdb_train_cnv.gen_mace_train_aiida(
    # path="/WAREHOUSE/projects/P2/nnp/mace_training_data",
    aiida_group_list=[
        # Bulks
        "correct_calcs_nnp",
        # Surfaces
        "fixed_kpoint_surfaces",
        # Clusters
        "sp_cluster_batch_2_finished",
        "small_cluster_complement_20230929T000201",
    ],
    filter_dict=filter_dict,
)
