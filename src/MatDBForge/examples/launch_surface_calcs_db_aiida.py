import pathlib as pl

from MatDBForge.core import initial_db as mdb_indb
from MatDBForge.core import utils as mdb_ut
from MatDBForge.workflows import aiida_utils as mdb_aut

# k-spacing values for every phase to be
# included in the INCAR
KSPACING = {
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

# Dictionary containing settings for selecting aiida's computer and code.
# Example:
#
# QUEUE_DICT = {
#    20: {
#     "node_cpus": 48,
#     "code_string": "vasp-5.4.4_mn@marenostrum1",
#     "qos": "class_a",
#     "max_wallclock_seconds": 3600,
#     "max_memory_kb": 96000000,
#     "options_resources": {
#         "parallel_env": "class_a",
#         "tot_num_mpiprocs": 48,
#     },
#     "multiple": 1,
# },
#
QUEUE_DICT = {
    10: {
        "type": "slurm",
        "node_cpus": 48,
        "code_string": "vasp-5.4.4_mn@marenostrum1",
        "qos": "debug",
        "max_wallclock_seconds": 8000,
        "max_memory_kb": 96000000,
        "options_resources": {
            "tot_num_mpiprocs": 48,
        },
        "multiple": 1,
    },
    20: {
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
            "tot_num_mpiprocs": 24,
        },
        "multiple": 2,
    },
}

# POTCAR equivalent
# Potential_family is chosen among the list given by
# 'verdi data vasp-potcar listfamilies'
POTENTIAL_FAMILY = "vasp-5.4-PBE-2023"

# The potential mapping selects which potential to use
# This could for instance be {'Si': 'Si_GW'} to use the GW ready potential
# We use a specific function which just uses the default potential for
# every atom.
POTENTIAL_MAPPING = mdb_aut.generate_potential_mapping()

# Paths for the source and target dataframe.
SOURCE_DF = (
    "/WAREHOUSE/P2/initial_databases/relaxed_structures_initialdb/surfaces/"
    "surfaces_bulks-database_11102023-161158_pymatgen.xz"
)
TARGET_DF = "/WAREHOUSE/surface_database.pkl"
LOG_PATH = "/WAREHOUSE/P2/calc_logs"

# Which calculation to run.
# Refer to the CalcType class for more information.
CALC_TYPE = mdb_aut.CalcType.single_point_surface

# Maximum size of each calculation batch.
# A maximum of MAX_BATCH calculations will run at once.
MAX_BATCH = 300

# Skip all chunks from 0 up to the the chunk number set in START_ON.
START_ON = 3

# Group name prefix
GROUP_NAME = "surface_run_kpoints_fixed"


if __name__ == "__main__":
    # Start logger
    mdb_ut.init_logger(source=pl.Path(__file__).stem, log_path=LOG_PATH)

    # Loading the initial structures dataframe
    initial_db = mdb_indb.CuZnInitialDatabase(SOURCE_DF)

    # Limiting the number of allowed structures per phase
    # initial_db.limit_structure_number_phases(
    #     structure_limit=500,
    #     phases_to_use=["alpha", "beta-prime", "gamma", "epsilon", "delta", "eta"],
    #     structure_types=["cluster"],
    # )

    # Selecting the desired structures
    sel_struct_df = initial_db.df[initial_db.df["surface"]]

    mdb_aut.run_dataframe_vasp_simulations_aiida(
        sel_struct_df=sel_struct_df,
        initial_db=initial_db,
        calc_type=CALC_TYPE,
        group_name=GROUP_NAME,
        kspacing_dict=KSPACING,
        max_batch=MAX_BATCH,
        start_on=START_ON,
        potential_family=POTENTIAL_FAMILY,
        potential_mapping=POTENTIAL_MAPPING,
        queue_dict=QUEUE_DICT,
        dry_run=False,
    )
