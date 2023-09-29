import pathlib as pl
from datetime import datetime

import MatDBForge.core.clusters as mdb_clust
import MatDBForge.core.utils as mdb_utils
from MatDBForge.core import initial_db as indb

MAX_SIZE = 5
NUM_STRUCT = 4
NUM_REPEAT = 10

# Gathering current time for filenames
now = datetime.now()
date_time = now.strftime("%Y-%m-%dT%H%M%S")

# Logger preparation
mdb_utils.init_logger(source=pl.Path(__file__).stem)

# Logging generation parameters
mdb_utils.custom_print(
    (
        f"Current run params: MAX_SIZE = {MAX_SIZE},"
        f" NUM_STRUCT = {NUM_STRUCT}, NUM_REPEAT = {NUM_REPEAT}"
    ),
    "debug",
)

# Creating a InitialDatabase object.
indb = indb.CuZnInitialDatabase(
    f"clusters_{date_time}_final",
    max_num_atoms=128,
)
mdb_utils.custom_print(indb, "done")

# Selecting phases to use
selected_phases = [
    phase
    for phase in indb.DB_PHASE_DIAGRAM.phases
    if phase.name not in ["m1", "m2", "m3", "m4"]
]

mdb_utils.custom_print("Adding base clusters...", "info")

# Generating cluster structures for different sizes
# using current phase, and applying
# replacements and perturbations to each generated structure.
# Structures are then stored into the database.
cluster_list = indb.generate_clusters(
    size_range=range(3, MAX_SIZE + 1),
    num_struct=NUM_STRUCT,
    num_repeat=NUM_REPEAT,
    add_dimer=True,
    save_in_db=True,
)

mdb_utils.custom_print(indb, "done")
print()

# For every phase, iterate over all structures and
# generate N_STRUCT*N_REPEAT random replacements.
mdb_utils.custom_print("Applying replacements...", "info")
for phase in selected_phases:
    mdb_utils.custom_print(f"Working with '{phase.name}' phase...", "debug")
    replaced_clusters = mdb_clust.apply_replacement_cluster_db(
        db_obj=indb,
        phase=phase,
        num_struct=NUM_STRUCT,
        num_repeat=NUM_REPEAT,
        similarity_check=True,
        save_in_db=False,
    )
    mdb_utils.custom_print(f"Phase '{phase.name}' done!", "done")


# Checking for similarity after replacement
unique_repl_clusters = mdb_utils.similarity_check_list(
    db_obj=indb,
    replaced_structures=replaced_clusters,
    save_in_db=True,
)

mdb_utils.clear_previous_print()
print()

# Getting all generated structures (base and perturb)
# and applying a perturbation.
mdb_utils.custom_print("Applying perturbations...", "info")
perturbed_clusters = mdb_clust.apply_gauss_perturb_db(
    db_obj=indb,
    center=0.04,
    repeat=NUM_REPEAT,
)

mdb_utils.custom_print(indb, "done")

indb.save_database(
    path="/home/psanz/teklahome/projects/p2-CuZn/relaxed_structures_initialdb/cluster_database",
    suffix="database",
)
