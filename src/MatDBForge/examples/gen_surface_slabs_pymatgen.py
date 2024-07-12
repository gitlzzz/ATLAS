from MatDBForge.core import initial_db as mdb_indb
from MatDBForge.core import surfaces as mdb_surf
import MatDBForge.core.utils as mdb_ut
import time
import pathlib as pl


NUM_STRUCT = 3
NUM_REPEAT = 1


db_path = (
    # "/home/psanz/teklahome/projects/p2-CuZn/relaxed_structures_initialdb/initial_db"
    "/tmp"
)
LOG_PATH = "/tmp"

# Getting current time
ctime = time.strftime("%Y%m%dT%H%M%S")
mdb_ut.init_logger(source=pl.Path(__file__).stem, log_path=LOG_PATH)

structures = mdb_indb.CuZnInitialDatabase(
    # f"{db_path}/initial-database_19052023-165058_final.pkl",
    "/tmp/initial-database_11102023-161158_new_cluster.xz",
    max_num_atoms=128,
)

mdb_ut.custom_print(structures, "done")

structures.df["surface"] = None
structures.df["surface"].fillna(False, inplace=True)
structures.df["surface"] = structures.df["surface"].astype("boolean")

# Filtering phases
selected_phases = [
    phase
    for phase in mdb_indb.CuZnInitialDatabase.DB_PHASE_DIAGRAM.phases
    if phase.name not in ["m1", "m2", "m3", "m4"]
]

mdb_ut.custom_print("Generating surfaces from initial structures...", "debug")
for phase in selected_phases:
    # Line break for aesthetic purposes
    print()

    # Creating surfaces from the base structures, generating
    # different supercells and applying replacements.
    mdb_ut.custom_print(f"Current phase: {phase}.", "info")
    slabs = mdb_surf.generate_surfaces_pymatgen(
        db_obj=structures,
        phase=phase,
        overwrite_max_num_atoms=128,
        min_miller_index=1,
        max_miller_index=3,
        min_slab_size=4,  # Angs
        # max_slab_size=14,  # Angs
        num_diff_layer_size=3,  # 3
        min_vacuum_size=12,  # Angs
        get_supercells=True,
        # get_replacements=True,
        num_replacement_structs=2,
        num_replacement_repeats=2,
        fixed_layers=2,
        limit_per_phase=250,
        limit_supercell=500,
    )
    print('slabs: ', slabs)

    mdb_surf.apply_replacement_surface(
        db_obj=structures,
        slabs_to_replace=slabs,
        save_in_db=True,
        num_replacement_repeats=NUM_REPEAT,
        num_replacement_structs=NUM_STRUCT,
        limit_replacements=500,
    )

    mdb_ut.custom_print(
        "Checking for repeated structures...",
        "info",
    )
    structures.find_repeat_structures(
        delete=True,
        filters=["surface"],
        phase=phase,
    )

    mdb_ut.custom_print(
        "Applying a random perturbation to the surfaces...",
        "info",
    )
    mdb_surf.apply_gauss_perturb_db(
        db_obj=structures,
        repeat=NUM_REPEAT,
        filters=["surface"],
        phase=phase,
    )


print()

mdb_ut.custom_print(structures, "done")
print()

structures.save_database(path="/tmp/", suffix="surfaces_pymatgen")

# surfaces = structures.df.loc[structures.df.surface]
# print(surfaces)
