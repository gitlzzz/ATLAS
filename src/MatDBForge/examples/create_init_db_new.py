import time

from MatDBForge.core import initial_db as indb
from MatDBForge.core import utils as mdb_ut

# Desired phases
PHASES = [
    "alpha",
    "beta-prime",
    "gamma",
    "delta",
    "epsilon",
    "eta",
]

# The following parameters will set the number of structures
# to be generated from each base structure
#
# Number of replacement percentages to be computed per base structure
NUM_STRUCT = 5

# Number of random replacements to be done for a single percentage
NUM_REPEAT = 2

# Where the optimized base structures are located
RELAX_STRUCT_PATH = "/home/psanz/teklahome/projects/p2-CuZn/relaxed_structures_initialdb/relaxed_structures"

# Where to store the initial database once ready
SAVE_PATH = (
    # "/home/psanz/teklahome/projects/p2-CuZn/relaxed_structures_initialdb/initial_db"
    "/tmp"
)

current_time = time.strftime("%d%m%Y-%H%M%S")
db_name = "initial-database_" + current_time

structures = indb.CuZnInitialDatabase(
    database_name=db_name, use_offset=True, max_num_atoms=128
)

# Initial structures obtained with DFT relaxation are read from a given path
structures.read_base_structures(path=RELAX_STRUCT_PATH, target_structures=PHASES)

# Generating perturbed structures for every base structure.
# for phase in PHASES:
for phase in PHASES:
    # Getting properties for the current phase
    phase = structures.DB_PHASE_DIAGRAM.get_phase(phase)

    mdb_ut.custom_print(f"Generating structures for '{phase.name}' phase.", "info")

    # Generating NUM_STRUCT*NUM_REPEAT structures for the given phase.
    structures.generate_bulk_structures(
        prototype=phase.prototype,
        phase=phase,
        num_struct=NUM_STRUCT,
        num_repeats=NUM_REPEAT,
        get_different_supercells=True,
    )

print()
mdb_ut.custom_print(f"Done! {len(structures.df.index)} structures generated.", "done")
print()

# Checking for duplicate structures, and deleting the ones that are repeated.
mdb_ut.custom_print(
    "Checking for repeated structures...",
    "info",
)
structures.find_repeat_structures(delete=True)
mdb_ut.custom_print(structures, "info")

# Displacing structures around PES minima by modifying the
# relaxed cell lattice parameters.
mdb_ut.custom_print("Adding structures displaced around the minimum...", "info")
structures.perturb_min_displacement(frac_max=0.05, repeat=5)
mdb_ut.custom_print("Displacements around minimum done.", "done")
mdb_ut.custom_print(structures, "info")
print()

# Adding a random perturbation to structures.
mdb_ut.custom_print("Adding a random perturbation to structures...", "info")
structures.perturb_gauss(center=0.04, repeat=3, filters=["replacement", "base"])
mdb_ut.custom_print("Random perturbation done.", "done")
mdb_ut.custom_print(structures, "info")
print()

# Saving database
mdb_ut.custom_print("Checking for incorrect phase assignation...", "info")
mdb_ut.check_incorrect_ratios(structures.df, indb.CuZnInitialDatabase.DB_PHASE_DIAGRAM)
structures.save_database(path=SAVE_PATH, suffix="test_file_structure")
