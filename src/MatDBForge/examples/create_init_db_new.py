import pymatgen.io.vasp as vasp
import pandas as pd

import pandas as pd
import time
from MatDBForge.core import utils as ut
from MatDBForge.core import initial_db_utils as indb

# List containing the MP ids for the target structures
# to be included and perturbed.
TARGET_STRUCTURES = [
    "mp-30",  # Fm3-m, Cu alpha
    "mp-987",  # Pm-3m space group, beta' phase
    "mp-1368",  # I-43m, gamma phase
    "mp-1216020",  # P1 space group, gamma-epsilon (m3) phase
    "mp-972042",  # P6₃/mmc space group, epsilon phas
    "mp-1215518",  # P-6m2, narrow phase at high T above alpha?
    "mp-1215401",  # P-3m1, beta-derived
    "mp-79",  # P6₃/mmc, Zn eta
]

PHASES = [
    "alpha",
    "m1",
    "beta-prime",
    "m2",
    "gamma",
    "m3",
    "epsilon",
    "m4",
    "eta",
]

# Number of structures to be generated from each
# base structure

# Number of replacement percentages to be computed per base structure
NUM_STRUCT = 50

# Number of random replacements to be done for a single percentage
NUM_REPEAT = 5

RELAX_STRUCT_PATH = "/home/psanz/teklahome/projects/p2-CuZn/relaxed_structures_initialdb/relaxed_structures"
SAVE_PATH = (
    "/home/psanz/teklahome/projects/p2-CuZn/relaxed_structures_initialdb/initial_db"
)

current_time = time.strftime("%d%m%Y-%H%M%S")
db_name = "initial-database_" + current_time

structures = indb.CuZnInitialDatabase(
    database_name=db_name, use_offset=True, max_num_atoms=64
)


# New version where structures obtained with DFT relaxation are read
# from a given path
structures.read_base_structures(path=RELAX_STRUCT_PATH)

# Generating perturbed structures for every base structure.
for phase in PHASES:
    # Getting properties for the current phase
    props = structures.CUZN_PHASES.get(phase)

    ut.custom_print(f"Generating structures for '{phase}' phase.", "info")

    # Generating NUM_STRUCT*NUM_REPEAT structures for the given phase.
    structures.generate_phase_structures(
        prototype=props.get("prototype", None),
        phase=phase,
        num_struct=NUM_STRUCT,
        num_repeats=NUM_REPEAT,
        get_different_supercells=True,
    )

print()
ut.custom_print(f"Done! {len(structures.df.index)} structures generated.", "done")

print()

# Checking for duplicate structures, and deleting the ones that are repeated.
structures.find_repeat_structures(delete=True)
ut.custom_print(structures, "info")
print()

# Displacing structures around PES minima by modifying the
# relaxed cell lattice parameters.
structures.perturb_min_displacement(frac_max=0.05, repeat=40)

ut.custom_print("Displacements around minimum done.", "done")
ut.custom_print(structures, "info")
print()

# Adding a random perturbation to structures.
structures.perturb_gauss(center=0.04, repeat=7)

ut.custom_print("Random perturbation done.", "done")
ut.custom_print(structures, "info")
print()

# Saving database
structures.save_database(path=SAVE_PATH, suffix="main_structs_small")
