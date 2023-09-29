from MatDBForge.core import initial_db as indb
import MatDBForge.core.utils as ut

NUM_STRUCT = 3
NUM_REPEAT = 2

db_path = (
    "/home/psanz/teklahome/projects/p2-CuZn/relaxed_structures_initialdb/initial_db"
)

structures = indb.CuZnInitialDatabase(
    # f"{db_path}/initial-database_19052023-165058_final.pkl",
    # "/tmp/initial-database_12072023-165311_test_new_db_style.pkl",
    "/tmp/initial-database_07092023-122416_test_file_structure.xz",
    max_num_atoms=128,
)

ut.custom_print(structures, "done")

structures.df["surface"] = None
structures.df["surface"].fillna(False, inplace=True)
structures.df["surface"] = structures.df["surface"].astype("boolean")

# Filtering phases
selected_phases = [
    phase
    for phase in indb.DB_PHASE_DIAGRAM.phases
    if phase.name not in ["m1", "m2", "m3", "m4"]
]

ut.custom_print("Generating surfaces from initial structures...", "debug")
for phase in selected_phases:
    # Line break for aesthetic purposes
    print()

    # Creating surfaces from the base structures, generating different supercells
    # and applying replacements.
    ut.custom_print(f"Current phase: {phase}.", "info")
    structures.generate_surfaces_pure(
        phase=phase,
        overwrite_max_num_atoms=64,
        max_miller_index=2,
        min_slab_size=4, # Angs
        max_slab_size=14, # Angs
        num_diff_layer_size=3, # 3
        min_vacuum_size=12, # Angs
        get_supercells=False,
        get_replacements=True,
        num_replacement_structs=2,
        num_replacement_repeats=2,
        fixed_layers=2,
        limit_per_phase=250,
    )

    ut.custom_print("Checking for repeated structures...","info",)
    structures.find_repeat_structures(delete=True, filters=[('surface', 'replacement')], phase=phase)

print()

ut.custom_print("Applying a random perturbation to the surfaces...", "info")
structures.perturb_gauss(filters=['surface'], repeat=3)
ut.custom_print(structures, "done")
print()

structures.save_database(path="/tmp/", suffix="testing_site_properties")

# surfaces = structures.df.loc[structures.df.surface]
# print(surfaces)
