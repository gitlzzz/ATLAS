## MLIP Benchmarks

Generate a mlip benchmarks template file using `mdb_gen_configuration_file -t mlip_benchmarks`.

:::{attention}
All keys are mandatory unless stated otherwise.
:::


### General benchmark settings. - `[general]`


- `output_dir`: (str) Directory to save benchmark results.
  - Default is `'./mlip_evaluation'`.

- `metal`: (str) Metal symbol for the benchmark systems.
  - Default is `'Cu'`.

- `device`: (optional, str) Device to run calculations on.
  - Default is `'cuda'`.
  - Possible values are: `cuda`, `cpu`.

- `dtype`: (optional, str) Data type for calculations.
  - Default is `'float64'`.
  - Possible values are: `float32`, `float64`.

- `no_rich_ui`: (optional, bool) Disable Rich UI and use plain text output.
  - Default is `False`.

### Model specifications for benchmarking. - `[models]`


- `model_files`: (optional, list[str]) List of paths to .model files.
  - Default is `[]`.

- `aiida_pks`: (optional, list[int]) List of AiiDA workchain PKs/UUIDs to load models from.
  - Default is `[]`.

- `foundation_models`: (optional, list[str]) List of foundation model specifications. Format: 'library:model_name'.
  - Default is `[]`.

### Settings for slab generation in benchmarks. - `[slab_generation]`


- `surface_indices`: (optional, list[int]) Miller indices for the surface.
  - Default is `[1, 1, 1]`.

- `supercell_size`: (optional, list[int]) Size of the supercell (x, y, z).
  - Default is `[3, 3, 4]`.

- `vacuum`: (optional, float) Vacuum layer in Angstrom.
  - Default is `10.0`.

### Molecular dynamics simulation parameters. - `[md_parameters]`


- `temp`: (optional, float) MD temperature in Kelvin.
  - Default is `300.0`.

- `n_steps`: (optional, int) Number of MD steps.
  - Default is `10000`.

- `timestep`: (optional, float) MD timestep in femtoseconds.
  - Default is `2.0`.

- `friction`: (optional, float) Friction coefficient for Langevin dynamics.
  - Default is `0.005`.

### Selection of benchmarks to run. - `[benchmarks]`


- `run_energy_md`: (optional, bool) Run the energy MD benchmark.
  - Default is `False`.

- `run_accuracy_test_set`: (optional, bool) Run energy and force error benchmark on a test set.
  - Default is `False`.

- `run_elastic_properties`: (optional, bool) Run elastic properties benchmark.
  - Default is `False`.

- `run_defect_formation_energy`: (optional, bool) Run defect formation energy benchmark.
  - Default is `False`.

- `run_surface_energies`: (optional, bool) Run surface energies benchmark.
  - Default is `False`.

- `run_phonon_dispersion`: (optional, bool) Run phonon dispersion benchmark.
  - Default is `False`.

- `run_high_temp_md`: (optional, bool) Run high-temperature MD benchmark.
  - Default is `False`.

- `run_melting_point`: (optional, bool) Run melting point calculation benchmark.
  - Default is `False`.

- `run_gsfe`: (optional, bool) Run Generalized Stacking Fault Energy (GSFE) benchmark.
  - Default is `False`.

- `run_learning_curves`: (optional, bool) Plot learning curves from AL runs.
  - Default is `False`.

- `run_final_db_size`: (optional, bool) Compare final database sizes from AL runs.
  - Default is `False`.

- `run_md_count`: (optional, bool) Count total MD calculations performed during AL loops.
  - Default is `False`.

- `run_evaluate_database`: (optional, bool) Evaluate models against a user-provided structure database.
  - Default is `False`.

- `run_magic_cluster`: (optional, bool) Run magic number cluster benchmark.
  - Default is `False`.

### Test set configuration for accuracy benchmarks. - `[test_set]`

:::{attention}
This section is optional.
:::


- `test_set_path`: (optional, str) Path to the held-out test set for accuracy benchmarks.
  - Example: `'/path/to/test_set.xyz'`.

### Configuration for database evaluation benchmark. - `[database_evaluation]`

:::{attention}
This section is optional.
:::


- `database_path`: (optional, str) Path to the structure database file for evaluation.
  - Example: `'/path/to/database.xyz'`.

### Magic cluster benchmark settings. - `[magic_cluster]`

:::{attention}
This section is optional.
:::


- `magic_cluster_dft_refs`: (optional, str) Path to JSON file with DFT reference energies for magic clusters.
  - Example: `'/path/to/dft_refs.json'`.

- `magic_cluster_sizes`: (optional, list[int]) List of magic number cluster sizes to test.
  - Default is `[13, 19, 55, 147, 309, 561]`.

### Surface energy benchmark specific settings. - `[surface_energy_benchmark]`

:::{attention}
This section is optional.
:::


- `dft_refs`: (optional, str) Path to JSON file with DFT reference energies.
  - Example: `'/path/to/surface_energies.json'`.

- `bulk_structure`: (optional, str) Path to DFT-optimized bulk structure file.
  - Example: `'/path/to/bulk.xyz'`.

- `slab_structures`: (optional, dict) Dictionary of slab structures with surface indices as keys.
  - Example: `{'100': '/path/to/slab_100.xyz', '110': '/path/to/slab_110.xyz'}`.

### Melting point benchmark settings. - `[melting_point_benchmark]`


- `supercell_size`: (optional, list[int]) Supercell size for melting point calculation (x, y, z).
  - Default is `[6, 6, 6]`.

- `solid_temp_K`: (optional, float) Temperature in Kelvin for solid phase.
  - Default is `1100.0`.

- `liquid_temp_K`: (optional, float) Temperature in Kelvin for liquid phase.
  - Default is `1600.0`.

- `nve_initial_T_test_K`: (optional, float) Initial temperature in Kelvin for NVE test.
  - Default is `800.0`.

- `melting_point_supercell_path`: (optional, str) Path to the supercell structure file for melting point calculation.
  - Example: `'/path/to/supercell.xyz'`.
