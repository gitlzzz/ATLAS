## MLIP Benchmarks

The MLIP Benchmarks tool allows you to run a series of benchmarks to evaluate the performance of Machine Learning Interatomic Potentials (MLIPs) trained with MatDBForge.
These benchmarks include:
- Accuracy tests on a given dataset.
- Melting point benchmark via the coexistence method.
- Monovacancy formation energy calculations.
- Surface energy calculations for various crystallographic facets.

Generate a mlip benchmarks template file using `mdb_gen_configuration_file -t mlip_benchmarks`.

:::{attention}
All keys are mandatory unless stated otherwise.
:::


### General - `[general]`

General benchmark settings.


- :alt:`output_dir`: (str) Directory to save benchmark results.
  - Default is `'./mlip_evaluation'`.

- :alt:`metal`: (str) Metal symbol for the benchmark systems.
  - Default is `'Cu'`.

- :alt:`device`: (optional, str) Device to run calculations on.
  - Default is `'cuda'`.
  - Possible values are: `cuda`, `cpu`.

- :alt:`dtype`: (optional, str) Data type for calculations.
  - Default is `'float64'`.
  - Possible values are: `float32`, `float64`.

- :alt:`no_rich_ui`: (optional, bool) Disable Rich UI and use plain text output.
  - Default is `False`.

### Models - `[models]`

Model specifications for benchmarking.


- :alt:`model_files`: (optional, list[str]) List of paths to .model files.
  - Default is `[]`.

- :alt:`aiida_pks`: (optional, list[int]) List of AiiDA workchain PKs/UUIDs to load models from.
  - Default is `[]`.

- :alt:`foundation_models`: (optional, list[str]) List of foundation model specifications. Format: 'library:model_name'.
  - Default is `[]`.

### Slab_Generation - `[slab_generation]`

Settings for slab generation in benchmarks.


- :alt:`surface_indices`: (optional, list[int]) Miller indices for the surface.
  - Default is `[1, 1, 1]`.

- :alt:`supercell_size`: (optional, list[int]) Size of the supercell (x, y, z).
  - Default is `[3, 3, 4]`.

- :alt:`vacuum`: (optional, float) Vacuum layer in Angstrom.
  - Default is `10.0`.

### Md_Parameters - `[md_parameters]`

Molecular dynamics simulation parameters.


- :alt:`temp`: (optional, float) MD temperature in Kelvin.
  - Default is `300.0`.

- :alt:`n_steps`: (optional, int) Number of MD steps.
  - Default is `10000`.

- :alt:`timestep`: (optional, float) MD timestep in femtoseconds.
  - Default is `2.0`.

- :alt:`friction`: (optional, float) Friction coefficient for Langevin dynamics.
  - Default is `0.005`.

### Benchmarks - `[benchmarks]`

Selection of benchmarks to run.


- :alt:`run_energy_md`: (optional, bool) Run the energy MD benchmark.
  - Default is `False`.

- :alt:`run_accuracy_test_set`: (optional, bool) Run energy and force error benchmark on a test set.
  - Default is `False`.

- :alt:`run_elastic_properties`: (optional, bool) Run elastic properties benchmark.
  - Default is `False`.

- :alt:`run_defect_formation_energy`: (optional, bool) Run defect formation energy benchmark.
  - Default is `False`.

- :alt:`run_surface_energies`: (optional, bool) Run surface energies benchmark.
  - Default is `False`.

- :alt:`run_phonon_dispersion`: (optional, bool) Run phonon dispersion benchmark.
  - Default is `False`.

- :alt:`run_high_temp_md`: (optional, bool) Run high-temperature MD benchmark.
  - Default is `False`.

- :alt:`run_melting_point`: (optional, bool) Run melting point calculation benchmark.
  - Default is `False`.

- :alt:`run_gsfe`: (optional, bool) Run Generalized Stacking Fault Energy (GSFE) benchmark.
  - Default is `False`.

- :alt:`run_learning_curves`: (optional, bool) Plot learning curves from AL runs.
  - Default is `False`.

- :alt:`run_final_db_size`: (optional, bool) Compare final database sizes from AL runs.
  - Default is `False`.

- :alt:`run_md_count`: (optional, bool) Count total MD calculations performed during AL loops.
  - Default is `False`.

- :alt:`run_evaluate_database`: (optional, bool) Evaluate models against a user-provided structure database.
  - Default is `False`.

- :alt:`run_magic_cluster`: (optional, bool) Run magic number cluster benchmark.
  - Default is `False`.

### Test_Set - `[test_set]`

Test set configuration for accuracy benchmarks.

:::{attention}
This section is optional.
:::


- :alt:`test_set_path`: (optional, str) Path to the held-out test set for accuracy benchmarks.
  - Example: `'/path/to/test_set.xyz'`.

### Database_Evaluation - `[database_evaluation]`

Configuration for database evaluation benchmark.

:::{attention}
This section is optional.
:::


- :alt:`database_path`: (optional, str) Path to the structure database file for evaluation.
  - Example: `'/path/to/database.xyz'`.

### Magic_Cluster - `[magic_cluster]`

Magic cluster benchmark settings.

:::{attention}
This section is optional.
:::


- :alt:`magic_cluster_dft_refs`: (optional, str) Path to JSON file with DFT reference energies for magic clusters.
  - Example: `'/path/to/dft_refs.json'`.

- :alt:`magic_cluster_sizes`: (optional, list[int]) List of magic number cluster sizes to test.
  - Default is `[13, 19, 55, 147, 309, 561]`.

### Surface_Energy_Benchmark - `[surface_energy_benchmark]`

Surface energy benchmark specific settings.

:::{attention}
This section is optional.
:::


- :alt:`dft_refs`: (optional, str) Path to JSON file with DFT reference energies.
  - Example: `'/path/to/surface_energies.json'`.

- :alt:`bulk_structure`: (optional, str) Path to DFT-optimized bulk structure file.
  - Example: `'/path/to/bulk.xyz'`.

- :alt:`slab_structures`: (optional, dict) Dictionary of slab structures with surface indices as keys.
  - Example:

```python
{'100': '/path/to/slab_100.xyz', '110': '/path/to/slab_110.xyz'}
```

### Melting_Point_Benchmark - `[melting_point_benchmark]`

Melting point benchmark settings.


- :alt:`supercell_size`: (optional, list[int]) Supercell size for melting point calculation (x, y, z).
  - Default is `[6, 6, 6]`.

- :alt:`solid_temp_K`: (optional, float) Temperature in Kelvin for solid phase.
  - Default is `1100.0`.

- :alt:`liquid_temp_K`: (optional, float) Temperature in Kelvin for liquid phase.
  - Default is `1600.0`.

- :alt:`nve_initial_T_test_K`: (optional, float) Initial temperature in Kelvin for NVE test.
  - Default is `800.0`.

- :alt:`melting_point_supercell_path`: (optional, str) Path to the supercell structure file for melting point calculation.
  - Example: `'/path/to/supercell.xyz'`.
