
```{role} alt
:class: code-alt
```
```{role} codeheader
:class: code-header
```

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


- {alt}`output_dir`:
  - **Description**: Directory to save benchmark results.
  - **Type**: `(str)`
  - **Default**: `'./mlip_evaluation'`.

- {alt}`metal`:
  - **Description**: Metal symbol for the benchmark systems.
  - **Type**: `(str)`
  - **Default**: `'Cu'`.

- {alt}`device`:
  - **Description**: Device to run calculations on.
  - **Type**: `(optional, str)`
  - **Default**: `'cuda'`.
  - Possible values are: `cuda`, `cpu`.

- {alt}`dtype`:
  - **Description**: Data type for calculations.
  - **Type**: `(optional, str)`
  - **Default**: `'float64'`.
  - Possible values are: `float32`, `float64`.

- {alt}`no_rich_ui`:
  - **Description**: Disable Rich UI and use plain text output.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

### Models - `[models]`

Model specifications for benchmarking.


- {alt}`model_files`:
  - **Description**: List of paths to .model files.
  - **Type**: `(optional, list[str])`
  - **Default**: `[]`.

- {alt}`aiida_pks`:
  - **Description**: List of AiiDA workchain PKs/UUIDs to load models from.
  - **Type**: `(optional, list[int])`
  - **Default**: `[]`.

- {alt}`foundation_models`:
  - **Description**: List of foundation model specifications. Format: 'library:model_name'.
  - **Type**: `(optional, list[str])`
  - **Default**: `[]`.

### Slab_Generation - `[slab_generation]`

Settings for slab generation in benchmarks.


- {alt}`surface_indices`:
  - **Description**: Miller indices for the surface.
  - **Type**: `(optional, list[int])`
  - **Default**: `[1, 1, 1]`.

- {alt}`supercell_size`:
  - **Description**: Size of the supercell (x, y, z).
  - **Type**: `(optional, list[int])`
  - **Default**: `[3, 3, 4]`.

- {alt}`vacuum`:
  - **Description**: Vacuum layer in Angstrom.
  - **Type**: `(optional, float)`
  - **Default**: `10.0`.

### Md_Parameters - `[md_parameters]`

Molecular dynamics simulation parameters.


- {alt}`temp`:
  - **Description**: MD temperature in Kelvin.
  - **Type**: `(optional, float)`
  - **Default**: `300.0`.

- {alt}`n_steps`:
  - **Description**: Number of MD steps.
  - **Type**: `(optional, int)`
  - **Default**: `10000`.

- {alt}`timestep`:
  - **Description**: MD timestep in femtoseconds.
  - **Type**: `(optional, float)`
  - **Default**: `2.0`.

- {alt}`friction`:
  - **Description**: Friction coefficient for Langevin dynamics.
  - **Type**: `(optional, float)`
  - **Default**: `0.005`.

### Benchmarks - `[benchmarks]`

Selection of benchmarks to run.


- {alt}`run_energy_md`:
  - **Description**: Run the energy MD benchmark.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`run_accuracy_test_set`:
  - **Description**: Run energy and force error benchmark on a test set.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`run_elastic_properties`:
  - **Description**: Run elastic properties benchmark.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`run_defect_formation_energy`:
  - **Description**: Run defect formation energy benchmark.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`run_surface_energies`:
  - **Description**: Run surface energies benchmark.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`run_phonon_dispersion`:
  - **Description**: Run phonon dispersion benchmark.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`run_high_temp_md`:
  - **Description**: Run high-temperature MD benchmark.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`run_melting_point`:
  - **Description**: Run melting point calculation benchmark.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`run_gsfe`:
  - **Description**: Run Generalized Stacking Fault Energy (GSFE) benchmark.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`run_learning_curves`:
  - **Description**: Plot learning curves from AL runs.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`run_final_db_size`:
  - **Description**: Compare final database sizes from AL runs.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`run_md_count`:
  - **Description**: Count total MD calculations performed during AL loops.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`run_evaluate_database`:
  - **Description**: Evaluate models against a user-provided structure database.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

- {alt}`run_magic_cluster`:
  - **Description**: Run magic number cluster benchmark.
  - **Type**: `(optional, bool)`
  - **Default**: `False`.

### Test_Set - `[test_set]`

Test set configuration for accuracy benchmarks.

:::{attention}
This section is optional.
:::


- {alt}`test_set_path`:
  - **Description**: Path to the held-out test set for accuracy benchmarks.
  - **Type**: `(optional, str)`
  - **Example**: `'/path/to/test_set.xyz'`.

### Database_Evaluation - `[database_evaluation]`

Configuration for database evaluation benchmark.

:::{attention}
This section is optional.
:::


- {alt}`database_path`:
  - **Description**: Path to the structure database file for evaluation.
  - **Type**: `(optional, str)`
  - **Example**: `'/path/to/database.xyz'`.

### Magic_Cluster - `[magic_cluster]`

Magic cluster benchmark settings.

:::{attention}
This section is optional.
:::


- {alt}`magic_cluster_dft_refs`:
  - **Description**: Path to JSON file with DFT reference energies for magic clusters.
  - **Type**: `(optional, str)`
  - **Example**: `'/path/to/dft_refs.json'`.

- {alt}`magic_cluster_sizes`:
  - **Description**: List of magic number cluster sizes to test.
  - **Type**: `(optional, list[int])`
  - **Default**: `[13, 19, 55, 147, 309, 561]`.

### Surface_Energy_Benchmark - `[surface_energy_benchmark]`

Surface energy benchmark specific settings.

:::{attention}
This section is optional.
:::


- {alt}`dft_refs`:
  - **Description**: Path to JSON file with DFT reference energies.
  - **Type**: `(optional, str)`
  - **Example**: `'/path/to/surface_energies.json'`.

- {alt}`bulk_structure`:
  - **Description**: Path to DFT-optimized bulk structure file.
  - **Type**: `(optional, str)`
  - **Example**: `'/path/to/bulk.xyz'`.

- {alt}`slab_structures`:
  - **Description**: Dictionary of slab structures with surface indices as keys.
  - **Type**: `(optional, dict)`
  - **Example**:

```python
{'100': '/path/to/slab_100.xyz', '110': '/path/to/slab_110.xyz'}
```

### Melting_Point_Benchmark - `[melting_point_benchmark]`

Melting point benchmark settings.


- {alt}`supercell_size`:
  - **Description**: Supercell size for melting point calculation (x, y, z).
  - **Type**: `(optional, list[int])`
  - **Default**: `[6, 6, 6]`.

- {alt}`solid_temp_K`:
  - **Description**: Temperature in Kelvin for solid phase.
  - **Type**: `(optional, float)`
  - **Default**: `1100.0`.

- {alt}`liquid_temp_K`:
  - **Description**: Temperature in Kelvin for liquid phase.
  - **Type**: `(optional, float)`
  - **Default**: `1600.0`.

- {alt}`nve_initial_T_test_K`:
  - **Description**: Initial temperature in Kelvin for NVE test.
  - **Type**: `(optional, float)`
  - **Default**: `800.0`.

- {alt}`melting_point_supercell_path`:
  - **Description**: Path to the supercell structure file for melting point calculation.
  - **Type**: `(optional, str)`
  - **Example**: `'/path/to/supercell.xyz'`.
