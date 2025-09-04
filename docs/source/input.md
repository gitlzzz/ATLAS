# Input specification

The input format of MDB is [TOML](https://toml.io/en/). The syntax from TOML is unchanged. The available parameters are different depending on the selected tool.

Users are advised to use the `mdb_gen_configuration_file` utility to generate a template file which can be customized. However, the configuration files can be created from scratch using the sections from below and the appropiate TOML syntax as in the following example:

```toml
[database]
database_name = 'test'
...


[database.plot_db]
show = true

...

[generation]
generate_type = ['bulk', 'surface', 'cluster']

```

Please, check the tool's corresponding section to learn more about all the available options.

## Database Generation

Generate a database generation template file using `mdb_gen_configuration_file -t database_generation`.

:::{attention}
All keys are mandatory unless stated otherwise.
:::


### General settings and file paths for the database. - `[database]`


- `database_name`: (str) Name of the database to be used for internal reference and as the filename. Example: 'my_material_db'.

- `min_num_atoms`: (int) Minimum number of atoms allowed in the generated structures. Default is 64.

- `max_num_atoms`: (int) Maximum number of atoms allowed in the generated structures. Default is 128.

- `min_cell_size`: (float) Minimum cell size in Angstrom. Default is 5.0.

- `relax_struct_path`: (optional, str) Path to a folder containing DFT optimized structures. Default is ''.

- `database_path`: (str, PosixPath) Path where the final database will be saved. Default is ''.

- `rng_seed`: (optional, int) Numerical value used to fix the RNG seed. If not specified, it will be chosen randomly each run. Example: 42.

- `overwrite_db`: (optional, bool) Allow database overwrite. If false, and the database exists, the new database name will include a timestamp. Default is False.

#### Settings for the composition of the database. - `[database.composition]`


- `size`: (int) Maximum number of structures to generate for the database. Default is 7500.

##### Fraction of different structure types. The sum of the fractions must be equal to 1.0. - `[database.composition.ratios]`


- `bulk`: (float) Fraction of structures that will be bulk. Default is 0.4.

- `surface`: (float) Fraction of structures that will be surfaces. Default is 0.6.

- `cluster`: (optional, float) Fraction of structures that will be clusters. Default is 0.0.

#### Display and Export Options for the phase diagram plot. - `[database.plot_db]`


- `show`: (optional, bool) Whether to display the database with a phase diagram after creation. Default is True.

- `format`: (optional, str) Format for the figure. Default is 'png'.

##### Matplotlib rcParams for the plot. - `[database.plot_db.rc_params]`


- `font.family`: (str) Font family for the phase diagram plot. Default is 'monospace'.

- `font.size`: (int) Font size for the phase diagram plot. Default is 14.

#### ASE GUI display options. - `[database.show_db_ase]`

:::{attention}
This section is optional.
:::


- `show`: (optional, bool) Whether to display the database using ASE GUI after creation. Default is False.

#### Export options for the database. - `[database.export]`


- `export`: (bool) Whether to export the database. Default is True.

- `format`: (str) Export format supported by ASE (e.g., 'extxyz'). Default is 'extxyz'.

- `file_path`: (str, PosixPath) Path where the exported file will be saved. Default is ''.

- `file_name`: (str) Name of the exported file. Default is 'export_db_filename'.

### Description of the phase diagram. - `[phase_diagram]`


- `material_name`: (str) Internal name for the material in the phase diagram. Default is 'default_material_name'.

- `element_list`: (list[str]) List of elements to include in the phase diagram. Example: ['Cu', 'O'].

- `base_element`: (str) Symbol of the most abundant element in the phase. Example: 'Cu'.

#### Defines a specific phase within the phase diagram. Multiple phases can be added. - `[phase_diagram.phase.XXXXX]`

This key describes settings for dynamic entries. Several entries can be added by using different key names.

The key name (`XXXXX`) is used as the reference name. **Replace XXXXX with a name of your choice.**

Example parameters for each entry:


- `name`: (str) Name to be used as reference for the phase. Example: 'alpha'.

- `cluster_element`: (optional, str) Symbol of the element defining the cluster.

- `prototype`: (str) Materials Project ID of the prototypical structure. Example: 'mp-30'.

- `offset`: (float) Fraction of composition allowed over and under the phase limits. Default is 0.1.

- `limit_max_num_structures`: (optional, int) Maximum number of structures to generate for this phase. Default is 100.

- `allow_modifications`: (optional, bool) Allow modifications (supercells, replacements, etc.) to the base structure. Default is True.

- `use_cache`: (optional, bool) Store structures in cache to speed up generation. Can consume a lot of disk space. Default is False.

Parameters using `composition.` prefix:

- `composition.min`: (float) Minimum composition as a fraction of the current phase element. Example: 0.1.
- `composition.max`: (float) Maximum composition as a fraction of the current phase element. Example: 0.25.

Parameters using `replacements.` prefix:

- `replacements.replace`: (optional, bool) Whether to replace specific elements. Elements in element_list will be considered for replacement and replaced by a single element species. Default is False.
- `replacements.element_list`: (optional, list[str]) List of elements to be replaced. Example: ['Ti'].
- `replacements.replace_with`: (optional, str) Element to replace with. Example: 'Ir'.

### Structure generation settings. - `[generation]`


- `generate_type`: (list[str]) Types of structures to generate. Default is ['bulk', 'surface', 'cluster'].

#### Bulk structure generation settings. - `[generation.bulk]`


- `num_struct`: (int) Number of structures to generate. Default is 25.

- `num_repeat`: (int) Number of repeats for each structure. Default is 5.

- `supercell_max_idx`: (int) Maximum Miller index for the bulk supercells. Default is 2.

#### Surface structure generation settings. - `[generation.surface]`


- `min_miller_index`: (int) Minimum Miller index used to generate surface structures. Default is 1.

- `max_miller_index`: (int) Maximum Miller index used to generate surface structures. Default is 3.

- `min_slab_size_ang`: (optional, float) Minimum slab thickness in Angstrom. Default is 7.0.

- `min_vacuum_size_ang`: (float) Minimum size of the vacuum layer in Angstroms. Default is 12.0.

- `get_supercells`: (bool) Whether to generate supercells for surface structures. Default is True.

- `fixed_layers`: (int) Number of fixed layers in the surface slab. Default is 3.

- `max_number_supercells`: (int) Maximum number of surface supercells to generate. Default is 200.

- `save_in_db`: (bool) Whether to save generated surfaces in the database. Default is True.

- `num_replacements`: (int) Number of replacement percentages to generate for each structure. Default is 20.

- `num_repeat_replace`: (int) Number of repeats for each replacement. Default is 2.

- `frac_slabs_save`: (optional, float) Fraction of slabs to save after generation. Default is 0.1.

- `frac_supercells_save`: (optional, float) Fraction of unreplaced supercells to save after generation. Default is 0.1.

- `max_slab_num`: (int) Maximum number of slabs to gather from the slab generation. Default is 15.

- `n_workers`: (optional, int) Maximum number of workers for parallel processing.

### Lattice deformation settings. - `[deformation]`

:::{attention}
This section is optional.
:::


- `lattice_frac_deform_max`: (float) Maximum deformation value as a percentage of the lattice side length. Default is 0.05.

- `lattice_frac_deform_min`: (float) Minimum deformation value as a percentage of the lattice side length. Default is 0.01.

- `num_repeats`: (int) Number of repeats for each structure with random deformations. Default is 5.

- `limit_max_num_deformations`: (int) Maximum number of lattice deformations to generate. Default is 100.

### Perturbation settings. - `[perturbation]`

:::{attention}
This section is optional.
:::


- `filter_struct_types`: (list[str]) Types of structures to which the perturbation will be applied. Default is ['bulk', 'surface'].

- `limit_max_num_perturbs`: (int) Maximum number of perturbations to generate. Default is 100.

- `num_repeats`: (int) Number of repeats for each structure with random perturbations. Default is 1.

- `perturbation_ang`: (optional, float) Perturbation magnitude in Angstrom. Default is 0.04.

### Adsorbate placement settings. - `[adsorbates]`

:::{attention}
This section is optional.
:::


- `filter_struct_types`: (optional, list[str]) Types of structures to which adsorbates will be added. Default is ['surface'].

- `limit_max_num_perturbs`: (optional, int) Maximum number of structures with adsorbates to generate. Default is 100.

- `num_repeats`: (int) Number of repeats for each structure. Default is 1.

- `adsorbate_species`: (list[str]) List of adsorbate species to consider. Example: ['H', 'H2O'].

### Settings for filtering out incorrect structures. - `[struct_filters]`


#### Filter for structures with atoms that have no neighbors. - `[struct_filters.no_neighbors]`


- `cov_rad_multiplier`: (optional, float) Multiplier applied to the covalent radii to be used as cutoff radius for the neighbor check. Default is 1.2.

#### Filter for layer distances in surface slabs. - `[struct_filters.layer_distance]`

:::{attention}
This section is optional.
:::


- `max_layer_distance_ang`: (optional, float) Maximum accepted distance between layers in Angstrom. Default is 4.0.

#### Filter for duplicate slabs. - `[struct_filters.duplicate_slabs]`

:::{attention}
This section is optional.
:::


- `tolerance`: (optional, float) Tolerance for the duplicate slabs filter. Default is 0.2.

### Settings for vacancy generation. - `[vacancies]`


- `filter_struct_types`: (list[str]) Types of structures to which vacancies will be applied. Default is ['bulk', 'surface'].

- `limit_max_num_vacancies`: (optional, int) Maximum number of structures with vacancies to generate. Default is 400.

- `num_repeats`: (int) Number of repeats for each structure with different random vacancies. Default is 3.

- `max_vacancy_percentage`: (float) Maximum vacancies to generate as a percentage of the total number of atoms. Default is 0.75.

- `min_vacancy_percentage`: (float) Minimum vacancies to generate as a percentage of the total number of atoms. Default is 0.025.

- `element_list`: (list[str]) List of elements to consider for the vacancies. Example: ['O'].

### Settings for targeted structural modifications. - `[targeted_modification]`

:::{attention}
This section is optional.
:::


#### Apply perturbations to the central atom in octahedral sites. - `[targeted_modification.central_atom_octahedral]`

:::{attention}
This section is optional.
:::


- `filter_phases`: (optional, list[str]) Only apply the modification to the following phases. Example: ['rutile', 'original_IrO2'].

- `filter_struct_types`: (optional, list[str]) Types of structures to which the modification will be applied. Default is ['bulk', 'surface'].

- `central_element`: (optional, str) Symbol of the central element of the octahedral site. Example: 'Ir'.

- `num_repeats`: (optional, int) Number of repeats for each structure with different perturbations. Default is 3.

- `limit_max_num_modifications`: (optional, int) Maximum number of modified structures to generate. Default is 200.

- `max_perturbation_ang`: (optional, float) Maximum perturbation movement of the central atom in Angstrom. Default is 0.2.

### Settings for descriptors and concave hull generation. - `[concave_hull]`

:::{attention}
This section is optional.
:::


- `gen_concave_hull`: (optional, bool) Whether to generate the concave hull of the descriptors for all structures in the database. Default is False.

- `descriptor`: (optional, str) Descriptor to use for the concave hull generation. Default is 'SOAP'.

- `dim_reduction`: (optional, str) Dimensionality reduction method for the concave hull generation. Default is 'autoencoder'.

- `plot_filename`: (optional, str) Filename for the figure displaying the concave hull. Default is 'descriptors_concave_hull.png'.

## DFT Calculations

Generate a dft template file using `mdb_gen_configuration_file -t dft`.

:::{attention}
All keys are mandatory unless stated otherwise.
:::


### General settings for the DFT script. - `[general]`


- `log_path`: (str, PosixPath) Path where the logs will be stored. Default is '/tmp/'.

- `result_file_path`: (str, PosixPath) Path for the results file (extxyz format). Default is 'dft_calculation_results'.

- `source_db`: (optional, str, PosixPath) Path to the source database file (.extxyz or mdb .xz format).

- `aiida_group_name`: (str) Name of the AiiDA group for the calculations. Example: 'my_dft_run'.

- `max_batch`: (int) Maximum number of structures to process in one batch. Default is 100.

- `queue_check_interval_seconds`: (int) Interval in seconds to check the queue for submitting new calculations. Default is 240.

- `start_on_struct_idx`: (int) Number of structures to skip before starting the calculations. Default is 0.

- `dry_run`: (optional, bool) If True, a dry-run is performed and no calculations are submitted. Default is False.

- `selected_structure_type`: (optional, str) If specified, only structures of this type will be processed.

### DFT calculation settings. - `[calculation]`


- `calc_type`: (str) Type of calculation. Default is 'static'.

- `aiida_potential_family`: (str) AiiDA potential family name. Example: 'vasp-5.4-PBE-2023'.

- `potential_mapping`: (optional, dict) Mapping of elements to specific potential labels. Example: {'Si': 'Si_GW'}.

### K-point settings. - `[kpoints]`


- `kspacing`: (dict) K-spacing in Å⁻¹ for different phases or a single value for all structures. Example: {'alpha': 0.125, 'default': 0.15}.

### Queue settings for HPC schedulers (e.g., SLURM). - `[queue]`


- `code_string`: (str) Name of the code as defined in AiiDA. Example: 'vasp@my_cluster'.

- `account`: (optional, str) Account to be used for the calculations.

- `qos`: (optional, str) Quality of service parameter.

- `node_cpus`: (optional, int) Number of CPUs per node.

- `max_wallclock_seconds`: (int) Maximum wallclock time in seconds. Default is 16200.

- `max_memory_kb`: (optional, int) Maximum memory per node in KB.

- `multiple`: (optional, int) Whether to use multiple nodes.

- `custom_scheduler_commands`: (optional, str) Custom scheduler commands.

#### Scheduler resource options. - `[queue.options_resources]`


- `tot_num_mpiprocs`: (int) Total number of MPI processes. Default is 24.

### Settings for the AiiDA-VASP plugin. - `[aiida_vasp]`

:::{attention}
This section is optional.
:::


- `critical_notifications`: (optional, dict) Errors and warnings to be treated as critical (general).

#### Parser settings for aiida-vasp. - `[aiida_vasp.parser_settings]`


- `add_kpoints`: (optional, bool) Whether to add k-points information to the parsed results. Default is True.

##### Critical error and warning notifications to be treated as important. - `[aiida_vasp.parser_settings.critical_notifications]`


- `add_edddav_zhegv`: (optional, bool) Add EDDDAV ZHEGV error notification. Default is True.

- `add_eddrmm_zhegv`: (optional, bool) Add EDDRMM ZHEGV error notification. Default is True.

- `add_not_hermitian`: (optional, bool) Add not hermitian error notification. Default is True.

- `add_brmix`: (optional, bool) Add BRMIX error notification. Default is True.

- `add_bandocc`: (optional, bool) Add band occupation error notification. Default is False.

### INCAR settings for different structure types. - `[incar]`


- `bulk`: (optional, dict) INCAR settings for bulk structures.

- `surface`: (optional, dict) INCAR settings for surface structures.

- `cluster`: (optional, dict) INCAR settings for cluster structures.

## Active Learning Loop

Generate a active learning template file using `mdb_gen_configuration_file -t active_learning`.

:::{attention}
All keys are mandatory unless stated otherwise.
:::


### General active learning settings. - `[active_learning]`


- `aiida_profile`: (str) Name of the AiiDA profile to be used.

- `run_name`: (str) Internal name for the run.

- `init_db_path`: (str, PosixPath) Path to the folder containing the initial database.

- `results_dir`: (str, PosixPath) Path for final results. A folder named run_{uuid} will be created inside.

- `log_path`: (optional, str, PosixPath) Path for the log file. Defaults to results_dir if not specified.

- `final_db_name`: (str) Name for the final database (extxyz format). Default is 'final_data_test'.

- `max_iterations`: (int) Maximum number of AL loop iterations. Default is 3.

- `model_acc_multiplier`: (float) Multiplier for model accuracy threshold. Higher values mean more DFT calculations. Default is 10.0.

- `load_init_models`: (optional, list[int]) Load initial models from a list of AiiDA UUIDs/PKs. Default is [].

- `al_mode`: (optional, str) Active learning mode. Default is 'data_acquisition'.

- `dft_method`: (optional, str) Selection of DFT calculator method. Default is 'vasp'.

### Settings for the data reduction AL mode. - `[data_reduction]`


- `large_database_path`: (str, PosixPath) Path to the large database file from which to select structures.

- `initial_selection_size`: (int) Number of structures to select from the large database for initial training. Default is 100.

- `initial_selection_method`: (str) Selection method for initial structures. Default is 'lowest_energy'.

- `structures_per_iteration`: (int) Number of structures to select per iteration from the large database. Default is 50.

- `iterative_selection_method`: (str) Selection method for iterative structures. Default is 'uncertainty'.

### Settings for the AL seed generation for MD. - `[al_seed]`


- `seed_size_frac`: (float) Sets total structures in an MD seed as a fraction of the training db size. Default is 0.01.

- `seed_min_num_structs`: (optional, int) Minimum number of structures in an MD seed. Default is 25.

- `seed_max_num_structs`: (int) Maximum number of structures in the MD seed. Default is 500.

- `delete_seed_structs`: (optional, bool) Whether to delete structures from the seed database even if they are in domain. Default is True.

#### MD seed selection mode settings. - `[al_seed.seed_select_settings]`


- `seed_select_type`: (str) MD seed selection mode. Default is 'random'.

- `small_first_max_size`: (int) Maximum size in number of atoms for structures selected with small_first mode. Default is 50.

- `small_first_max_iter`: (int) Apply small_first mode for the first n iterations. Default is 5.

#### Settings for the seed ranking methods. - `[al_seed.seed_ranking_settings]`

:::{attention}
This section is optional.
:::


- `seed_ranking_algorithm`: (optional, str) Algorithm used for seed selection. Default is 'random'.

##### Uses Farthest Point Sampling (FPS) from the descriptors of an initially selected structure. - `[al_seed.seed_ranking_settings.descriptor_fps]`


- `descriptor_type`: (optional, str) What descriptors to use. Default is 'soap'.

- `initial_structure`: (optional, str) Whether to gather a structure at random or select the one with the lowest energy available. Default is 'random'.

###### Entry containing settings that depend on the descriptor type selected. - `[al_seed.seed_ranking_settings.descriptor_fps.descriptor]`

:::{attention}
This section is optional.
:::


- `r_cut`: (optional, float) Cutoff radius for SOAP descriptor. Default is 6.0.

- `n_max`: (optional, int) Maximum number of radial basis functions for SOAP. Default is 8.

- `l_max`: (optional, int) Maximum degree of spherical harmonics for SOAP. Default is 6.

- `periodic`: (optional, bool) Whether to consider the system as periodic. Default is True.

- `average`: (optional, str) Averaging mode for SOAP descriptor. Default is 'off'.

- `model_path`: (optional, str) Path to the trained MACE model.

- `device`: (optional, str) What device to use for MACE. Default is 'cpu'.

- `dtype`: (optional, str) Floating point number precision for MACE. Default is 'float32'.

### Settings for extrapolation checks. - `[extrapolation]`


- `disagreement_check_type`: (optional, str) Approach for energy and force (E&F) committee disagreement check. Default is 'training'.

- `check_extrapolation_type`: (optional, str) Method for extrapolation check. Default is 'none'.

#### Settings for the concave hull extrapolation check. - `[extrapolation.concave_hull]`

:::{attention}
This section is optional.
:::


- `target_alpha_range_min`: (optional, float) No description available. Default is 3.0.

- `target_alpha_range_max`: (optional, float) No description available. Default is 8.0.

### Settings for MD simulations. - `[md]`


- `ignore_container`: (optional, bool) Whether to ignore the container specified in the container settings for the MD calculations. Default is False.

#### MD simulation parameters. - `[md.parameters]`


- `temperature_list_K`: (list[float]) List of different temperatures (in K) for MD simulations. Default is [300.0, 500.0, 900.0].

- `max_temp_multiplier`: (float) Multiplier for MD temperature to determine the upper bound of the temperature. Default is 1.3.

- `num_steps`: (int) Total number of timesteps for each MD simulation. Default is 1000.

- `timestep_duration_ps`: (float) Duration of each timestep in picoseconds. Default is 0.003.

- `langevin_friction_ps-1`: (float) Friction coefficient for the Langevin thermostat in ps⁻¹. Default is 10.0.

- `gather_traj_cnt_lattice`: (bool) Consider constant lattice when gathering trajectories. Default is True.

- `use_kokkos`: (bool) Whether to use Kokkos to run the MD. Default is True.

- `device`: (str) Device for the MACE model in MD simulations. Default is 'cuda'.

- `default_dtype`: (str) Default data type for the MACE model in MD simulations. Default is 'float32'.

- `al_keep_struct_every_n_ps`: (float) Keep a structure every N picoseconds of MD simulation. Default is 0.5.

- `log_save_interval`: (optional, int) Log energy and force information every N MD steps. Default is 1.

- `max_energy_threshold_per_atom`: (optional, float) Maximum energy threshold per atom in eV. Default is 1000.0.

#### Settings for MD trajectory filters. - `[md.filters]`


- `save_filtered_structures`: (optional, bool) Whether to save filtered structures. Default is False.

##### Filter for structures with atoms that have no neighbors. - `[md.filters.check_atoms_no_neighbor]`


- `enable`: (bool) No description available. Default is True.

- `covalent_radius_multiplier`: (float) Multiplier for covalent radii to define cutoff for neighbor check. Default is 1.05.

##### Filter for layer distances in surface slabs. - `[md.filters.layer_distance]`


- `enable`: (bool) No description available. Default is True.

- `max_layer_distance_ang`: (float) Maximum accepted distance between layers in Angstrom. Default is 3.5.

##### Filter for exploding structures based on covalent radius limits. - `[md.filters.exploding_structures]`


- `enable`: (optional, bool) Whether to enable the exploding structures filter. Default is True.

- `cov_rad_multiplier_max`: (optional, float) Maximum multiplier for covalent radius threshold. Default is 10.0.

- `cov_rad_multiplier_min`: (optional, float) Minimum multiplier for covalent radius threshold. Default is 1.5.

#### AiiDA metadata and scheduler options for MD simulations. - `[md.metadata]`


- `code`: (str) AiiDA code name for MD software. Example: 'mace_lammps@cluster'.

- `computer`: (str) AiiDA computer name for MD calculations. Example: 'my_cluster'.

- `options`: (optional, dict) AiiDA scheduler options for MD calculations.

### Settings for containerized code execution. - `[code]`


#### Container settings for code execution. - `[code.container]`


- `use_container`: (optional, bool) Whether to use a containerized version of the code. Default is False.

- `image_name`: (optional, str) Path to the container image on calculation nodes. Example: '/path/to/container.sif'.

- `engine_command`: (optional, str) Command template for the container engine. Example: 'singularity exec --bind .:/mdb_data --nv --contain --writable-tmpfs {image_name}'.

- `prepend_text`: (optional, str) Text to prepend to job scripts for container setup. Example: 'module load singularity
export PATH=$PATH:.'.

### Settings for MACE model training. - `[mace_train]`


- `result_force_weight`: (optional, float) Weight of the force when considering model performance in weighted sum calculation. Default is 0.1.

- `test_fraction`: (optional, float) Fraction of the training data to be used for testing. Default is 0.1.

- `code`: (str) AiiDA code name for MACE training. Example: 'mace_train@cluster'.

- `computer`: (str) AiiDA computer name for MACE training. Example: 'my_cluster'.

- `ignore_container`: (optional, bool) Whether to ignore container settings for MACE training. Default is False.

- `multihead_finetuning`: (optional, bool) Whether to use multihead finetuning. Default is False.

- `metadata`: (optional, dict) AiiDA metadata and scheduler options for MACE training.

- `train_settings`: (optional, dict) MACE training parameters and hyperparameters.

### Settings for committee evaluation using multiple MACE models. - `[committee_eval]`


- `committee_num_models`: (optional, int) Total number of MACE models in the committee. Default is 4.

- `openmp_threads`: (optional, int) Number of OpenMP threads for MACE CPU evaluation. Default is 24.

- `ignore_container`: (optional, bool) Whether to ignore container settings for committee evaluation. Default is False.

- `prepend_text`: (optional, str) Text to prepend to job scripts for committee evaluation.

- `metadata`: (optional, dict) AiiDA metadata and scheduler options for committee evaluation.

#### Settings for MACE evaluator. - `[committee_eval.mace]`


- `device`: (optional, str) Device for MACE evaluation. Default is 'cpu'.

- `default_dtype`: (optional, str) Default data type for MACE evaluation. Default is 'float32'.

- `batch_size`: (optional, int) Batch size for MACE evaluation. Default is 32.

- `compute_stress`: (optional, bool) Whether to compute stress during evaluation. Default is False.

### Settings for descriptor computation and dimensionality reduction. - `[descriptors]`


- `dimensionality_reduction_method`: (optional, str) Dimensionality reduction method for MACE descriptors. Default is 'none'.

- `ignore_container`: (optional, bool) Whether to ignore container settings for descriptor computation. Default is False.

- `metadata`: (optional, dict) AiiDA metadata and scheduler options for descriptor computation.

#### Settings for autoencoder-based dimensionality reduction. - `[descriptors.autoencoder]`


##### Training settings for the autoencoder. - `[descriptors.autoencoder.train_settings]`


- `device`: (optional, str) Device for autoencoder training. Default is 'cuda'.

- `model_path`: (optional, str) Path to save the autoencoder model. Default is 'autoencoder_model.pth'.

- `load_model`: (optional, bool) Whether to load the model from the model path. Default is False.

- `dataset`: (optional, str) Path to the training dataset. Default is 'all_descriptors.npz'.

- `l1_hidden_dim`: (optional, int) Number of units in the first hidden layer. Default is 256.

- `l2_hidden_dim`: (optional, int) Number of units in the second hidden layer. Default is 32.

- `bottleneck_dim`: (optional, int) Dimensionality of the bottleneck (latent space). Default is 2.

- `bias_flag`: (optional, bool) Flag to include bias terms in the layers. Default is True.

- `num_epochs`: (optional, int) Number of epochs to train the model. Default is 50.

- `batch_size`: (optional, int) Batch size for training. Default is 2048.

- `patience`: (optional, int) Patience for early stopping. Default is 5.

- `lr`: (optional, float) Learning rate for the optimizer. Default is 0.001.

- `weight_decay`: (optional, float) L2 regularization parameter. Default is '1e-5'.

- `loss`: (optional, str) Loss function type. Default is 'mse'.

- `train_frac`: (optional, float) Fraction of the data to use for training. Default is 0.8.

- `valid_frac`: (optional, float) Fraction of the data to use for validation. Default is 0.1.

- `test_frac`: (optional, float) Fraction of the data to use for testing. Default is 0.1.

- `wandb`: (optional, bool) Whether to log metrics to wandb. Default is False.

- `wandb_name`: (optional, str) Name of the wandb run. Default is ''.

- `wandb_project`: (optional, str) Name of the wandb project. Default is ''.

### DFT settings specific to active learning (different from top-level dft section). - `[dft]`


- `ignore_container`: (optional, bool) Whether to ignore container settings for DFT calculations. Default is False.

- `dft_method`: (optional, str) Selection of DFT calculator. Default is 'mace'.

#### MACE settings as DFT calculator. - `[dft.mace]`


- `mace_potential_path`: (str) Path to MACE potential file. Example: 'model.model'.

- `metadata`: (optional, dict) AiiDA metadata for MACE calculations.

- `options`: (optional, dict) AiiDA scheduler options for MACE calculations.

##### Options for MACE that will be passed as arguments during execution. - `[dft.mace.settings]`


- `device`: (optional, str) Device for MACE calculations. Default is 'cuda'.

- `default_dtype`: (optional, str) Default data type for MACE calculations. Default is 'float64'.

- `batch_size`: (optional, int) Batch size for MACE calculations. Default is 11.

- `compute_stress`: (optional, bool) Whether to compute stress. Default is False.

#### VASP settings as DFT calculator. - `[dft.vasp]`

:::{attention}
This section is optional.
:::


- `calc_type`: (optional, str) Type of calculation. Default is 'static'.

- `dft_calc_limit`: (optional, int) Maximum number of DFT calculations to perform per AL step. Default is 200.

- `potential_family`: (str) VASP potential family name. Example: 'vasp-5.4-PBE-2024'.

- `structure_types`: (optional, list[str]) List of structure types to process. Default is ['bulk', 'surface', 'cluster'].

- `kspacing`: (optional, dict) K-spacing settings for different phases or default value. Example: {'MDB_DEFAULT': 0.15, 'alpha': 0.135}.

- `incar`: (optional, dict) INCAR settings for VASP calculations. Example: {'istart': 0, 'icharg': 2, 'gga': 'Pe', 'encut': 450}.

##### Queue settings for VASP calculations. - `[dft.vasp.queue]`


- `queue_type`: (str) Scheduler type. Default is 'slurm'.

- `computer`: (str) AiiDA computer name for VASP calculations. Example: 'aiida-computer-name'.

- `code_string`: (str) Name of the VASP code as defined in AiiDA. Example: 'vasp@computer'.

- `withmpi`: (optional, bool) Whether to run with MPI. Default is False.

- `qos`: (optional, str) Quality of service parameter. Example: 'gp_partition'.

- `account`: (optional, str) Account to be used for calculations. Example: 'account_name'.

- `node_cpus`: (optional, int) Number of CPUs per node. Default is 48.

- `max_wallclock_seconds`: (optional, int) Maximum wallclock time in seconds. Default is 28800.

- `options_resources`: (optional, dict) Resource options for the scheduler. Example: {'tot_num_mpiprocs': 112, 'num_machines': 1}.

- `multiple`: (optional, int) Multiple factor for resources. Default is 1.

- `custom_scheduler_commands`: (optional, str) Custom scheduler commands.

##### Surface-specific INCAR settings. - `[dft.vasp.surface]`


- `incar`: (optional, dict) INCAR settings specific to surface calculations. Example: {'ldipol': True, 'idipol': 3, 'ispin': 2}.

##### Cluster-specific INCAR settings. - `[dft.vasp.cluster]`


- `incar`: (optional, dict) INCAR settings specific to cluster calculations. Example: {'ldipol': True, 'dipol': [0.5, 0.5, 0.5], 'idipol': 4}.

##### AiiDA-VASP specific settings. - `[dft.vasp.aiida_vasp]`


- `parser_settings`: (optional, dict) Contains entries to include in the results gathered using the aiida-vasp parser settings Example: {'add_trajectory': False, 'add_bands': False, 'add_charge_density': False, 'add_dos': False, 'add_kpoints': False, 'add_energies': True, 'add_misc': True, 'add_structure': False, 'add_projectors': False, 'add_born_charges': False, 'add_dielectrics': False, 'add_hessian': False, 'add_dynmat': False, 'add_wavecar': False, 'add_forces': False, 'add_stress': False}.
