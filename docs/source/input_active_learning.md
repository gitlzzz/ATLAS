## Active Learning Loop

Generate a active learning template file using `mdb_gen_configuration_file -t active_learning`.

:::{attention}
All keys are mandatory unless stated otherwise.
:::


### Active Learning - `[active_learning]`

General active learning settings.


- `aiida_profile`: (str) Name of the AiiDA profile to be used.

- `enable_ntfysh`: (optional, bool) Enable notifications using ntfy.sh for the run.
  - Default is `False`.

- `run_name`: (str) Internal name for the run.

- `init_db_path`: (str, PosixPath) Path to the folder containing the initial database.

- `results_dir`: (str, PosixPath) Path for final results. A folder named run_{uuid} will be created inside.

- `log_path`: (optional, str, PosixPath) Path for the log file. Defaults to results_dir if not specified.

- `final_db_name`: (str) Name for the final database (extxyz format).
  - Default is `'final_data_test'`.

- `max_iterations`: (int) Maximum number of AL loop iterations.
  - Default is `3`.

- `load_init_models`: (optional, list[int]) Load initial models from a list of AiiDA UUIDs/PKs. The format is `[UUID1/PK1, UUID2/PK2, ...]`.
  - Default is `[]`.

- `load_descriptor_calc`: (optional, str) Load a single descriptor calculation from a AiiDA UUID/PK.
  - Default is `''`.

- `load_md_calcs`: (optional, list[int]) Load MD calculations from a list of AiiDA UUIDs/PKs.
  - Default is `[]`.

- `al_mode`: (optional, str) Active learning mode.
  - Default is `'data_acquisition'`.
  - Possible values are: `md`, `data_reduction`, `data_acquisition`.

### Data Reduction Settings - `[data_reduction]`

Settings for the data reduction AL mode.


- `large_database_path`: (str, PosixPath) Path to the large database file from which to select structures.

- `initial_selection_size`: (int) Number of structures to select from the large database for initial training.
  - Default is `100`.

- `initial_selection_method`: (str) Selection method for initial structures.
  - Default is `'lowest_energy'`.
  - Possible values are: `random`, `lowest_energy`, `fps`.

- `structures_per_iteration`: (int) Number of structures to select per iteration from the large database.
  - Default is `50`.

- `iterative_selection_method`: (str) Selection method for iterative structures.
  - Default is `'uncertainty'`.
  - Possible values are: `random`, `fps`, `uncertainty`, `lowest_energy`.

### Active Learning Seed Settings - `[al_seed]`

Settings for the AL seed generation for MD.


- `seed_size_frac`: (float) Sets total structures in an MD seed as a fraction of the training db size.
  - Default is `0.01`.

- `seed_min_num_structs`: (optional, int) Minimum number of structures in an MD seed.
  - Default is `25`.

- `seed_max_num_structs`: (int) Maximum number of structures in the MD seed.
  - Default is `500`.

- `delete_seed_structs`: (optional, bool) Whether to delete structures from the seed database even if they are in domain.
  - Default is `True`.

#### Seed Selection Settings - `[al_seed.seed_select_settings]`

MD seed selection mode settings.


- `seed_select_type`: (str) MD seed selection mode.
  - Default is `'random'`.
  - Possible values are: `random`, `small_first`.

- `small_first_max_size`: (int) Maximum size in number of atoms for structures selected with small_first mode.
  - Default is `50`.

- `small_first_max_iter`: (int) Apply small_first mode for the first n iterations.
  - Default is `5`.

#### Seed Ranking Settings for Seed Selection - `[al_seed.seed_ranking_settings]`

Settings for the seed ranking methods.

:::{attention}
This section is optional.
:::


- `seed_ranking_algorithm`: (optional, str) Algorithm used for seed selection.
  - Default is `'random'`.
  - Possible values are: `random`, `descriptor_fps`.

##### Farthest Point Sampling (FPS) Configuration - `[al_seed.seed_ranking_settings.descriptor_fps]`

Farthest Point Sampling (FPS) ranking.


- `descriptor_type`: (optional, str) What descriptors to use for the seed selection process.
  - Default is `'soap'`.
  - Possible values are: `soap`, `mace`.

- `initial_structure`: (optional, str) Whether to gather a structure at random or select the one with the lowest energy available.
  - Default is `'random'`.
  - Possible values are: `random`, `lowest_energy`.

###### FPS Seed Selection - Selected Descriptor Settings - `[al_seed.seed_ranking_settings.descriptor_fps.descriptor]`

Entry containing settings that depend on the descriptor type selected.

:::{attention}
This section is optional.
:::


- `r_cut`: (optional, float) Cutoff radius for SOAP descriptor.
  - Default is `6.0`.

- `n_max`: (optional, int) Maximum number of radial basis functions for SOAP.
  - Default is `8`.

- `l_max`: (optional, int) Maximum degree of spherical harmonics for SOAP.
  - Default is `6`.

- `periodic`: (optional, bool) Whether to consider the system as periodic.
  - Default is `True`.

- `average`: (optional, str) Averaging mode for SOAP descriptor.
  - Default is `'off'`.
  - Possible values are: `inner`, `outer`, `off`.

- `model_path`: (optional, str) Path to the trained MACE model.

- `device`: (optional, str) What device to use for MACE.
  - Default is `'cpu'`.
  - Possible values are: `cpu`, `cuda`.

- `dtype`: (optional, str) Floating point number precision for MACE.
  - Default is `'float32'`.
  - Possible values are: `float32`, `float64`.

- `enable_cueq`: (optional, bool) Enable CUEQ for MACE.
  - Default is `False`.

### Interpolation Check Settings - `[interpolation]`

Settings for the interpolation check. The interpolation check determines whether a structure is within the model's domain or requires a DFT calculation, by checking the committee models disagreement, considering the model accuracy threshold.


- `model_acc_multiplier`: (float) Multiplier for model accuracy threshold. Higher values mean more DFT calculations.
  - Default is `10.0`.

### Extrapolation Check Settings - `[extrapolation]`

Settings for extrapolation checks.


- `disagreement_check_type`: (optional, str) Approach for energy and force (E&F) committee disagreement check. With `training`, compare E&F with a threshold obtained from the training RMSE values multiplied by a threshold. With `md_threshold`, compare E&F with a threshold obtained from the standard deviaiton of the MD frames.
  - Default is `'training'`.
  - Possible values are: `training`, `md_threshold`.

- `check_extrapolation_type`: (optional, str) Method for extrapolation check. With `min-max` or `basic`, check for extrapolation using the range of the MACE descriptors. With `alpha-shape` or `advanced`, check for extrapolation using the concave hull of the MACE descriptors. With `disabled` or `none`, disable the extrapolation check, only leaving committee disagreement for EF for the domain.
  - Default is `'none'`.
  - Possible values are: `disabled`, `none`, `basic`, `min-max`, `alpha-shape`, `advanced`.

#### Concave Hull Extrapolation Check Settings - `[extrapolation.concave_hull]`

Settings for the concave hull / alpha shape extrapolation check.

:::{attention}
This section is optional.
:::


- `target_alpha_range_min`: (optional, float) Minimum alpha value for the concave hull.
  - Default is `3.0`.

- `target_alpha_range_max`: (optional, float) Maximum alpha value for the concave hull.
  - Default is `8.0`.

- `default_alpha_if_issues`: (optional, float) Default alpha value if there are issues with the concave hull generation.
  - Default is `5.0`.

- `nn_dist_scale_factor`: (optional, float) Scaling factor for the alpha candidate calculation, where `alpha_candidate = nn_dist_scale_factor / mean_nn_dist`
  - Default is `1.5`.

- `frac_points_allowed_out`: (optional, float) Maximum fraction of points allowed to be outside the concave hull. If the fraction of points outside the hull exceeds this value, alpha will be decreased iteratively until the condition is met or alpha reaches zero. Value is expressed as a fraction, thus 0.002 means 0.2%.
  - Default is `0.002`.

### Active Learning Safeguard Settings - `[safeguard]`

Settings for active learning safeguard mechanisms. The safeguard will run long MD simulations on selected structures and perform an uncertainty quantification check in order to determine if the active learning loop is robust enough to stop at the current point.

:::{attention}
This section is optional.
:::


- `enable`: (bool) Whether to enable the safeguard mechanism.
  - Default is `False`.

- `target_structure_mode`: (str) Type of structures to include in the safeguard. Multiple types can be selected. If `base` is provided, the structures labelled as 'base' in the initial database will be used as target structures. If `target` is provided, a selection of targeted must be included through the `struct_target_list` option below.
  - Default is `'base'`.
  - Possible values are: `base`, `target`.

- `struct_target_list`: (optional, list[str, int]) List of structure IDs or paths to be used as target structures in the safeguard. Only used if `target_structure_mode` is set to `target`.

- `ignore_container`: (optional, bool) Whether to ignore the container specified in the container settings for the safeguard.
  - Default is `False`.

#### Metadata - `[safeguard.metadata]`

AiiDA metadata and scheduler options for the safeguard.


- `computer`: (str) AiiDA computer name for safeguard calculations.
  - Example: `'my_cluster'`.

- `prepend_text`: (optional, str) Text to prepend to job scripts for AiiDA.
  - Example: `'module load singularity
export PATH=$PATH:.'`.

- `options`: (dict) AiiDA scheduler options for safeguard calculations.

#### Safeguard MD Parameters - `[safeguard.md_parameters]`

MD simulation parameters for safeguard. These settings will override the general MD settings for the safeguard simulations.


- `temperature_list_K`: (list[float]) List of different temperatures (in K) for MD simulations.
  - Default is `[300.0, 500.0, 900.0]`.

- `max_temp_multiplier`: (float) Multiplier for MD temperature to determine the upper bound of the temperature.
  - Default is `1.3`.

- `num_steps`: (int) Total number of timesteps for each MD simulation.
  - Default is `1000`.

- `timestep_duration_ps`: (float) Duration of each timestep in picoseconds.
  - Default is `0.003`.

- `langevin_friction_ps-1`: (float) Friction coefficient for the Langevin thermostat in ps⁻¹.
  - Default is `10.0`.

- `gather_traj_cnt_lattice`: (bool) Consider constant lattice when gathering trajectories.
  - Default is `True`.

- `use_kokkos`: (bool) Whether to use Kokkos to run the MD.
  - Default is `True`.

- `device`: (str) Device for the MACE model in MD simulations.
  - Default is `'cuda'`.
  - Possible values are: `cpu`, `cuda`.

- `enable_cueq`: (optional, bool) Enable CUEQ for MACE.
  - Default is `False`.

- `default_dtype`: (str) Default data type for the MACE model in MD simulations.
  - Default is `'float32'`.
  - Possible values are: `float32`, `float64`.

- `al_keep_struct_every_n_ps`: (float) Keep a structure every N picoseconds of MD simulation.
  - Default is `0.5`.

- `log_save_interval`: (optional, int) Log energy and force information every N MD steps.
  - Default is `1`.

- `max_energy_threshold_per_atom`: (optional, float) Maximum energy threshold per atom in eV.
  - Default is `1000.0`.

- `num_cpus_large_struct`: (optional, int) Number of CPUs to use for structures larger than `large_struct_size`.
  - Default is `16`.

- `md_thermostat`: (optional, str) Thermostat used in the MD simulation.
  - Default is `'langevin'`.
  - Possible values are: `langevin`, `nvt`, `npt`, `nose-hoover`.

### Molecular Dynamics Settings - `[md]`

Settings for MD simulations.


- `ignore_container`: (optional, bool) Whether to ignore the container specified in the container settings for the MD calculations.
  - Default is `False`.

#### Parameters - `[md.parameters]`

MD simulation parameters.


- `temperature_list_K`: (list[float]) List of different temperatures (in K) for MD simulations.
  - Default is `[300.0, 500.0, 900.0]`.

- `sample_frames_during_md`: (optional, bool) Whether to sample frames during MD simulations for AL.
  - Default is `False`.

- `num_at_large_struct`: (optional, int) Number of structures with a number of atoms larger than `large_struct_size` to consider for MD simulations.
  - Default is `'None'`.

- `max_temp_multiplier`: (float) Multiplier for MD temperature to determine the upper bound of the temperature.
  - Default is `1.3`.

- `num_steps`: (int) Total number of timesteps for each MD simulation.
  - Default is `1000`.

- `timestep_duration_ps`: (float) Duration of each timestep in picoseconds.
  - Default is `0.003`.

- `langevin_friction_ps-1`: (float) Friction coefficient for the Langevin thermostat in ps⁻¹.
  - Default is `10.0`.

- `gather_traj_cnt_lattice`: (bool) Consider constant lattice when gathering trajectories.
  - Default is `True`.

- `use_kokkos`: (bool) Whether to use Kokkos to run the MD.
  - Default is `True`.

- `device`: (str) Device for the MACE model in MD simulations.
  - Default is `'cuda'`.
  - Possible values are: `cpu`, `cuda`.

- `enable_cueq`: (optional, bool) Enable CUEQ for MACE.
  - Default is `False`.

- `default_dtype`: (str) Default data type for the MACE model in MD simulations.
  - Default is `'float32'`.
  - Possible values are: `float32`, `float64`.

- `al_keep_struct_every_n_ps`: (float) Keep a structure every N picoseconds of MD simulation.
  - Default is `0.5`.

- `log_save_interval`: (optional, int) Log energy and force information every N MD steps.
  - Default is `1`.

- `max_energy_threshold_per_atom`: (optional, float) Maximum energy threshold per atom in eV.
  - Default is `1000.0`.

- `num_cpus_large_struct`: (optional, int) Number of CPUs to use for structures larger than `large_struct_size`.
  - Default is `16`.

- `md_thermostat`: (optional, str) Thermostat used in the MD simulation.
  - Default is `'langevin'`.
  - Possible values are: `langevin`, `nvt`, `npt`, `nose-hoover`.

#### MD Trajectory Filters - `[md.filters]`

Settings for MD trajectory filters.


- `save_filtered_structures`: (optional, bool) Whether to save filtered structures.
  - Default is `False`.

##### Check_Atoms_No_Neighbor - `[md.filters.check_atoms_no_neighbor]`

Filter for structures with atoms that have no neighbors.


- `enable`: (bool) No description available.
  - Default is `True`.

- `covalent_radius_multiplier`: (float) Multiplier for covalent radii to define cutoff for neighbor check.
  - Default is `1.05`.

##### Layer_Distance - `[md.filters.layer_distance]`

Filter for layer distances in surface slabs.


- `enable`: (bool) No description available.
  - Default is `True`.

- `max_layer_distance_ang`: (float) Maximum accepted distance between layers in Angstrom.
  - Default is `3.5`.

##### Exploding_Structures - `[md.filters.exploding_structures]`

Filter for exploding structures based on covalent radius limits.


- `enable`: (optional, bool) Whether to enable the exploding structures filter.
  - Default is `True`.

- `cov_rad_multiplier_max`: (optional, float) Maximum multiplier for covalent radius threshold.
  - Default is `10.0`.

- `cov_rad_multiplier_min`: (optional, float) Minimum multiplier for covalent radius threshold.
  - Default is `1.5`.

- `explode_check_interval_perc`: (optional, float) Interval percentage (as a fraction) of MD steps to check for exploding structures.
  - Default is `0.1`.

#### MD Metadata and Scheduler Options (AiiDA) - `[md.metadata]`

AiiDA metadata and scheduler options for MD simulations.


- `code`: (optional, str) AiiDA code name for MD software.
  - Example: `'mace_lammps@cluster'`.

- `computer`: (str) AiiDA computer name for MD calculations.
  - Example: `'my_cluster'`.

- `prepend_text`: (optional, str) Text to prepend to job scripts for AiiDA.
  - Example: `'module load singularity
export PATH=$PATH:.'`.

- `options`: (optional, dict) AiiDA scheduler options for MD calculations.

### Code and Containerization Settings - `[code]`

Settings for containerized code execution.


#### Container - `[code.container]`

Container settings for code execution.


- `use_container`: (optional, bool) Whether to use a containerized version of the code.
  - Default is `False`.

- `image_name`: (optional, str) Path to the container image on calculation nodes.
  - Example: `'/path/to/container.sif'`.

- `engine_command`: (optional, str) Command template for the container engine.
  - Example: `'singularity exec --bind .:/mdb_data --nv --contain --writable-tmpfs {image_name}'`.

- `prepend_text`: (optional, str) Text to prepend to job scripts for container setup.
  - Example: `'module load singularity
export PATH=$PATH:.'`.

### MLIP Training Settings - `[mace_train]`

Settings for MACE model training.


- `result_force_weight`: (optional, float) Weight of the force when considering model performance in weighted sum calculation.
  - Default is `0.1`.

- `test_fraction`: (optional, float) Fraction of the training data to be used for testing.
  - Default is `0.1`.

- `code`: (str) AiiDA code name for MACE training.
  - Example: `'mace_train@cluster'`.

- `computer`: (str) AiiDA computer name for MACE training.
  - Example: `'my_cluster'`.

- `ignore_container`: (optional, bool) Whether to ignore container settings for MACE training.
  - Default is `False`.

- `multihead_finetuning`: (optional, bool) Whether to use multihead finetuning.
  - Default is `False`.

- `metadata`: (optional, dict) AiiDA metadata and scheduler options for MACE training.

- `train_settings`: (optional, dict) MACE training parameters and hyperparameters.

### Committee Evaluation Settings - `[committee_eval]`

Settings for committee evaluation using multiple MLIP models.


- `committee_num_models`: (optional, int) Total number of MLIP models in the committee.
  - Default is `4`.

- `openmp_threads`: (optional, int) Number of OpenMP threads for MLIP CPU evaluation.
  - Default is `24`.

- `ignore_container`: (optional, bool) Whether to ignore container settings for committee evaluation.
  - Default is `False`.

- `prepend_text`: (optional, str) Text to prepend to job scripts for committee evaluation.

- `metadata`: (optional, dict) AiiDA metadata and scheduler options for committee evaluation.

#### Commitee Evaluation - MACE Settings - `[committee_eval.mace]`

Settings for MACE evaluator.


- `device`: (optional, str) Device for MACE evaluation.
  - Default is `'cpu'`.
  - Possible values are: `cpu`, `cuda`.

- `default_dtype`: (optional, str) Default data type for MACE evaluation.
  - Default is `'float32'`.
  - Possible values are: `float32`, `float64`.

- `batch_size`: (optional, int) Batch size for MACE evaluation.
  - Default is `32`.

- `compute_stress`: (optional, bool) Whether to compute stress during evaluation.
  - Default is `False`.

### Descriptor Computation Settings - `[descriptors]`

Settings for descriptor computation and dimensionality reduction.


- `dimensionality_reduction_method`: (optional, str) Dimensionality reduction method for MACE descriptors.
  - Default is `'none'`.
  - Possible values are: `autoencoder`, `pca`, `none`.

- `ignore_container`: (optional, bool) Whether to ignore container settings for descriptor computation.
  - Default is `False`.

- `metadata`: (optional, dict) AiiDA metadata and scheduler options for descriptor computation.

- `descriptor_type`: (optional, str) Type of descriptor to compute.
  - Default is `'mace'`.
  - Possible values are: `mace`, `soap`.

- `dtype`: (optional, str) Data type of descriptor to compute.
  - Default is `'float32'`.
  - Possible values are: `float32`, `float64`.

- `device`: (optional, str) Device for descriptor computation.
  - Default is `'cpu'`.
  - Possible values are: `cpu`, `cuda`.

#### Autoencoder - `[descriptors.autoencoder]`

Settings for autoencoder-based dimensionality reduction.


##### Train_Settings - `[descriptors.autoencoder.train_settings]`

Training settings for the autoencoder.


- `device`: (optional, str) Device for autoencoder training.
  - Default is `'cuda'`.
  - Possible values are: `cpu`, `cuda`.

- `dtype`: (optional, str) Data type for autoencoder training.
  - Default is `'float32'`.
  - Possible values are: `float32`, `float64`.

- `model_path`: (optional, str) Path to save the autoencoder model.
  - Default is `'autoencoder_model.pth'`.

- `load_model`: (optional, bool) Whether to load the model from the model path.
  - Default is `False`.

- `dataset`: (optional, str) Path to the training dataset.
  - Default is `'all_descriptors.npz'`.

- `l1_hidden_dim`: (optional, int) Number of units in the first hidden layer.
  - Default is `256`.

- `l2_hidden_dim`: (optional, int) Number of units in the second hidden layer.
  - Default is `32`.

- `bottleneck_dim`: (optional, int) Dimensionality of the bottleneck (latent space).
  - Default is `2`.

- `bias_flag`: (optional, bool) Flag to include bias terms in the layers.
  - Default is `True`.

- `num_epochs`: (optional, int) Number of epochs to train the model.
  - Default is `50`.

- `batch_size`: (optional, int) Batch size for training.
  - Default is `2048`.

- `patience`: (optional, int) Patience for early stopping.
  - Default is `5`.

- `lr`: (optional, float) Learning rate for the optimizer.
  - Default is `0.001`.

- `weight_decay`: (optional, float) L2 regularization parameter.
  - Default is `'1e-5'`.

- `loss`: (optional, str) Loss function type.
  - Default is `'mse'`.
  - Possible values are: `mse`, `mae`.

- `train_frac`: (optional, float) Fraction of the data to use for training.
  - Default is `0.8`.

- `valid_frac`: (optional, float) Fraction of the data to use for validation.
  - Default is `0.1`.

- `test_frac`: (optional, float) Fraction of the data to use for testing.
  - Default is `0.1`.

- `wandb`: (optional, bool) Whether to log metrics to wandb.
  - Default is `False`.

- `wandb_name`: (optional, str) Name of the wandb run.
  - Default is `''`.

- `wandb_project`: (optional, str) Name of the wandb project.
  - Default is `''`.

### DFT Calculation Settings - `[dft]`

DFT settings specific to active learning (different from top-level dft section).


- `ignore_container`: (optional, bool) Whether to ignore container settings for DFT calculations.
  - Default is `False`.

- `dft_method`: (optional, str) Selection of DFT calculator.
  - Default is `'mace'`.
  - Possible values are: `vasp`, `mace`.

- `calc_type`: (optional, str) Type of calculation.
  - Default is `'single_point'`.
  - Possible values are: `single_point`, `relax`, `static`.

- `dft_calc_limit`: (optional, int) Maximum number of DFT calculations to perform per AL step.
  - Default is `'None'`.

#### MACE DFT Calculator Settings - `[dft.mace]`

MACE settings as DFT calculator.

:::{attention}
This section is optional.
:::


- `mace_potential_path`: (str) Path to MACE potential file.
  - Example: `'model.model'`.

- `metadata`: (optional, dict) AiiDA metadata for MACE calculations.

- `options`: (optional, dict) AiiDA scheduler options for MACE calculations.

##### Settings - `[dft.mace.settings]`

Options for MACE that will be passed as arguments during execution.


- `device`: (optional, str) Device for MACE calculations.
  - Default is `'cuda'`.
  - Possible values are: `cpu`, `cuda`.

- `default_dtype`: (optional, str) Default data type for MACE calculations.
  - Default is `'float64'`.
  - Possible values are: `float32`, `float64`.

- `batch_size`: (optional, int) Batch size for MACE calculations.
  - Default is `11`.

- `compute_stress`: (optional, bool) Whether to compute stress.
  - Default is `False`.

##### Filter - `[dft.mace.filter]`

Settings for filtering structures after DFT calculations, based on energy and force thresholds.

:::{attention}
This section is optional.
:::


- `filter_dft_calcs`: (optional, bool) Whether to filter structures based on energy and force thresholds.
  - Default is `False`.

- `threshold_E_meV`: (optional, float) Energy threshold in meV/atom for filtering structures.
  - Default is `1000.0`.

- `threshold_F_meV`: (optional, float) Force threshold in meV/Å for filtering structures.
  - Default is `10000.0`.

#### VASP DFT Calculator Settings - `[dft.vasp]`

VASP settings as DFT calculator.

:::{attention}
This section is optional.
:::


- `calc_type`: (optional, str) Type of calculation.
  - Default is `'static'`.
  - Possible values are: `static`, `relax`.

- `dft_calc_limit`: (optional, int) Maximum number of DFT calculations to perform per AL step.
  - Default is `200`.

- `potential_family`: (str) VASP potential family name.
  - Example: `'vasp-5.4-PBE-2024'`.

- `structure_types`: (optional, list[str]) List of structure types to process.
  - Default is `['bulk', 'surface', 'cluster']`.
  - Possible values are: `bulk`, `surface`, `cluster`.

- `kspacing`: (optional, dict) K-spacing settings for different phases or default value.
  - Example:

```python
{'MDB_DEFAULT': 0.15, 'alpha': 0.135}
```

- `incar`: (optional, dict) INCAR settings for VASP calculations.
  - Example:

```python
{'encut': 450, 'gga': 'Pe', 'icharg': 2, 'istart': 0}
```

##### Filter - `[dft.vasp.filter]`

Settings for filtering structures after DFT calculations, based on energy and force thresholds.

:::{attention}
This section is optional.
:::


- `filter_dft_calcs`: (optional, bool) Whether to filter structures based on energy and force thresholds.
  - Default is `False`.

- `threshold_E_meV`: (optional, float) Energy threshold in meV/atom for filtering structures.
  - Default is `1000.0`.

- `threshold_F_meV`: (optional, float) Force threshold in meV/Å for filtering structures.
  - Default is `10000.0`.

##### Queue - `[dft.vasp.queue]`

Queue settings for VASP calculations.


- `queue_type`: (str) Scheduler type.
  - Default is `'slurm'`.

- `computer`: (str) AiiDA computer name for VASP calculations.
  - Example: `'aiida-computer-name'`.

- `code_string`: (str) Name of the VASP code as defined in AiiDA.
  - Example: `'vasp@computer'`.

- `withmpi`: (optional, bool) Whether to run with MPI.
  - Default is `False`.

- `qos`: (optional, str) Quality of service parameter.
  - Example: `'gp_partition'`.

- `account`: (optional, str) Account to be used for calculations.
  - Example: `'account_name'`.

- `node_cpus`: (optional, int) Number of CPUs per node.
  - Default is `48`.

- `max_wallclock_seconds`: (optional, int) Maximum wallclock time in seconds.
  - Default is `28800`.

- `options_resources`: (optional, dict) Resource options for the scheduler.
  - Example:

```python
{'num_machines': 1, 'tot_num_mpiprocs': 112}
```

- `multiple`: (optional, int) Multiple factor for resources.
  - Default is `1`.

- `custom_scheduler_commands`: (optional, str) Custom scheduler commands.

##### Surface - `[dft.vasp.surface]`

Surface-specific INCAR settings.


- `incar`: (optional, dict) INCAR settings specific to surface calculations.
  - Example:

```python
{'idipol': 3, 'ispin': 2, 'ldipol': True}
```

##### Cluster - `[dft.vasp.cluster]`

Cluster-specific INCAR settings.


- `incar`: (optional, dict) INCAR settings specific to cluster calculations.
  - Example:

```python
{'dipol': [0.5, 0.5, 0.5], 'idipol': 4, 'ldipol': True}
```

##### AiiDA-VASP Settings - `[dft.vasp.aiida_vasp]`

AiiDA-VASP specific settings.


- `parser_settings`: (optional, dict) Contains entries to include in the results gathered using the aiida-vasp parser settings
  - Example:

```python
{   'add_bands': False,
    'add_born_charges': False,
    'add_charge_density': False,
    'add_dielectrics': False,
    'add_dos': False,
    'add_dynmat': False,
    'add_energies': True,
    'add_forces': False,
    'add_hessian': False,
    'add_kpoints': False,
    'add_misc': True,
    'add_projectors': False,
    'add_stress': False,
    'add_structure': False,
    'add_trajectory': False,
    'add_wavecar': False}
```

- `critical_notifications`: (optional, dict) Critical error and warning notifications. These represent VASP errors and warnings to be treated as critical, which will result in an error code being thrown by the aiida calculation job. The example contains all defults.
  - Example:

```python
{   'add_bandocc': True,
    'add_brmix': True,
    'add_cnormn': True,
    'add_denmp': True,
    'add_dentet': True,
    'add_edddav_zhegv': True,
    'add_eddrmm_zhegv': True,
    'add_edwav': True,
    'add_fexcp': True,
    'add_fock_acc': True,
    'add_magmom': True,
    'add_no_potimm': True,
    'add_non_collinear': True,
    'add_not_hermitian': True,
    'add_psmaxn': True,
    'add_pzstein': True,
    'add_real_optlay': True,
    'add_rhosyg': True,
    'add_rspher': True,
    'add_set_indpw_full': True,
    'add_sgrcon': True}
```
