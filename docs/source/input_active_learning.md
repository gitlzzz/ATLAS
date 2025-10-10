## Active Learning Loop

Generate a active learning template file using `mdb_gen_configuration_file -t active_learning`.

:::{attention}
All keys are mandatory unless stated otherwise.
:::


### General active learning settings. - `[active_learning]`


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

- `model_acc_multiplier`: (float) Multiplier for model accuracy threshold. Higher values mean more DFT calculations.
  - Default is `10.0`.

- `load_init_models`: (optional, list[int]) Load initial models from a list of AiiDA UUIDs/PKs.
  - Default is `[]`.

- `al_mode`: (optional, str) Active learning mode.
  - Default is `'data_acquisition'`.
  - Possible values are: `md`, `data_reduction`, `data_acquisition`.

### Settings for the data reduction AL mode. - `[data_reduction]`


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

### Settings for the AL seed generation for MD. - `[al_seed]`


- `seed_size_frac`: (float) Sets total structures in an MD seed as a fraction of the training db size.
  - Default is `0.01`.

- `seed_min_num_structs`: (optional, int) Minimum number of structures in an MD seed.
  - Default is `25`.

- `seed_max_num_structs`: (int) Maximum number of structures in the MD seed.
  - Default is `500`.

- `delete_seed_structs`: (optional, bool) Whether to delete structures from the seed database even if they are in domain.
  - Default is `True`.

#### MD seed selection mode settings. - `[al_seed.seed_select_settings]`


- `seed_select_type`: (str) MD seed selection mode.
  - Default is `'random'`.
  - Possible values are: `random`, `small_first`.

- `small_first_max_size`: (int) Maximum size in number of atoms for structures selected with small_first mode.
  - Default is `50`.

- `small_first_max_iter`: (int) Apply small_first mode for the first n iterations.
  - Default is `5`.

#### Settings for the seed ranking methods. - `[al_seed.seed_ranking_settings]`

:::{attention}
This section is optional.
:::


- `seed_ranking_algorithm`: (optional, str) Algorithm used for seed selection.
  - Default is `'random'`.
  - Possible values are: `random`, `descriptor_fps`.

##### Farthest Point Sampling (FPS) ranking.` - `[al_seed.seed_ranking_settings.descriptor_fps]`


- `descriptor_type`: (optional, str) What descriptors to use.
  - Default is `'soap'`.
  - Possible values are: `soap`, `mace`.

- `initial_structure`: (optional, str) Whether to gather a structure at random or select the one with the lowest energy available.
  - Default is `'random'`.
  - Possible values are: `random`, `lowest_energy`.

###### Entry containing settings that depend on the descriptor type selected. - `[al_seed.seed_ranking_settings.descriptor_fps.descriptor]`

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

### Settings for extrapolation checks. - `[extrapolation]`


- `disagreement_check_type`: (optional, str) Approach for energy and force (E&F) committee disagreement check. With `training`, compare E&F with a threshold obtained from the training RMSE values multiplied by a threshold. With `md_threshold`, compare E&F with a threshold obtained from the standard deviaiton of the MD frames.
  - Default is `'training'`.
  - Possible values are: `training`, `md_threshold`.

- `check_extrapolation_type`: (optional, str) Method for extrapolation check. With `min-max` or `basic`, check for extrapolation using the range of the MACE descriptors. With `alpha-shape` or `advanced`, check for extrapolation using the concave hull of the MACE descriptors. With `disabled` or `none`, disable the extrapolation check, only leaving committee disagreement for EF for the domain.
  - Default is `'none'`.
  - Possible values are: `disabled`, `none`, `basic`, `min-max`, `alpha-shape`, `advanced`.

#### Settings for the concave hull extrapolation check. - `[extrapolation.concave_hull]`

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

### Settings for MD simulations. - `[md]`


- `ignore_container`: (optional, bool) Whether to ignore the container specified in the container settings for the MD calculations.
  - Default is `False`.

#### MD simulation parameters. - `[md.parameters]`


- `temperature_list_K`: (list[float]) List of different temperatures (in K) for MD simulations.
  - Default is `[300.0, 500.0, 900.0]`.

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

#### Settings for MD trajectory filters. - `[md.filters]`


- `save_filtered_structures`: (optional, bool) Whether to save filtered structures.
  - Default is `False`.

##### Filter for structures with atoms that have no neighbors. - `[md.filters.check_atoms_no_neighbor]`


- `enable`: (bool) No description available.
  - Default is `True`.

- `covalent_radius_multiplier`: (float) Multiplier for covalent radii to define cutoff for neighbor check.
  - Default is `1.05`.

##### Filter for layer distances in surface slabs. - `[md.filters.layer_distance]`


- `enable`: (bool) No description available.
  - Default is `True`.

- `max_layer_distance_ang`: (float) Maximum accepted distance between layers in Angstrom.
  - Default is `3.5`.

##### Filter for exploding structures based on covalent radius limits. - `[md.filters.exploding_structures]`


- `enable`: (optional, bool) Whether to enable the exploding structures filter.
  - Default is `True`.

- `cov_rad_multiplier_max`: (optional, float) Maximum multiplier for covalent radius threshold.
  - Default is `10.0`.

- `cov_rad_multiplier_min`: (optional, float) Minimum multiplier for covalent radius threshold.
  - Default is `1.5`.

- `explode_check_interval_perc`: (optional, float) Interval percentage (as a fraction) of MD steps to check for exploding structures.
  - Default is `0.1`.

#### AiiDA metadata and scheduler options for MD simulations. - `[md.metadata]`


- `code`: (optional, str) AiiDA code name for MD software.
  - Example: `'mace_lammps@cluster'`.

- `computer`: (str) AiiDA computer name for MD calculations.
  - Example: `'my_cluster'`.

- `prepend_text`: (optional, str) Text to prepend to job scripts for AiiDA.
  - Example: `'module load singularity
export PATH=$PATH:.'`.

- `options`: (optional, dict) AiiDA scheduler options for MD calculations.

### Settings for containerized code execution. - `[code]`


#### Container settings for code execution. - `[code.container]`


- `use_container`: (optional, bool) Whether to use a containerized version of the code.
  - Default is `False`.

- `image_name`: (optional, str) Path to the container image on calculation nodes.
  - Example: `'/path/to/container.sif'`.

- `engine_command`: (optional, str) Command template for the container engine.
  - Example: `'singularity exec --bind .:/mdb_data --nv --contain --writable-tmpfs {image_name}'`.

- `prepend_text`: (optional, str) Text to prepend to job scripts for container setup.
  - Example: `'module load singularity
export PATH=$PATH:.'`.

### Settings for MACE model training. - `[mace_train]`


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

### Settings for committee evaluation using multiple MACE models. - `[committee_eval]`


- `committee_num_models`: (optional, int) Total number of MACE models in the committee.
  - Default is `4`.

- `openmp_threads`: (optional, int) Number of OpenMP threads for MACE CPU evaluation.
  - Default is `24`.

- `ignore_container`: (optional, bool) Whether to ignore container settings for committee evaluation.
  - Default is `False`.

- `prepend_text`: (optional, str) Text to prepend to job scripts for committee evaluation.

- `metadata`: (optional, dict) AiiDA metadata and scheduler options for committee evaluation.

#### Settings for MACE evaluator. - `[committee_eval.mace]`


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

### Settings for descriptor computation and dimensionality reduction. - `[descriptors]`


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

#### Settings for autoencoder-based dimensionality reduction. - `[descriptors.autoencoder]`


##### Training settings for the autoencoder. - `[descriptors.autoencoder.train_settings]`


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

### DFT settings specific to active learning (different from top-level dft section). - `[dft]`


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

#### MACE settings as DFT calculator. - `[dft.mace]`

:::{attention}
This section is optional.
:::


- `mace_potential_path`: (str) Path to MACE potential file.
  - Example: `'model.model'`.

- `metadata`: (optional, dict) AiiDA metadata for MACE calculations.

- `options`: (optional, dict) AiiDA scheduler options for MACE calculations.

##### Options for MACE that will be passed as arguments during execution. - `[dft.mace.settings]`


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

##### Settings for filtering structures after DFT calculations, based on energy and force thresholds. - `[dft.mace.filter]`

:::{attention}
This section is optional.
:::


- `filter_dft_calcs`: (optional, bool) Whether to filter structures based on energy and force thresholds.
  - Default is `False`.

- `threshold_E_meV`: (optional, float) Energy threshold in meV/atom for filtering structures.
  - Default is `1000.0`.

- `threshold_F_meV`: (optional, float) Force threshold in meV/Å for filtering structures.
  - Default is `10000.0`.

#### VASP settings as DFT calculator. - `[dft.vasp]`

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
  - Example: `{'MDB_DEFAULT': 0.15, 'alpha': 0.135}`.

- `incar`: (optional, dict) INCAR settings for VASP calculations.
  - Example: `{'istart': 0, 'icharg': 2, 'gga': 'Pe', 'encut': 450}`.

##### Settings for filtering structures after DFT calculations, based on energy and force thresholds. - `[dft.vasp.filter]`

:::{attention}
This section is optional.
:::


- `filter_dft_calcs`: (optional, bool) Whether to filter structures based on energy and force thresholds.
  - Default is `False`.

- `threshold_E_meV`: (optional, float) Energy threshold in meV/atom for filtering structures.
  - Default is `1000.0`.

- `threshold_F_meV`: (optional, float) Force threshold in meV/Å for filtering structures.
  - Default is `10000.0`.

##### Queue settings for VASP calculations. - `[dft.vasp.queue]`


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
  - Example: `{'tot_num_mpiprocs': 112, 'num_machines': 1}`.

- `multiple`: (optional, int) Multiple factor for resources.
  - Default is `1`.

- `custom_scheduler_commands`: (optional, str) Custom scheduler commands.

##### Surface-specific INCAR settings. - `[dft.vasp.surface]`


- `incar`: (optional, dict) INCAR settings specific to surface calculations.
  - Example: `{'ldipol': True, 'idipol': 3, 'ispin': 2}`.

##### Cluster-specific INCAR settings. - `[dft.vasp.cluster]`


- `incar`: (optional, dict) INCAR settings specific to cluster calculations.
  - Example: `{'ldipol': True, 'dipol': [0.5, 0.5, 0.5], 'idipol': 4}`.

##### AiiDA-VASP specific settings. - `[dft.vasp.aiida_vasp]`


- `parser_settings`: (optional, dict) Contains entries to include in the results gathered using the aiida-vasp parser settings
  - Example: `{'add_trajectory': False, 'add_bands': False, 'add_charge_density': False, 'add_dos': False, 'add_kpoints': False, 'add_energies': True, 'add_misc': True, 'add_structure': False, 'add_projectors': False, 'add_born_charges': False, 'add_dielectrics': False, 'add_hessian': False, 'add_dynmat': False, 'add_wavecar': False, 'add_forces': False, 'add_stress': False}`.
