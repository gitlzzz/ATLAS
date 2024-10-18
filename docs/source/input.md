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

Generate a database generation template file using `mdb_gen_configuration_file -t initial_db`.

:::{attention}
All keys are mandatory unless stated otherwise.
:::

### General Database Description - `[database]`

This section defines the general settings and file paths for the database.

- `database_name:` (str) Name of the database to be used for internal reference and as the filename.
- `min_num_atoms`: (int) Minimum number of atoms allowed in the generated structures.
- `max_num_atoms`: (int) Maximum number of atoms allowed in the generated structures.
- `min_cell_size`: (float) Minimum cell size in Angstrom.
- `relax_struct_path`: (str) Path to a folder containing DFT optimized structures (optional).
- `database_path`: (str) Path where the final database will be saved.
- `rng_seed`: (optional, int) Numerical value used to fix the RNG seed. If not specified, it will be chosen randomly each run.
- `overwrite_db`:  (optional, bool) Allow database overwrite. Default is false. If false, and the database exists, the new database name will include a timestamp.

#### Display and Export Options - `[database.plot_db]`

- `show`: (bool) Whether to display the database with a phase diagram after creation.
- `format`: (str, optional) Format for the figure, such as 'png' or 'svg'. Default is 'png'.
- `rc_params."font.family"`: (str) Font family for the phase diagram plot.
- `rc_params."font.size"`: (int) Font size for the phase diagram plot.

#### ASE Display Options - `[database.show_db_ase]`

- `show`: (bool) Whether to display the database using ASE GUI after creation.

#### Export Options - `[database.export]`

- `export`: (bool) Whether to export the database.
- `format`: (str) Export format supported by ASE (e.g., 'extxyz').
- `file_path`: (str) Path where the exported file will be saved.
- `file_name`: (str) Name of the exported file.

### Phase Diagram Description - `[phase_diagram]`

This section defines the settings related to the phase diagram of the material.

- `material_name`: (str) Internal name for the material in the phase diagram.
- `element_list`: (list[str]) List of elements to include in the phase diagram.
- `base_element`: (str) Symbol of the most abundant element in the phase.

### Specific Phase Settings - `[phase_diagram.phase.XXXXX]`

This key describes the settings for a specific phase within the phase diagram. Several phases can be added to describe the entire phase diagram by adding new keys with different phase names.

The key name (`XXXXX`) is used as the reference name for the phase (e.g., `alpha`, `beta`, `gamma`, `liquid`, `amorphous`, ...). **Replace XXXXX with a phase name.**.

An example for a phase:

```toml
[phase_diagram.phase.alpha]
cluster_element = "Cu"
prototype = "mp-30"
composition.Cu.min = 0.627
composition.Zn.max = 0.373
composition.Zn.min = 0.0
composition.Cu.max = 1.0
offset = 0.1
limit_max_num_structures = 500
allow_modifications = true
replacements.replace = true
replacements.element_list = ["Cu"]
replacements.replace_with = ["Fe"]
```

Every phase can be provided with options to customize it:

- `cluster_element:` (str) Symbol of the element defining the cluster.
- `prototype`: (str) Materials Project ID of the prototypical structure (e.g., `mp-30` for Cu alpha).
- `composition.X.min`: (float) Minimum composition of element X.
- `composition.X.max`: (float) Maximum composition of element X.
- `composition.Y.min`: (float) Minimum composition of element Y.
- `composition.Y.max`: (float) Maximum composition of element Y.
- `offset`: (float) Fraction of composition allowed over and under the phase limits.
- `replacements.replace`: (bool) Whether to replace specific elements.
- `replacements.element_list`: (list[str]) List of elements to be replaced.
- `replacements.replace_with`: (str) Element to replace with.
- `limit_max_num_structures`: (optional, int) Maximum number of structures to generate for the current phase. This limit is enforced after the initial phase generation, but subsequent operations (e.g., perturbations, vacancy generation) may add new structures, potentially exceeding the specified limit.
- `allow_modifications`: (optional, bool) Whether to allow applying modifications (supercells, replacements, perturbations, modifications...) to the base structure, or keep the base structure as the only structure for the phase. Default is true, allowing modifications to the base structure to be applied.

### Structure Generation Settings - `[generation]`

This key describes the settings related to the generation of structures.

- `generate_type`: (list[str]) Types of structures to generate. Options: `'bulk'`, `'surface'`, `'cluster'`.

### Bulk Structure Generation - `[generation.bulk]`

This key describes the settings for the generation of bulk structures.

- `num_struct`: (int) Number of structures to generate.
- `num_repeat`: (int) Number of repeats for each structure.
- `supercell_max_idx`: (int) Maximum Miller index for the bulk supercells.

### Surface Structure Generation - `[generation.surface]`

This key describes the settings related to the generation of surface structures.

- `min_miller_index`: (int)  Minimum Miller index used to generate surface structures.
- `max_miller_index`: (int)  Maximum Miller index used to generate surface structures.
- `min_vacuum_size_ang`: (float) Minimum size of the vacuum layer in Angstroms.
- `get_supercells`: (bool) Whether to generate supercells for surface structures.
- `fixed_layers`: (int) Number of fixed layers in the surface slab.
- `max_number_supercells`: (int) Maximum number of surface supercells to generate.
- `save_in_db`: (bool) Whether to save generated surfaces in the database.
- `num_replacements`: (int) Number of replacement percentages to generate for each structure.
- `num_repeat_replace`: (int) Number of repeats for each replacement.
- `frac_slabs_save`: (float) Fraction of slabs to save after generation. This avoids having too many slab structures with the same composition.
- `frac_supercells_save`: (float) Fraction of unreplaced supercells to save after generation. This avoids having too many slab structures with the same composition.

### Lattice Deformation Settings - `[displacement]`

This key describes the settings related to the lattice deformation of structures.

- `lattice_frac_displ_max`: (float) Maximum displacement value as a percentage of the lattice side length.
- `lattice_frac_displ_min`: (float) Minimum displacement value as a percentage of the lattice side length.
- `num_repeats`: (int) Number of repeats for each structure with random displacements.

### Perturbation Settings - `[perturbation]`

This key describes the settings related to the perturbation of structures.

- `filter_struct_types`: (list) Types of structures to which the perturbation will be applied. Valid types: `'bulk'`, `'surface'`, `'cluster'`.
- `limit_max_num_perturbs`: (int) Maximum number of perturbations to generate.
- `num_repeats`: (int) Number of repeats for each structure, with each repeat getting different random perturbations.

### Vacancy Generation Settings - `[vacancies]`

This section describes the settings for generating vacancies in structures.

- `filter_struct_types`: (list[str]) Structure types to apply vacancies. Valid types: `'bulk'`, `'surface'`, `'cluster'`.
- `limit_max_num_vacancies`: (int) Maximum number of structures with vacancies.
- `num_repeats`: (int) Number of repeats for each structure with random vacancies.
- `max_vacancy_percentage`: (float) Maximum vacancy percentage of total atoms.
- `min_vacancy_percentage`: (float) Minimum vacancy percentage of total atoms.
- `element_list`: (list[str]) List of elements to consider for the vacancies

## Input Example: Database Generation

```{literalinclude} ../../src/MatDBForge/data/input_files/database_generation_settings.toml
```

## Active Learning Loop

Generate a database generation template file using `mdb_gen_configuration_file -t active_learning`.

:::{attention}
All keys are mandatory unless stated otherwise.
:::

### Active learning - `[al_learning]`

This key describes the main active learning settings:

- `aiida_profile`: (str) Name of the aiida profile to be used.
- `run_name`: (str) Internal name for the run
- `init_db_path`: (str) Path to the folder where the initial database is contained.
- `results_dir`: (str) Path for final results. Will be created if not existent.
It will contain a folder named `run_{uuid}`.
- `final_db_name`: (str) Name for the final database. The database will be stored in the extxyz format
- `max_iterations`: (int) Maximum number of AL loop iterations.
- `model_acc_multiplier`: (float)
Multiplier for model accuracy. Loosens model accuracy threshold. Tighter thresholds (lower values) will result in more DFT calculations. Any values that meet: $$RMSE_E\ or\ RMSE_F > chem\_acc \cdot chem\_acc\_multiplier$$ will be considered wrong.

- `al_keep_struct_every_n_ps`: (float) Every how many ps of MD simulation keep a structure.
Influences the total number of energy evaluations and possibly DFT calculations.
<!-- - `check_extrapolation`: (bool) Whether to check for extrapolation using the MACE descriptors -->
- `dft_method`: (str) Selection of energy/force calculator. Options: "vasp", "mace"
- `load_init_models`: (list[int], optional) # Load initial models from several aiida uuids/pk.

### Active Learning Seed Generation - `[al_seed]`

Parameters to configure the AL seed generation

- `seed_size_frac`: (float) Fraction of the training db set to be used to create the AL seed. This influences the amount of MD calculations and E F evaluations.
- `seed_max_num_structs`: (int) Maximum number of structures in the MD seed.


### Extrapolation Settings - `[extrapolation]`
This section contains keys which adjust the extrapolation settings.

- `check_extrapolation_type`: (str, optional) Whether to check for extrapolation. Default is `advanced`. Currently two options are allowed:
    - `basic`: Check for extrapolation using the ranges of the MACE descriptors.
    - `advanced`: Check for extrapolation using the concave hull of the MACE descriptors, after reducing dimensionality with an autoencoder trained on the current iteration data descriptors.


#### Seed selection settings - `[al_seed.seed_select_settings]`

Parameters to tune the structure selection while creating the AL seeds.

- `seed_select_type`: MD seed selection mode. `random` selects random structures from the seed pool `small_first` selects random structures smaller than small_first_max_size for the first small_first_max_iter iters. random (default) / small_first
- `small_first_max_size`: (int) Maximum size in number of atoms for the structures selected with small_first mode
- `small_first_max_iter`: (int) Apply small_first mode for the first n iterations

### MD Settings - `[md]`

Settings for MD simulations using LAMMPS

#### MD Parameters - `[md.parameters]`

- `temperature_list_K`: (list[float]) List of different temperatures (in K) to be used for the MD simulations. Example: [300, 350, 400]
- `max_temp_multiplier`: (int) Multiplier for the user-specified MD temperature used to determine the upper bound of the temperature at the end of the simulation run. Set to 1 to disable the multiplier.
- `num_steps` (int) Total number of MD steps to be run in each MD simulation
- `timestep_duration_ps`: (float) Duration of each timestep. In LAMMPS, [timestep size depends on the choice of units](https://docs.lammps.org/timestep.html). If metal (default) units are set, the timestep is in ps.
- `gather_traj_cnt_lattice`: (bool) Consider constant lattice when gathering trajectories
- `use_kokkos`: (bool) Whether to use kokkos to run the LAMMPS MD on gpu

#### MD Filters - `[md.filters]`

Contains settings related to the filtering of structures obtained from MD calculations. Filtering allows the removal of some types of incorrect structures that might pollute the training database

- `check_atoms_no_neighbor` (bool):  Check for structures that have atoms with no neighbors. Specific setting for the neighbor check MD filter.

:::{warning}
In version 0.6, only use the `check_atoms_no_neighbor` when generating bulks, surfaces or clusters that have no adsorbed molecules.
:::

- `layer_distance.max_layer_distance_ang` (float):  Specific setting for the layer distance MD filter.Maximum accepted distance between layers (in Angstrom).

#### MD Queue - `[md.queue]`

Contains settings related to the MD calculation setup and usage of AiiDA. As of now, the only MD code available is LAMMPS-MACE.
The queue key can take any option from its [matching AiiDA input](https://aiida.readthedocs.io/projects/aiida-core/en/stable/topics/calculations/usage.html#options). Below are listed the bare minimum options for running calculations using an SGE scheduler:

- `code` (str): AiiDA code name for MD software to be used
 AiiDA scheduler options for MD
- `metadata.options.resources.parallel_env` (str):  name for the parallel environment
- `metadata.options.resources.tot_num_mpiprocs` (int): Total number of mpi processors
- `metadata.options.queue_name` (str):  Name of the SGE queue
- `metadata.options.max_memory_kb` (int):  Max memory allowed in kB.
- `metadata.options.max_wallclock_seconds` (int):  Maximum requested wall time.
- `metadata.options.withmpi` (bool): Use MPI.
- `metadata.options.custom_scheduler_commands` (str): Use this to include extra options, such as GPU allocation on SGE or extra commands. Use triple quoted strings to allow for several options. E.g.:

```python
'''
#$ -l gpu=1
#$ -l hostname="tekla2189"
'''
```

### Committee Evaluation Settings - `[committee_eval]`

Contains settings related to the NNP model evaluation (MACE as of now).

- `committee_num_models`: (int) Total number of MACE models.  Increases resources necessary for training.
- `openmp_threads` (int): Number of OpenMP threads to be used for MACE CPU evaluation.
- `prepend_text`: prepend text for the aiida [PortableCode](https://aiida.readthedocs.io/projects/aiida-core/en/stable/topics/data_types.html#portablecode). Use triple quoted strings to allow for several options.

#### MACE Evaluation Settings - `[committee_eval.mace]`

Contains settings for the MACE evaluator.

- `device`: (str) Device to use for the training, either `cpu` or `cuda`
- `default_dtype`: (str) Whether to use single precision or double precision floating point numbers. Either `float32` or  `float64`
- `batch_size`: (int) Size of the training batch
- `compute_stress`: (bool) Whether or not to compute stress (true/false)

#### MACE Evaluation Scheduler - `[committee_eval.metadata.options]

Contains settings related to the scheduler and AiiDA for the MACE Evaluations. This key can take any option from its [matching AiiDA input](https://aiida.readthedocs.io/projects/aiida-core/en/stable/topics/calculations/usage.html#options). Below are listed the bare minimum options for running calculations using an SGE scheduler:

- `parser_name` (str):  "mace-committee-eval-parser"
- `resources.parallel_env` (str):  "c128m1024ib_mpi_32slots"
- `resources.tot_num_mpiprocs`: 32
- `queue_name`(str) :  "c128m1024ibgpu4.q"
- `max_wallclock_seconds`: 117280000
- `max_memory_kb`: 102400000
- `withmpi`: false
- `computer` = "tekla2-new-test"
- `custom_scheduler_commands`:

```
'''#$ -l gpu=1
$ -l hostname="tekla2189"
'''
```

### Descriptor Settings - `[descriptors]`


- `dimensionality_reduction_method`: (str, optional) Dimensionality reduction method for MACE descriptors. Options: `autoencoder`, `pca`, `none`. If not set, the default is none, which means no dimensionality reduction.

#### Descriptor Scheduler Options - `[descriptors.metadata]`

AiiDA metadata settings for MACE descriptor calculation

- `computer` = ''
- `account` = ''
- `qos` = ''
- `parser_name` = ''
- `prepend_text` = ''
- `options.max_wallclock_seconds` = ''
- `options.resources.num_cores_per_mpiproc` = ''
- `options.resources.tot_num_mpiprocs` = ''
- `options.resources.num_machines` = ''
- `options.withmpi` = ''


#### Autoencoder settings - `[descriptors.autoencoder.train_settings]`

This section contains keys related to the Autoencoder training

- `device`: (str) Device to train the model. Either `cpu` or `cuda`.
- `model_path`: (str): Filename or path to save the model.
- `load_model`: (bool) Load the model from the model path
- `dataset`: (str) Path to the training dataset
- `l1_hidden_dim`: (int) Number of units in the first hidden layer, default `256`.
- `l2_hidden_dim`: (int) Number of units in the second hidden layer, default `32`.
- `bottleneck_dim`: (int) Dimensionality of the bottleneck (latent space), default `2`.
- `bias_flag`: (bool) Flag to include bias terms in the layers
- `num_epochs`: (int) Number of epochs to train the model
- `batch_size`: (int) Batch size for training
- `patience`: (int) Patience for early stopping
- `lr`: (float) Learning rate for the optimizer
- `weight_decay`: (float) L2 regularization
- `loss`: (str) Loss function type (one of: "mse", "mae", "weighted_mse")
- `train_frac`: (float) Fraction of the data to use for training
- `valid_frac`: (float) Fraction of the data to use for validation
- `test_frac`: (float) Fraction of the data to use for testing
- `wandb`: (bool) Whether to log metrics to wandb
- `wandb_name`: (str) Name of the wandb run
- `wandb_project`: (str)  Name of the wandb project


### MACE Scheduler - `[mace_train]`

MACE training code and scheduler settings (aiida)

- `result_force_weight` (float): Weight of the force when considering model performance used in:  $$Weighted\ sum = RMSE_E + (F_{weight} * RMSE_F)$$
The lowest weighted_sum will be considered as the most performant model.
- `code`: (str) AiiDA code name
- `metadata.options.resources.parallel_env`: (str)
- `metadata.options.resources.tot_num_mpiprocs` = 32
- `metadata.options.parser_name` = "mace-training-parser"
- `metadata.options.queue_name` = "c128m1024ibgpu4.q"
- `metadata.options.max_wallclock_seconds` = 117280000
- `metadata.options.max_memory_kb` = 102400000
- `metadata.options.withmpi` = true
- `metadata.options.custom_scheduler_commands`: (str)

```
'''#$ -l gpu=1
$ -l hostname="tekla2188"
'''
```

#### MACE Train Settings - `[mace_train.train_settings]`

MACE Training Settings. Check the [MACE documentation on training](https://mace-docs.readthedocs.io/en/latest/guide/training.html) for more information. Here are some sample values used for training in one of our case studies:

- `name`(str)
- `energy_key` (str):  "energy"
- `valid_fraction` (float):  0.1
- `foundation_model` (str):  Either `small`/`medium`/`large` or a path pointing to a foundation model in the machine where the training will be running.
- `config_type_weights`: { Default = 1.0 }
- `weight_decay`: 9.34e-07
- `E0s`: "average"
- `num_interactions`: 2
- `model`: "MACE"
- `correlation`: 3
- `hidden_irreps` (str): "16x0e + 16x1o"
- `lr` (float): 0.0056
- `r_max`: 6.0
- `max_ell`: 3
- `max_L`: 2
- `batch_size`: 64
- `max_num_epochs`: 35
- `swa`: true
- `ema`: true
- `ema_decay`: 0.99
- `amsgrad`: true
- `restart_latest`: true
- `device` (str): Either `cuda`/`cpu`
- `default_dtype` (str): Either `float32`/`float64`
- `wandb`(bool): false

### DFT Settings - `[dft]`

### MACE as DFT calculator - `[dft.mace]`

MACE Settings as DFT calculator. Ignored if dft_method = "vasp"
Options intended for MACE will be passed as arguments during MACE execution. The scheduler options will be used in the builder.metadata.options from AiiDA.

- `mace_potential_path`(str): Path to MACE potential file
- `device` (str): "cuda"           # Options: "cpu", "cuda"
- `default_dtype` (str): "float32" # Options: "float32", "float64"
- `batch_size` (int): 64
- `compute_stress`(bool): true     # Options: true, false
- `options.parser_name`: "mace-eval-parser"
- `options.code_string`: "mace_run_eval_gpu@tekla2-new-test"
- `options.resources`: { parallel_env = "c128m1024ib_mpi_32slots", tot_num_mpiprocs = 32 }
- `options.max_wallclock_seconds`: 117280000
- `options.withmpi`: false
- `options.max_memory_kb`: 102400000
- `options.custom_scheduler_commands`(str): Additional options for the scheduler, such as setting the hostname:

```
'''
#$ -l gpu=1
#$ -l hostname="tekla2189"
'''
```

### VASP as DFT calculator - `[dft.vasp]`

Settings for VASP as DFT calculator using the [aiida-vasp](https://aiida-vasp.readthedocs.io/en/latest/) plugin. Ignored if `dft_method = "mace"`.

- `potential_family` = "vasp-5.3-PBE"
- `potential_family` = "vasp-5.4-PBE-2023"
- `structure_types` = ['bulk', 'surface', 'cluster']

See [the potentials section in the aiida-vasp documentation](https://aiida-vasp.readthedocs.io/en/latest/getting_started/potentials.html) to setup the potentials for VASP.

### Scheduler settings for aiida-vasp - `[dft.vasp.queue]`

- `queue.type` = "sge"
- `queue.node_cpus` = 12
- `queue.code_string` = "vasp-std-5.4.4-new@tekla2"
- `queue.options_resources` = { parallel_env = "c12m48ib_mpi", tot_num_mpiprocs = 12 }
- `queue.multiple` = 1
- `queue.custom_scheduler_commands` = '#$ -l hostname="tekla2044"'

### VASP k-spacing - `[dft.vasp.kspacing]`

**WIP**

Description of the phase diagram.

- `alpha` = 0.135088484104361
- `m1` = 0.100530964914873
- `beta`-prime = 0.102415920507027
- `m2` = 0.100530964914873
- `gamma` = 0.141371669411541
- `m3` = 0.166504410640259
- `epsilon` = 0.105557513160617
- `eta` = 0.0993371597065093
- `m4` = 0.0948760981384118
- `delta` = 0.0994491889005363

### VASP INCAR - `[dft.vasp.incar]`

General incar settings to be used as a template for all calculations. Different types of calculations, i.e., relaxations, bulks, clusters and surfaces, can have different options. Type-specific options must be specified in the corresponding key for each type (see below) and will overwrite the keys on the general incar in this section.

- `istart`: 0
- `icharg`: 2
- `gga`: "Pe"
- `ispin`: 1
- `encut`: 450    # electronic steps
- `ediff`: 1e-6
- `ismear`: 0
- `sigma`: 0.03
- `algo`: "Fast"
- `lreal`: "Auto"
- `nelm`: 60      # ionic steps
- `ibrion`: -1
- `nsw`: 1
- `ediffg`: -0.03
- `isif`: 2
- `potim`: 0.3
- `lwave`: false
- `lcharg`: false
- `ncore`: 4
- `lelf`: false
- `ivdw`: 11      # van der Waals

#### VASP INCAR for relaxations - `[dft.vasp.relax.incar]`

- `ibrion` = 2
- `nsw` = 350
- `isif` = 3

#### VASP INCAR for surfaces - `[dft.vasp.surface.incar]`

- `ldipol` = true
- `idipol` = 3

#### VASP INCAR for clusters - `[dft.vasp.cluster.incar]`

- `ldipol` = true
- `dipol` = [0.5, 0.5, 0.5]
- `idipol` = 4

## Input Example: Active Learning
```{literalinclude} ../../src/MatDBForge/data/input_files/active_learning_settings.toml
```
