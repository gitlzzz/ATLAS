## DFT Calculations

Generate a dft template file using `mdb_gen_configuration_file -t dft`.

:::{attention}
All keys are mandatory unless stated otherwise.
:::


### General settings for the DFT script. - `[general]`


- `log_path`: (str, PosixPath) Path where the logs will be stored.
  - Default is `'/tmp/'`.

- `result_file_path`: (str, PosixPath) Path for the results file (extxyz format).
  - Default is `'dft_calculation_results'`.

- `source_db`: (optional, str, PosixPath) Path to the source database file (.extxyz or mdb .xz format).

- `aiida_group_name`: (str) Name of the AiiDA group for the calculations.
  - Example: `'my_dft_run'`.

- `max_batch`: (int) Maximum number of structures to process in one batch.
  - Default is `100`.

- `queue_check_interval_seconds`: (int) Interval in seconds to check the queue for submitting new calculations.
  - Default is `240`.

- `start_on_struct_idx`: (int) Number of structures to skip before starting the calculations.
  - Default is `0`.

- `dry_run`: (optional, bool) If True, a dry-run is performed and no calculations are submitted.
  - Default is `False`.

- `selected_structure_type`: (optional, str) If specified, only structures of this type will be processed.
  - Possible values are: `bulk`, `surface`, `cluster`.

### DFT calculation settings. - `[calculation]`


- `calc_type`: (str) Type of calculation.
  - Default is `'static'`.
  - Possible values are: `static`, `relaxation`.

- `aiida_potential_family`: (str) AiiDA potential family name.
  - Example: `'vasp-5.4-PBE-2023'`.

- `potential_mapping`: (optional, dict) Mapping of elements to specific potential labels.
  - Example: `{'Si': 'Si_GW'}`.

### K-point settings. - `[kpoints]`


- `kspacing`: (dict) K-spacing in Å⁻¹ for different phases or a single value for all structures.
  - Example: `{'alpha': 0.125, 'default': 0.15}`.

### Queue settings for HPC schedulers (e.g., SLURM). - `[queue]`


- `code_string`: (str) Name of the code as defined in AiiDA.
  - Example: `'vasp@my_cluster'`.

- `account`: (optional, str) Account to be used for the calculations.

- `qos`: (optional, str) Quality of service parameter.

- `node_cpus`: (optional, int) Number of CPUs per node.

- `max_wallclock_seconds`: (int) Maximum wallclock time in seconds.
  - Default is `16200`.

- `max_memory_kb`: (optional, int) Maximum memory per node in KB.

- `multiple`: (optional, int) Whether to use multiple nodes.

- `custom_scheduler_commands`: (optional, str) Custom scheduler commands.

#### Scheduler resource options. - `[queue.options_resources]`


- `tot_num_mpiprocs`: (int) Total number of MPI processes.
  - Default is `24`.

- `parallel_env`: (str) Parallel environment to be used.
  - Default is `' '`.

### Settings for the AiiDA-VASP plugin. - `[aiida_vasp]`

:::{attention}
This section is optional.
:::


- `critical_notifications`: (optional, dict) Errors and warnings to be treated as critical (general).

#### Parser settings for aiida-vasp. - `[aiida_vasp.parser_settings]`


- `add_kpoints`: (optional, bool) Whether to add k-points information to the parsed results.
  - Default is `True`.

##### Critical error and warning notifications to be treated as important. - `[aiida_vasp.parser_settings.critical_notifications]`


- `add_edddav_zhegv`: (optional, bool) Add EDDDAV ZHEGV error notification.
  - Default is `True`.

- `add_eddrmm_zhegv`: (optional, bool) Add EDDRMM ZHEGV error notification.
  - Default is `True`.

- `add_not_hermitian`: (optional, bool) Add not hermitian error notification.
  - Default is `True`.

- `add_brmix`: (optional, bool) Add BRMIX error notification.
  - Default is `True`.

- `add_bandocc`: (optional, bool) Add band occupation error notification.
  - Default is `False`.

### INCAR settings for different structure types. - `[incar]`


- `bulk`: (optional, dict) INCAR settings for bulk structures.

- `surface`: (optional, dict) INCAR settings for surface structures.

- `cluster`: (optional, dict) INCAR settings for cluster structures.
