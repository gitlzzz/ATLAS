## DFT Calculations

Generate a dft template file using `mdb_gen_configuration_file -t dft`.

:::{attention}
All keys are mandatory unless stated otherwise.
:::


### General - `[general]`

General settings for the DFT script.


- :alt:`log_path`: (str, PosixPath) Path where the logs will be stored.
  - Default is `'/tmp/'`.

- :alt:`result_file_path`: (str, PosixPath) Path for the results file (extxyz format).
  - Default is `'dft_calculation_results'`.

- :alt:`source_db`: (optional, str, PosixPath) Path to the source database file (.extxyz or mdb .xz format).

- :alt:`aiida_group_name`: (str) Name of the AiiDA group for the calculations.
  - Example: `'my_dft_run'`.

- :alt:`max_batch`: (int) Maximum number of structures to process in one batch.
  - Default is `100`.

- :alt:`queue_check_interval_seconds`: (int) Interval in seconds to check the queue for submitting new calculations.
  - Default is `240`.

- :alt:`start_on_struct_idx`: (int) Number of structures to skip before starting the calculations.
  - Default is `0`.

- :alt:`dry_run`: (optional, bool) If True, a dry-run is performed and no calculations are submitted.
  - Default is `False`.

- :alt:`selected_structure_type`: (optional, str) If specified, only structures of this type will be processed.
  - Possible values are: `bulk`, `surface`, `cluster`.

### Calculation - `[calculation]`

DFT calculation settings.


- :alt:`calc_type`: (str) Type of calculation.
  - Default is `'static'`.
  - Possible values are: `static`, `relaxation`.

- :alt:`aiida_potential_family`: (str) AiiDA potential family name.
  - Example: `'vasp-5.4-PBE-2023'`.

- :alt:`potential_mapping`: (optional, dict) Mapping of elements to specific potential labels.
  - Example:

```python
{'Si': 'Si_GW'}
```

### Kpoints - `[kpoints]`

K-point settings.


- :alt:`kspacing`: (dict) K-spacing in Å⁻¹ for different phases or a single value for all structures.
  - Example:

```python
{'alpha': 0.125, 'default': 0.15}
```

### Queue - `[queue]`

Queue settings for HPC schedulers (e.g., SLURM).


- :alt:`code_string`: (str) Name of the code as defined in AiiDA.
  - Example: `'vasp@my_cluster'`.

- :alt:`account`: (optional, str) Account to be used for the calculations.

- :alt:`qos`: (optional, str) Quality of service parameter.

- :alt:`node_cpus`: (optional, int) Number of CPUs per node.

- :alt:`max_wallclock_seconds`: (int) Maximum wallclock time in seconds.
  - Default is `16200`.

- :alt:`max_memory_kb`: (optional, int) Maximum memory per node in KB.

- :alt:`multiple`: (optional, int) Whether to use multiple nodes.

- :alt:`custom_scheduler_commands`: (optional, str) Custom scheduler commands.

#### Options_Resources - `[queue.options_resources]`

Scheduler resource options.


- :alt:`tot_num_mpiprocs`: (int) Total number of MPI processes.
  - Default is `24`.

- :alt:`parallel_env`: (str) Parallel environment to be used.
  - Default is `' '`.

### Aiida_Vasp - `[aiida_vasp]`

Settings for the AiiDA-VASP plugin.

:::{attention}
This section is optional.
:::


- :alt:`critical_notifications`: (optional, dict) Errors and warnings to be treated as critical (general).

#### Parser_Settings - `[aiida_vasp.parser_settings]`

Parser settings for aiida-vasp.


- :alt:`add_kpoints`: (optional, bool) Whether to add k-points information to the parsed results.
  - Default is `True`.

##### Critical_Notifications - `[aiida_vasp.parser_settings.critical_notifications]`

Critical error and warning notifications to be treated as important.


- :alt:`add_edddav_zhegv`: (optional, bool) Add EDDDAV ZHEGV error notification.
  - Default is `True`.

- :alt:`add_eddrmm_zhegv`: (optional, bool) Add EDDRMM ZHEGV error notification.
  - Default is `True`.

- :alt:`add_not_hermitian`: (optional, bool) Add not hermitian error notification.
  - Default is `True`.

- :alt:`add_brmix`: (optional, bool) Add BRMIX error notification.
  - Default is `True`.

- :alt:`add_bandocc`: (optional, bool) Add band occupation error notification.
  - Default is `False`.

### Incar - `[incar]`

INCAR settings for different structure types.


- :alt:`bulk`: (optional, dict) INCAR settings for bulk structures.

- :alt:`surface`: (optional, dict) INCAR settings for surface structures.

- :alt:`cluster`: (optional, dict) INCAR settings for cluster structures.
