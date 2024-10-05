## 0.6.10 (2024-10-03)

### BREAKING CHANGE

- The `system` key in the input TOML has been renamed to `database`, and users must now specify the path for the database and export options under this new key.
- `phase.name` is now slugified, which may affect downstream usage when accessing phase names.

### Feat

- **al_dashboard**: new logging option + rich cli output when running dashboard
- **al_dashboard**: allow running dashboard in a separate process
- **al_loop**: LAMMPS MD dump data now stored as compressed files
- **al_loop**: added safeguard check for filtered trajectories
- **al_loop**: enhance active learning workflow with new md filters and descriptor settings
- **al_loop**: seed structure deletion is now optional in AL workflows
- **db_gen**: Improve surface generation with additional replacement and saving options
- **db_gen**: Enhance structure handling and surface generation logic
- **db_gen**: add phase structure limiting and improve phase handling
- **db_gen**: added support for limiting the number of structures in lattice displacement operations.
- **db_gen**: add element-specific vacancy generation and improve installation docs
- **db_gen**: add `apply_replacement_no_db()`
- **db_gen**: add support for random vacancy generation and structure enhancements
- **db_gen**: add displacement tracking and enhance structure generation
- **db-gen**: add enhanced lattice perturbation and filtering options
- **db_gen**: `perturb_min_displacement` can now use different structures, although it still defaults to using base structures only unless specified otherwise to maintain compability.
- **db_gen**: Enhanced `perturb_min_displacement` to support structure filtering, phase selection, and random number generator seeding for lattice perturbations.
  - **db_gen**: Introduced parameters `filters`, `only_use_base`, `use_phase`, and `rng_seed` to `perturb_min_displacement` for greater control over perturbation strategies.
- **db_gen**: Improved structure type conversion in vacancy generation and perturbation methods by using `to_bulk()`, `to_surface()`, and `to_cluster()` methods.
- **core**: Updated string representations for `Structure` and `Phase` classes to improve readability of phase and structure details.
- **core**: Minor bug fixes and improvements to database and structure handling functions.
- **core**: add deprecation utilities and enhance warnings handling
- **core**: added dev optional setup option
- **core**: added precommits
- **database**: enhance database generation with plotting, exporting, and structure replacement options
- **database**: enhance phase diagram and database initialization logic
- **phase_diagram**: upgrade phase diagram logic to support binary and ternary systems
- **test**: added first test templates
- **utils**: add support for adsorbate addition on surfaces in database
- **utils**: extend API key gathering to support environment variable

### Fix

- **al_dashboard**: changed dashboard progressbar header
- **al_dashboard**: changed dashboard progressbar header
- **al_dashboard**: updated dashboard launch mechanism
- **al_dashboard**: update dashboard counter
- **al_loop**: removed duplicate key
- **al_loop**: fixed typo in send_calc_or_remove_structures()`
- **al_loop**: added structure selection safeguard for `dft_structres` list
- **al_loop**: improved structure selection safeguard
- **al_loop**: resolve path on input files
- **al_loop**: update documentation and modified wrong function call
- **al_loop**: allow `TrainMACEModelCalculationParser` to gather any model type
- **al_loop**: applied uniform formatting to most files in the library
- **al_loop**: improve iteration display and change progressbar iteration
- **al_loop**: remove filtered md structures from db and fix descriptor executable
- **al_loop**: add cwd in remote as path for PortableCodes to run
- **al_loop**: changed forces unit
- **al_loop**: allow to use MACE_forces key on extxyz files
- **al_loop**: allow to use functions from `aiida_utils` without using aiida
- **al_loop**: updated imports on test files
- **al_loop**: updated aiida-core dependency to avoid an error from AiiDA (see AiiDA #6519)
- **al_loop**: only load MACE models on first iteration
- **al_loop**: changes to `send_calc_or_remove_structures` to avoid broadcasting errors: - Added padding to extrapolating_frames array when size is different to error_f_structures or error_e_structures - Disabled shrinking from array masks to maintain array size.
- **al_loop**: added different padding strategies for energy and forces in `model_res_dict_to_arr()`
- **al_loop**: updated `model_res_dict_to_arr`
- **al_loop**: added nan padding to uneven model_res_dict_to_arr arrays
- **al_loop**: mace model as absolute path (iv)
- **al_loop**: mace model as absolute path (iii)
- **al_loop**: mace model as absolute path (ii)
- **cli**: update toml handling for Python 3.11 compatibility
- **core**: fix phase name selection when filtering from db
- **gen_db**: fixed wrong argument for `gen_surfaces_diff_miller`
- **gen_db**: Reenabled ignore small structures during bulk generation.
- **gen_db**: Changed fixed seed with random generated one in case it is not given by the user.
- **gen_db**: Removed unused comments.
- **gen_db**: added conditional to avoid all types of perturbed structures being saved as cluster.
- **gen_db**: changed `fix_bottom_layers` message to warning.
- **gen_db**: renamed `run_gen_initial_database` to `cli_run_gen_initial_database`
- **gen_db**: fixed import related to `cli_run_gen_initial_database`
- **gen_db**: converted numbers to float and int type after parsing database generation input .toml file
- **gen_db**: changed incorrect import in cli_generate_init_db.py
- **gen_db**: Added function for fixing bottom layers: `fix_bottom_layers`, however it is unfinished.
- **logs**: database report now showing in logs
- **tests**: improved compatibility with tests

### Refactor

- **al_dashboard**: reduced line length on rich cli outputs
- **al_loop**: improved README.md writing
- **core**: replace deprecated structure conversion with new methods
- **core**: renamed cli files
- **core**: separated command line scripts into individual files
