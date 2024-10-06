## 0.6.11 (2024-10-06)

### Feat

- **structure**: added function to create mdb.Structure from InitialDatabase row.
- **gen_db**: added allow_modifications flag to config gile phase entries in order to avoid applying changes to desired base structures.
- **gen_db**: added targeted modification `central_atom_octahedral` to database generation procedure
- **database**: added attributes to Structure to identify targeted modification and al_loop_step.
- **al_loop**: add option to disable multiheads_finetuning for MACE 0.3.7

### Fix

- **database**: modified database plot generation to avoid overlapping

## 0.6.10 (2024-10-03)

### BREAKING CHANGE

- The `system` key in the input TOML has been renamed to `database`, and users must now specify the path for the database and export options under this new key.
- `phase.name` is now slugified, which may affect downstream usage when accessing phase names.

### Feat

- **core**: improve surface generation with additional replacement and saving options
- **core**: add displacement tracking and enhance structure generation
- **core**: enhance structure handling and surface generation logic
- **core**: add phase structure limiting and improve phase handling
- **core**: add element-specific vacancy generation and improve installation docs
- **core**: add enhanced lattice perturbation and filtering options                                                                                                                                                   ─╯    - Enhanced `perturb_min_displacement` to support structure filtering, phase selection, and random number generator seeding for lattice perturbations.    - Introduced parameters `filters`, `only_use_base`, `use_phase`, and `rng_seed` to `perturb_min_displacement` for greater control over perturbation strategies.    - Improved structure type conversion in vacancy generation and perturbation methods by using `to_bulk()`, `to_surface()`, and `to_cluster()` methods.    - Updated string representations for `Structure` and `Phase` classes to improve readability of phase and structure details.    - Added support for limiting the number of structures in lattice displacement operations.    - Minor bug fixes and improvements to database and structure handling functions.    - `perturb_min_displacement` can now use different structures, although it still defaults to using base structures only unless specified otherwise to maintain compability.
- **core**: add deprecation utilities and enhance warnings handling
- **database**: add support for random vacancy generation and structure enhancements
- **database**: enhance database generation with plotting, exporting, and structure replacement options
- **active_learning**: seed structure deletion is now optional in AL workflows
- **utils**: add support for adsorbate addition on surfaces in database
- **utils**: extend API key gathering to support environment variable
- **database**: enhance phase diagram and database initialization logic
- **phase_diagram**: upgrade phase diagram logic to support binary and ternary systems
- added first test templates
- add `apply_replacement_no_db()`
- LAMMPS MD dump data now stored as compressed files
- new logging option + rich cli output when running dashboard
- added dev optional setup option
- allow running dashboard in a separate process
- added precommits
- enhance active learning workflow with new md filters and descriptor settings

### Fix

- **gen_db**: fixed wrong argument for `gen_surfaces_diff_miller`
- **logs**: database report now showing in logs
- **core**: fix phase name selection when filtering from db
- several fixes: - Reenabled ignore small structures during bulk generation. - Changed fixed seed with random generated one in case it is not given by the user. - Removed unused comments.
- added conditional to avoid all types of perturbed structures being saved as cluster.
- **cli**: update toml handling for Python 3.11 compatibility
- changed `fix_bottom_layers` message to warning.
- fixed database generation functions: - renamed `run_gen_initial_database` to `cli_run_gen_initial_database` - fixed import related to `cli_run_gen_initial_database` - converted numbers to float and int type after parsing database generation input .toml file - Added function for fixing bottom layers: `fix_bottom_layers`, however it is an unfinished which must be improved. It won't be used until finished.
- updated aiida-core dependency to avoid error #6519
- removed duplicate key
- fixed typo in send_calc_or_remove_structures()`
- added structure selection safeguard for `dft_structres` list
- improved structure selection safeguard
- added safeguard check for filtered trajectories
- updated imports on test files
- only load MACE models on first iteration
- changes to `send_calc_or_remove_structures` to avoid broadcasting errors: - Added padding to extrapolating_frames array when size is different to error_f_structures or error_e_structures - Disabled shrinking from array masks to maintain array size.
- added different padding strategies for energy and forces in `model_res_dict_to_arr()`
- updated `model_res_dict_to_arr`
- Changed dashboard progressbar header
- Changed dashboard progressbar header
- added nan padding to uneven model_res_dict_to_arr arrays
- updated dashboard launch mechanism
- mace model as absolute path (iv)
- mace model as absolute path (iii)
- mace model as absolute path (ii)
- changed incorrect import in cli_generate_init_db.py
- improved compatibility with tests
- resolve path on input files
- update documentation and modified wrong function call
- update dashboard counter
- allow TrainMACEModelCalculationParser to gather any model type
- improved README.md writing
- applied uniform formatting to most files in the library
- improve iteration display and change progressbar iteration
- remove filtered md structures from db and fix descriptor executable
- add cwd in remote as path for PortableCodes to run
- changed forces unit
- allow to use MACE_forces key on extxyz files
- allow to use functions from `aiida_utils` without using aiida

### Refactor

- **core**: replace deprecated structure conversion with new methods
- reduced line length on rich cli outputs
- renamed cli files
- separated command line scripts into individual files

## v0.3.9 (2024-06-19)
