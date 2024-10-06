## 0.6.11 (2024-10-06)

### Feat

- **structure**: added function to create mdb.Structure from InitialDatabase row. (98f6e22)
- **gen_db**: added allow_modifications flag to config gile phase entries in order to avoid applying changes to desired base structures. (8543078)
- **gen_db**: added targeted modification `central_atom_octahedral` to database generation procedure (bdccd9f)
- **database**: added attributes to Structure to identify targeted modification and al_loop_step. (46dbf80)
- **al_loop**: add option to disable multiheads_finetuning for MACE 0.3.7 (9c69386)

### Fix

- **database**: modified database plot generation to avoid overlapping (e80fedc)

## 0.6.10 (2024-10-03)

### BREAKING CHANGE

- The `system` key in the input TOML has been renamed to `database`, and users must now specify the path for the database and export options under this new key. (fbd88e2)
- `phase.name` is now slugified, which may affect downstream usage when accessing phase names. (a42ea14)

### Feat

- **core**: improve surface generation with additional replacement and saving options (643ec8c)
- **core**: add displacement tracking and enhance structure generation (c63745d)
- **core**: enhance structure handling and surface generation logic (fa6ddf3)
- **core**: add phase structure limiting and improve phase handling (8a3e1a8)
- **core**: add element-specific vacancy generation and improve installation docs (08d5694)
- **core**: add enhanced lattice perturbation and filtering options                                                                                                                                                   ─╯    - Enhanced `perturb_min_displacement` to support structure filtering, phase selection, and random number generator seeding for lattice perturbations.    - Introduced parameters `filters`, `only_use_base`, `use_phase`, and `rng_seed` to `perturb_min_displacement` for greater control over perturbation strategies.    - Improved structure type conversion in vacancy generation and perturbation methods by using `to_bulk()`, `to_surface()`, and `to_cluster()` methods.    - Updated string representations for `Structure` and `Phase` classes to improve readability of phase and structure details.    - Added support for limiting the number of structures in lattice displacement operations.    - Minor bug fixes and improvements to database and structure handling functions.    - `perturb_min_displacement` can now use different structures, although it still defaults to using base structures only unless specified otherwise to maintain compability. (b5626fc)
- **core**: add deprecation utilities and enhance warnings handling (37312dc)
- **database**: add support for random vacancy generation and structure enhancements (e04a3ac)
- **database**: enhance database generation with plotting, exporting, and structure replacement options (fbd88e2)
- **active_learning**: seed structure deletion is now optional in AL workflows (9617489)
- **utils**: add support for adsorbate addition on surfaces in database (146727f)
- **utils**: extend API key gathering to support environment variable (739e299)
- **database**: enhance phase diagram and database initialization logic (a42ea14)
- **phase_diagram**: upgrade phase diagram logic to support binary and ternary systems (21cc63e)
- added first test templates (9352880)
- add `apply_replacement_no_db()` (f20e6bf)
- LAMMPS MD dump data now stored as compressed files (0c95f2c)
- new logging option + rich cli output when running dashboard (75cdf58)
- added dev optional setup option (0ec4afe)
- allow running dashboard in a separate process (a6d18ae)
- added precommits (5eeea1e)
- enhance active learning workflow with new md filters and descriptor settings (fe2da24)

### Fix

- **gen_db**: fixed wrong argument for `gen_surfaces_diff_miller` (dc45ed8)
- **logs**: database report now showing in logs (52bf135)
- **core**: fix phase name selection when filtering from db (6d9037e)
- several fixes: - Reenabled ignore small structures during bulk generation. - Changed fixed seed with random generated one in case it is not given by the user. - Removed unused comments. (0240a46)
- added conditional to avoid all types of perturbed structures being saved as cluster. (01fd2c0)
- **cli**: update toml handling for Python 3.11 compatibility (afc94d4)
- changed `fix_bottom_layers` message to warning. (631f420)
- fixed database generation functions: - renamed `run_gen_initial_database` to `cli_run_gen_initial_database` - fixed import related to `cli_run_gen_initial_database` - converted numbers to float and int type after parsing database generation input .toml file - Added function for fixing bottom layers: `fix_bottom_layers`, however it is an unfinished which must be improved. It won't be used until finished. (f6e4466)
- updated aiida-core dependency to avoid error #6519 (5be149b)
- removed duplicate key (2ef361c)
- fixed typo in send_calc_or_remove_structures()` (381a64a)
- added structure selection safeguard for `dft_structres` list (03f9779)
- improved structure selection safeguard (c721582)
- added safeguard check for filtered trajectories (e5ef27b)
- updated imports on test files (46467f2)
- only load MACE models on first iteration (e82744b)
- changes to `send_calc_or_remove_structures` to avoid broadcasting errors: - Added padding to extrapolating_frames array when size is different to error_f_structures or error_e_structures - Disabled shrinking from array masks to maintain array size. (295ff37)
- added different padding strategies for energy and forces in `model_res_dict_to_arr()` (f5b0b21)
- updated `model_res_dict_to_arr` (624c1f9)
- Changed dashboard progressbar header (2c3e9b0)
- Changed dashboard progressbar header (cc22899)
- added nan padding to uneven model_res_dict_to_arr arrays (a75c095)
- updated dashboard launch mechanism (a2ae5be)
- mace model as absolute path (iv) (f703582)
- mace model as absolute path (iii) (ac43885)
- mace model as absolute path (ii) (b6e104c)
- changed incorrect import in cli_generate_init_db.py (7355f16)
- improved compatibility with tests (005f4b3)
- resolve path on input files (df7a8b5)
- update documentation and modified wrong function call (69739d1)
- update dashboard counter (1afc7d7)
- allow TrainMACEModelCalculationParser to gather any model type (d3d55b0)
- improved README.md writing (af6d94d)
- applied uniform formatting to most files in the library (daa62dd)
- improve iteration display and change progressbar iteration (e03b413)
- remove filtered md structures from db and fix descriptor executable (13f7b83)
- add cwd in remote as path for PortableCodes to run (e0c3101)
- changed forces unit (6ffd858)
- allow to use MACE_forces key on extxyz files (8d36397)
- allow to use functions from `aiida_utils` without using aiida (61bfad6)

### Refactor

- **core**: replace deprecated structure conversion with new methods (f12aa96)
- reduced line length on rich cli outputs (2f59483)
- renamed cli files (c5e4e73)
- separated command line scripts into individual files (2df2e93)

## v0.3.9 (2024-06-19)
