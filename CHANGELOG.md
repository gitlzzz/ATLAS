## 0.8.0 (2024-10-07)

### Feat

- **db_gen**: added function to check for updates ([#b0e73ab](https://github.com/pol-sb/MatDBForge/commit/b0e73abfc50e2755c11b7a1714d09ddb944d3925))

### Fix

- **db_gen**: fixed bug that prevented creation of 'central_atom_perturbation' modified structures ([#f31d6c4](https://github.com/pol-sb/MatDBForge/commit/f31d6c4b3aaafc82649c31c6b7272f01df62abc6))
- **db_gen**: improved pie charts ([#d9d996d](https://github.com/pol-sb/MatDBForge/commit/d9d996d1c6c48a23c6916f350375c1b856f39b5a))
- **db_gen**: refactored code to avoid circular imports ([#5c682a1](https://github.com/pol-sb/MatDBForge/commit/5c682a1624459df6f58de3ca5d947baec82a08d6))

## 0.7.0 (2024-10-07)

### Feat

- **db_gen**: add parameter for disallowing database overwrite.
- **db_gen**: added pie charts to composition plots

### Fix

- **db_gen**: increased consistency with uuid and phase label storage.
- **db_gen**: increased consistency when saving structures with `targeted_modification`.

## 0.6.11 (2024-10-06)

### Feat

- **structure**: added function to create mdb.Structure from InitialDatabase row. ([#98f6e22](https://github.com/pol-sb/MatDBForge/commit/98f6e224ad47f161dfee8097502a9f8fbf5e4c6c))
- **gen_db**: added allow_modifications flag to config gile phase entries in order to avoid applying changes to desired base structures. ([#8543078](https://github.com/pol-sb/MatDBForge/commit/85430787fd6111dc2639f6583b2f5e29db747c18))
- **gen_db**: added targeted modification `central_atom_octahedral` to database generation procedure ([#bdccd9f](https://github.com/pol-sb/MatDBForge/commit/bdccd9f70bfdc9a5efe2008000872e73979a29d8))
- **database**: added attributes to Structure to identify targeted modification and al_loop_step. ([#46dbf80](https://github.com/pol-sb/MatDBForge/commit/46dbf80307a332f57648548e29ceabc8aa2e233f))
- **al_loop**: add option to disable multiheads_finetuning for MACE 0.3.7 ([#9c69386](https://github.com/pol-sb/MatDBForge/commit/9c69386f7cc126856be9428c929e9878e67a76e7))

### Fix

- **database**: modified database plot generation to avoid overlapping ([#e80fedc](https://github.com/pol-sb/MatDBForge/commit/e80fedcd7fd220adc1a2f81cba7b0432e3166840))

## 0.6.10 (2024-10-03)

### BREAKING CHANGE

- The `system` key in the input TOML has been renamed to `database`, and users must now specify the path for the database and export options under this new key. ([#fbd88e2](https://github.com/pol-sb/MatDBForge/commit/fbd88e23ee869691428d779da6ccd83da4dc2aed))
- `phase.name` is now slugified, which may affect downstream usage when accessing phase names. ([#a42ea14](https://github.com/pol-sb/MatDBForge/commit/a42ea145324f855c75cf3a84ebc74435733fcc91))

### Feat

- **core**: improve surface generation with additional replacement and saving options ([#643ec8c](https://github.com/pol-sb/MatDBForge/commit/643ec8c28b33811b60715f43f6e329d3910753f3))
- **core**: add displacement tracking and enhance structure generation ([#c63745d](https://github.com/pol-sb/MatDBForge/commit/c63745d63f3cc5382c45159350710f90ec40e8c0))
- **core**: enhance structure handling and surface generation logic ([#fa6ddf3](https://github.com/pol-sb/MatDBForge/commit/fa6ddf3cb8d4c1887c5f3c16b0f437404c1d9f5d))
- **core**: add phase structure limiting and improve phase handling ([#8a3e1a8](https://github.com/pol-sb/MatDBForge/commit/8a3e1a8cbc7985d0568cb4a375d1954bc467b256))
- **core**: add element-specific vacancy generation and improve installation docs ([#08d5694](https://github.com/pol-sb/MatDBForge/commit/08d5694cdbb897805a8775dfb06f96bd667913ad))
- **core**: add enhanced lattice perturbation and filtering options                                                                                                                                                   ─╯    - Enhanced `perturb_min_displacement` to support structure filtering, phase selection, and random number generator seeding for lattice perturbations.    - Introduced parameters `filters`, `only_use_base`, `use_phase`, and `rng_seed` to `perturb_min_displacement` for greater control over perturbation strategies.    - Improved structure type conversion in vacancy generation and perturbation methods by using `to_bulk()`, `to_surface()`, and `to_cluster()` methods.    - Updated string representations for `Structure` and `Phase` classes to improve readability of phase and structure details.    - Added support for limiting the number of structures in lattice displacement operations.    - Minor bug fixes and improvements to database and structure handling functions.    - `perturb_min_displacement` can now use different structures, although it still defaults to using base structures only unless specified otherwise to maintain compability. ([#b5626fc](https://github.com/pol-sb/MatDBForge/commit/b5626fcf58730a1e4be7ad84cee79609c86ad5f4))
- **core**: add deprecation utilities and enhance warnings handling ([#37312dc](https://github.com/pol-sb/MatDBForge/commit/37312dc2d1bd393be2c8789240978fbe7663e1b3))
- **database**: add support for random vacancy generation and structure enhancements ([#e04a3ac](https://github.com/pol-sb/MatDBForge/commit/e04a3ac0a6f8d8d5476ea43c062497516b4843b4))
- **database**: enhance database generation with plotting, exporting, and structure replacement options ([#fbd88e2](https://github.com/pol-sb/MatDBForge/commit/fbd88e23ee869691428d779da6ccd83da4dc2aed))
- **active_learning**: seed structure deletion is now optional in AL workflows ([#9617489](https://github.com/pol-sb/MatDBForge/commit/96174895213ed41b87592c39ef40f2e344867ea9))
- **utils**: add support for adsorbate addition on surfaces in database ([#146727f](https://github.com/pol-sb/MatDBForge/commit/146727fdfe10f2f842f58f99ee16f349a66341bd))
- **utils**: extend API key gathering to support environment variable ([#739e299](https://github.com/pol-sb/MatDBForge/commit/739e299ee38f3f8b219f29e00195f1fe714f3038))
- **database**: enhance phase diagram and database initialization logic ([#a42ea14](https://github.com/pol-sb/MatDBForge/commit/a42ea145324f855c75cf3a84ebc74435733fcc91))
- **phase_diagram**: upgrade phase diagram logic to support binary and ternary systems ([#21cc63e](https://github.com/pol-sb/MatDBForge/commit/21cc63e57b08407fd087e6d09764e868e874a29f))
- added first test templates ([#9352880](https://github.com/pol-sb/MatDBForge/commit/93528802017739106a59d2c2c3ba3959c3beb089))
- add `apply_replacement_no_db()` ([#f20e6bf](https://github.com/pol-sb/MatDBForge/commit/f20e6bf07cfad240ac43ec683cbf5a7089860bdf))
- LAMMPS MD dump data now stored as compressed files ([#0c95f2c](https://github.com/pol-sb/MatDBForge/commit/0c95f2cd5c766eaf69b1bfa0a880a707b835555f))
- new logging option + rich cli output when running dashboard ([#75cdf58](https://github.com/pol-sb/MatDBForge/commit/75cdf581a2253907512bd50bea3dc07d626effb6))
- added dev optional setup option ([#0ec4afe](https://github.com/pol-sb/MatDBForge/commit/0ec4afe8c70f1cecdbcfb5b162dba03603402cae))
- allow running dashboard in a separate process ([#a6d18ae](https://github.com/pol-sb/MatDBForge/commit/a6d18ae9cbf7f09b74d43241b47b7353d123c1e4))
- added precommits ([#5eeea1e](https://github.com/pol-sb/MatDBForge/commit/5eeea1e14cab762876bc7be1e57b6fbb666bb20f))
- enhance active learning workflow with new md filters and descriptor settings ([#fe2da24](https://github.com/pol-sb/MatDBForge/commit/fe2da243955248a6a6d22dd7ca062cbaeb0b1da5))

### Fix

- **gen_db**: fixed wrong argument for `gen_surfaces_diff_miller` ([#dc45ed8](https://github.com/pol-sb/MatDBForge/commit/dc45ed8fa03359ed97c653dd74ebcd7969769635))
- **logs**: database report now showing in logs ([#52bf135](https://github.com/pol-sb/MatDBForge/commit/52bf13564d0c9122e77809e0676c88de5bdba04c))
- **core**: fix phase name selection when filtering from db ([#6d9037e](https://github.com/pol-sb/MatDBForge/commit/6d9037ee37bc27744af83dfeed5fc0022dd5efc7))
- several fixes: - Reenabled ignore small structures during bulk generation. - Changed fixed seed with random generated one in case it is not given by the user. - Removed unused comments. ([#0240a46](https://github.com/pol-sb/MatDBForge/commit/0240a4626156e1eb1d1937c0c4280d60202dcca8))
- added conditional to avoid all types of perturbed structures being saved as cluster. ([#01fd2c0](https://github.com/pol-sb/MatDBForge/commit/01fd2c0ec1a93c0a392df8d64c3ffd6e0742637f))
- **cli**: update toml handling for Python 3.11 compatibility ([#afc94d4](https://github.com/pol-sb/MatDBForge/commit/afc94d4aa040030b6d43bc77b9c26079d457365a))
- changed `fix_bottom_layers` message to warning. ([#631f420](https://github.com/pol-sb/MatDBForge/commit/631f420928135411e86d7de187c3249ca935ef40))
- fixed database generation functions: - renamed `run_gen_initial_database` to `cli_run_gen_initial_database` - fixed import related to `cli_run_gen_initial_database` - converted numbers to float and int type after parsing database generation input .toml file - Added function for fixing bottom layers: `fix_bottom_layers`, however it is an unfinished which must be improved. It won't be used until finished. ([#f6e4466](https://github.com/pol-sb/MatDBForge/commit/f6e44665f16b551ba78ccb3fe745872058a6c30a))
- updated aiida-core dependency to avoid error #6519 ([#5be149b](https://github.com/pol-sb/MatDBForge/commit/5be149bbc34c62deb17e5e5f94b11cb3414a2c83))
- removed duplicate key ([#2ef361c](https://github.com/pol-sb/MatDBForge/commit/2ef361c565e35bb1edec55741420009c4b68bdab))
- fixed typo in send_calc_or_remove_structures()` ([#381a64a](https://github.com/pol-sb/MatDBForge/commit/381a64a71d2ac0e2ecf902ffd6f48042bbdadf20))
- added structure selection safeguard for `dft_structres` list ([#03f9779](https://github.com/pol-sb/MatDBForge/commit/03f9779b3555aab84c5aaadbe2c8e05ee30c523a))
- improved structure selection safeguard ([#c721582](https://github.com/pol-sb/MatDBForge/commit/c721582b13f685189614ee950556868e0a67d2cd))
- added safeguard check for filtered trajectories ([#e5ef27b](https://github.com/pol-sb/MatDBForge/commit/e5ef27b5875cc3df5601f1e31c66cca3b8a99aca))
- updated imports on test files ([#46467f2](https://github.com/pol-sb/MatDBForge/commit/46467f21ce014842488fa251d4d00dfc3cf48639))
- only load MACE models on first iteration ([#e82744b](https://github.com/pol-sb/MatDBForge/commit/e82744b1f93c46e1780544cf6cf0d2326e859a70))
- changes to `send_calc_or_remove_structures` to avoid broadcasting errors: - Added padding to extrapolating_frames array when size is different to error_f_structures or error_e_structures - Disabled shrinking from array masks to maintain array size. ([#295ff37](https://github.com/pol-sb/MatDBForge/commit/295ff37fca18bb47f60ba3dc184e41fe552f2eae))
- added different padding strategies for energy and forces in `model_res_dict_to_arr()` ([#f5b0b21](https://github.com/pol-sb/MatDBForge/commit/f5b0b219d93be56dcfa4070bb962f3dfaeccb56c))
- updated `model_res_dict_to_arr` ([#624c1f9](https://github.com/pol-sb/MatDBForge/commit/624c1f95749ec20a52c869101dca739b71d10dca))
- Changed dashboard progressbar header ([#2c3e9b0](https://github.com/pol-sb/MatDBForge/commit/2c3e9b0ce11713c0b711b14045337a56276a54d6))
- Changed dashboard progressbar header ([#cc22899](https://github.com/pol-sb/MatDBForge/commit/cc2289983bee20a85e3af1762fb5bd1f2a16dbf3))
- added nan padding to uneven model_res_dict_to_arr arrays ([#a75c095](https://github.com/pol-sb/MatDBForge/commit/a75c095c7e3d31a945cdcf6356ee19d293fb460c))
- updated dashboard launch mechanism ([#a2ae5be](https://github.com/pol-sb/MatDBForge/commit/a2ae5be51e9a909481e4a783aab50452e2b64b12))
- mace model as absolute path (iv) ([#f703582](https://github.com/pol-sb/MatDBForge/commit/f703582484e98afdd38a189d528a61544dcda4b7))
- mace model as absolute path (iii) ([#ac43885](https://github.com/pol-sb/MatDBForge/commit/ac4388551e8abd3d2e24e28c2f42f01e83c849fc))
- mace model as absolute path (ii) ([#b6e104c](https://github.com/pol-sb/MatDBForge/commit/b6e104c66328430558feac63bbc28a9f29909480))
- changed incorrect import in cli_generate_init_db.py ([#7355f16](https://github.com/pol-sb/MatDBForge/commit/7355f16f92941703f0efb6d7b10f6e9493143086))
- improved compatibility with tests ([#005f4b3](https://github.com/pol-sb/MatDBForge/commit/005f4b3bedb9fab65af37e2c0866d6d150a9ca51))
- resolve path on input files ([#df7a8b5](https://github.com/pol-sb/MatDBForge/commit/df7a8b52c1b04c3e50dbf4f28c5f4a9f8ff708e6))
- update documentation and modified wrong function call ([#69739d1](https://github.com/pol-sb/MatDBForge/commit/69739d1ee2d524ca0e735d29ee8e369c5516c3fa))
- update dashboard counter ([#1afc7d7](https://github.com/pol-sb/MatDBForge/commit/1afc7d7f5b34c3557210abe70777fceb029ed5e7))
- allow TrainMACEModelCalculationParser to gather any model type ([#d3d55b0](https://github.com/pol-sb/MatDBForge/commit/d3d55b01ca07bbac56fc571359f8eacaf0873d13))
- improved README.md writing ([#af6d94d](https://github.com/pol-sb/MatDBForge/commit/af6d94d02fe76ac89a3953e21a11a0bcf4bccc16))
- applied uniform formatting to most files in the library ([#daa62dd](https://github.com/pol-sb/MatDBForge/commit/daa62dd4c09a311965aa0d74f9b88a00e540aecf))
- improve iteration display and change progressbar iteration ([#e03b413](https://github.com/pol-sb/MatDBForge/commit/e03b41331515c56bb9de9422fc7414180fcb25f3))
- remove filtered md structures from db and fix descriptor executable ([#13f7b83](https://github.com/pol-sb/MatDBForge/commit/13f7b83074967988e93e3c51a7b9709af965ea41))
- add cwd in remote as path for PortableCodes to run ([#e0c3101](https://github.com/pol-sb/MatDBForge/commit/e0c3101c4273830a10cbe4349ab096587221af2f))
- changed forces unit ([#6ffd858](https://github.com/pol-sb/MatDBForge/commit/6ffd8581a867124dab7bd13d77eb1220e5e7fbbc))
- allow to use MACE_forces key on extxyz files ([#8d36397](https://github.com/pol-sb/MatDBForge/commit/8d36397c0a059c7fdbd20048bce12decf52f8e5f))
- allow to use functions from `aiida_utils` without using aiida ([#61bfad6](https://github.com/pol-sb/MatDBForge/commit/61bfad6f7be6ea07dbdadfcbce4d99af661ceeb2))

### Refactor

- **core**: replace deprecated structure conversion with new methods ([#f12aa96](https://github.com/pol-sb/MatDBForge/commit/f12aa96420401968bd7a7509c5f03f764062ae52))
- reduced line length on rich cli outputs ([#2f59483](https://github.com/pol-sb/MatDBForge/commit/2f5948334ce1dcfd81710a89fd863899b0839aec))
- renamed cli files ([#c5e4e73](https://github.com/pol-sb/MatDBForge/commit/c5e4e7358b449859dddab63cb046733ffc028de7))
- separated command line scripts into individual files ([#2df2e93](https://github.com/pol-sb/MatDBForge/commit/2df2e9368325d16fef4d96dc55fdf4113336a2ab))

## v0.3.9 (2024-06-19)
