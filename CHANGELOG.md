# Changelog

## 0.11.4 (2024-10-15)

### Fix

- **al_loop**: fixed `check_extrapolation_type` key being gathered from wrong dictionary ([#6386024](https://github.com/pol-sb/MatDBForge/commit/6386024a1c2c7a496b733bfa475b84b9f2a4b87e))

### Misc

- **docs**: added input description for new extrapolation settings. ([#7794c68](https://github.com/pol-sb/MatDBForge/commit/7794c68f2ab4e90ad4f64904b0bdd26d5a416fa6))
- **core**: added shapely dep. ([#05894e6](https://github.com/pol-sb/MatDBForge/commit/05894e6d593155ac7b5ee1dcb86f662f295a7623))

## 0.11.3 (2024-10-14)

### Fix

- **db_gen**: perturbed structures now save material id ([#e859d7e](https://github.com/pol-sb/MatDBForge/commit/e859d7e99ac0d6c09f1cd333e35e92634d73203c))

### Misc

- **db_gen**: added limit to number of phases in pie chart, remaining ones are grouped into `others`. ([#2fed433](https://github.com/pol-sb/MatDBForge/commit/2fed433d60cf11af274bbc511c957d2c126270cd))
- **db_gen**: improved output and added deprecation warnings ([#2f3b778](https://github.com/pol-sb/MatDBForge/commit/2f3b7786beb044905bb8934bb3f1fb77431e59eb))
- **structure**: simplified structure conversion ([#1575d91](https://github.com/pol-sb/MatDBForge/commit/1575d9183765b9a9729f71210f43f727f07aa722))
- **structure**: updated structure representation ([#af9a608](https://github.com/pol-sb/MatDBForge/commit/af9a60822878fc896bf3e87fb1490f50fd12f29c))
- **al_loop**: updated logger to rich logger ([#74ba05e](https://github.com/pol-sb/MatDBForge/commit/74ba05ef9fc516cddf53e147624679d3e779cc27))
- **core**: improved report when missing MDB config file ([#a4abfdd](https://github.com/pol-sb/MatDBForge/commit/a4abfddb7c01699ac1a37b94abe4eb6873e6c16a))

## 0.11.2 (2024-10-14)

### Fix

- **core**: added __version__ directly to __init__.py ([#5429dd1](https://github.com/pol-sb/MatDBForge/commit/5429dd163c00b34f8627754a6c7f941a99c47a94))

### Misc

- **docs**: updated formatting on CHANGELOG.md ([#7119a22](https://github.com/pol-sb/MatDBForge/commit/7119a22193eb01716bbe02a1f3cf6ebd9ecc6ccf))

## 0.11.1 (2024-10-14)

### Fix

- **al_loop**: added check when submitting MACE energy evaluation to avoid empty lists being submitted. ([#c5744a7](https://github.com/pol-sb/MatDBForge/commit/c5744a7f4bb275b3fbbde1dfa2b60a71d71769bc))
- **core**: fixed changelog format. ([#db3a77f](https://github.com/pol-sb/MatDBForge/commit/db3a77f28ca64ab2439bc6d13799d1b0f4dcffe1))

### Misc

- **core**: updated changelog format. ([#4686f22](https://github.com/pol-sb/MatDBForge/commit/4686f22aa43ab09bf62bbadd3f737a1880ca1ad1))

## 0.11.0 (2024-10-13)

### Feature

- **gen_db**: add bar chart and structure dict sorting. ([#9c7e277](https://github.com/pol-sb/MatDBForge/commit/9c7e2770dfcd4348a502d14a3726b43e5714eea8))
- **al_loop**: added function to plot concave hull during the active learning loop ([#8948657](https://github.com/pol-sb/MatDBForge/commit/8948657b402c541ed4d834c2b5fc0222d784ea9b))
- **al_loop**: add implementation of advanced extrapolation with concave hull and autoencoder support: ([#8ef5165](https://github.com/pol-sb/MatDBForge/commit/8ef51659a52ce30952986f7914ad8e94ca0b7a88))

### Fix

- **gen_db**: use rcparams to update matplotlib rcparams globally. ([#675e683](https://github.com/pol-sb/MatDBForge/commit/675e683b2b62616c619ebba2019e5acbf1818139))
- **gen_db**: allow runs without specifying optional paramaters. ([#6244b23](https://github.com/pol-sb/MatDBForge/commit/6244b23f983957c87997ea9b38260fd6180df52a))
- **al_loop**: added parsing for `multiheads_finetuning` key for MACE training ([#50fd490](https://github.com/pol-sb/MatDBForge/commit/50fd4902dd4f3e88bf2dd8f2e603a005f7f80a0b))

### Misc

- **gen_db**: rename central perturbation key. ([#b13304a](https://github.com/pol-sb/MatDBForge/commit/b13304ae4069b22fad44db77c88f8eee02db041a))
- **gen_db**: added debug print ([#c26dd9e](https://github.com/pol-sb/MatDBForge/commit/c26dd9e5da682dec93fc7c88704b263477eb4a24))
- **core**: pinned `mace-torch` library version ([#e40001e](https://github.com/pol-sb/MatDBForge/commit/e40001ee1dae14db1a76aa2b0c2501f9234d7cae))

## 0.10.0 (2024-10-08)

### Feature

- **extrapolation**: add rng seed for autoencoder training ([#5f9df8c](https://github.com/pol-sb/MatDBForge/commit/5f9df8cfd565c6a14d951baa01475f55b20e252a))
- **extrapolation**: include aiida entry points for autoencoder train and parsing ([#0868684](https://github.com/pol-sb/MatDBForge/commit/0868684f14bbf79b5ee029adac28ae2cc9d29b36))
- **extrapolation**: added entry point for autoencoder training ([#7cbc062](https://github.com/pol-sb/MatDBForge/commit/7cbc0629a71c9b7c7adef2dbbe27a57e071781bf))
- **db_gen**: added flag to avoid db overwriting, which adds a timestamp to the db_name ([#a6a61ee](https://github.com/pol-sb/MatDBForge/commit/a6a61ee10c98d87d3f19f0db33bd4fa045d7c724))

### Misc

- **core**: changed logginghandler to `RichHandler` ([#98954f9](https://github.com/pol-sb/MatDBForge/commit/98954f94dbf92ae2fb4d0aea1063557bdb5a9a0d))

## 0.9.1 (2024-10-08)

### Misc

- **core**: improved changelog generation ([#a99c322](https://github.com/pol-sb/MatDBForge/commit/a99c3227a5e70a7c8fee4205a4365ee8907631e4))

## 0.9.0 (2024-10-08)

### Feature

- **db_gen**: add parameter for selecting the composition plot image format. ([#ce24695](https://github.com/pol-sb/MatDBForge/commit/ce246956fae9c3a54541c70e9a98c0728ad3f9ef))

### Fix

- **gen_db**: Changed `overwrite_db` default to false ([#8978885](https://github.com/pol-sb/MatDBForge/commit/89788855edbd0c66b918459ee5b28384a1566a9e))
- **structure**: Removed deprecated function ([#626001b](https://github.com/pol-sb/MatDBForge/commit/626001b524fdb5b88fd1e729f677f92958f32d86))

### Misc

- **core**: added changelog template ([#d8f3215](https://github.com/pol-sb/MatDBForge/commit/d8f32151e052d1dd6cdfca7e9f34a9b1ebf18a55))
- **gen_db**: improved database generation plot ([#17df15e](https://github.com/pol-sb/MatDBForge/commit/17df15e91a81f9f4b013a3bb732cf4f8d43a0e5e))
- **core**: increased clarity of update warning ([#a26f3da](https://github.com/pol-sb/MatDBForge/commit/a26f3da318bd5c039e9ccc38752750de585f22a7))
- **core**: updated CHANGELOG.md ([#b5f59d4](https://github.com/pol-sb/MatDBForge/commit/b5f59d41f73baa6f6ceec3e5fc255a4295eb16b0))

## 0.8.0 (2024-10-07)

### Feature

- **db_gen**: added function to check for updates ([#b0e73ab](https://github.com/pol-sb/MatDBForge/commit/b0e73abfc50e2755c11b7a1714d09ddb944d3925))

### Fix

- **db_gen**: fixed bug that prevented creation of 'central_atom_perturbation' modified structures ([#f31d6c4](https://github.com/pol-sb/MatDBForge/commit/f31d6c4b3aaafc82649c31c6b7272f01df62abc6))
- **db_gen**: improved pie charts ([#d9d996d](https://github.com/pol-sb/MatDBForge/commit/d9d996d1c6c48a23c6916f350375c1b856f39b5a))
- **db_gen**: refactored code to avoid circular imports ([#5c682a1](https://github.com/pol-sb/MatDBForge/commit/5c682a1624459df6f58de3ca5d947baec82a08d6))

## 0.7.0 (2024-10-07)

### Feature

- **db_gen**: add parameter for disallowing database overwrite. ([#3a69386](https://github.com/pol-sb/MatDBForge/commit/3a6938675c184a829a5bab907a86d8bb500d7472))
- **db_gen**: added pie charts to composition plots ([#594e3c6](https://github.com/pol-sb/MatDBForge/commit/594e3c61831fef6149c1aee2ffc90a028526fa50))

### Fix

- **db_gen**: increased consistency with uuid and phase label storage. ([#7c38868](https://github.com/pol-sb/MatDBForge/commit/7c3886853d09353aaa8a203bee57b22df90af69f))
- **db_gen**: increased consistency when saving structures with `targeted_modification`. ([#331263e](https://github.com/pol-sb/MatDBForge/commit/331263efbd3e8e8ae1fa43eedf33aa107e8c612e))

## 0.6.11 (2024-10-06)

### Feature

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

### Feature

- **core**: improve surface generation with additional replacement and saving options ([#643ec8c](https://github.com/pol-sb/MatDBForge/commit/643ec8c28b33811b60715f43f6e329d3910753f3))
- **core**: add displacement tracking and enhance structure generation ([#c63745d](https://github.com/pol-sb/MatDBForge/commit/c63745d63f3cc5382c45159350710f90ec40e8c0))
- **core**: enhance structure handling and surface generation logic ([#fa6ddf3](https://github.com/pol-sb/MatDBForge/commit/fa6ddf3cb8d4c1887c5f3c16b0f437404c1d9f5d))
- **core**: add phase structure limiting and improve phase handling ([#8a3e1a8](https://github.com/pol-sb/MatDBForge/commit/8a3e1a8cbc7985d0568cb4a375d1954bc467b256))
- **core**: add element-specific vacancy generation and improve installation docs ([#08d5694](https://github.com/pol-sb/MatDBForge/commit/08d5694cdbb897805a8775dfb06f96bd667913ad))
- **core**: add enhanced lattice perturbation and filtering options: ([#b5626fc](https://github.com/pol-sb/MatDBForge/commit/b5626fcf58730a1e4be7ad84cee79609c86ad5f4))
  - Enhanced `perturb_min_displacement` to support structure filtering, phase selection, and random number generator seeding for lattice perturbations.
  - Introduced parameters `filters`, `only_use_base`, `use_phase`, and `rng_seed` to `perturb_min_displacement` for greater control over perturbation strategies.
  - Improved structure type conversion in vacancy generation and perturbation methods by using `to_bulk()`, `to_surface()`, and `to_cluster()` methods.
  - Updated string representations for `Structure` and `Phase` classes to improve readability of phase and structure details.
  - Added support for limiting the number of structures in lattice displacement operations.
  - Minor bug fixes and improvements to database and structure handling functions.    - `perturb_min_displacement` can now use different structures, although it still defaults to using base structures only unless specified otherwise to maintain compability. ([#b5626fc](https://github.com/pol-sb/MatDBForge/commit/b5626fcf58730a1e4be7ad84cee79609c86ad5f4))
- **core**: add deprecation utilities and enhance warnings handling ([#37312dc](https://github.com/pol-sb/MatDBForge/commit/37312dc2d1bd393be2c8789240978fbe7663e1b3))
- **database**: add support for random vacancy generation and structure enhancements ([#e04a3ac](https://github.com/pol-sb/MatDBForge/commit/e04a3ac0a6f8d8d5476ea43c062497516b4843b4))
- **database**: enhance database generation with plotting, exporting, and structure replacement options ([#fbd88e2](https://github.com/pol-sb/MatDBForge/commit/fbd88e23ee869691428d779da6ccd83da4dc2aed))
- **active_learning**: seed structure deletion is now optional in AL workflows ([#9617489](https://github.com/pol-sb/MatDBForge/commit/96174895213ed41b87592c39ef40f2e344867ea9))
- **utils**: add support for adsorbate addition on surfaces in database ([#146727f](https://github.com/pol-sb/MatDBForge/commit/146727fdfe10f2f842f58f99ee16f349a66341bd))
- **utils**: extend API key gathering to support environment variable ([#739e299](https://github.com/pol-sb/MatDBForge/commit/739e299ee38f3f8b219f29e00195f1fe714f3038))
- **database**: enhance phase diagram and database initialization logic ([#a42ea14](https://github.com/pol-sb/MatDBForge/commit/a42ea145324f855c75cf3a84ebc74435733fcc91))
- **phase_diagram**: upgrade phase diagram logic to support binary and ternary systems ([#21cc63e](https://github.com/pol-sb/MatDBForge/commit/21cc63e57b08407fd087e6d09764e868e874a29f))
- **core**: added first test templates ([#9352880](https://github.com/pol-sb/MatDBForge/commit/93528802017739106a59d2c2c3ba3959c3beb089))
- **gen_db**: add `apply_replacement_no_db()` ([#f20e6bf](https://github.com/pol-sb/MatDBForge/commit/f20e6bf07cfad240ac43ec683cbf5a7089860bdf))
- **active_learning**: LAMMPS MD dump data now stored as compressed files ([#0c95f2c](https://github.com/pol-sb/MatDBForge/commit/0c95f2cd5c766eaf69b1bfa0a880a707b835555f))
- **dashboard**: new logging option + rich cli output when running dashboard ([#75cdf58](https://github.com/pol-sb/MatDBForge/commit/75cdf581a2253907512bd50bea3dc07d626effb6))
- **core**: added dev optional setup option ([#0ec4afe](https://github.com/pol-sb/MatDBForge/commit/0ec4afe8c70f1cecdbcfb5b162dba03603402cae))
- **dashboard**: allow running dashboard in a separate process ([#a6d18ae](https://github.com/pol-sb/MatDBForge/commit/a6d18ae9cbf7f09b74d43241b47b7353d123c1e4))
- **core**: added precommits ([#5eeea1e](https://github.com/pol-sb/MatDBForge/commit/5eeea1e14cab762876bc7be1e57b6fbb666bb20f))
- **active_learning**: enhance active learning workflow with new md filters and descriptor settings ([#fe2da24](https://github.com/pol-sb/MatDBForge/commit/fe2da243955248a6a6d22dd7ca062cbaeb0b1da5))

### Fix

- **gen_db**: fixed wrong argument for `gen_surfaces_diff_miller` ([#dc45ed8](https://github.com/pol-sb/MatDBForge/commit/dc45ed8fa03359ed97c653dd74ebcd7969769635))
- **logs**: database report now showing in logs ([#52bf135](https://github.com/pol-sb/MatDBForge/commit/52bf13564d0c9122e77809e0676c88de5bdba04c))
- **core**: fix phase name selection when filtering from db ([#6d9037e](https://github.com/pol-sb/MatDBForge/commit/6d9037ee37bc27744af83dfeed5fc0022dd5efc7))
- **gen_db**: several fixes: ([#0240a46](https://github.com/pol-sb/MatDBForge/commit/0240a4626156e1eb1d1937c0c4280d60202dcca8))
  - Reenabled ignore small structures during bulk generation.
  - Changed fixed seed with random generated one in case it is not given by the user.
  - Removed unused comments.
- **gen_db**: added conditional to avoid all types of perturbed structures being saved as cluster. ([#01fd2c0](https://github.com/pol-sb/MatDBForge/commit/01fd2c0ec1a93c0a392df8d64c3ffd6e0742637f))
- **cli**: update toml handling for Python 3.11 compatibility ([#afc94d4](https://github.com/pol-sb/MatDBForge/commit/afc94d4aa040030b6d43bc77b9c26079d457365a))
- **gen_db**: changed `fix_bottom_layers` message to warning. ([#631f420](https://github.com/pol-sb/MatDBForge/commit/631f420928135411e86d7de187c3249ca935ef40))
- **gen_db**: fixed database generation functions: ([#f6e4466](https://github.com/pol-sb/MatDBForge/commit/f6e44665f16b551ba78ccb3fe745872058a6c30a))
  - renamed `run_gen_initial_database` to `cli_run_gen_initial_database`
  - fixed import related to `cli_run_gen_initial_database`
  - converted numbers to float and int type after parsing database generation input .toml file
  - Added function for fixing bottom layers: `fix_bottom_layers`, however it is an unfinished which must be improved. It won't be used until finished.

- **core**: updated aiida-core dependency to avoid error #6519 ([#5be149b](https://github.com/pol-sb/MatDBForge/commit/5be149bbc34c62deb17e5e5f94b11cb3414a2c83))
- **active_learning**: removed duplicate key ([#2ef361c](https://github.com/pol-sb/MatDBForge/commit/2ef361c565e35bb1edec55741420009c4b68bdab))
- **active_learning**: fixed typo in send_calc_or_remove_structures()` ([#381a64a](https://github.com/pol-sb/MatDBForge/commit/381a64a71d2ac0e2ecf902ffd6f48042bbdadf20))
- **active_learning**: added structure selection safeguard for `dft_structres` list ([#03f9779](https://github.com/pol-sb/MatDBForge/commit/03f9779b3555aab84c5aaadbe2c8e05ee30c523a))
- **active_learning**: improved structure selection safeguard ([#c721582](https://github.com/pol-sb/MatDBForge/commit/c721582b13f685189614ee950556868e0a67d2cd))
- **active_learning**: added safeguard check for filtered trajectories ([#e5ef27b](https://github.com/pol-sb/MatDBForge/commit/e5ef27b5875cc3df5601f1e31c66cca3b8a99aca))
- **core**: updated imports on test files ([#46467f2](https://github.com/pol-sb/MatDBForge/commit/46467f21ce014842488fa251d4d00dfc3cf48639))
- **active_learning**: only load MACE models on first iteration ([#e82744b](https://github.com/pol-sb/MatDBForge/commit/e82744b1f93c46e1780544cf6cf0d2326e859a70))
- **active_learning**: changes to `send_calc_or_remove_structures` to avoid broadcasting errors: ([#295ff37](https://github.com/pol-sb/MatDBForge/commit/295ff37fca18bb47f60ba3dc184e41fe552f2eae))
  - Added padding to extrapolating_frames array when size is different to error_f_structures or error_e_structures
  - Disabled shrinking from array masks to maintain array size.
- **active_learning**: added different padding strategies for energy and forces in `model_res_dict_to_arr()` ([#f5b0b21](https://github.com/pol-sb/MatDBForge/commit/f5b0b219d93be56dcfa4070bb962f3dfaeccb56c))
- **active_learning**: updated `model_res_dict_to_arr` ([#624c1f9](https://github.com/pol-sb/MatDBForge/commit/624c1f95749ec20a52c869101dca739b71d10dca))
- **dashboard**: Changed dashboard progressbar header ([#2c3e9b0](https://github.com/pol-sb/MatDBForge/commit/2c3e9b0ce11713c0b711b14045337a56276a54d6))
- **dashboard**: Changed dashboard progressbar header ([#cc22899](https://github.com/pol-sb/MatDBForge/commit/cc2289983bee20a85e3af1762fb5bd1f2a16dbf3))
- **active_learning**: added nan padding to uneven model_res_dict_to_arr arrays ([#a75c095](https://github.com/pol-sb/MatDBForge/commit/a75c095c7e3d31a945cdcf6356ee19d293fb460c))
- **dashboard**: updated dashboard launch mechanism ([#a2ae5be](https://github.com/pol-sb/MatDBForge/commit/a2ae5be51e9a909481e4a783aab50452e2b64b12))
- **active_learning**:mace model as absolute path (iv) ([#f703582](https://github.com/pol-sb/MatDBForge/commit/f703582484e98afdd38a189d528a61544dcda4b7))
- **active_learning**:mace model as absolute path (iii) ([#ac43885](https://github.com/pol-sb/MatDBForge/commit/ac4388551e8abd3d2e24e28c2f42f01e83c849fc))
- **active_learning**:mace model as absolute path (ii) ([#b6e104c](https://github.com/pol-sb/MatDBForge/commit/b6e104c66328430558feac63bbc28a9f29909480))
- **gen_db**: changed incorrect import in cli_generate_init_db.py ([#7355f16](https://github.com/pol-sb/MatDBForge/commit/7355f16f92941703f0efb6d7b10f6e9493143086))
- **core**: improved compatibility with tests ([#005f4b3](https://github.com/pol-sb/MatDBForge/commit/005f4b3bedb9fab65af37e2c0866d6d150a9ca51))
- **active_learning**: resolve path on input files ([#df7a8b5](https://github.com/pol-sb/MatDBForge/commit/df7a8b52c1b04c3e50dbf4f28c5f4a9f8ff708e6))
- **docs**: update documentation and modified wrong function call ([#69739d1](https://github.com/pol-sb/MatDBForge/commit/69739d1ee2d524ca0e735d29ee8e369c5516c3fa))
- **dashboard**: update dashboard counter ([#1afc7d7](https://github.com/pol-sb/MatDBForge/commit/1afc7d7f5b34c3557210abe70777fceb029ed5e7))
- **active_learning**: allow TrainMACEModelCalculationParser to gather any model type ([#d3d55b0](https://github.com/pol-sb/MatDBForge/commit/d3d55b01ca07bbac56fc571359f8eacaf0873d13))
- **docs**: improved README.md writing ([#af6d94d](https://github.com/pol-sb/MatDBForge/commit/af6d94d02fe76ac89a3953e21a11a0bcf4bccc16))
- **core**: applied uniform formatting to most files in the library ([#daa62dd](https://github.com/pol-sb/MatDBForge/commit/daa62dd4c09a311965aa0d74f9b88a00e540aecf))
- **dashboard**: improve iteration display and change progressbar iteration ([#e03b413](https://github.com/pol-sb/MatDBForge/commit/e03b41331515c56bb9de9422fc7414180fcb25f3))
- **active_learning**: remove filtered md structures from db and fix descriptor executable ([#13f7b83](https://github.com/pol-sb/MatDBForge/commit/13f7b83074967988e93e3c51a7b9709af965ea41))
- **active_learning**: add cwd in remote as path for PortableCodes to run ([#e0c3101](https://github.com/pol-sb/MatDBForge/commit/e0c3101c4273830a10cbe4349ab096587221af2f))
- **active_learning**: changed forces unit ([#6ffd858](https://github.com/pol-sb/MatDBForge/commit/6ffd8581a867124dab7bd13d77eb1220e5e7fbbc))
- **active_learning**: allow to use MACE_forces key on extxyz files ([#8d36397](https://github.com/pol-sb/MatDBForge/commit/8d36397c0a059c7fdbd20048bce12decf52f8e5f))
- **core**: allow to use functions from `aiida_utils` without using aiida ([#61bfad6](https://github.com/pol-sb/MatDBForge/commit/61bfad6f7be6ea07dbdadfcbce4d99af661ceeb2))

### Refactor

- **core**: replace deprecated structure conversion with new methods ([#f12aa96](https://github.com/pol-sb/MatDBForge/commit/f12aa96420401968bd7a7509c5f03f764062ae52))
- **core**: reduced line length on rich cli outputs ([#2f59483](https://github.com/pol-sb/MatDBForge/commit/2f5948334ce1dcfd81710a89fd863899b0839aec))
- **core**: renamed cli files ([#c5e4e73](https://github.com/pol-sb/MatDBForge/commit/c5e4e7358b449859dddab63cb046733ffc028de7))
- **core**: separated command line scripts into individual files ([#2df2e93](https://github.com/pol-sb/MatDBForge/commit/2df2e9368325d16fef4d96dc55fdf4113336a2ab))

## v0.3.9 (2024-06-19)
development version
