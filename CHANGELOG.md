# Changelog

## 0.20.3 (2025-01-28)

### Fix

- **al_loop**: fix incorrect latent space saving generating huge descriptor files ([#7a8c469](https://github.com/pol-sb/MatDBForge/commit/7a8c469702f4d531ac3cc16a68aab787349214a7))
- **al_loop**: enable containerized code in simple active learning loop ([#664f697](https://github.com/pol-sb/MatDBForge/commit/664f697278ba04cff3048b8a88167d3e49c1598e))

## 0.20.2 (2025-01-24)

### Fix

- **dft**: increased number of steps for IsolatedAtoms calculations ([#afbd09b](https://github.com/pol-sb/MatDBForge/commit/afbd09bb3f850c7e5bf608a054de9c73e4130557))
- **al_loop**: made code compatible with MACE multihead replay fine-tuning ([#08944e2](https://github.com/pol-sb/MatDBForge/commit/08944e2f7d63255ce7f6a52cccf13dd86469701c))
- **dft**: added IsolatedAtoms to initial database generation ([#96395ed](https://github.com/pol-sb/MatDBForge/commit/96395ed190c8efcb00850e82c1521270929e7827))

### Misc

- **core**: updated documentation badge ([#79f34bf](https://github.com/pol-sb/MatDBForge/commit/79f34bf7b40180b1226c9718058db8f066b3505f))

## 0.20.1 (2025-01-23)

### Fix

- **al_loop**: added TODO block with outlier filtering for VASP calculations ([#b592722](https://github.com/pol-sb/MatDBForge/commit/b59272205594d5d7bb2cfec8663ffbcbf0d126d2))
- **al_loop**: added error when evaluating structures with missing element in MACE models ([#8c1d97a](https://github.com/pol-sb/MatDBForge/commit/8c1d97a25a077c931cf2f9674be3aae711132a46))
- **al_loop**: improve error handling when mace evaluations fail ([#25bdb8c](https://github.com/pol-sb/MatDBForge/commit/25bdb8c760ee91942ba3864c6ee2cd9dd19f4bc8))
- **core**: added default "unknown" version to version check in case it fails ([#df94bde](https://github.com/pol-sb/MatDBForge/commit/df94bde589e5cc48cde8992df4a8c311c81cb82f))

### Misc

- **al_loop**: applied formatting and improved log messages ([#d53e86d](https://github.com/pol-sb/MatDBForge/commit/d53e86d65545ffa227804f6998ce98d92416cb73))
- **core**: updated pre-commit config ([#9c87a53](https://github.com/pol-sb/MatDBForge/commit/9c87a53e80f6ff9e414c70ee3552de17cb34e880))

## 0.20.0 (2025-01-20)

### Feature

- **al_loop**: add `mdb_check_descr_combined.py` script for descriptor gathering and extrapolation check for simple active learning loop: - Introduced `mdb_check_descr_combined.py`, a comprehensive script for checking extrapolation in the current iteration. - Implements descriptor generation using MACE and optional dimensionality reduction via an autoencoder. - Supports advanced extrapolation checks, including concave hull computation and visualization using `GMT.jl` and `ConcaveHull.jl`. ([#784ecdc](https://github.com/pol-sb/MatDBForge/commit/784ecdc85a11691dbe426e3e90b9e4284ed930b7))
- **autoencoder**: Make autoencoder training compatible with on the fly active learning loop and provide improved path handling and error checks: ([#2b07fcb](https://github.com/pol-sb/MatDBForge/commit/2b07fcb9f256bbd2d610203fe234faaa5d7d73ba))
- **al_loop**: Enhance MD structure processing in `mdb_process_structure.py` ([#4f3a44b](https://github.com/pol-sb/MatDBForge/commit/4f3a44b0939d2586ea72da93da790ff361e4ecae))
- **al_loop**: Add combined descriptor generation and concave hull analysis to simplified active learning loop ([#2943d0c](https://github.com/pol-sb/MatDBForge/commit/2943d0cd681472581322d9f6c50939e2b8fffd2a))

### Fix

- **md**: add minimum number of atoms safeguard to `check_disconn_neighbors` md filter. ([#82caa11](https://github.com/pol-sb/MatDBForge/commit/82caa11a679dcd38d39e31fab4dbc6d45aae181c))
- **core**: improve version check ([#6e94117](https://github.com/pol-sb/MatDBForge/commit/6e941170f2589137c6cbc2a6dfcbb618b8988e7d))
- **al_loop**: Working ver. for `ProcessMDSeedStructCalculation` and `GetDescriptorsCombined` aling their parsers. ([#2947e88](https://github.com/pol-sb/MatDBForge/commit/2947e889d9853da9835ebdbadadb7299b226c6ef))
- **docs**: fixed versions.html jinja template with missing elif statement ([#c53bfca](https://github.com/pol-sb/MatDBForge/commit/c53bfca9fafd4d1f26b75677a392e55531e716c3))
- **docs**: updated README.md documentation links to master branch of documentation ([#9232175](https://github.com/pol-sb/MatDBForge/commit/9232175caace654ed2634a5910d3157aba83d4f5))
- **dft**: fixed typo in `run_dataframe_vasp_aiida_queue` ([#e92dd82](https://github.com/pol-sb/MatDBForge/commit/e92dd82e70f77ca963103ce6ab495d3a332d1cc7))

### Misc

- **docs**: update docs templates and config ([#2713b52](https://github.com/pol-sb/MatDBForge/commit/2713b5264ba07c2cda68638dec16b26ef1ef9703))
- **docs**: add short commit hash to master/main branch in documentation ([#e541bfe](https://github.com/pol-sb/MatDBForge/commit/e541bfee1ad73029a71419777af371d1b39a40ce))
- **dft**: added aiida-vasp parsing settings to `dft_settings.toml` ([#2f07c1c](https://github.com/pol-sb/MatDBForge/commit/2f07c1c04040d85fce9886b825f136f6479e0e11))
- **core**: updated documentation.yml for gh-actions ([#1c63ed7](https://github.com/pol-sb/MatDBForge/commit/1c63ed707e60050491902582d9ddadc2ceca04ed))
- **core**: updated documentation.yml for gh-actions ([#7ba647a](https://github.com/pol-sb/MatDBForge/commit/7ba647a6b7e43fb1e5b4b331d336200353d0ed68))
- **core**: add documentation badge to README ([#631f5af](https://github.com/pol-sb/MatDBForge/commit/631f5af320e69866aa1b33aa7fc41095f1efe502))
- **core**: updated documentation.yml for gh-actions ([#30b0c89](https://github.com/pol-sb/MatDBForge/commit/30b0c89f26e2af57cafdca0c721fac338c4d5740))

## 0.19.12 (2024-12-28)

### Misc

- **core**: updated documentation.yml for gh-actions ([#2b1106b](https://github.com/pol-sb/MatDBForge/commit/2b1106b7c34cb717367b06e9e3015e8d54745322))

## 0.19.11 (2024-12-28)

### Misc

- **core**: add versioning information to sphinx documentation ([#036379d](https://github.com/pol-sb/MatDBForge/commit/036379dad1aedcb158d6258a6a5f456cf0db7708))

## 0.19.10 (2024-12-28)

### Fix

- **dft**: add `aiida_vasp.parser_settings` as input. Improve logging. ([#ff6f05a](https://github.com/pol-sb/MatDBForge/commit/ff6f05a2b9fd4d2fa93f0774b8b4c5f4f73bdd0d))

### Misc

- **dft**: add structure count to  running calculations print statement ([#c63939d](https://github.com/pol-sb/MatDBForge/commit/c63939dac44b644c13a6c084ba496d7cb44253b3))

## 0.19.9 (2024-12-23)

### Fix

- **core**: reformat `aiida_utils` and fix errors: - address indexing error when skipping completed calculations - create new structure and assign to list when updating in `update_db_with_dft_results` ([#4049eff](https://github.com/pol-sb/MatDBForge/commit/4049effdbc861d27ded9037b207795446b76d575))
- **core**: add tags argument to  git fetch command for update checker ([#30b3d4f](https://github.com/pol-sb/MatDBForge/commit/30b3d4fc219e9057685a1d03017e896d97500bba))

## 0.19.8 (2024-12-20)

### Fix

- **dft**: fix issue with phase collection ([#8dac04a](https://github.com/pol-sb/MatDBForge/commit/8dac04a08a45b391552fc5199a829bb5e1095e10))

### Misc

- **dft**: reformatted `conversion.py` and updated call to extras on `get_struct_type()` ([#4434341](https://github.com/pol-sb/MatDBForge/commit/44343413a799a6d52cea571a90a29c122ecc4796))

## 0.19.7 (2024-12-20)

### Fix

- **dft**: fixed unfinished loops and changed logging information during dft calculations ([#36262e3](https://github.com/pol-sb/MatDBForge/commit/36262e36b0b0fa69134a916b16d04072078c9b24))

### Misc

- **core**: reenabled 'empty' custom print type ([#236570a](https://github.com/pol-sb/MatDBForge/commit/236570a56181501556c84f1f4540f4f5a32d65c0))

## 0.19.6 (2024-12-19)

### Fix

- **dft**: enable usage of `calc_performed` to skip completed structures ([#28ec65e](https://github.com/pol-sb/MatDBForge/commit/28ec65e4a5b62fb6edc3ff7df0822896b4be5379))

## 0.19.5 (2024-12-18)

### Fix

- **dft**: simplified calculation skipping logic in `update_db_with_dft_results()` and added try/except block to active_learning_utils ([#87267a3](https://github.com/pol-sb/MatDBForge/commit/87267a3a78c29d817413503d309c686303030ab5))

## 0.19.4 (2024-12-17)

### Fix

- **dft**: changed logic for broken calculations in `update_db_with_dft_results()` ([#f717470](https://github.com/pol-sb/MatDBForge/commit/f717470f51276b6082f2fd8ccad00afcd4cdc646))

## 0.19.3 (2024-12-17)

### Fix

- **dft**: changed `finished` with `finished_ok` in `update_db_with_dft_results` check ([#1128626](https://github.com/pol-sb/MatDBForge/commit/1128626d7fecdeb0cf031a85efcd289220b28f7b))

## 0.19.2 (2024-12-16)

### Fix

- **dft**: added try block for different result gathering structures ([#dabc40f](https://github.com/pol-sb/MatDBForge/commit/dabc40fdc67d4224b54eb5ddf5e32d3259298076))

## 0.19.1 (2024-12-14)

### Fix

- **dft**: added `update_db_with_dft_results` function to save DFT simulation results into an output file ([#f701b81](https://github.com/pol-sb/MatDBForge/commit/f701b819f1b4064297b5c3c7237f07c5c0252708))

## 0.19.0 (2024-12-14)

### Feature

- **gen_db**: new adsorbate placement strategy (thank you @lll0606) and several improvements to dastabase generation ([#315a42b](https://github.com/pol-sb/MatDBForge/commit/315a42b125cc6e4b01b3a17b8b091d0f5e26b518))
- **dft**: added `mdb_run_dft_database` to entry points ([#5dfa2aa](https://github.com/pol-sb/MatDBForge/commit/5dfa2aa1c004694f4652d5f399f1b918f86065bc))
- **dft**: introduced `mdb_run_dft_database` utility to run dft calculations for mdb databases ([#c342594](https://github.com/pol-sb/MatDBForge/commit/c3425944e872939d385c3ac6303c83a01b8bb3cd))

### Fix

- **core**: removed forced logger handler clearing from git tag gathering strategy ([#394a6b5](https://github.com/pol-sb/MatDBForge/commit/394a6b52e2807af1fd1e5d437929dcc90f696f14))

### Misc

- **core**: updated clarity of several exceptions ([#610d435](https://github.com/pol-sb/MatDBForge/commit/610d43579e545424f6f0436854899e8c4702ec0f))

## 0.18.3 (2024-12-04)

### Fix

- **al_loop**: reintroduced standard version of `TrainMACEModelCalculationParser()` ([#016279c](https://github.com/pol-sb/MatDBForge/commit/016279c7f51a6ac6777347442203418c014e93b9))

## 0.18.2 (2024-12-02)

### Fix

- **al_loop**: add prepend_path to mdb_process_structure to allow compatibility with docker ([#80a1289](https://github.com/pol-sb/MatDBForge/commit/80a1289093b25f3a1e5e2db4499f6a0e9bdbd0ab))
- **al_loop**: change `get_concave_hull` aiida computer definition in `SimpleActiveLearningWorkChain` ([#1f391b6](https://github.com/pol-sb/MatDBForge/commit/1f391b6283c428ab0e8c2c9bd5c2bc3c7d01711a))

## 0.18.1 (2024-11-28)

### Fix

- **al_loop**: moved juliacall deps. to `get_concave_hull_julia()` ([#9854aed](https://github.com/pol-sb/MatDBForge/commit/9854aedeb84e165b462759c40e5eb91495a8ac21))
- **al_loop**: re-added wandb dependency for autoencoder training ([#ab9bddb](https://github.com/pol-sb/MatDBForge/commit/ab9bddb4014def18aa1da63ab2e4c8131f9193e9))

### Misc

- **al_loop**: added message marking start of procedure to `mdb_process_structure` script ([#04d617e](https://github.com/pol-sb/MatDBForge/commit/04d617e85d2fb143becec52d2964ba7bafbd7f5f))

## 0.18.0 (2024-11-27)

### Feature

- **al_loop**: Included Maxwell-Boltzmann initialization of velocities for simple MD. ([#ffaa323](https://github.com/pol-sb/MatDBForge/commit/ffaa3236eba82f9af36804d363c3fe539d1dea7f))

### Fix

- **initial_db**: commented julia imports from beginning of file to avoid errors on installs not using julia. ([#3111077](https://github.com/pol-sb/MatDBForge/commit/3111077a6be9eb8c7f7731a60b76170d45ff9a86))
- **al_loop**: allow to generate plot for resumed calculations ([#477619e](https://github.com/pol-sb/MatDBForge/commit/477619e3451404e9dc2b4da40ef72d1d117299dd))
- **al_loop**: improved active learning report plot ([#64e3107](https://github.com/pol-sb/MatDBForge/commit/64e310768042010cdbf187a3211525f986c48bd8))
- **al_loop**: removed hardcoded value in active_learning_utils ([#19186fe](https://github.com/pol-sb/MatDBForge/commit/19186feff4d9a1c7cf936942c0a79e3a26922a00))
- **al_loop**: modified descriptor get settings ([#00dc19e](https://github.com/pol-sb/MatDBForge/commit/00dc19ef10ac28b400a52a906fb453d7a77c18d8))
- **al_loop**: added alternative E and F gathering in gather_md_E_F_data ([#78fcc91](https://github.com/pol-sb/MatDBForge/commit/78fcc91b38079e5d709df603dcc8f1476b84ebb2))
- **al_loop**: improved `active_learning_utils.py` so they are more general ([#247ce3f](https://github.com/pol-sb/MatDBForge/commit/247ce3f690ee55eed58e17b21dd220833f73e617))
- **al_loop**: replaced hardcoded settings with parameter dict for descriptor gathering ([#25c1433](https://github.com/pol-sb/MatDBForge/commit/25c14339c0beab1765df0ea6c7d6aa49ec6fb885))
- **al_loop**: changed parameter gathering so that `mdb_process_structure` works on non-gpu machines ([#fdccbc7](https://github.com/pol-sb/MatDBForge/commit/fdccbc79c4bf8b1cba33617e95f6456843df1cf3))

### Misc

- **al_loop**: add inputs for enhanced error plot generation ([#2190828](https://github.com/pol-sb/MatDBForge/commit/2190828a163dc0d0df8f86b8a6433dd0e5ff4cb9))
- **al_loop**: improve error plot generation ([#96699bf](https://github.com/pol-sb/MatDBForge/commit/96699bf833f8b0880ebea0dc35a3c7f18e44c700))
- **al_loop**: move error plot to default al_loop report plot ([#0577b60](https://github.com/pol-sb/MatDBForge/commit/0577b609ef17a1e90f26c2c3bba37baca6822112))
- **docs**: update README.md ([#6b26420](https://github.com/pol-sb/MatDBForge/commit/6b26420067cc617686dba75957c67e7d099734bd))
- **docs**: update install docs, bump up python version and add julia ([#e09ccc8](https://github.com/pol-sb/MatDBForge/commit/e09ccc81cccba2a295f74c6b26a8549549fdbdd7))
- **core**: refactored all python scripts using new ruff settings ([#4d010b2](https://github.com/pol-sb/MatDBForge/commit/4d010b20fd31cf4d79ade6a66d9ed573577e6ebd))
- **core**: moved ruff settings to pyproject.toml and updated precommit configuration ([#655b2fc](https://github.com/pol-sb/MatDBForge/commit/655b2fcde1218ab9f6229c8668e476868dbf3fac))

## 0.17.0 (2024-11-08)

### Feature

- **al_loop**: introduce first version of simplified active learning workchain ([#14e4332](https://github.com/pol-sb/MatDBForge/commit/14e4332cb612715f102a8d7cd44a76fa1c7152c5))

## 0.16.6 (2024-11-06)

### Fix

- **al_loop**: fixed report logger printing double text ([#19859ab](https://github.com/pol-sb/MatDBForge/commit/19859abaac8dbb5a8f610aa845afa1b49effa3ff))
- **al_loop**: improved report plot ([#cc28601](https://github.com/pol-sb/MatDBForge/commit/cc286010823e38bda6ea72283dd48516a5c2737c))
- **al_loop**: define `num_cpus_large_struct` to use with large structures in md ([#429d79f](https://github.com/pol-sb/MatDBForge/commit/429d79fa14964730266b890f91b8f758c32fb708))
- **al_loop**: fixed input ([#2211e2c](https://github.com/pol-sb/MatDBForge/commit/2211e2cd86012fbbf914f8ac7e4403b0ab6329c4))
- **al_loop**: fixed input ([#b92194c](https://github.com/pol-sb/MatDBForge/commit/b92194ce5cd68b190c9aec9ccd11493bc512af9b))

### Misc

- **docs**: updated admonition css ([#bf24ee2](https://github.com/pol-sb/MatDBForge/commit/bf24ee26ab08c0cca0cbfe3a5043a164032392f1))
- **al_loop**: refactored code ([#bf4058d](https://github.com/pol-sb/MatDBForge/commit/bf4058d37dfa0e61265749070a1e3d91dd4dd720))

## 0.16.5 (2024-11-05)

### Fix

- **al_loop**: added run call to resume cli command ([#29da12d](https://github.com/pol-sb/MatDBForge/commit/29da12d6792e3e0ae04d05c83b47e4550efe9e43))
- **al_loop**: changed default value for `seed_min_num_structs` ([#5d270ae](https://github.com/pol-sb/MatDBForge/commit/5d270aea21718b7b68dac1ac625ca2b2e70dcba4))

## 0.16.4 (2024-11-05)

### Fix

- **al_loop**: replaced `min_seed_frac` with `seed_min_num_structs` for md seeds ([#eda71f8](https://github.com/pol-sb/MatDBForge/commit/eda71f87622179f34ad1f3fe1d2775d8b9d60e55))

## 0.16.3 (2024-11-05)

### Fix

- **al_loop**: mdb version now logged at the start of al_loop ([#d4cea8c](https://github.com/pol-sb/MatDBForge/commit/d4cea8cd67a6d51c59a61d8716badad66966a536))
- **al_loop**: mdb version now logged at the start of al_loop ([#ec030db](https://github.com/pol-sb/MatDBForge/commit/ec030dbb1b92346e6027206b0658d2c7734902c8))

## 0.16.2 (2024-11-05)

### Fix

- **al_loop**: added minimum md seed size ([#2464cf2](https://github.com/pol-sb/MatDBForge/commit/2464cf28a082440421f83690f7c31a39ad5112a8))
- **al_loop**: added `can_do_advanced_extrapolation` check on iteration outline to allow for basic runs with no latent space computation ([#a9d4070](https://github.com/pol-sb/MatDBForge/commit/a9d4070e3e9087ed17691f90efe42f8a26213c1a))

## 0.16.1 (2024-11-05)

### Fix

- **core**: changed git tag gathering strategy ([#bbe48bb](https://github.com/pol-sb/MatDBForge/commit/bbe48bba58ec918a887c1e67715fe15b704ada4e))
- **al_loop**: add initial database size to al_report ([#b04f2c3](https://github.com/pol-sb/MatDBForge/commit/b04f2c38d59dbc08bcb345c8a821e9e3513f24f2))
- **al_loop**: fixed conditional block preventing al_loop report generation ([#53373f2](https://github.com/pol-sb/MatDBForge/commit/53373f29a3b4932560d8cd49987095c289844e98))

### Misc

- **docs**: update theme + add params for simple md ([#1c82597](https://github.com/pol-sb/MatDBForge/commit/1c825973417502bfa1be175ddfbb9765511f3ce7))
- **docs**: add params for simple md ([#0667eb1](https://github.com/pol-sb/MatDBForge/commit/0667eb1475f311cb60f70e58ab4984f34ee78d02))
- **docs**: modified `custom.css` ([#82ed9e8](https://github.com/pol-sb/MatDBForge/commit/82ed9e879d70e5406084c9ab98197ab3dde94378))
- **docs**: modified `custom.css` ([#d80f6c1](https://github.com/pol-sb/MatDBForge/commit/d80f6c1605d74996a8077f5eb43ebbfb741b2a5d))
- **docs**: update docs ([#2fe7908](https://github.com/pol-sb/MatDBForge/commit/2fe7908cbd608e481648931d66c200e026e027bb))
- **docs**: update docs ([#78dc954](https://github.com/pol-sb/MatDBForge/commit/78dc954d9400cec07a60f01a086d127f38591633))
- **docs**: moved calculation limiting info to `customization.md` ([#8682d2c](https://github.com/pol-sb/MatDBForge/commit/8682d2cdf1dfcd31b7589dc471bbdc4396c49e4e))
- **docs**: updated CLI usage documentation ([#8594113](https://github.com/pol-sb/MatDBForge/commit/859411335e3ad3cd8ec93e063e1e2b30271a0bab))
- **docs**: updated CLI usage documentation ([#3c5202d](https://github.com/pol-sb/MatDBForge/commit/3c5202d2767e457a72e58fa7e636d772922e38f1))
- **docs**: updated CLI usage documentation ([#d04766b](https://github.com/pol-sb/MatDBForge/commit/d04766b6e3479447b19520006982d7f8e680ea30))
- **docs**: add documentation for CLI usage ([#f626608](https://github.com/pol-sb/MatDBForge/commit/f6266088046bd36e329d682c3fba52e6ebbc4d0e))

## 0.16.0 (2024-10-31)

### Feature

- **al_loop**: add resume function to al_loop ([#bc92d44](https://github.com/pol-sb/MatDBForge/commit/bc92d44c77c58f1c5d75c54ca9242a569579a8e9))
- **al_loop**: add `copy_input_toml_file` function to al_loop ([#a4309e5](https://github.com/pol-sb/MatDBForge/commit/a4309e5f723eeabe3686c185f7f9c8ffaaa45745))
- **al_loop**: add al_loop report subcommand ([#26d1a49](https://github.com/pol-sb/MatDBForge/commit/26d1a49d62f82753d594176fbb94102dffc14da5))
- **al_loop**: add al_loop report subcommand ([#7def029](https://github.com/pol-sb/MatDBForge/commit/7def02977922bd0c30ff02008a81254868a6b628))

### Fix

- **al_loop**: improved al_loop report representation ([#47fb827](https://github.com/pol-sb/MatDBForge/commit/47fb827df14125434bafba7b4c9f4d9c895b1e2c))
- **core**: silenced git output when checking for updates ([#73548db](https://github.com/pol-sb/MatDBForge/commit/73548db650c2537d23bd1084133061a17b613eac))

## 0.15.0 (2024-10-25)

### Feature

- **al_loop**: first iteration of the ASE-MD+descriptor+extrapolation PortableCode ([#28b4430](https://github.com/pol-sb/MatDBForge/commit/28b4430208d62c6fa0013b3a161bd88335aeb9a1))

### Fix

- **al_loop**: slugify model names to remove unsupported characters ([#4fd825b](https://github.com/pol-sb/MatDBForge/commit/4fd825bcfb4f06c1fd6d0a3e7519b1db17d393cb))

## 0.14.0 (2024-10-23)

### Feature

- **code**: add utilities for cache dir generation ([#2f5f9cb](https://github.com/pol-sb/MatDBForge/commit/2f5f9cbf0d6c735b2887cd5ca5a4813a0f78c343))
- **gen_db**: use cache in surface generation ([#2279ae0](https://github.com/pol-sb/MatDBForge/commit/2279ae00b4142bd70ceb450be10e85d899b4f47e))
- **phase**: introduce cache to structure generation ([#fe25d80](https://github.com/pol-sb/MatDBForge/commit/fe25d80ea869b43a72ee0e9801ddf32c3dca87eb))
- **gen_db**: added parallel processing to `gen_surfaces_diff_miller` to speed up db generation ([#1a49a96](https://github.com/pol-sb/MatDBForge/commit/1a49a9661255b4abe45509e123ac030308982954))

### Fix

- **autoencoder**: correctly assign device and dtype to tensors ([#2469f2d](https://github.com/pol-sb/MatDBForge/commit/2469f2d3f346134705c1b228b12a0b58f6b98e4c))

### Misc

- **code**: update `cli_run_initial_config` to create cache folder ([#e6dc73d](https://github.com/pol-sb/MatDBForge/commit/e6dc73d3af7c86ada7f149f699f1e8f2dab73e5f))
- **docs**: update cache docs ([#1ae899a](https://github.com/pol-sb/MatDBForge/commit/1ae899a023dd80af290f868d74a2dd940814255f))
- **docs**: fixed typo ([#2e408f3](https://github.com/pol-sb/MatDBForge/commit/2e408f3505b2d2ab01793a220f4e7e17e67fc1a9))
- **core**: moved `init_conf` to `code_utils.py` (ii) ([#7e034d7](https://github.com/pol-sb/MatDBForge/commit/7e034d7b4c338c277158b96ff48ba6c26b30d7fa))
- **core**: moved `init_conf` to `code_utils.py` ([#0d6894c](https://github.com/pol-sb/MatDBForge/commit/0d6894c46a1359042790f7b8414b217213705566))
- **core**: fixed dependency versions ([#1b88075](https://github.com/pol-sb/MatDBForge/commit/1b8807548e47d115b6cbf753e01d9078e2e6fc7f))

## 0.13.0 (2024-10-21)

### Feature

- **gen_db**: improve handling of material prototypes and phase replacements ([#299ed8a](https://github.com/pol-sb/MatDBForge/commit/299ed8a5b894606c62ec4f55a2db49ce2e13dea8))
- **autoencoder**: added option for dtype to autoencoder ([#0669b82](https://github.com/pol-sb/MatDBForge/commit/0669b826aceeae28a658c96daaf1b960158d5b0c))

### Fix

- **gen_db**: keep base structures when limiting number of structures per phase ([#bc556db](https://github.com/pol-sb/MatDBForge/commit/bc556db487143c389933a673bd6dcfc75c4ef7f8))
- **gen_db**: moved vacancy generation so it is applied to each phase ([#6271e8e](https://github.com/pol-sb/MatDBForge/commit/6271e8e04c9a0cef04867b6523c5c9ca47b127ac))
- **al_loop**: updated inputs for  during loop ([#dbebeed](https://github.com/pol-sb/MatDBForge/commit/dbebeed13ab7692387ea59038fdadef4825bcf97))
- **al_loop**: correctly gather computer in `generate_descriptors()` ([#410e94f](https://github.com/pol-sb/MatDBForge/commit/410e94fe179065ecd5f2044407f0c0cb5bbe4e1c))
- **al_loop**: replaced deprecated `User.objects` with `User.collection` ([#e721d93](https://github.com/pol-sb/MatDBForge/commit/e721d937a1e03051fbb0d45cd0d69452765eedc4))
- **al_loop**: added `mdb_calc_limit` check before calcjob submission ([#e1f5017](https://github.com/pol-sb/MatDBForge/commit/e1f501799614bb231c86d1ba74c0fb37f6a2afaa))
- **core**: removed git output from `get_last_tagged_version` ([#d74ff10](https://github.com/pol-sb/MatDBForge/commit/d74ff10cbef1079f25b2fb8f8e3d498618d42384))

### Misc

- **core**: fix typo in `CHANGELOG.md` ([#0ee378e](https://github.com/pol-sb/MatDBForge/commit/0ee378e1a3cb4dd678957a273ae7abcdc76b9726))
- **structure**: catch FutureWarning for new structure dataframe concatenation ([#b5796c6](https://github.com/pol-sb/MatDBForge/commit/b5796c64af47d9e7f42f476bdcd1003dcb49e39d))
- **docs**: updated docs to include `mdb_calc_limit` and improved phase explanation. ([#77121dc](https://github.com/pol-sb/MatDBForge/commit/77121dc141389f9cbc9f5995b44d339386b13368))
- **docs**: updated docs ([#b2e4a73](https://github.com/pol-sb/MatDBForge/commit/b2e4a73230c1da08df93e8c712a92d2f029c2286))
- **docs**: updated docs ([#cb5304f](https://github.com/pol-sb/MatDBForge/commit/cb5304ff6f83c0aaff7ae147e404bc35e382b021))
- **docs**: updated docs ([#b0b2e7c](https://github.com/pol-sb/MatDBForge/commit/b0b2e7c25a18cd46bada965e32553c2c471b7bd0))
- **docs**: added favicon ([#f865d30](https://github.com/pol-sb/MatDBForge/commit/f865d30592ef32becc4e4343f46a5ad26621e117))
- **docs**: added favicon ([#83c015e](https://github.com/pol-sb/MatDBForge/commit/83c015e5d193f6747186192d8a68d2cf300b6841))
- **core**: updated `InitialDatabase` and `Phase` __repr__ methods ([#593cd76](https://github.com/pol-sb/MatDBForge/commit/593cd76c916767187fe9dac72a9ee8f6b81444f4))
- **core**: marked conf.py as a version file ([#78c8b63](https://github.com/pol-sb/MatDBForge/commit/78c8b6367cf670fd70f1d350c95a84e5aaeb7a5f))
- **docs**: updated documentation ([#2e300ba](https://github.com/pol-sb/MatDBForge/commit/2e300ba157b6988620e3cafe502b150f158810fe))
- **docs**: updated docs ([#95760e1](https://github.com/pol-sb/MatDBForge/commit/95760e1757a05a8484c6f2da876fc9f1c321b76a))
- **docs**: updated docs ([#ae57904](https://github.com/pol-sb/MatDBForge/commit/ae57904e0f3fd6a3c0043a7a95a0ca475fbb265d))
- **docs**: updated docs ([#4b7c4e9](https://github.com/pol-sb/MatDBForge/commit/4b7c4e93d703608936e1a3887e657eb1b895cbce))
- **docs**: updated docs ([#7284302](https://github.com/pol-sb/MatDBForge/commit/728430279879fd5407cf056c6e988850903c4657))
- **docs**: updated `index.rst` ([#e29248f](https://github.com/pol-sb/MatDBForge/commit/e29248fa94bdb2d32e5039390ed2a60dcd0fa2a6))
- **docs**: updated `README.md` ([#8513aa1](https://github.com/pol-sb/MatDBForge/commit/8513aa1da2048079c06a85986a487edaccd76bad))
- **docs**: updated documentation ([#f2e2707](https://github.com/pol-sb/MatDBForge/commit/f2e27071afb75983d00bb112698a594fd0cd65ca))
- **core**: added pip cache to github workflows ([#9082035](https://github.com/pol-sb/MatDBForge/commit/9082035dfa88a919a912796f8fc62f2ea7c1ef2d))
- **core**: updated dependencies ([#c4ec121](https://github.com/pol-sb/MatDBForge/commit/c4ec121064862109169595643e993c073ead4f3d))
- **docs**: updated workflows ([#37557f5](https://github.com/pol-sb/MatDBForge/commit/37557f5903d35a5983eded5080333bdc753f54e4))
- **docs**: updated workflows ([#d7813a4](https://github.com/pol-sb/MatDBForge/commit/d7813a411f756361b11384a225a8835ca4066253))
- **docs**: updated workflows ([#9fa22b3](https://github.com/pol-sb/MatDBForge/commit/9fa22b373d74e869b1614e0d6f94f33b63fd4f10))
- **docs**: updated `README.md` ([#32f5e58](https://github.com/pol-sb/MatDBForge/commit/32f5e588aa44794cf9472b1a2e4f8cc87c75e2e2))
- **docs**: added doc files ([#b47aca1](https://github.com/pol-sb/MatDBForge/commit/b47aca140841c91cc9dd7fb4bed74168498f1cff))
- **docs**: added workflow file ([#bd052e4](https://github.com/pol-sb/MatDBForge/commit/bd052e4f1547f3c924d5e62ae5de4601f469afe8))
- **docs**: added `.nojekyll` file ([#24ac3cc](https://github.com/pol-sb/MatDBForge/commit/24ac3ccfef7e5efefa3ff62ddeed753cbfe29f87))
- **docs**: updated docs ([#97066e2](https://github.com/pol-sb/MatDBForge/commit/97066e2ec2fd92c44fd8e308f50d6a8fccbbc438))
- **docs**: updated docs ([#9b5f688](https://github.com/pol-sb/MatDBForge/commit/9b5f688055800271a3191f6c244209c729ed4749))
- **al_loop**: fixed logger output ([#d7e70af](https://github.com/pol-sb/MatDBForge/commit/d7e70afbb26fbf9d68029ff9733e91d60ab2c497))
- **docs**: updated documentation ([#77c720e](https://github.com/pol-sb/MatDBForge/commit/77c720ecf32dd7085269e00fa356dc1012c9c734))

## 0.12.0 (2024-10-15)

### Feature

- **gen_db**: added descriptor concave hull option to db_gen ([#01dbe76](https://github.com/pol-sb/MatDBForge/commit/01dbe76814f3832aee1022647b9ff73f5d0ce7a6))

### Misc

- **core**: added `commitizen` to dev dependencies. ([#a8f2989](https://github.com/pol-sb/MatDBForge/commit/a8f298934972d3b89f0e76b6608cbda927f53da6))

## 0.11.4 (2024-10-15)

### Fix

- **al_loop**: fixed `check_extrapolation_type` key being gathered from wrong dictionary ([#6386024](https://github.com/pol-sb/MatDBForge/commit/6386024a1c2c7a496b733bfa475b84b9f2a4b87e))

### Misc

- **docs**: added input description for new extrapolation settings. ([#7794c68](https://github.com/pol-sb/MatDBForge/commit/7794c68f2ab4e90ad4f64904b0bdd26d5a416fa6))
- **core**: added shapely dep. ([#05894e6](https://github.com/pol-sb/MatDBForge/commit/05894e6d593155ac7b5ee1dcb86f662f295a7623))

## 0.11.3 (2024-10-14)

### Fix

- **gen_db**: perturbed structures now save material id ([#e859d7e](https://github.com/pol-sb/MatDBForge/commit/e859d7e99ac0d6c09f1cd333e35e92634d73203c))

### Misc

- **gen_db**: added limit to number of phases in pie chart, remaining ones are grouped into `others`. ([#2fed433](https://github.com/pol-sb/MatDBForge/commit/2fed433d60cf11af274bbc511c957d2c126270cd))
- **gen_db**: improved output and added deprecation warnings ([#2f3b778](https://github.com/pol-sb/MatDBForge/commit/2f3b7786beb044905bb8934bb3f1fb77431e59eb))
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
