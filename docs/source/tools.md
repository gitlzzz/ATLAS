# CLI Arguments

This section describes the command line arguments for various scripts provided by the MatDBForge library. These arguments allow users to control the behavior and configuration of active learning loops, database generation, and other features. Each argument is explained along with its usage and default value (if applicable).

## Run Active Learning - `mdb_active_learning`


Handles MatDBForge active learning loop runs. Running the `mdb_active_learning` command without any subcommand or the gui subcommand will start a new run.
In order to do this, users must provide a TOML settings file which can be configured as seen in the [inputs section](input.md#active-learning-loop).

Other available subcommands allow to generate reports or resume loops using files generated from previous runs.

**Usage:** `mdb_active_learning <SUBCOMMAND> <OPTIONS>`

### General Arguments

- `-c, --config_file` (PATH)
  - **Description**: Path pointing to a TOML settings file. By default, `active_learning_settings.toml` will be searched in the current working directory (CWD).
  - **Type**: `Path`
  - **Default**: `./active_learning_settings.toml`

### Subcommands

#### `report`
Generates a report for a MatDBForge running/finished active learning loop. This produce a plot containing information about the training and seed databases sizes, the NNP performance and the number of added and removed structures.

**Usage**: `mdb_active_learning report [-h] (--loop_id <ID> | --log_path <PATH>)`

- `--loop_id, -i` (`<ID>`)
  - **Description**: AiiDA PK/UUID of the active learning loop.
- `--log_path, -log` (`<PATH>`)
  - **Description**: Path to the MatDBForge log of the active learning loop.

#### `resume`
Resumes an active learning loop using a results folder. A resumed run will use the settings from the `.toml` file contained in the results folder.

**Usage**: `mdb_active_learning resume [-h] --dir_resume <PATH> [--config_file <PATH>]`

- `--dir_resume, -d` (`<PATH>`)
  - **Description**: Path to the results directory of a previous active learning loop run.
  - **Required**: Yes
- `--config_file, -c` (`<PATH>`)
  - **Description**: Path pointing to a TOML settings file. Optional, as all calculation folders should contain a TOML file.
  - **Default**: `None`

#### `gui`
Launches a dashboard to track the active learning loop.

**Usage:** `run_active_learning gui [-h] [--update_interval n_sec] [--port port] [--debug] [--online]`

- `--update_interval` (`n_sec`)
  - **Description**: Refresh time interval in seconds.
  - **Type**: `int`
  - **Default**: `60`
- `--port` (`port`)
  - **Description**: Port to use for the webapp.
  - **Type**: `int`
  - **Default**: `8000`
- `--debug`
  - **Description**: Enable Flask debug mode.
  - **Default**: `False`
- `--online`
  - **Description**: Enable online mode.
  - **Default**: `False`


## Active Learning Dashboard - `monitor_al_loop`

Monitor a MatDBForge active learning loop using a dashboard.

**Usage:** `monitor_al_loop [-h] [--process_id UUID/PK] [--update_interval n_sec] [--port port] [--debug] [--online]`

- `--process_id` (`UUID/PK`)
  - **Description**: Process id (pk/uuid) of the WorkChain to monitor.
  - **Type**: `str`
- `--update_interval` (`n_sec`)
  - **Description**: Refresh time interval in seconds.
  - **Type**: `int`
  - **Default**: `60`
- `--port` (`port`)
  - **Description**: Port to use for the webapp.
  - **Type**: `int`
  - **Default**: `8000`
- `--debug`
  - **Description**: Enable Flask debug mode.
  - **Action**: `store_const`
  - **Default**: `False`
- `--online`
  - **Description**: Enable online mode.
  - **Default**: `False`


## Generate Configuration File - `mdb_gen_configuration_file`

Generates default configuration files for MatDBForge in the TOML format. The generated `.toml` files can be used as a template for active learning runs and initial database generation.

**Usage:** `mdb_gen_configuration_file [-h] -t TYPE [-p PATH] [-o]`

- `-t, --config_type` (`TYPE`)
  - **Description**: Type of the configuration file to be generated. Available types:
    - `active_learning`: Configuration file for an active learning loop.
    - `initial_db`: Configuration file for initial database generation.
  - **Type**: `str`
  - **Choices**: `["active_learning", "initial_db"]`
  - **Required**: Yes
- `-p, --path` (`PATH`)
  - **Description**: Path to store the configuration file. Defaults to the CWD.
  - **Type**: `Path`
  - **Default**: `.`
- `-o, --overwrite`
  - **Description**: Whether to overwrite the destination file if it already exists.
  - **Default**: `False`


## Generate Initial Database - `mdb_gen_init_db`

Generates an initial database for MatDBForge. A `.toml` configuration is required, which is described in the [corresponding inputs section](input.md#database-generation)

**Usage:** `mdb_gen_init_db [-h] [-c PATH]`

- `-c, --config_file` (`PATH`)
  - **Description**: Path to a TOML settings file. By default, `database_generation_settings.toml` will be searched in the CWD.
  - **Type**: `Path`
  - **Default**: `./database_generation_settings.toml`


## Run DFT Database - `mdb_run_dft_database`

Runs VASP DFT calculations for a database of structures using AiiDA-VASP. A .toml configuration file is required to specify the input settings for HPC and VASP. Refer to the [corresponding inputs section](input.md#database-batch-dft-execution) for more details about the available configuration options.

**Usage**: `mdb_run_dft_database [-h] --db_file FILE --config FILE`

- `--db_file FILE, -i FILE`
  - **Description**: Path to the extxyz file containing the database of structures.
  - **Type**: `Path`
  - **Required**: Yes

- `--config FILE, -c FILE`
  - `Description`: Path to the TOML file with HPC and VASP input configuration.
  - `Type`: `Path`
  - `Required`: Yes


## Autoencoder for Dimensionality Reduction - `mdb_train_autoencoder`

Train an autoencoder model for dimensionality reduction using the generated descriptors.
The model is trained on descriptors (either `SOAP` or `MACE`) which are provided through a numpy array in the `.npy` format that holds vstacked arrays of atomic descriptors.

**Usage:** `mdb_train_autoencoder [-h] [--device DEVICE] [--dtype DTYPE] [--model_path MODEL_PATH] [--load_model LOAD_MODEL] [--rng_seed RNG_SEED] [--dataset DATASET] [--l1_hidden_dim L1_HIDDEN_DIM] [--l2_hidden_dim L2_HIDDEN_DIM] [--bottleneck_dim BOTTLENECK_DIM] [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE] [--patience PATIENCE] [--lr LR] [--weight_decay WEIGHT_DECAY] [--bias_flag] [--loss LOSS] [--train_frac TRAIN_FRAC] [--valid_frac VALID_FRAC] [--test_frac TEST_FRAC] [--wandb] [--wandb_name WANDB_NAME] [--wandb_project WANDB_PROJECT]`

### Arguments

- `--device`
  - **Description**: Device to run the model on. Options: `cpu`, `cuda`, or `auto`.
  - **Type**: `str`
  - **Default**: `cpu`
- `--dtype`
  - **Description**: Data type for the model.
  - **Type**: `str`
  - **Default**: `float32`
- `--model_path`
  - **Description**: Path to save the model.
  - **Type**: `str`
  - **Default**: `autoencoder_model.pth`
- `--load_model`
  - **Description**: Load the model from the model path.
  - **Type**: `bool`
  - **Default**: `False`
- `--rng_seed`
  - **Description**: Seed for the random number generator.
  - **Type**: `int`
- `--dataset`
  - **Description**: Path to the training dataset.
  - **Type**: `str`
  - **Default**: `all_descriptors.npy`
- `--l1_hidden_dim`
  - **Description**: Number of units in the first hidden layer.
  - **Type**: `int`
  - **Default**: `256`
- `--l2_hidden_dim`
  - **Description**: Number of units in the second hidden layer.
  - **Type**: `int`
  - **Default**: `32`
- `--bottleneck_dim`
  - **Description**: Dimensionality of the bottleneck (latent space).
  - **Type**: `int`
  - **Default**: `2`
- `--num_epochs`
  - **Description**: Number of epochs to train the model.
  - **Type**: `int`
  - **Default**: `50`
- `--batch_size`
  - **Description**: Batch size for training.
  - **Type**: `int`
  - **Default**: `4096`
- `--patience`
  - **Description**: Patience for early stopping.
  - **Type**: `int`
  - **Default**: `5`
- `--lr`
  - **Description**: Learning rate for the optimizer.
  - **Type**: `float`
  - **Default**: `0.001`
- `--weight_decay`
  - **Description**: L2 regularization.
  - **Type**: `float`
  - **Default**: `1e-5`
- `--bias_flag`
  - **Description**: Add this argument to unclude bias terms in the layers.
  - **Action**: `store_true`
- `--loss`
  - **Description**: Loss function type. Options: `mse`, `weighted_mse`.
  - **Type**: `str`
  - **Default**: `mse`
- `--train_frac`
  - **Description**: Fraction of data used for training.
  - **Type**: `float`
  - **Default**: `0.8`
- `--valid_frac`
  - **Description**: Fraction of data used for validation.
  - **Type**: `float`
  - **Default**: `0.1`
- `--test_frac`
  - **Description**: Fraction of data used for testing.
  - **Type**: `float`
  - **Default**: `0.1`
- `--wandb`
  - **Description**: Add this argument to log metrics to Weights & Biases.
  - **Action**: `store_true`
- `--wandb_name`
  - **Description**: Name of the Weights & Biases run.
  - **Type**: `str`
- `--wandb_project`
  - **Description**: Name of the Weights & Biases project.
  - **Type**: `str`
