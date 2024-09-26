# MatDBForge

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="media/logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="media/logo_light.png">
    <img alt="MatDBForge: A workflow for materials NNP generation" src="media/logo_light.png">
  </picture>

MatDBForge is a Python library that enables the generation of chemical structures database for training NNPs (Neural Network Potential) for heterogeneous catalysis. It provides functionalities to create and manage a database of materials structures for training machine learning models, and allows to interact with workflow tools in order to automate the structure labelling.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Example](#example-training-a-mace-nnp-from-scratch)
- [Package Structure](#package-structure)

## Installation

To install MatDBForge, you can use pip in a python virtual environment or conda environment. Any python versions above `python3.9` should work. Most development has been made with `python3.11` in mind, which can be installed through the OS's package manager or conda.

First, create a virtual environment and activate it:

```bash
# Create and activate the environment
python3.11 -m venv mdb
source mdb/bin/activate
```

Next, clone the repository:

```bash
# Clone the reposittory
git clone https://github.com/pol-sb/MatDBForge.git
```

Next, install the library using pip in the desired python environment:

```shell
# Install the library using pip
python3 -m pip install MatDBForge
```

Finally, initialize configuration files by running the initial configuration command. Then, enter your [Materials Project API key](https://next-gen.materialsproject.org/api) in the path displayed in the output to finish the setup process:

```shell
# Run the last setup step
mdb_init_setup
```

The active learning (AL) loop uses the [AiiDA](https://github.com/aiidateam/aiida-core) library for managing the workflow. In order to run the AL loop in compute clusters, codes and computers must be conifigured in AiiDA.

DFT calculations with VASP use the [aiida-vasp](https://aiida-vasp.readthedocs.io/en/latest/) plugin, which needs additional configuration. Please, follow the [instructions on their website](https://aiida-vasp.readthedocs.io/en/latest/getting_started/general.html).

## Usage

The goal of this library is to provide workflows, functions and utilities for streamlining the training of neural networks potentials (NNPs) by means of Active Learning (AL) Loops.

During the library installation, several entry points will be added so that the user can easily run the different utilities:

- `mdb_conf_gen`: Generate a `.toml` template configuration file to be used in any of the different operation modes of the code.
- `mdb_gen_init_db`: Generate a database containing structures for NNP training.
- `mdb_active_learning`: Launch an AL loop using a configuration file and a labelled initial database.
- `mdb_monitor_al_loop:` Launch a flask dashboard locally to monitor a running active learning loop. Open <http://127.0.0.1:8000> (or port specified in the launch arguments) in a browser to visualize the dashboard.

All of the entry points provide usage documentation when launched with the `-h`/`--help` argument, e.g.:

```bash
$> mdb_conf_gen --help

>>> usage: gen_default_config [-h] -t TYPE [-p PATH] [-o]
>>>
>>> Generate MDB default configuration files in the TOML format.
>>>
>>> options:
>>>   -h, --help
                     show this help message and exit
>>>   -t TYPE, --config_type TYPE
>>>                  Type of the configuration file to be generated. Available types are:
>>>                         - active_learning: Configuration file for active learning loop.
>>>                         - initial_db: Configuration file for initial database generation.
>>>   -p PATH, --path PATH
                        Path in which to store the file.
                        Will use the CWD by default. Folders will be created if necessary.
>>>  -o, --overwrite
                        Whether to overwrite the destination file, if existent.
```

The utilities for generation and running the AL loop use inputs in the TOML format. Users are advised to use `mdb_conf_gen` to generate a template file which can be customized.

A description of all the possible parameters is available in the documentation for the input files: [Input](./docs/input.md).

## Example: Training a MACE NNP from scratch

This example will showcase the training of a MACE potential in a pure Cu database.

### 1. Initial database generation

In order to generate the database, parameters for generation need to be listed in a .toml configuration file. Use the `mdb_conf_gen` command to generate a template file with instructions that can be customized easily. Available options are listed and described here: [./docs/input.md](./docs/input.md#database-generation)

```bash
# Generate a configuration file for the database generation.
mdb_conf_gen -t initial_db
```

After performing any desired changes to the created configuration file, a database can be generated using the `mdb_gen_init_db` with the path to the configuration file:

```bash
# Generate the initial database
mdb_gen_init_db -c ./path/to/config_file.toml
```

This database will be generated as an extxyz file. This file must be labelled in order to be suitable for the AL Loop.

### 2. Database labelling

The structures can be labelled automatically with VASP, or as a quick testing using a pretrained MACE model.

- For **MACE labelling**, the following command can be used. For more information, check the [MACE documentation](https://mace-docs.readthedocs.io/en/latest/guide/evaluation.html):

```bash
mace_eval_configs  --configs ./unlabelled_db.xyz  --model /model/path cu_model_zan.model --output ./labelled_db.xyz --device cpu --batch_size 5
```

- **(👷 WIP 🕒)** In order to use **VASP for structure labelling**, check the scripts in the [examples](./src/MatDBForge/examples/) directory.

### 3. Run active learning loop

Generate a settings file, customize it using [the options here](./docs/input.md#active-learning-loop) and run the active learning loop:

```bash
# Generate a template file
mdb_conf_gen -t active_learning

# Run the active learning loop, piping its outputs to a file.
# Without the '-c', the program will search for the 'active_learning_settings.toml'
# in the current directory
mdb_active_learning gui --n_sec 60 2>&1 | tee ./run_mdb_al.log
```

The progress of the AL Loop can be monitored by checking its output, or opening the dashboard running at <http://127.0.0.1:8000>.

After the loop is done, a database in the extxyz format and a model file for the potential will be returned.

## Package structure

The main functionalities are organized into the following modules:

- `workflows`: Contains functions and methods that allow to connect the database with workflow tools, mainly with the goal of performing DFT calculations.
- `core`: Includes core functionalities and utilities used by the library, such as the generation and management of the database.
- `active_learning`: Contains classes and functions leveraged during the active learning loops.
- `examples`: Provides example scripts that demonstrate the usage of the library.

The following examples demonstrate the usage of MatDBForge:

- [launch_sp_calcs_db_aiida.py](src/MatDBForge/examples/launch_sp_calcs_db_aiida.py): This example script demonstrates how to launch single-point calculations for a given set of structures and store the results in the database.
- [create_init_db_new.py](src/MatDBForge/examples/create_init_db_new.py): This example script showcases how to create and initialize a new database with initial data.

Please refer to the examples in the examples directory for more details on how to utilize the library for your specific needs.
