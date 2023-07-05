# MatDBForge

![MatDBForge logo](media/logo.png)

MatDBForge is a Python library that enables the generation of a chemical compounds database for training NNPs (Neural Network Potential). It provides functionalities to create and manage a database of materials structures for training machine learning models, and allows to interact with workflow tools in order to automate the structure labelling.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [License](#license)

## Installation

To install MatDBForge, you can use pip.

First, clone the repository:

```
git clone git@github.com:pol-sb/MatDBForge.git
```

Then enter the folder containing the repository and install the library using pip in the desired python environment.

```shell
cd MatDBForge
python3 -m pip install MatDBForge
```

## Usage

The goal of this library is to provide functions and utilities for creating and managing a chemical compounds database. The main functionalities are organized into the following modules:

- `workflows`: Contains functions and methods that allow to connect the database with workflow tools, mainly with the goal of performing DFT calculations.
- `core`: Includes core functionalities and utilities used by the library, such as the generation and management of the database.
- `examples`: Provides example scripts that demonstrate the usage of the library.

The intented approach is that the user imports any of the modules and uses them as needed.


## Examples

The following examples demonstrate the usage of MatDBForge:

- [launch_sp_calcs_db_aiida.py](src/MatDBForge/examples/aunch_sp_calcs_db_aiida.py): This example script demonstrates how to launch single-point calculations for a given set of structures and store the results in the database.
- [create_init_db_new.py](src/MatDBForge/examples/create_init_db_new.py): This example script showcases how to create and initialize a new database with initial data.

Please refer to the examples in the examples directory for more details on how to utilize the library for your specific needs.

