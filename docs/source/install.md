# Installation

## Installation procedure

To install MatDBForge, you can use pip in a python virtual environment or conda environment. Development has been made with `python3.11` in mind, which can be installed through the OS's package manager or conda. Additionally, [Julia](https://julialang.org/downloads/) is needed to accelerate some parts of the active learning loop.

1. First, **create a virtual environment** and activate it. This can be done using `conda` or python `venv`. One example for an ubuntu system, using python3.11 and venv:

```bash
# Install python3.11 and venv
sudo apt install python3.11 python3.11-venv

# Install Julia if you dont have it.
# Follow the instructions in:
# https://julialang.org/downloads/

# Using python venv - create and activate the environment
python3 -m venv mdb
source mdb/bin/activate
```

2. Next, **clone the repository**:

```bash
# Clone the reposittory
git clone https://github.com/pol-sb/MatDBForge.git
```

3. Next, **install the library** using pip in the desired python environment:

```shell
# Install the library in the venv using pip
python3 -m pip install ./MatDBForge
```

4. Finally, initialize configuration files by running the **initial configuration command** (`mdb_init_setup`). Then, enter your [Materials Project API key](https://next-gen.materialsproject.org/api) in the path displayed in the output to finish the setup process:

```shell
# Run the last setup step - configuration initialization
mdb_init_setup
```

:::{note}
If the user is only interested in database generation, the setup can be completed only up until this point, skipping the AiiDA setup.
:::

The **active learning (AL)** loop uses the [AiiDA](https://github.com/aiidateam/aiida-core) library for managing the workflow. In order to run the AL loop in compute clusters, codes and computers must be conifigured in AiiDA. See the [AiiDA installation guide](https://aiida.readthedocs.io/projects/aiida-core/en/stable/installation/guide_quick.html) for installation instructions. DFT calculations with VASP use the [aiida-vasp](https://aiida-vasp.readthedocs.io/en/latest/) plugin, which needs additional configuration. Please, follow the [instructions on their website](https://aiida-vasp.readthedocs.io/en/latest/getting_started/general.html).

The steps required to set up the active learning loop with the simplest AiiDA configuration are the following:

1. Set up an aiida profile and database with `verdi presto`.
2. Create the AiiDA computer and code entries for MatDBForge and aiida-vasp.
3. Add the potential datasets for aiida-vasp ([information here](https://aiida-vasp.readthedocs.io/en/latest/getting_started/potentials.html)).
