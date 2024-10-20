# Installation

## Installation procedure

To install MatDBForge, you can use pip in a python virtual environment or conda environment. Any python versions above `python3.9` should work. Most development has been made with `python3.11` in mind, which can be installed through the OS's package manager or conda.

1. First, **create a virtual environment** and activate it. This can be done using `conda` or python `venv`. One example for an ubuntu system, using python3.11 and venv:

```bash
# Install python3.11 and venv
sudo apt install python3.11 python3.11-venv

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

## Limiting the number of submitted calculations

MatDBForge includes a way of artificially limiting the number of submitted calculations.

In some supercomputer clusters there is a limit on the number of jobs that can be submitted at the same time, and  going over this limit will result in the job not entering the queue.
AiiDA will reattempt to submit after the time given by `exponential_backoff_retry` is elapsed and for `task_maximum_attempts` attempts, and if the number of attempts is exceeded, the calculation will be paused, holding the loop.

This limit will be set by the supercomputer's scheduler, in the case of SLURM, it can be checked with `sacctmgr show qos XXXX` under the `MaxSubmitPU` column.

The artificial limit can be set by adding the `mdb_calc_limit` property to the desired AiiDA Computer's metadata and it will be considered beforing a calculation in the library. The following code fragment shows how to do it for any Computer using its label. It can be executed line by line in the python interpreter launched via the `verdi shell` command or as a script.

```python
from aiida import orm


# Replace `computer_label_example` with the actual label of the AiiDA computer.
computer_name = 'computer_label_example'

# Load the computer
computer = orm.load_computer(computer_name)

# Set the custom property
computer.set_property(name='mdb_calc_limit', value=350)
```
