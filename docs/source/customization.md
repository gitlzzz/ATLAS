# Customizing your MatDBForge installation

## Limiting the number of simultaneously submitted calculations

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
