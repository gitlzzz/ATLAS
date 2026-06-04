---
name: Bug report
about: Create a report to help us improve
title: "[BUG]"
labels: ''
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. ATLAS command/subcommand used (e.g. `atl_gen_init_db generate`, `atl_active_learning run`, etc.)
2. Config type used (e.g. `initial_db`, `active_learning`, `run_dft_database`)
3. Config file (attach or paste relevant snippet)
4. Expected vs actual output / error traceback

**Relevant logs or error output**
If applicable, paste the error traceback or relevant log output. Look for:
- `atl*.log` or console output
- If applicable, AiiDA report on related nodes (`verdi process report <PK>`)
- If applicable, scheduler stderr (e.g. Slurm `.err` files)

**Environment (please complete the following information):**
 - ATLAS version: [e.g. 0.15.0, or output of `pip show ATLAS`]
 - Python version
 - Installation method: [e.g. `uv`, `pip install -e '.[dev,mace]'`, `conda`, container]
 - MACE/torch version: [e.g. mace-torch==0.3.11, torch==2.4.0]

**Configuration details:**
 - Database composition and example structure
 - DFT method: [vasp / mace]
 - Extrapolation method: [basic / advanced / none]
 - Committee size: [number of models]
 - Containerized execution: [yes / no]
 - Foundation model used: [e.g. mace:mp-0, off-0]
 - NN-MD type (ase, lammps)
 - Extrapolation type
 - Seed selection type

**System (if applicable):**
 - OS: [e.g. Ubuntu 22.04, CentOS 7]
 - HPC scheduler: [e.g. Slurm, SGE, none (local)]
 - GPU: [e.g. NVIDIA A100, none (CPU-only)]
 - AiiDA profile: [e.g. `verdi profile show` output]

**Additional context**
Add any other context about the problem here, such as:
- Did this work in a previous version?
- Are you running inside a container (Docker/Singularity)?
- Can you share a minimal reproducible example?
