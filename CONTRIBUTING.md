# Contributing to ATLAS

Thank you for your interest in contributing to ATLAS. This guide is aimed at external collaborators who want to modify the codebase, add features, or fix bugs. 

Here are listed a few guidelines to ensure compatibility with the existing codebase and to make the contribution process smoother, however, these are not strict rules and we are open to suggestions and improvements to the development workflow. Please reach out in the [discussions](https://github.com/LopezGroup-ICIQ/ATLAS/discussions) if you have any ideas on how to make it better.

## Getting Started

### Prerequisites

- **Python 3.11** (minimum version, however 3.12 and 3.13 have been tested successfully)
- Access to a Linux environment with GPU (for MACE-related work)

### Installation

```bash
# Install in editable mode with development dependencies
pip install -e '.[dev]'

# If you need GPU/MACE support
pip install -e '.[dev,mace]'
```

After installation, initialize config files:

```bash
atl_init_setup
```

This will prompt you for a Materials Project Registry (MPR) API key that will be stored in a file in the configuration directory. You can obtain one for free from [Materials Project](https://next-gen.materialsproject.org/api).

### AiiDA Setup

Several ATLAS components (DFT calculations, active learning workflows, MACE training) integrate with [AiiDA](https://aiida.readthedocs.io/) for workflow orchestration and database management.

**If you only need to modify core library code** (structure handling, database generation, CLI tools, filters), you do **not** need AiiDA installed.

**If you need to work with AiiDA-dependent components** (active learning, DFT submissions, CalcJobs, parsers, workchains), you will need:

1. An AiiDA installation: [aiida-core docs](https://aiida.readthedocs.io/projects/aiida-core/)
2. A configured profile: `verdi presto` (quick setup for development)
3. aiida-vasp: [aiida-vasp docs](https://aiida-vasp.readthedocs.io/)
4. aiida-lammps (for MD simulations): [aiida-lammps docs](https://aiida-lammps.readthedocs.io/)

After any change to CalcJob, Parser, or WorkChain classes, reinstall the package for AiiDA to pick up the new entry points:

```bash
pip install -e .
```

## Development Workflow

### Pre-commit Hooks

Install pre-commit to run linting and checks automatically before each commit:

```bash
pre-commit install
```

The hooks run:
1. Documentation regeneration from schema (if `config_schema.yaml` changes)
2. Ruff linting (`ruff --fix --show-fixes`)
3. End-of-file fixer
4. YAML and TOML validation

You can also run manually:

```bash
pre-commit run --all-files
```

### Commit Convention

ATLAS uses [commitizen](https://commitizen-tools.github.io/commitizen/) with conventional commits. Use `cz run` (or `cz bump` for releases) to guide commit messages.

Commit format: `<type>(<scope>): <summary>`

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`

**Scopes:** `al_loop`, `schema`, `report`, `core`, `md`, `dft`, `safeguard`, `init_db`, `domain_validity`, `benchmarks`

Examples:
```
feat(al_loop): add safeguard check before early stopping
fix(core): correct vacuum direction heuristic for slabs
docs(schema): update MACE training settings documentation
refactor(active_learning): consolidate descriptor generation
```

### Running Tests

```bash
pytest tests/
```

Tests are concentrated in `core/` modules (Structure, PhaseDiagram, Clusters, Exceptions, Code Utils). The active learning and AiiDA workflow modules have limited test coverage as of now, so contributions adding tests to these areas are welcome.

Some tests require an active AiiDA profile.

## Code Style

ATLAS uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

- Line length: 88 characters
- NumPy-style docstrings
- Single quotes
- Imports sorted (isort)

Ruff configuration is in `pyproject.toml`. Run `ruff check --fix src/` to auto-fix issues.

## Adding New AiiDA Components

If you're adding a new CalcJob, Parser, WorkChain, or custom data type:

1. **Place the class** in the appropriate module under `src/ATLAS/`.
2. **Register entry points** in `pyproject.toml` under the relevant `[project.entry-points."aiida.*"]` section.
3. **Reinstall** with `pip install -e .` for AiiDA to discover the new entry points.
4. **Follow existing patterns**, review similar components in the codebase for naming conventions and structure.

Key conventions:
- CalcJobs use `spec.input()` / `spec.output()` for inputs and outputs
- Parsers set `node.exit_code` on failure; use exit codes >= 400 for application-level errors
- WorkChains use `spec.expose()` to expose inner workchain inputs/outputs
- Every ASE `Atoms` object should carry `atoms.info['atl_id']` (UUID) for tracking

## Documentation

API documentation is generated with Sphinx and `sphinx-multiversion`:

```bash
sphinx-multiversion docs _build/html
```

**Input schema changes:** If you modify `src/ATLAS/data/config_schema.yaml`, run:

```bash
pre-commit run update-docs-from-schema --files src/ATLAS/data/config_schema.yaml
```

This regenerates `docs/source/input.md`.

## Reporting Issues

When opening an issue, please include:

- **ATLAS version** (`atl_active_learning --version` or check `pyproject.toml`)
- **Python version** (`python --version`)
- **Relevant configuration snippets** (after redacting, never share API keys or credentials!)
- **Steps to reproduce**
- **Error messages / logs** (full traceback, not just the last line)

For AiiDA-related issues:
- Include the node PK: `verdi node show <pk>`
- Attach the calculation report: `verdi node report <pk>`
- Note which AiiDA plugins are involved (aiida-core, aiida-vasp, aiida-lammps versions)

## Pull Requests

1. Create a feature branch from `main`
2. Make your changes, following the commit convention above
3. Run `pre-commit run --all-files` to ensure linting passes
4. Run the test suite: `pytest tests/`
5. Open a pull request with a clear description of the changes

---

Thank you for contributing to ATLAS!
