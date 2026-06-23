"""Application-wide GUI parameters."""

import re

_ACRONYMS = {
    'dft': 'DFT',
    'md': 'MD',
    'npt': 'NPT',
    'nve': 'NVE',
    'nvt': 'NVT',
    'hpc': 'HPC',
    'vasp': 'VASP',
    'lammps': 'LAMMPS',
    'mace': 'MACE',
    'aiida': 'AiiDA',
    'rmse': 'RMSE',
    'incar': 'INCAR',
    'potcar': 'POTCAR',
    'poscar': 'POSCAR',
    'kpoints': 'KPOINTS',
    'al': 'AL',
    'db': 'DB',
    'ase': 'ASE',
    'api': 'API',
    'ssh': 'SSH',
    'toml': 'TOML',
    'yaml': 'YAML',
    'xml': 'XML',
    'hdf5': 'HDF5',
    'soap': 'SOAP',
    'paw': 'PAW',
    'pbe': 'PBE',
    'lda': 'LDA',
    'gga': 'GGA',
    'sge': 'SGE',
    'slurm': 'SLURM',
    'mpi': 'MPI',
    'cpu': 'CPU',
    'gpu': 'GPU',
    'xyz': 'XYZ',
    'bh': 'BH',
    'lj': 'LJ',
}

_ACRONYM_RE = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in _ACRONYMS) + r')\b',
    re.IGNORECASE,
)


def pretty_label(key: str) -> str:
    """Turn a snake_case schema key into a human-readable label.

    ``key.replace('_', ' ').title()`` but with acronyms preserved.
    """
    text = key.replace('_', ' ').title()
    return _ACRONYM_RE.sub(lambda m: _ACRONYMS[m.group().lower()], text)


class ApplicationParameters:
    """Holds application-wide parameters."""

    FONT_FAMILIES_REGULAR = ['Noto Sans', 'Sans Serif', 'Arial']
    FONT_FAMILIES_MONOSPACE = ['Fira Code', 'monospace']
