"""Build adsorption structures for database generation.

Pure ASE/pymatgen helpers (no AiiDA/MACE) that place small molecules or clusters
on surface slabs:

- :func:`build_adsorbate_molecule` — an ASE ``Atoms`` molecule from a small
  built-in library (H, O, OH, H2O, CO, CO2, N2, NH3, O2, H2) or ``ase.build.molecule``.
- :func:`find_adsorption_sites` — high-symmetry ontop/bridge/hollow sites on a slab
  via pymatgen's ``AdsorbateSiteFinder``.
- :func:`place_molecule_on_slab` / :func:`place_cluster_on_slab` — put an adsorbate
  above the slab at a given site/height.
- :func:`generate_adsorbed_structures` — enumerate placements for a slab.

The database-facing wrapper that iterates the structure DataFrame and persists the
results lives in ``atlas.core.utils`` (``add_adsorbates_to_surfaces``).
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import molecule as ase_molecule

# Small custom library for fragments ASE's g2 doesn't provide directly, plus
# common adsorbates. Coordinates in Angstrom; the lowest atom is the anchor.
_CUSTOM_MOLECULES: dict[str, tuple[list[str], list[list[float]]]] = {
    'H': (['H'], [[0.0, 0.0, 0.0]]),
    'O': (['O'], [[0.0, 0.0, 0.0]]),
    'N': (['N'], [[0.0, 0.0, 0.0]]),
    'C': (['C'], [[0.0, 0.0, 0.0]]),
    'OH': (['O', 'H'], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]]),
    'O2': (['O', 'O'], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.21]]),
    'H2': (['H', 'H'], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
    'N2': (['N', 'N'], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.10]]),
    'CO': (['C', 'O'], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.13]]),
}


def build_adsorbate_molecule(species: str) -> Atoms:
    """Return an ASE ``Atoms`` adsorbate for ``species``.

    Tries the built-in library first, then ``ase.build.molecule`` (g2). The
    molecule is translated so its lowest atom sits at z=0 (the anchor point).
    """
    if species in _CUSTOM_MOLECULES:
        symbols, positions = _CUSTOM_MOLECULES[species]
        mol = Atoms(symbols=symbols, positions=positions)
    else:
        try:
            mol = ase_molecule(species)
        except (KeyError, NotImplementedError, ValueError) as exc:
            raise ValueError(f"Unknown adsorbate species '{species}'.") from exc

    # Anchor the lowest atom at z = 0.
    positions = mol.get_positions()
    positions[:, 2] -= positions[:, 2].min()
    mol.set_positions(positions)
    return mol


def find_adsorption_sites(slab: Atoms, site_types: list[str] | None = None) -> dict:
    """Return high-symmetry adsorption sites for an ASE slab.

    Uses pymatgen's ``AdsorbateSiteFinder``. Returns a dict mapping site type
    (``'ontop'``, ``'bridge'``, ``'hollow'``) to a list of cartesian coordinates.
    """
    from pymatgen.analysis.adsorption import AdsorbateSiteFinder
    from pymatgen.core.surface import Slab
    from pymatgen.io.ase import AseAtomsAdaptor

    structure = AseAtomsAdaptor.get_structure(slab)
    # AdsorbateSiteFinder expects a pymatgen Slab; wrap with a (0,0,1) miller
    # guess (only the surface normal / top sites matter for placement).
    slab_pmg = Slab(
        lattice=structure.lattice,
        species=structure.species,
        coords=structure.frac_coords,
        miller_index=(0, 0, 1),
        oriented_unit_cell=structure,
        shift=0.0,
        scale_factor=np.eye(3),
    )
    finder = AdsorbateSiteFinder(slab_pmg)
    sites = finder.find_adsorption_sites(positions=site_types or ['ontop', 'bridge', 'hollow'])
    # `sites` includes an 'all' aggregate; drop it and keep per-type cartesian coords.
    return {k: [np.asarray(p) for p in v] for k, v in sites.items() if k != 'all'}


def place_molecule_on_slab(
    slab: Atoms,
    molecule: Atoms,
    site_xy: np.ndarray,
    height: float = 2.0,
) -> Atoms:
    """Place ``molecule`` above ``slab`` at lateral ``site_xy`` and ``height`` (Angstrom)."""
    combined = slab.copy()
    mol = molecule.copy()

    top_z = slab.get_positions()[:, 2].max()
    offset = np.array([site_xy[0], site_xy[1], top_z + height])
    mol.set_positions(mol.get_positions() + offset)

    combined += mol
    return combined


def place_cluster_on_slab(cluster: Atoms, slab: Atoms, height: float = 2.5) -> Atoms:
    """Center ``cluster`` laterally over ``slab`` and lift it ``height`` above the top."""
    combined = slab.copy()
    clus = cluster.copy()

    slab_pos = slab.get_positions()
    slab_top = slab_pos[:, 2].max()
    slab_xy = slab_pos[:, :2].mean(axis=0)

    clus_pos = clus.get_positions()
    clus_xy = clus_pos[:, :2].mean(axis=0)
    clus_bottom = clus_pos[:, 2].min()

    shift = np.array(
        [
            slab_xy[0] - clus_xy[0],
            slab_xy[1] - clus_xy[1],
            (slab_top + height) - clus_bottom,
        ]
    )
    clus.set_positions(clus_pos + shift)

    combined += clus
    return combined


def generate_adsorbed_structures(
    slab: Atoms,
    species_list: list[str],
    site_types: list[str] | None = None,
    height: float = 2.0,
    max_per_slab: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[tuple[Atoms, str, str]]:
    """Generate adsorbed structures for ``slab``.

    Returns a list of ``(atoms, adsorbate_type, site_type)`` tuples, one per
    (species, site) combination (optionally capped at ``max_per_slab``).
    """
    rng = rng or np.random.default_rng()
    sites = find_adsorption_sites(slab, site_types)

    combos: list[tuple[str, str, np.ndarray]] = []
    for species in species_list:
        for site_type, coords in sites.items():
            for coord in coords:
                combos.append((species, site_type, coord))

    if max_per_slab is not None and len(combos) > max_per_slab:
        idxs = rng.choice(len(combos), size=max_per_slab, replace=False)
        combos = [combos[i] for i in idxs]

    results: list[tuple[Atoms, str, str]] = []
    for species, site_type, coord in combos:
        mol = build_adsorbate_molecule(species)
        atoms = place_molecule_on_slab(slab, mol, coord[:2], height=height)
        results.append((atoms, species, site_type))
    return results
