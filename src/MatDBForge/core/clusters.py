import pathlib as pl
import MatDBForge.core.phase_diagram as mdb_phase
import MatDBForge.core.structure as mdb_struct
import MatDBForge.core.exceptions as mdb_exc
import MatDBForge.core.utils as mdb_utils
import MatDBForge.core.initial_db as mdb_indb
import numpy as np
import itertools as it
from pymatgen.core import Structure, Lattice
from pymatgen.io.vasp import Poscar

# Gathering some cluster related data
DATA_PATH = (pl.Path(f"{__file__}").parent / "../data").resolve()
CLUST_LIST = [clst.stem for clst in DATA_PATH.iterdir()]

# Default initial lattice size.
LATT = [20, 21, 22]

# Data sourced from Materials Project
# TODO: Get this using a query. Will probably need the structure name and phase?
ATOM_DATA = {"Cu": {"a": 3.5691940}}

# Maximum vacuum thickness, in Angs.
MAX_VAC = 12


def make_clean_cluster(indb_obj, size, phase: mdb_phase.Phase):
    curr_at_data = ATOM_DATA.get(phase.cluster_elem.symbol)

    folder = [
        fold for idx, fold in enumerate(CLUST_LIST) if phase.cluster_elem.symbol in fold
    ]
    if len(folder) == 0 or not curr_at_data:
        raise mdb_exc.AtomNotFoundForCluster

    at_posc = np.loadtxt(DATA_PATH / folder[0] / str(size), skiprows=1)

    # Generating initial structure with a large cell size.
    # It will be used to check the cluster size and the cell siize
    # reduction needed in order to get a vacuum thickness of MAX_VAC
    lattice = Lattice.from_parameters(
        a=LATT[0], b=LATT[1], c=LATT[2], alpha=90, beta=90, gamma=90
    )
    struct = Structure(
        lattice=lattice,
        coords=at_posc * curr_at_data["a"],
        species=list(it.repeat(phase.cluster_elem, size)),
        coords_are_cartesian=True,
    )

    # Vaccuum thickness for every axis,
    # considering if the cluster goes over the cell boundary.
    vac_thick = (
        np.min(struct.cart_coords, axis=0)
        + np.array(LATT)
        - np.max(struct.cart_coords, axis=0)
    )
    new_latt_size = (np.array(LATT) - vac_thick) + MAX_VAC
    lattice = Lattice.from_parameters(
        a=new_latt_size[0],
        b=new_latt_size[1],
        c=new_latt_size[2],
        alpha=90,
        beta=90,
        gamma=90,
    )

    # New structure with the desired vacuum thickness
    struct = Structure(
        lattice=lattice,
        coords=struct.cart_coords,
        species=list(it.repeat(phase.cluster_elem, size)),
        coords_are_cartesian=True,
    )

    # Centering and bottoming structure
    # struct = center_and_bottom_structure(struct)
    struct = center_structure(struct)
    # struct = bottom_structure(struct)

    Poscar(struct).write_file(
        f"/tmp/cluster_tests/test_{np.random.randint(1,1e6)}.poscar"
    )
    quit()

    cluster_name = f"base_cluster_{struct.formula}"
    clust_obj = mdb_struct.Cluster(
        material_name=cluster_name,
        structure=struct,
        base=True,
    )

    return clust_obj


def center_and_bottom_structure(
    structure: Structure,
    offset: int = 2,
) -> Structure:
    """
    Move the structure towards the bottom of the cell, leaving a
    offset wide margin at the bottom.

    Parameters
    ----------
    structure : Structure
        Target structure to move to the bottom.
    offset : int, optional
        Separation to be left between the bottom of the cell
        and the structure, by default 2, in Angstrom.


    Returns
    -------
    Structure
        Pymatgen structure containing a structure placed on the bottom,
        with the same attributes as the original.
    """

    # # Centering
    # avg = np.average(structure.cart_coords, axis=0)
    # print('avg: ', avg)
    # print('structure.lattice.abc: ', structure.lattice.abc)
    # dist = np.array(structure.lattice.abc) / 2 - avg
    # print('dist: ', dist)
    # modified_coords = structure.cart_coords + dist

    # # TODO: Center of mass should be computed after replacement.
    # atomic_mass_arr = np.array([atom.atomic_mass for atom in structure.species])
    # print('atomic_mass_arr: ', atomic_mass_arr)
    # print('modified_coords1: ', modified_coords)
    # total_mass = np.sum(atomic_mass_arr)

    # # Multiplying the coordinates by the atomic mass, adding an additional empty axis
    # modified_coords = np.multiply(modified_coords,atomic_mass_arr[:, None])
    # print('modified_coords_multiply: ', modified_coords)
    # modified_coords = np.sum(modified_coords, axis=0)
    # print('modified_coords_sum: ', modified_coords)

    # # Dividing by the total mass in order to get the center of mass
    # modified_coords /= total_mass
    # print('modified_coords: ', modified_coords)
    # print('total_mass: ', total_mass)
    # # quit()
    # # Getting the position closest to the bottom
    # bottom = np.min(modified_coords)
    # bottom_arr = np.zeros(shape=modified_coords.shape)

    # Getting the center of mass
    com = get_center_of_mass(structure)

    # Getting the position closest to the bottom
    bottom = np.min(structure.cart_coords)
    bottom_arr = np.zeros(shape=structure.cart_coords.shape)

    # Centering
    # avg = np.average(structure.cart_coords, axis=0)
    dist = np.array(structure.lattice.abc) / 2 - com
    print("dist: ", dist)
    modified_coords = structure.cart_coords + dist

    # Applying the offset
    bottom_arr[:, 2] += bottom - offset
    print("bottom_arr: ", bottom_arr)

    # Substracting the bottom position from the structure plus an offset
    modified_coords = modified_coords + bottom_arr
    print("modified_coords: ", modified_coords)

    # Generating new pymatgen structure with the modifications
    new_structure = Structure(
        lattice=structure.lattice,
        species=structure.species,
        coords=modified_coords,
        coords_are_cartesian=True,
        site_properties=structure.site_properties,
    )

    return new_structure


def center_structure(
    structure: Structure,
) -> Structure:
    """
    Move the center of mass of the structure towards the
    center of the cell.

    Parameters
    ----------
    structure : Structure
        Target structure to move to the center.

    Returns
    -------
    Structure
        Pymatgen structure containing a structure placed on the center,
        with the same attributes as the original.
    """

    # Getting the center of mass
    com = get_center_of_mass(structure)

    # Centering
    dist = np.array(structure.lattice.abc) / 2 - com
    modified_coords = structure.cart_coords + dist

    # Generating new pymatgen structure with the modifications
    new_structure = Structure(
        lattice=structure.lattice,
        species=structure.species,
        coords=modified_coords,
        coords_are_cartesian=True,
        site_properties=structure.site_properties,
    )

    return new_structure


def bottom_structure(
    structure: Structure,
    offset: int = 2,
) -> Structure:
    """
    Move the structure towards the bottom of the cell, leaving a
    offset wide margin at the bottom.

    Parameters
    ----------
    structure : Structure
        Target structure to move to the bottom.
    offset : int, optional
        Separation to be left between the bottom of the cell
        and the structure, by default 2, in Angstrom.


    Returns
    -------
    Structure
        Pymatgen structure containing a structure placed on the bottom,
        with the same attributes as the original.
    """

    # Getting the position closest to the bottom
    bottom = np.min(structure.cart_coords)
    bottom_arr = np.zeros(shape=structure.cart_coords.shape)

    # Applying the offset
    bottom_arr[:, 2] += bottom - offset

    # Substracting the bottom position from the structure plus an offset
    modified_coords = structure.cart_coords + bottom_arr

    # Generating new pymatgen structure with the modifications
    new_structure = Structure(
        lattice=structure.lattice,
        species=structure.species,
        coords=modified_coords,
        coords_are_cartesian=True,
        site_properties=structure.site_properties,
    )

    return new_structure


def get_center_of_mass(structure: Structure):
    """
    Get the center of mass (COM) of a given structure.
    The center of mass is computed by using:

        `COM = sum(r_i*m_i)/sum(m_i)`

    Where r_i are the coordinates for each atom and
    m_i their atomic masses.

    Parameters
    ----------
    structure : Structure
        Structure for which the COM will be found

    Returns
    -------
    np.array
        Coordinates of the COM
    """

    # Getting the atomic mass for each atom and the total mass
    atomic_mass_arr = np.array([atom.atomic_mass for atom in structure.species])
    total_mass = np.sum(atomic_mass_arr)

    # Multiplying the coordinates by the atomic mass, adding an additional empty axis
    com = np.multiply(structure.cart_coords, atomic_mass_arr[:, None])
    com = np.sum(com, axis=0)

    # Dividing by the total mass in order to get the center of mass
    com /= total_mass

    return com


def apply_replacement_cluster(
    db_obj: "mdb_indb.InitialDatabase",
    cluster: mdb_struct.Cluster,
    phase: mdb_phase.Phase,
    num_struct: int,
    num_repeat: int,
):
    replaced_clusters = []
    rng = np.random.default_rng()
    rnd_ratios = (phase.base_elem_comp_max - phase.base_elem_comp_min) * rng.random(
        size=num_struct
    ) + phase.base_elem_comp_min
    (other_elem,) = db_obj.ALLOY_SET - {phase.cluster_elem}

    structure_len = len(cluster.structure.species)

    for repl_ind, ratio in enumerate(rnd_ratios):
        for repeat_ind in range(num_repeat):
            n_replace = int(structure_len * ratio)
            if n_replace == 0:
                n_replace = 1

            other_elem_choices = rng.choice(
                a=structure_len,
                size=n_replace,
                replace=False,
                shuffle=True,
            )

            # Creating a new pymatgen structure using the base one as a template
            new_structure = cluster.structure.copy(sanitize=True)
            site_props_before = cluster.structure.site_properties

            # Replacing atoms in the structures
            for ind in other_elem_choices:
                new_structure.replace(ind, other_elem)

            # TODO: Instead of this, create a new structure
            # Copying site properties
            new_structure = new_structure.copy(
                sanitize=True, site_properties=site_props_before
            )

            cluster_name = (
                f"cluster_{new_structure.formula}_replacement"
                f"-{n_replace}_repeat-{repeat_ind}"
            )
            clust_obj = mdb_struct.Cluster(
                material_name=cluster_name,
                structure=new_structure,
                replacement=True,
                replacement_ind=repl_ind,
                phase=phase,
            )
            replaced_clusters.append(clust_obj)

    return replaced_clusters


def apply_gauss_perturb(repeat: int, cluster_list: list, center: float = 0.04):
    perturbed_clusters = []
    for cluster in cluster_list:
        for perturb_repeat_idx in range(repeat):
            # Applying displacement
            new_struct_perturb = mdb_utils.gauss_perturb(
                center=center, structure=cluster
            )

            mat_str = f"{cluster.material_name}_perturb_gauss_{perturb_repeat_idx+1}"
            clust_obj = mdb_struct.Cluster(
                material_name=mat_str,
                structure=new_struct_perturb,
                replacement_ind=cluster.replacement_ind,
                phase=cluster.phase,
                perturb=True,
            )
            perturbed_clusters.append(clust_obj)

    return perturbed_clusters
