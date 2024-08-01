"""DFT benchmark utilities for the MatDBForge package."""

import pathlib as plb

import numpy as np
import pymatgen.io.vasp as Vasp


def gen_kspacing_from_vasp_input(
    path: plb.Path,
    kpoints: Vasp.Kpoints = None,
    poscar: Vasp.Poscar = None,
) -> np.array:
    """
    Generate an array containing the kpoint density for the compound
    on the 3 axis, in 2pi*A^-1 units, using VASP files.
    This requires a POSCAR and a KPOINTS from vasp, which can be
    specified using either a single path or by passing their equivalent
    pymatgen objects as arguments.

    Parameters
    ----------
    path : plb.Path | str
        Path object or string representing a path where both required files are located
    kpoints : Vasp.Kpoints, optional
        pymatgen Kpoints object, by default None
    poscar : Vasp.Poscar, optional
        pymatgen Poscar object, by default None

    Returns
    -------
    np.array
        3 element array containing the kpoint density on each axis, in 2pi*A^-1 units.
    """
    # Read POSCAR and KPOINTS
    if path:
        read_poscar = Vasp.Poscar.from_file(path / "POSCAR")
        read_kpoints = Vasp.Kpoints.from_file(path / "KPOINTS")

    # If kpoints or poscar given by the user, use them instead of the ones
    # in the path.
    if kpoints:
        read_kpoints = kpoints
    if poscar:
        read_poscar = poscar

    # Computing kpt density
    kpt_dens_arr = compute_kpt_density(kpoints=read_kpoints, poscar=read_poscar)

    return kpt_dens_arr


def compute_kpt_density(kpoints: Vasp.Kpoints, poscar: Vasp.Poscar):
    # Getting the kpoint density
    # Getting array of kpoints
    arr_kpt_run = np.array(kpoints.kpts[0])

    # Getting lattice vectors
    l_mat = poscar.structure.lattice.matrix

    # Getting volume of the reciprocal cell
    v_mat = np.dot(np.cross(l_mat[0, :], l_mat[1, :]), l_mat[2, :])

    # Computing values for each axis
    a_rcpr = np.linalg.norm((np.cross(l_mat[1, :], l_mat[2, :])) / v_mat)
    b_rcpr = np.linalg.norm((np.cross(l_mat[0, :], l_mat[2, :])) / v_mat)
    c_rcpr = np.linalg.norm((np.cross(l_mat[0, :], l_mat[1, :])) / v_mat)

    # Computing the kpt density values in an array
    kpt_dens_arr = np.array((a_rcpr, b_rcpr, c_rcpr)) * (1 / arr_kpt_run)

    return kpt_dens_arr
