import itertools as it
import warnings
from multiprocessing import Pool
from typing import Union

import catkit.gen.surface as cts
import numpy as np
from pymatgen.core.structure import Lattice, Structure
from pymatgen.core.surface import Slab
from pymatgen.io.ase import AseAtomsAdaptor

import MatDBForge.core.phase_diagram as mdb_pd
import MatDBForge.core.structure as mdb_struct


def gen_perc_surfaces(
    db_obj,
    phase: mdb_pd.Phase,
    num_struct: int,
    current_perc: float,
    relative=True,
) -> list:
    """
    Generate num_struct percentages for a structure in a given phase.
    The percentages represent the ratio of the base element of the
    structure's phase.

    Parameters
    ----------
    phase : Phase
        Phase that will be used to define the limits of the percentage values by
        checking its base_elem range.
    num_struct : int
        Number of percentages that will be generated
    current_perc : float
        Percentage of the current structure
    relative : bool, optional
        Whether to return the percentages by themselves or relative
        to the percentage of the current structure, by default True

    Returns
    -------
    list[float]
        List of floats containing the generated percentages.
    """

    # Getting offset. If not found set to 0.
    offset = phase.offset

    # Randomly generating base_elem percentages for the new structures
    max_base_elem = (phase.base_elem_comp_min) + offset
    if max_base_elem > 1:
        max_base_elem = 1

    min_base_elem = (phase.base_elem_comp_max) - offset
    if min_base_elem < 0:
        min_base_elem = 0

    subst_base_elem_perc = (min_base_elem - max_base_elem) * np.random.ranf(
        size=num_struct
    ) + max_base_elem

    if relative:
        adjusted_perc = [(per - current_perc) for per in subst_base_elem_perc]
        return adjusted_perc
    else:
        return subst_base_elem_perc


def make_clean_surf(
    db_obj,
    bulk: Union[Structure, Slab],
    max_num_at: float,
    n_layers: int,
    miller_list: list,
    fixed: int,
):
    img_miller = []
    images = []

    for miller in miller_list:
        # Object that allows to generate the slab.
        # The attach_grap parameter is disabled to increase speed.
        gen = cts.SlabGenerator(
            bulk,
            miller_index=(miller),
            layers=n_layers,
            attach_graph=False,
            layer_type="angs",
            fixed=fixed,
            standardize_bulk="True",
            vacuum=7.5,
        )

        # Getting unique terminations for the current surface
        termination = gen.get_unique_terminations()

        for ind, t in enumerate(termination):
            img_miller.append(miller)
            imgsize = gen.get_slab(iterm=ind).get_global_number_of_atoms()
            slab_rep = int(max_num_at / imgsize)

            try:
                slab = gen.get_slab(iterm=ind, size=slab_rep)
            except Exception:
                break

            images.append(slab)

    return images, img_miller


def make_clean_surf_mp(
    db_obj,
    bulk: Union[Structure, Slab],
    max_num_at: float,
    n_layers: int,
    miller_list: list,
    fixed: int,
):
    img_miller = []
    images = []

    # gen = make_generator_slab
    # Original parameters:
    # fixed = 3

    with Pool() as p:
        slabs_worker = p.starmap(
            db_obj._gen_slab_pool,
            zip(
                miller_list,
                it.repeat(bulk),
                it.repeat(max_num_at),
                it.repeat(n_layers),
                it.repeat(fixed),
            ),
        )
        for slb, mill in slabs_worker:
            if isinstance(slb, list):
                for i in slb:
                    images.append(i)
            if isinstance(mill, list):
                for m in mill:
                    img_miller.append(m)

    return images, img_miller


def _gen_curr_surface(
    db_obj,
    phase,
    curr_bulk_ase,
    n_layers,
    n_at,
    max_miller_index,
    fixed_layers,
    get_supercells,
    limit_per_phase,
):
    # Filtering specific catkit warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    n_layers = int(n_layers)
    n_at = int(n_at)

    slabs = []
    miller = cts.get_unique_indices(
        bulk=curr_bulk_ase,
        max_index=max_miller_index,
    )

    slabs, miller_idx_slabs = make_clean_surf(
        db_obj=db_obj,
        bulk=curr_bulk_ase,
        n_layers=n_layers,
        max_num_at=n_at,
        miller_list=miller,
        fixed=fixed_layers,
    )

    # Will contain tuples as such: (Structure, miller_index_string)
    slabs_bottom = []
    for ind, (slab, mill) in enumerate(zip(slabs, miller_idx_slabs)):
        curr_surf_pymg = AseAtomsAdaptor().get_structure(slab)
        slab = slab_to_bottom(db_obj=db_obj, slab=curr_surf_pymg)
        mill_str = db_obj._get_miller_index_str(mill)

        # INFO: The _adjust_vacuum function does not work correctly.
        # As of now, the vacuum size is being defined during slab creation,
        # I suspect it is related to the way pymatgen handles lattices.
        #
        # if not db_obj._check_correct_vacuum_size(slab, min_vacuum_size):
        #     slab = db_obj._adjust_vacuum(slab, min_vacuum_size)

        slabs_bottom.append((slab, mill_str))

    prototype = phase.prototype
    generated_structures = []

    # Getting only the slabs and their miller index whose total size
    # is smaller than the maximum given for the InitialDatabase.
    slabs_size = [
        (slab, mill)
        for slab, mill in slabs_bottom
        if len(slab.sites) < db_obj.max_num_atoms
    ]

    # Storing the remaining slabs.
    for idx, (slab, mill) in enumerate(slabs_size):
        # Getting the current slab's miller index
        mill_str = db_obj._get_miller_index_str(mill)

        # Preparing the structure name
        surf_name = (
            f"{prototype}_{phase.name}_pure_surface_{mill_str}-{idx+1}"
            f"_{n_layers}-layers_{n_at}-max-at"
        )

        # Creating a new surface from the supercell
        curr_strct = mdb_struct.Surface(
            material_name=surf_name,
            material_id=prototype,
            surface_miller=mill_str,
            structure=slab,
            temperature=np.nan,
            perturb=False,
            base=False,
            calc_performed=False,
            phase=phase,
        )
        # Saving the bulk to the db.
        db_obj.df = curr_strct.save_to_db(db_obj.df)
        generated_structures.append(curr_strct)

    # Getting supercells
    if get_supercells:
        for idx, (slab, mill) in enumerate(slabs_size):
            super_list, idx_list, supercells = db_obj._find_supercell_indices(
                structure=curr_surf_pymg,
                max_atoms=db_obj.max_num_atoms,
                get_different_supercells=True,
                initial_supercell_size=3,
                verbose=False,
            )

            # Storing the supercells.
            for supercell, idx, sup_vec in zip(super_list, idx_list, supercells):
                if len(supercell.sites) <= db_obj.max_num_atoms:
                    # Dragging the slab to the bottom
                    supercell_bottom = db_obj._slab_to_bottom(curr_surf_pymg)

                    # Preparing the structure name
                    surf_name = (
                        f"{prototype}_{phase.name}_pure_surface-"
                        f"{n_layers}-layers_{n_at}-max-at_{db_obj._get_miller_index_str(mill)}-super-{idx+1}"
                    )

                    # Creating a new surface from the supercell
                    curr_strct = mdb_struct.Surface(
                        material_name=surf_name,
                        material_id=prototype,
                        structure=supercell_bottom,
                        temperature=np.nan,
                        perturb=False,
                        base=False,
                        calc_performed=False,
                        phase=phase,
                        supercell=sup_vec,
                    )

                    # Saving the bulk to the db.
                    db_obj.df = curr_strct.save_to_db(db_obj.df)
                    generated_structures.append(curr_strct)

    return generated_structures
    # return surf_name


def gen_slab_pool(db_obj, miller, bulk, max_num_at, n_layers, fixed):
    img_miller = []
    images = []

    gen = cts.SlabGenerator(
        bulk,
        miller_index=(miller),
        layers=n_layers,
        layer_type="angs",
        fixed=fixed,
        standardize_bulk="True",
        vacuum=7.5,
    )

    # Getting unique terminations for the current
    # surface
    termination = gen.get_unique_terminations()

    for ind, t in enumerate(termination):
        img_miller.append(miller)
        imgsize = gen.get_slab(iterm=ind).get_global_number_of_atoms()
        slab_rep = int(max_num_at / imgsize)

        try:
            slab = gen.get_slab(iterm=ind, size=slab_rep)
        except Exception:
            break

        images.append(slab)

    return images, img_miller


def adjust_vacuum(db_obj, slab: Slab, vacuum_size: float) -> Slab:
    # Getting 'c' vector for the cell
    vec_c = slab.lattice.c

    # Getting position of the topmost layer
    z_axis_max = max(slab.cart_coords[:, 2])

    #
    current_vacuum_size = vec_c - z_axis_max

    # Computing correct slab size
    corr_slab_size = z_axis_max + vacuum_size

    # Computing the difference between the correct slab
    diff = vec_c - corr_slab_size
    # print('vec_c: ', vec_c)
    # print('top layer: ', z_axis_max)
    # print('calculated distance:',vec_c-z_axis_max)
    # print('corr_slab_size: ', corr_slab_size)

    # Changing the 'c' vector
    if current_vacuum_size > vacuum_size:
        new_vec_c = vec_c - diff
    elif current_vacuum_size < vacuum_size:
        new_vec_c = vec_c + diff

    # Creating a new abc vector
    new_latt_abc = np.array(slab.lattice.abc)
    new_latt_abc[-1] = new_vec_c

    # Converting the abc vector into a 3x3 matrix
    new_latt_matrix = np.zeros([3, 3])
    diag = np.diag_indices(3)
    new_latt_matrix[diag] = new_latt_abc

    # Creating a lattice use the new matrix
    new_lattice = Lattice(matrix=new_latt_matrix)

    # Using the lattice to create a new slab
    new_slab = Structure(
        lattice=new_lattice,
        species=slab.species,
        coords=slab.cart_coords,
        coords_are_cartesian=True,
    )
    return new_slab


def slab_to_bottom(
    db_obj,
    slab: Union[Slab, Structure],
    offset: int = 2,
) -> Structure:
    """
    Move the slab towards the bottom of the cell, leaving a
    offset wide margin at the bottom.

    Parameters
    ----------
    slab : Union[Slab, Structure]
        Target slab to move to the bottom.
    offset : int, optional
        Separation to be left between the bottom of the cell
        and the slab, by default 2, in Angstrom.


    Returns
    -------
    Structure
        Pymatgen structure containing slab placed on the bottom,
        with the same attributes as the original.
    """

    # Getting the position closest to the bottom
    bottom = min(slab.cart_coords[:, 2])
    bottom_arr = np.zeros(shape=slab.cart_coords.shape)

    # Applying the offset
    bottom_arr[:, 2] += bottom - offset

    # Substracting the bottom position from the slab plus an offset
    modified_coords = slab.cart_coords - bottom_arr

    new_slab = Structure(
        lattice=slab.lattice,
        species=slab.species,
        coords=modified_coords,
        coords_are_cartesian=True,
        site_properties=slab.site_properties,
    )

    return new_slab


def check_correct_vacuum_size(
    db_obj,
    slab: Union[Slab, Structure],
    vacuum_size: float,
    tolerance: float = 0.5,
) -> bool:
    # Getting 'c' size for the cell
    vec_c = slab.lattice.c

    # Getting position of the topmost layer
    z_axis_max = max(slab.cart_coords[:, 2])

    # Getting vacuum layer thickness by substracting
    vac_layer_thickness = vec_c - z_axis_max

    # Checking if layer is greater or equal than vacuum_size
    if abs(vac_layer_thickness - vacuum_size) <= tolerance:
        return True
    else:
        return False
