"""Module containing functions to generate surfaces from structures."""

import itertools as it
from typing import Union

import numpy as np
from pymatgen.core.structure import Lattice, Structure
from pymatgen.core.surface import Slab, SlabGenerator

import MatDBForge.core.exceptions as mdb_exc
import MatDBForge.core.initial_db as mdb_indb
import MatDBForge.core.phase_diagram as mdb_pd
import MatDBForge.core.structure as mdb_struct
import MatDBForge.core.utils as mdb_ut


def gen_perc_surfaces(
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
    slab: Union[Slab, Structure],
    offset: int = 2,
    return_mdb_struct=True,
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
    # If the cell has any vector with z coordinates lower
    # than zero, change the offset to consider this
    if np.any(slab.lattice.matrix[:, 2] < 0):
        min_val = np.min(slab.lattice.matrix[:, 2])
        offset = offset + min_val

    # Getting the position closest to the bottom
    bottom = min(slab.cart_coords[:, 2])
    bottom_arr = np.zeros(shape=slab.cart_coords.shape)

    # Applying the offset
    bottom_arr[:, 2] += bottom - offset

    # Substracting the bottom position from the slab plus an offset
    modified_coords = slab.cart_coords - np.abs(bottom_arr)

    # Delete the else and the conditional structure
    new_slab = Slab(
        lattice=slab.lattice,
        species=slab.species,
        coords=modified_coords,
        coords_are_cartesian=True,
        site_properties=slab.site_properties,
        miller_index=slab.miller_index,
        oriented_unit_cell=slab.oriented_unit_cell,
        shift=slab.shift,
        scale_factor=slab.scale_factor,
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
    return abs(vac_layer_thickness - vacuum_size) <= tolerance


def get_miller_index_str(miller_source):
    """
    Generate a miller index string from several sources,
    either a Slab structure, a numpy array with the indices
    or a string.
    The intended use of this string is for labeling structures
    and helping identification.

    Parameters
    ----------
    miller_source : Slab | np.ndarray | str
        Information about the miller indices used to
        generate the string.

    Returns
    -------
    str
        Miller indices coded as a string, without including brackets.
        Negative signs are added in front of the symbols.

    """
    if isinstance(miller_source, Slab):
        curr_miller = str(miller_source.miller_index)
    elif isinstance(miller_source, (np.ndarray, tuple, list)):
        curr_miller = str(miller_source)
    elif isinstance(miller_source, list):
        curr_miller = "".join(map(str, miller_source))
    elif isinstance(miller_source, str):
        curr_miller = miller_source
    else:
        # Return None if the given structure is not a surface.
        return None

    replace_chars = ["'", ",", " ", "(", ")", "[", "]"]
    for char in replace_chars:
        curr_miller = curr_miller.replace(char, "")

    return curr_miller


def gen_surfaces_diff_miller(
    db_obj: "mdb_indb.InitialDatabase",
    phase: mdb_pd.Phase,
    # num_diff_layer_size: int,
    max_miller_index: int,
    min_miller_index: int = 2,
    min_slab_size: float = 6,
    min_vacuum_size: float = 10,
    get_supercells=False,
    fixed_layers: int = 0,
    min_num_atoms: int = 12,
    overwrite_max_num_atoms: int = None,
    # limit_per_phase: int = None,
    limit_supercell: int = None,
    save_in_db=False,
):
    # Getting the current phase from the phase name.
    if isinstance(phase, str):
        phase = mdb_indb.CuZnInitialDatabase.DB_PHASE_DIAGRAM.get_phase(phase)

    base_structs = db_obj.get_base_structs_current_phase(phase)

    if not overwrite_max_num_atoms:
        overwrite_max_num_atoms = db_obj.max_num_atoms

    # Checking if there are any base structures for the current
    # phase.
    if len(base_structs) == 0:
        err_msg = (
            f"No base structure could be found for phase {phase.name}."
            "\nThe database must contain base structures before "
            "running this function."
        )

        raise mdb_exc.BaseStructureNotFound(err_msg)

    for idx, row in base_structs.iterrows():
        total_slabs = []
        miller_indices = list(
            it.combinations_with_replacement(
                range(min_miller_index, max_miller_index + 1), 3
            )
        )[1:]

        for miller in miller_indices:
            slabgen = SlabGenerator(
                row.structure,
                miller_index=miller,
                min_slab_size=6,
                min_vacuum_size=12,
                center_slab=True,
                lll_reduce=True,
            )
            slabs = slabgen.get_slabs()
            for slab in slabs:
                total_slabs.append((slab_to_bottom(slab=slab), miller))

        generated_structures = []
        # Storing the remaining slabs.
        for idx, (slab, mill) in enumerate(total_slabs):
            # Getting the current slab's miller index
            mill_str = get_miller_index_str(mill)

            # Preparing the structure name
            surf_name = (
                f"{row.material_id}_{phase.name}_pure_surface_{mill_str}-{idx+1}"
                f"_min_vac-{min_vacuum_size}_min_slab-{min_slab_size}_{len(row.structure)}-max-at"
            )

            # Fix the bottom `fixed_layers` number of layers
            if fixed_layers:
                mdb_ut.custom_print(
                    "`fixed_layers` specified, but not implemented yet.",
                    "debug",
                )
                # slab = mdb_ut.fix_bottom_layers(slab, fixed_layers)

            # Creating a new surface from the supercell
            curr_strct = mdb_struct.Surface(
                material_name=surf_name,
                material_id=row.material_id,
                surface_miller=mill_str,
                structure=slab,
                temperature=np.nan,
                perturb=False,
                base=False,
                calc_performed=False,
                phase=phase,
            )

            # Saving the surface to the db.
            db_obj.df = curr_strct.save_to_db(db_obj.df)
            generated_structures.append(curr_strct)

        # Getting supercells
        supercell_list = []
        if get_supercells:
            for _, (slab, mill) in enumerate(total_slabs):
                super_list, idx_list, supercells = db_obj._find_supercell_indices(
                    structure=slab,
                    min_atoms=min_num_atoms,
                    max_atoms=overwrite_max_num_atoms,
                    get_different_supercells=True,
                    initial_supercell_size=5,
                    verbose=True,
                )

                # Storing the supercells.
                for supercell, _, sup_vec in zip(super_list, idx_list, supercells):
                    sup_len = len(supercell.sites)
                    if sup_len <= overwrite_max_num_atoms and sup_len >= min_num_atoms:
                        # Dragging the slab to the bottom
                        supercell_bottom = slab_to_bottom(slab=supercell)

                        # Preparing the structure name
                        surf_name = (
                            f"{row.material_id}_{phase.name}_pure_surface-"
                            f"_min_vac-{min_vacuum_size}_min_slab-{min_slab_size}_{len(row.structure)}-max-at_{get_miller_index_str(mill)}-super-{sup_vec}"
                        )

                        # Creating a new surface from the supercell
                        curr_strct = mdb_struct.Surface(
                            material_name=surf_name,
                            material_id=row.material_id,
                            structure=supercell_bottom,
                            temperature=np.nan,
                            perturb=False,
                            base=False,
                            calc_performed=False,
                            phase=phase,
                            supercell=sup_vec,
                        )

                        supercell_list.append(curr_strct)

            # Limiting the number of generated supercells to
            # the supercell limit.
            mdb_ut.custom_print(
                f"Length of the supercell list: {len(supercell_list)}", "debug"
            )
            if len(supercell_list) > limit_supercell:
                mdb_ut.custom_print(
                    (
                        f"Limiting the number of supercells ({len(supercell_list)})"
                        f" to {limit_supercell}."
                    ),
                    "debug",
                )

                supercell_list = np.random.choice(
                    supercell_list, size=limit_supercell, replace=False
                )

            generated_structures.extend(supercell_list)

        mdb_ut.custom_print(f"Generated {len(generated_structures)} surfaces.", "debug")

        # Saving the structures in the db.
        if save_in_db:
            mdb_ut.custom_print("Saving replaced structures in dataframe.", "debug")
            for slab in generated_structures:
                slab.save_to_db(db_obj=db_obj)
            mdb_ut.custom_print(
                f"Dataframe shape after saving: {db_obj.df.shape}", "debug"
            )

        return generated_structures


def apply_replacement_surface(
    db_obj: "mdb_indb.InitialDatabase",
    slabs_to_replace: list,
    save_in_db: bool = False,
    num_replacement_structs: int = 3,
    num_replacement_repeats: int = 2,
    limit_replacements: int = None,
):
    mdb_ut.custom_print(
        f"Applying replacements to {len(slabs_to_replace)} structures...", "debug"
    )
    rng = np.random.default_rng()

    replacement_list = []

    for _idx, gen_slab in enumerate(slabs_to_replace):
        # Getting current phase and structure length.
        slab_phase = gen_slab.phase

        # Getting the base element percentage of the current structure
        current_perc = slab_phase.get_base_elem_perc(gen_slab.structure)

        # Generating a list of random percentages inside the current phase
        # range.
        gen_percentages = gen_perc_surfaces(
            phase=slab_phase,
            num_struct=num_replacement_structs,
            current_perc=current_perc,
            relative=True,
        )

        # Going over the generated percentages
        for str_ind, n_atoms in enumerate(gen_percentages):
            # Repeating the replacement for each percentage, so that
            # num_replacement_repeats structures are generated with
            # the same ratio but different distribution.
            for repl in range(num_replacement_repeats):
                # Applying the replacement
                new_structure = mdb_ut.apply_replacement(
                    structure=gen_slab,
                    phase=slab_phase,
                    n_atoms=n_atoms,
                    rng=rng,
                )

                # Generating name
                if gen_slab.supercell:
                    supercell_vec_str = get_miller_index_str(gen_slab.supercell)
                    supercell_vec_str_name = f"super-{supercell_vec_str}_"
                else:
                    supercell_vec_str = gen_slab.surface_miller
                    supercell_vec_str_name = supercell_vec_str

                mat_name = (
                    f"{slab_phase.prototype}_{slab_phase.name}_surface"
                    f"-{supercell_vec_str_name}-{str_ind+1}"
                    f"_replacement-{repl + 1}"
                )

                # Creating a new Surface object for the
                # structure with replacement
                new_struct_symm = mdb_struct.Surface(
                    material_name=mat_name,
                    material_id=slab_phase.prototype,
                    surface_miller=supercell_vec_str,
                    structure=new_structure,
                    temperature=gen_slab.temperature,
                    perturb=False,
                    replacement=True,
                    replacement_ind=(str_ind + 1, repl + 1),
                    base=False,
                    calc_performed=False,
                    supercell=gen_slab.supercell,
                    phase=slab_phase,
                )

                replacement_list.append(new_struct_symm)

    mdb_ut.custom_print(
        f"Generated {len(replacement_list)} replaced surfaces.", "debug"
    )

    # Limiting the number of replaced surfaces
    if limit_replacements and len(replacement_list) > limit_replacements:
        replacement_list = np.random.choice(
            replacement_list, size=limit_replacements, replace=False
        )

        mdb_ut.custom_print(
            f"Limited number of replaced surfaces to {len(replacement_list)}.", "debug"
        )

    if save_in_db:
        mdb_ut.custom_print("Saving replaced surfaces in dataframe.", "debug")
        for slab in replacement_list:
            slab.save_to_db(db_obj=db_obj)
        mdb_ut.custom_print(f"Dataframe shape after saving: {db_obj.df.shape}", "debug")

    return replacement_list
