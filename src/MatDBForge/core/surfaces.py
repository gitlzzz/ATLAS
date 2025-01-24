"""Module containing functions to generate surfaces from structures."""

import itertools as it
import multiprocessing as mp
import pathlib as pl
import pickle
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from pymatgen.core.structure import Lattice, Structure
from pymatgen.core.surface import Slab, SlabGenerator

import MatDBForge.core.code_utils as mdb_cud
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
    slab: Slab | Structure,
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
    slab: Slab | Structure,
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


def process_row_parallel(
    row,
    phase,
    phase_diagram,
    max_miller_index,
    min_miller_index,
    min_slab_size,
    min_vacuum_size,
    get_supercells,
    num_replacements,
    num_repeat_replace,
    fixed_layers,
    min_num_atoms,
    overwrite_max_num_atoms,
    frac_slabs_save,
    frac_supercells_save,
    limit_total_num_struct,
    rng_seed,
    max_slab_num,
    row_idx,
    temp_folder,
):
    rng = np.random.default_rng(seed=rng_seed)

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
            min_slab_size=min_slab_size,
            min_vacuum_size=min_vacuum_size,
            center_slab=True,
            lll_reduce=True,
        )
        slabs = slabgen.get_slabs(filter_out_sym_slabs=True)
        for slab in slabs:
            total_slabs.append((slab_to_bottom(slab=slab), miller))

    # Removing slabs from memory
    del slabs

    # Randomly choosing a subset of the slabs
    max_slab_num = min(max_slab_num, len(total_slabs))
    mdb_cud.custom_print(
        (
            f"Generated {len(total_slabs)} slabs for {row.material_id}."
            f" Limited to {max_slab_num} slabs."
        ),
        "debug",
    )
    total_slabs_idx = rng.choice(range(len(total_slabs)), max_slab_num, replace=False)
    total_slabs = [total_slabs[idx] for idx in total_slabs_idx]

    generated_structures = []
    fix_layers_warn = True

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
        # TODO: Implement this feature and remove warning
        if fixed_layers and fix_layers_warn:
            mdb_cud.custom_print(
                "`fixed_layers` specified, but not implemented yet.",
                "debug",
            )
            fix_layers_warn = False
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
            targeted_modification=row.targeted_modification,
            phase=phase.name,
        )

        # Saving the surface to the db.
        # db_obj.df = curr_strct.save_to_db(db_obj.df)
        generated_structures.append(curr_strct)

    # Getting supercells
    supercell_list = []
    if get_supercells:
        for _, (slab, mill) in enumerate(total_slabs):
            super_list, idx_list, supercells = mdb_ut.find_supercell_indices(
                structure=slab,
                min_atoms=min_num_atoms,
                max_atoms=overwrite_max_num_atoms,
                get_different_supercells=True,
                initial_supercell_size=5,
                verbose=True,
            )

            # Storing the supercells.
            for supercell, _, sup_vec in zip(
                super_list, idx_list, supercells, strict=False
            ):
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
                        targeted_modification=row.targeted_modification,
                        calc_performed=False,
                        phase=phase.name,
                        supercell=sup_vec,
                        surface_miller=mill_str,
                    )

                    supercell_list.append(curr_strct)

    # Get replacements
    replacement_list = []
    for structure_obj, supr_idx in zip(supercell_list, idx_list, strict=False):
        structure = structure_obj.structure

        # Replacing some atoms using symmetry
        structure = mdb_ut.create_symmetrical_prototype(
            structure=structure,
            phase_diagram=phase_diagram,
            phase=phase,
            structure_obj=structure_obj,
        )
        # Preparing an array of randomly generated base elem percentages
        # for the new structures
        subst_base_elem_perc = mdb_ut.gen_base_elem_perc(phase, num_replacements)

        mdb_cud.custom_print(
            f"Random base element % for surface to gen: {subst_base_elem_perc*100}",
            "debug",
        )

        # Attempting to fix any percentages outside of the
        # current phase ratios.
        # n_at_replacement_upd is a list which contains the
        # target number of base atoms in the new structure.
        n_at_replacement_upd = mdb_ut.fit_replacements_phase(
            phase, structure, subst_base_elem_perc
        )

        # Replacing the atoms and generate 'num_repeat_replace' structures
        # for each percentage
        for str_ind, n_atoms in enumerate(n_at_replacement_upd):
            for repl in range(num_repeat_replace):
                # Applying the replacement
                new_structure = mdb_ut.apply_replacement(
                    structure=structure,
                    phase=phase,
                    n_target_at=n_atoms,
                    phase_diagram=phase_diagram,
                    rng=rng,
                )

                # Getting the supercell vector
                supercell_vec_str = get_miller_index_str(structure_obj.supercell)

                bulk_temp = np.nan

                # Creating a new Bulk object for the structure with replacement
                new_struct_symm = mdb_struct.Surface(
                    material_name=f"{row.material_id}_{phase.name}_super-{supercell_vec_str}-{supr_idx}_replacement-{str_ind+1}-{repl+1}",
                    material_id=row.material_id,
                    targeted_modification=structure_obj.targeted_modification,
                    structure=new_structure,
                    temperature=bulk_temp,
                    perturb=False,
                    replacement=True,
                    replacement_ind=(str_ind + 1, repl + 1),
                    base=False,
                    cluster=False,
                    calc_performed=False,
                    supercell=structure_obj.supercell,
                    surface_miller=structure_obj.surface_miller,
                    phase=phase.name,
                )
                replacement_list.append(new_struct_symm)

    # Getting a random subset of the initial slabs
    if len(generated_structures) > 0:
        generated_structures = rng.choice(
            generated_structures,
            int(len(generated_structures) * frac_slabs_save),
            replace=False,
        )

    # Getting a random subset of the inital supercells
    supercell_list = rng.choice(
        supercell_list,
        int(len(supercell_list) * frac_supercells_save),
        replace=False,
    )

    generated_structures = np.concatenate(
        (generated_structures, supercell_list, replacement_list)
    )

    # Limiting the number of generated supercells to
    # the supercell limit.
    mdb_cud.custom_print(
        f"Length of the supercell+replacement list: {len(generated_structures)}",
        "debug",
    )

    if len(generated_structures) > limit_total_num_struct:
        mdb_cud.custom_print(
            (
                f"Limiting the number of slabs ({len(generated_structures)})"
                f" to {limit_total_num_struct}."
            ),
            "debug",
        )

        generated_structures = np.random.choice(
            generated_structures, size=limit_total_num_struct, replace=False
        )

    with open(f"{temp_folder}/generated_structures_{row_idx}.pkl", "wb") as f:
        pickle.dump(
            generated_structures,
            file=f,
        )

    del total_slabs
    del supercell_list
    del replacement_list
    del generated_structures


def gen_surfaces_diff_miller_parallel(
    db_obj: "mdb_indb.InitialDatabase",
    phase: mdb_pd.Phase,
    max_miller_index: int,
    min_miller_index: int = 2,
    min_slab_size: float = 6,
    min_vacuum_size: float = 10,
    get_supercells=False,
    num_replacements: int = 10,
    num_repeat_replace: int = 3,
    fixed_layers: int = 0,
    min_num_atoms: int = 12,
    overwrite_max_num_atoms: int = None,
    save_in_db=False,
    rng_seed: int = 42,
    frac_slabs_save: float = 1.0,
    frac_supercells_save: float = 1.0,
    limit_total_num_struct: int = 0,
    max_slab_num: int = 15,
    n_workers=None,
):
    # Instantiating RNG
    # rng = np.random.default_rng(seed=rng_seed)

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

    # Adding a progress bar if the number of structures
    # is greater than 100
    if len(base_structs) > 10:
        transient = True
        disable = False
    else:
        transient = False
        disable = True

    from rich import progress

    # Limit n_workers to the number of structures to process
    # if below the number of available cpus.
    # Additionally, limit the user given n_workers to the number
    # of available cpus.
    if not n_workers:
        n_workers = mp.cpu_count() - 1
        n_workers = min(n_workers, len(base_structs))
    else:
        n_workers = min(n_workers, mp.cpu_count() - 1)
        n_workers = min(n_workers, len(base_structs))

    if phase.use_cache:
        temp_dir = mdb_cud.get_cache_path() / "mdb"
        temp_dir = temp_dir / f"mdb_gen_init_db_surface_{phase.name}"
        temp_dir.mkdir(parents=False, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp(
            prefix="mdb_gen_init_db_",
        )
    mdb_cud.custom_print(
        f"Storing surfaces on temporary/cache directory: '{temp_dir}'.", "debug"
    )

    generated_structures = []
    if len(list(pl.Path(temp_dir).glob("*"))) == 0:
        with progress.Progress(
            "[progress.description]{task.description}",
            progress.BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            progress.MofNCompleteColumn(),
            progress.TimeRemainingColumn(),
            progress.TimeElapsedColumn(),
            refresh_per_second=1,
            transient=transient,
            disable=disable,
        ) as progress:
            futures = []
            overall_progress_task = progress.add_task(
                "[green]Generating structures:", total=len(base_structs)
            )

            mdb_cud.custom_print(f"Using '{n_workers}' workers.", "debug")

            # iterate over the jobs we need to run
            with ProcessPoolExecutor(
                max_workers=n_workers,
            ) as executor:
                for (
                    idx,
                    row,
                ) in base_structs.iterrows():
                    # Prepare arguments for function call
                    args = (
                        row,
                        phase,
                        db_obj.phase_diagram,
                        max_miller_index,
                        min_miller_index,
                        min_slab_size,
                        min_vacuum_size,
                        get_supercells,
                        num_replacements,
                        num_repeat_replace,
                        fixed_layers,
                        min_num_atoms,
                        overwrite_max_num_atoms,
                        frac_slabs_save,
                        frac_supercells_save,
                        limit_total_num_struct,
                        rng_seed,
                        max_slab_num,
                        idx,
                        temp_dir,
                    )

                    futures.append(executor.submit(process_row_parallel, *args))

                # monitor the progress
                while (n_finished := sum([future.done() for future in futures])) < len(
                    futures
                ):
                    progress.update(
                        overall_progress_task,
                        completed=n_finished,
                        total=len(base_structs),
                    )
                    time.sleep(0.5)

                # Final update
                progress.update(
                    overall_progress_task,
                    completed=n_finished,
                    total=len(base_structs),
                )
    else:
        mdb_cud.custom_print(
            f"Gathering surfaces from temporary/cache directory: '{temp_dir}'.", "info"
        )
    # Reading stored temporary files
    for idx, _ in base_structs.iterrows():
        # Load the generated structures
        # with open(
        # f"/tmp/mdb_gen_init_db_sgem6zp2/generated_structures_{idx}.pkl", "rb"
        # ) as f:
        with open(f"{temp_dir}/generated_structures_{idx}.pkl", "rb") as f:
            generated_structures.append(pickle.load(f))

    # Concatenating lists
    generated_structures = np.concatenate(generated_structures)

    mdb_cud.custom_print(f"Generated {len(generated_structures)} surfaces.", "debug")

    # Saving the structures in the db.
    if save_in_db:
        mdb_cud.custom_print("Saving replaced structures in dataframe.", "debug")
        for slab in generated_structures:
            slab.save_to_db(db_obj=db_obj)
        mdb_cud.custom_print(
            f"Dataframe shape after saving: {db_obj.df.shape}", "debug"
        )

    # Removing temporary directory
    if not phase.use_cache:
        mdb_cud.custom_print(f"Removing temporary directory: '{temp_dir}'.", "debug")

        # Removing contents
        for file in pl.Path(temp_dir).glob("*"):
            file.unlink()

        # Removing directory
        pl.Path(temp_dir).rmdir()

    return generated_structures


def apply_replacement_surface(
    db_obj: "mdb_indb.InitialDatabase",
    slabs_to_replace: list,
    save_in_db: bool = False,
    num_replacement_structs: int = 3,
    num_replacement_repeats: int = 2,
    limit_replacements: int = None,
):
    mdb_cud.custom_print(
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

    mdb_cud.custom_print(
        f"Generated {len(replacement_list)} replaced surfaces.", "debug"
    )

    # Limiting the number of replaced surfaces
    if limit_replacements and len(replacement_list) > limit_replacements:
        replacement_list = rng.choice(
            replacement_list, size=limit_replacements, replace=False
        )

        mdb_cud.custom_print(
            f"Limited number of replaced surfaces to {len(replacement_list)}.", "debug"
        )

    if save_in_db:
        mdb_cud.custom_print("Saving replaced surfaces in dataframe.", "debug")
        for slab in replacement_list:
            slab.save_to_db(db_obj=db_obj)
        mdb_cud.custom_print(
            f"Dataframe shape after saving: {db_obj.df.shape}", "debug"
        )

    return replacement_list


class AdsorbateAdder:
    """Class to add adsorbates to surfaces."""

    def __init__(self, cutoff=1.5, coverage=1.0):
        """
        Initialize adsorbate adder with cutoff and coverage.

        Parameters
        ----------
        cutoff : float
            Distance threshold to determine if a site is exposed.
        coverage : float
            Fraction of sites to be used for adsorption.
        """
        self.cutoff = cutoff
        self.coverage = coverage

    def is_exposed(self, pos, all_positions, z_pos, z_max):
        """Check if an atom is exposed."""
        # Mark atom as exposed if it is among the highest z-axis atoms
        if abs(z_pos - z_max) < 0.1:
            return True

        # Standard exposure check
        for other_pos in all_positions:
            if (
                other_pos[2] > z_pos + 0.1
                and np.linalg.norm(other_pos[:2] - pos[:2]) < self.cutoff
            ):
                return False
        return True

    # TODO: Add way of checking the main element of the structure
    def find_surface_atoms(self, slab):
        """
        Identify exposed surface atoms, considering Ir and O.

        Parameters
        ----------
        slab : ASE Atoms object
            The slab structure.

        Returns
        -------
        surface_ir : list of tuples
            Indices and positions of exposed Ir atoms.
        surface_o : list of tuples
            Indices and positions of exposed O atoms.
        """
        positions = slab.get_positions()
        symbols = slab.get_chemical_symbols()
        z_positions = positions[:, 2]
        z_max = max(z_positions)

        surface_ir = []
        surface_o = []

        for i, (pos, symbol) in enumerate(zip(positions, symbols, strict=False)):
            if self.is_exposed(pos, positions, pos[2], z_max):
                if symbol == "Ir":
                    surface_ir.append((i, pos))
                elif symbol == "O":
                    surface_o.append((i, pos))

        # Apply coverage to limit the number of sites
        if self.coverage < 1.0:
            surface_ir = surface_ir[: int(len(surface_ir) * self.coverage)]
            surface_o = surface_o[: int(len(surface_o) * self.coverage)]

        return surface_ir, surface_o

    def add_adsorbate_at_distance(self, slab, ads, surface_pos, distance):
        """Add an adsorbate at a fixed distance from the surface atom."""
        direction = np.array([0.0, 0.0, 1.0])
        target_pos = surface_pos + direction * distance
        ads.translate(target_pos)
        slab.extend(ads)

    def add_adsorbates(self, slab, ir_sites, o_sites):
        """Add adsorbates to surface sites."""
        from ase.build import molecule

        structures = []

        # Add H2O on Ir sites
        for i, (_, pos) in enumerate(ir_sites):
            slab_copy = slab.copy()
            water = molecule("H2O")
            water.rotate(180, "x")  # O end down
            self.add_adsorbate_at_distance(slab_copy, water, pos, 2.0)
            structures.append(("H2O", i, slab_copy))

        # Add H on O sites
        for i, (_, pos) in enumerate(o_sites):
            slab_copy = slab.copy()
            hydrogen = molecule("H")
            self.add_adsorbate_at_distance(slab_copy, hydrogen, pos, 1.5)
            structures.append(("H", i, slab_copy))

        return structures
