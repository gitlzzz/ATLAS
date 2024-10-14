"""Module containing general utilities for MatDBForge."""

import json as js
import os
import pathlib
import pathlib as pl
import sys
import tempfile
import uuid

import numpy as np
import pandas as pd
from ase import Atoms, visualize
from ase.io import read as ase_read
from dscribe.descriptors import SOAP
from dscribe.kernels import AverageKernel
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element, Species
from pymatgen.io import ase as pmg_ase
from pymatgen.io.ase import AseAtomsAdaptor

import MatDBForge.core.initial_db as mdb_indb
import MatDBForge.core.phase_diagram as mdb_pd
import MatDBForge.core.structure as mdb_struct
from MatDBForge.core.code_utils import custom_print

LINE_UP = "\033[1A"
LINE_CLEAR = "\x1b[2K"
CONFIG_EXAMPLE = {"API_KEY": "XXXXX..."}


def get_config_path() -> pl.Path:
    # Try to get XDG_CONFIG_HOME, if it doesn't exist, return None
    config_path = os.environ.get("XDG_CONFIG_HOME", None)

    # Check if $HOME/.config exists and if it does, return the path
    if not config_path:
        config_folder = pl.Path().home() / ".config"
        if config_folder.exists():
            config_path = config_folder

    return pl.Path(config_path)


def init_config_dir(config_dir):
    # Create a 'mdb' directory inside the config directory
    config_dir = config_dir / "mdb"
    config_dir.mkdir(parents=True, exist_ok=True)

    # Create a 'secrets.json' file inside the 'mdb' directory
    try:
        with open(config_dir / "secrets.json", "x") as f:
            f.write("{\n" '"API_KEY": ""\n' "}")
        return config_dir
    except FileExistsError:
        return None


def clear_previous_print():
    print(LINE_UP, end=LINE_CLEAR)


def gather_secrets():
    """
    Gather Materials project API key from file/env var.

    The API key can be gathered from a secrets.json file that can
    be placed in the config directory or in the current working
    directory. If the file is not found, the function will check
    for an environment variable named 'MP_API_KEY'.

    Notes
    -----
        The json file should have the following structure:

        {
            "API_KEY": "XXXXXX"
        }


    Returns
    -------
    dict
        object containing the api key
    """
    config_path = get_config_path() / "mdb"

    if pathlib.Path("secrets.json").exists():
        with open("secrets.json") as f:
            secrets = js.load(f)

    elif pathlib.Path(config_path, "secrets.json").exists():
        path = pathlib.Path(config_path, "secrets.json")
        with open(path) as f:
            secrets = js.load(f)
    elif os.environ.get("MP_API_KEY"):
        secrets = {"API_KEY": os.environ.get("MP_API_KEY")}

    else:
        print()
        custom_print(
            "[bold red blink]Materials Project secrets missing: "
            "'secrets.json' not found![/]\n"
            "[bold red]Please, either run 'mdb_init_setup', set the 'MP_API_KEY'"
            " environment variable,\nor add a 'secrets.json' file in the"
            f" following directory: '{config_path}',\nwith the following format:[/]\n"
            f"{CONFIG_EXAMPLE}",
            "error",
        )
        secrets = None
        sys.exit(420)

    return secrets


def check_incorrect_ratios(df, curr_phase_diag):
    for _id, row in df.iterrows():
        if not row.base and not row.material_name.endswith("_symm"):
            strct = row.structure.get_sorted_structure()
            name = row.material_name
            phase = curr_phase_diag.get_phase(row.phase)
            offset = phase.offset
            tot_atoms = len(strct.species)
            one_at_perc = 1 / tot_atoms

            tot_cu = strct.species.count(Species("Cu")) + strct.species.count(
                Element("Cu")
            )
            tot_zn = strct.species.count(Species("Zn")) + strct.species.count(
                Element("Zn")
            )

            # Checking the total atom number
            if tot_cu + tot_zn != tot_atoms:
                raise ValueError(
                    "Total count does not match."
                    f" tot_cu: {tot_cu}, tot_zn: {tot_zn}, total: {tot_atoms}."
                    f" Species: {set(strct.species)}"
                )

            perc = round(tot_zn / tot_atoms, 2)

            offset_min = round(phase.base_elem_comp_min - offset, 2)
            if offset_min < 0:
                offset_min = 0

            offset_max = round(phase.base_elem_comp_max + offset, 2)
            if offset_max > 1:
                offset_max = 1

            # Checking if the current structure is between the phase ratio
            # percentages.
            if not (offset_min - 0.1 <= perc <= offset_max + 0.1):
                # If the structure could be fixed by adding or removing an atom
                if (offset_min <= perc + one_at_perc <= offset_max) or (
                    offset_min <= perc - one_at_perc <= offset_max
                ):
                    custom_print(
                        (
                            f"{name}: {perc:.2f} Zn outside of ({offset_min:.2f} -"
                            f" {offset_max:.2f}) range"
                        ),
                        "error",
                    )
                else:
                    pass


def _display_indb_dataframe(structures, data=None):
    """
    Display all of the given structures using the ase gui.

    Parameters
    ----------
    structures : list
        List of structures
    data : dict, optional
        Dictionary containing additional data, by default None
    """
    atoms_obj_list = []
    for structure in structures:
        if not isinstance(structure, Atoms):
            struct_ase = pmg_ase.AseAtomsAdaptor().get_atoms(structure)
        else:
            struct_ase = structure

        atoms_obj_list.append(struct_ase)

    visualize.view(atoms_obj_list, data=data)


def display_dataframe_ase(dataframe):
    structures = dataframe.structure

    data_dict = {"filename": dataframe.material_name}
    _display_indb_dataframe(structures, data=data_dict)


def display_struct_list_ase(struct_list):
    new_struct_list = []
    for strc in struct_list:
        if isinstance(strc, mdb_struct.Structure):
            new_struct_list.append(strc.structure)
        else:
            new_struct_list.append(strc)

    data_dict = {}
    _display_indb_dataframe(new_struct_list, data=data_dict)


def similarity_check_list(
    db_obj, replaced_structures, r_cut=None, n_max=None, l_max=None, save_in_db=True
):
    custom_print(
        f"Checking replacements for {len(replaced_structures)} structures.", "debug"
    )

    # Checking for similarity after replacement
    uuid_list = _check_repeat_struct_list(
        replaced_structures, r_cut=r_cut, l_max=2, n_max=2
    )
    print("uuid_list: ", len(uuid_list))

    # Deleting equivalent structures
    replaced_structures = _del_structure_list_by_uuid(replaced_structures, uuid_list)
    custom_print(
        f"{len(replaced_structures)} structures after duplicate check", "debug"
    )

    if save_in_db:
        custom_print("Saving to db...", "debug")
        for _idx, cluster in enumerate(replaced_structures):
            db_obj._save_row(structure=cluster)

    return replaced_structures


def gauss_perturb(structure: Structure, center: float = 0.04):
    MDB_STRUCT_TYPES = (
        mdb_struct.Structure,
        mdb_struct.Bulk,
        mdb_struct.Cluster,
        mdb_struct.Surface,
    )
    if isinstance(structure, MDB_STRUCT_TYPES):
        structure_obj = structure
        structure = structure.structure

    new_structure = structure.copy()
    new_structure.perturb(distance=center * 2, min_distance=center / 2)

    if isinstance(structure, MDB_STRUCT_TYPES):
        structure_obj.structure = new_structure
        new_structure = structure_obj

    return new_structure


def _check_repeat_struct_list(structure_list, r_cut=6, n_max=8, l_max=6):
    print("r_cut: ", r_cut)

    species_list = [el.Z for el in mdb_indb.CuZnInitialDatabase.ALLOY_SET]

    # Setting up the SOAP descriptor
    soap = SOAP(
        species=species_list,
        periodic=True,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sparse=False,
    )

    soap_structs = []
    pymg_structure_list = [struct.structure for struct in structure_list]
    uuid_list = [struct.unique_id for struct in structure_list]

    for pym_struct in pymg_structure_list:
        # Converting to ase structure
        ase_struct = AseAtomsAdaptor().get_atoms(pym_struct)

        # Create soap descriptors for current system and storing it
        struct_soap = soap.create(ase_struct, n_jobs=-1, verbose=True)
        soap_structs.append(struct_soap)

    # Calculating similarity with an average kernel and a gaussan metric. The
    # result will be a full similarity matrix.
    kernel = AverageKernel(metric="rbf", gamma=1)
    simi_matrix = kernel.create(soap_structs)

    # For every structure, check if it is repeated more than once (itself)
    repeat_struct_uuid = []
    i_max, j_max = simi_matrix.shape

    # for struct_idx, row in enumerate(simi_matrix):
    for struct_idx, _struct in enumerate(structure_list):
        check_col = simi_matrix[struct_idx, struct_idx + 1 : j_max]

        # Get position in current row

        # Getting repeats
        row_repeats = np.isclose(check_col, 1)
        n_repeats = np.count_nonzero(row_repeats)

        # If structure is repeated
        if n_repeats > 0:
            len_diff = j_max - len(row_repeats)
            # For every structure, repeated or not
            for repeat_idx, repeat in enumerate(row_repeats):
                # If a structure is repeated
                if repeat:
                    # Compute the postion on the entire array column/row
                    list_idx = repeat_idx + len_diff
                    # Get the correspondng structure and add it to the list
                    repeat_struct_uuid.append(uuid_list[list_idx])

    repeat_struct_uuid = list(set(repeat_struct_uuid))

    struct_size = len(pymg_structure_list[0].species)
    custom_print(
        (
            f"Duplicate check for size {struct_size} - Total selected structures:"
            f"  {len(structure_list)}"
            f", equivalent: {len(repeat_struct_uuid)}"
            f" ({(len(repeat_struct_uuid)/len(structure_list))*100:.2f}%)"
        ),
        "debug",
    )
    return repeat_struct_uuid


def _del_structure_list_by_uuid(structure_list, dupl_uuid_list):
    all_uuid_set = set([stru.unique_id for stru in structure_list])
    dupl_uuid_set = set(dupl_uuid_list)
    unique_struct_set = all_uuid_set.difference(dupl_uuid_set)

    unique_structure_list = []
    for uniq_struc_uuid in unique_struct_set:
        for structure in structure_list:
            if structure.unique_id == uniq_struc_uuid:
                unique_structure_list.append(structure)

    return unique_structure_list


def apply_replacement(structure: Structure, phase, n_atoms: int | float, rng=None):
    if not rng:
        rng = np.random.default_rng()

    if isinstance(
        structure, (mdb_struct.Structure, mdb_struct.Surface, mdb_struct.Bulk)
    ):
        structure = structure.structure

    structure_len = len(structure.species)
    curr_comp = structure.composition

    # We assume that if the n_atoms is a fractional number, it must
    # represent the ratio of atoms in the structure, so we convert
    # that to a number of atoms.
    if isinstance(n_atoms, float) and n_atoms < 1:
        n_atoms = int(n_atoms * structure_len)

    # If no replacements are going to be made, this is probably due to
    # a low percentage being rounded to 0, thus we attempt to make at
    # least one replacement.
    if n_atoms == 0:
        n_atoms = 1

    # Getting current structure composition information
    # The current procedure assumes that all of the atom species in the structure
    # will have been replaced beforehand with the base atom,
    # although this results in more randomness.
    base_elem = phase.base_elem
    (other_elem,) = mdb_indb.CuZnInitialDatabase.ALLOY_SET - {base_elem}

    # If the structure only has one type of Element, and that is not the base
    # element, this changes with what to replace.
    if not curr_comp.as_dict().get(base_elem.symbol):
        base_elem = structure.composition.elements[0]
        (other_elem,) = mdb_indb.CuZnInitialDatabase.ALLOY_SET - {base_elem}
        other_atom_change = n_atoms

    else:
        # Getting how many base atoms must be changed in order for the
        # structure to meet the current percentage requirements.
        target_atoms_base = curr_comp[base_elem] - abs(n_atoms)

        # Getting how many atoms of the other element must be changed
        other_atom_change = int(curr_comp[other_elem] - target_atoms_base)
        # print('other_atom_change: ', other_atom_change)

    # Choosing which species of the structure to change with the other atom.
    other_elem_choices = rng.choice(
        a=structure_len,
        size=abs(int(other_atom_change)),
        replace=False,
        shuffle=True,
    )

    # Creating a new pymatgen structure using the base one as a template
    new_structure = structure.copy(sanitize=True)
    site_props_before = structure.site_properties

    # Replacing atoms in the structures
    for ind in other_elem_choices:
        new_structure.replace(ind, other_elem)

    # TODO: Instead of this, create a new structure
    # Copying site properties
    new_structure = new_structure.copy(sanitize=True, site_properties=site_props_before)

    return new_structure


def apply_replacement_no_db(
    structure: Structure,
    phase,
    n_atoms: int,
    replace_elem: str | Element,
    rng=None,
):
    MDB_STRUCT_TYPES = (
        mdb_struct.Structure,
        mdb_struct.Bulk,
        mdb_struct.Cluster,
        mdb_struct.Surface,
    )

    if not rng:
        rng = np.random.default_rng()

    if isinstance(structure, MDB_STRUCT_TYPES):
        structure = structure.structure

    # Converting ase.Atoms to pymatgen structures
    elif isinstance(structure, Atoms):
        is_ase = True
        structure = AseAtomsAdaptor.get_structure(structure)

    structure_len = len(structure.species)

    # We assume that if the n_atoms is a fractional number, it must
    # represent the ratio of atoms in the structure, so we convert
    # that to a number of atoms.
    if isinstance(n_atoms, float) and n_atoms < 1:
        n_atoms = int(n_atoms * structure_len)

    # If no replacements are going to be made, this is probably due to
    # a low percentage being rounded to 0, thus we attempt to make at
    # least one replacement.
    if n_atoms == 0:
        n_atoms = 1
    other_atom_change = n_atoms

    # Choosing which species of the structure to change with the other atom.
    other_elem_choices = rng.choice(
        a=structure_len,
        size=abs(int(other_atom_change)),
        replace=False,
        shuffle=True,
    )

    # Creating a new pymatgen structure using the base one as a template
    new_structure = structure.copy(sanitize=True)
    site_props_before = structure.site_properties

    # Replacing atoms in the structures
    for ind in other_elem_choices:
        new_structure.replace(ind, replace_elem)

    # TODO: Instead of this, create a new structure
    # Copying site properties
    new_structure = new_structure.copy(sanitize=True, site_properties=site_props_before)

    if is_ase:
        new_structure = AseAtomsAdaptor.get_atoms(new_structure)

    return new_structure


def apply_central_atom_octahedral(
    db_obj: "mdb_indb.InitialDatabase",
    filter_phase_list: list[str],
    filter_struct_types: list[str],
    central_element: str | Element,
    num_repeats: int,
    max_perturbation_ang: float,
    limit_num_structures: int,
    seed: int,
):
    # Setting up the RNG
    rng = np.random.default_rng(seed=seed)

    # Filter phase and structure types
    filtered_df, _, _ = apply_filters_db(
        db_obj, phase=filter_phase_list, filters=filter_struct_types
    )

    modified_structs = []
    ids_to_remove = []
    # Iterate over all filtered structures
    for _, row in filtered_df.iterrows():
        for repeat_idx in range(num_repeats):
            # Get mdb Structure
            mdb_struct_row = mdb_struct.Structure().from_db_row(
                row=row, columns=filtered_df.columns
            )
            new_structure = mdb_struct_row.structure.copy()

            # Identify indices of the central element
            if isinstance(central_element, Element):
                central_element = str(central_element)
            central_element_indices = new_structure.indices_from_symbol(central_element)

            # Perurbate the identified central atoms
            perturb_vec = rng.normal(loc=0, scale=max_perturbation_ang, size=(1, 3))[0]
            for site in central_element_indices:
                new_structure.translate_sites(indices=site, vector=perturb_vec)

            # Update structure attributes (targeted_modification, uuid?) and
            # add structure to db
            mdb_struct_row.structure = new_structure
            mdb_struct_row.targeted_modification = "central_atom_perturbation"
            mdb_struct_row.unique_id = str(uuid.uuid4())
            mdb_struct_row.base = row.base
            mdb_struct_row.material_name = (
                f"{mdb_struct_row.material_name}_perturb_central_{repeat_idx}"
            )
            modified_structs.append(mdb_struct_row)

        ids_to_remove.append(row.unique_id)

    # Limit the number of structures
    if limit_num_structures:
        limit_num_structures = min(limit_num_structures, len(modified_structs))

        modified_structs = rng.choice(
            modified_structs, limit_num_structures, replace=False
        )

    # Use uuid to remove original structures from the database
    for curr_uuid in ids_to_remove:
        db_obj.df = db_obj.df[db_obj.df["unique_id"] != str(curr_uuid)]

    # Add modified structures to the database
    for struc in modified_structs:
        struc.save_to_db(db_obj=db_obj)

    return db_obj


def apply_filters_db(db_obj, filters, phase: mdb_pd.Phase | str | list = None):
    filtered_df = db_obj

    if isinstance(db_obj, mdb_indb.InitialDatabase):
        filtered_df = db_obj.df

    remaining_df = filtered_df

    custom_print(f"Applying filters: {filters}.", "debug")

    # Applying filters. Filter lists are applied with an OR logic.
    if filters:
        appl_filter_db_list = []
        for filt in filters:
            appl_filt = filtered_df.loc[filtered_df[filt]]
            appl_filter_db_list.append(appl_filt)
            custom_print(
                f"Applied filter: '{filt}' - {len(appl_filt)} structs filtered.",
                "debug",
            )

        filtered_df = pd.concat(appl_filter_db_list, axis=0)

    # Getting which phases to check from the user.
    phase_list = []
    if phase:
        if isinstance(phase, list):
            for curr_phase in phase:
                if isinstance(curr_phase, str):
                    curr_phase = db_obj.phase_diagram.get_phase(curr_phase)

                if curr_phase:
                    phase_list.append(curr_phase.name)

        else:
            if isinstance(phase, str):
                phase = db_obj.phase_diagram.get_phase(phase)
            custom_print(f"Using phase: {phase.name}.", "debug")
            phase_list = [phase.name]

        custom_print(f"phase_list: {phase_list}.", "debug")

    # If no phase is given, getting the unique phases in the dataframe
    else:
        phase_list = filtered_df.phase.unique()
        custom_print(
            (f"No phase given. " f"Checking on all phases: {phase_list}."),
            "debug",
        )

    # Getting the current phase structures
    # In order to access a method from an object saved in a df, we need to do a
    # check like the following:
    idxs_list = []
    for idx, val in enumerate(filtered_df["phase"]):
        name = val if isinstance(val, str) else val.name
        if name in phase_list:
            idxs_list.append(idx)

    filtered_df = filtered_df.iloc[idxs_list]

    # Getting the remaining structures after selecting the phase
    remaining_df = remaining_df.loc[remaining_df.index.difference(filtered_df.index)]

    custom_print(
        f"Number of filtered structures: {filtered_df.shape[0]}",
        "debug",
    )
    custom_print(
        f"Number of remaining unfiltered structures: {remaining_df.shape[0]}",
        "debug",
    )

    return filtered_df, remaining_df, phase_list


def _apply_perturbation_mdb_struct(center, row, per_idx):
    # Applying displacement
    new_struct_perturb = gauss_perturb(center=center, structure=row.structure)

    # Creating perturbed cluster object
    mat_str = f"{row.material_name}_perturb_gauss_{per_idx+1}"
    if row.bulk:
        perturb_struct = mdb_struct.Bulk(
            material_name=mat_str,
            structure=new_struct_perturb,
            replacement_ind=row.replacement_ind,
            vacancy=row.vacancy,
            targeted_modification=row.targeted_modification,
            phase=row.phase,
            perturb=True,
        )
    elif row.surface:
        perturb_struct = mdb_struct.Surface(
            material_name=mat_str,
            structure=new_struct_perturb,
            replacement_ind=row.replacement_ind,
            targeted_modification=row.targeted_modification,
            phase=row.phase,
            perturb=True,
        )
    elif row.cluster:
        perturb_struct = mdb_struct.Cluster(
            material_name=mat_str,
            structure=new_struct_perturb,
            replacement_ind=row.replacement_ind,
            targeted_modification=row.targeted_modification,
            phase=row.phase,
            perturb=True,
        )
    return perturb_struct


def apply_gauss_perturb_db(
    repeat: int,
    db_obj: "mdb_indb.InitialDatabase",
    filters: list,
    phase: mdb_pd.Phase,
    center: float = 0.04,
    limit_num_structures: int = None,
):
    perturbed_structs = []

    if not isinstance(db_obj, mdb_indb.InitialDatabase):
        raise TypeError(
            f"'{apply_gauss_perturb_db.__name__}' expects a MatDBForge "
            f"database object, not a {type(db_obj)}."
        )

    # Filtering structures to perturb
    filtered_df, _, _ = apply_filters_db(db_obj, filters, phase)

    # Iterating over all filtered database rows to get the unperturbed surfaces
    custom_print(
        f"Perturbation will be applied to: {filtered_df.shape[0]} structures.", "debug"
    )
    for _, row in filtered_df.iterrows():
        if row.perturb:
            continue
        for per_idx in range(repeat):
            clust_obj = _apply_perturbation_mdb_struct(center, row, per_idx)

            perturbed_structs.append(clust_obj)

    custom_print(f"Total structs perturbed: {len(perturbed_structs)}", "debug")

    if limit_num_structures:
        custom_print(
            f"Limiting number of perturbations to  {limit_num_structures}", "debug"
        )

        limit_num_structures = np.min([limit_num_structures, len(perturbed_structs)])

        perturbed_structs = np.random.choice(
            perturbed_structs, limit_num_structures, replace=False
        )

    # Saving in database
    custom_print("Saving perturbed surfaces in dataframe...", "debug")
    for surface in perturbed_structs:
        db_obj._save_row(structure=surface)
    custom_print(f"Dataframe shape after saving: {db_obj.df.shape}.", "debug")

    return perturbed_structs


def limit_num_structures_phase(
    db_obj: "mdb_indb.InitialDatabase",
    phase: "mdb_pd.Phase",
    num_limit: int,
    rng_seed: int,
):
    # Instantiating RNG
    rng = np.random.default_rng(seed=rng_seed)

    # Getting the current phase structures (df_filtered)
    df_filtered = db_obj.df.loc[db_obj.df["phase"] == phase.name]

    # Getting the remaining structures after selecting
    # the phase (df_remaining)
    df_remaining = db_obj.df.loc[db_obj.df["phase"] != phase.name]

    # Getting a num_limit size random sample of the structures from
    # the current phase (df_filt_sample)
    num_limit = np.min([num_limit, df_filtered.shape[0]])
    df_filt_sampl_idx = rng.choice(
        range(df_filtered.shape[0]), num_limit, replace=False
    )
    df_filt_sampl = df_filtered.iloc[df_filt_sampl_idx]

    # Adding df_filt_sample to the df_remaining
    df_remaining = pd.concat([df_remaining, df_filt_sampl], axis=0)
    db_obj.df = df_remaining
    return db_obj


def add_adsorbates(
    repeat: int,
    db_obj: "mdb_indb.InitialDatabase",
    filters: list,
    phase: mdb_pd.Phase,
    adsorbate_species: list[str],
    limit_num_structures: int = None,
):
    from acat.build.adlayer import RandomPatternGenerator as RPG
    from acat.settings import site_heights

    # Create temporary file to store the trajectory using tmpfile
    tmp_file = tempfile.NamedTemporaryFile(suffix=".traj").name

    adsorb_structs = []

    if not isinstance(db_obj, mdb_indb.InitialDatabase):
        raise TypeError(
            f"'{apply_gauss_perturb_db.__name__}' expects a MatDBForge "
            f"database object, not a {type(db_obj)}."
        )

    # Filtering structures to perturb
    filtered_df, _, _ = apply_filters_db(db_obj, filters, phase)

    # Iterating over all filtered database rows to get desired surfaces
    custom_print(
        f"Perturbation will be applied to: {filtered_df.shape[0]} structures.", "debug"
    )
    for _, row in filtered_df.iterrows():
        struct = row.structure
        if not isinstance(row.structure, Atoms):
            struct = AseAtomsAdaptor().get_atoms(row.structure)

        print("struct: ", type(struct))
        # Add more heights to the CHOO*
        heights = {k: v + 0.5 for k, v in site_heights.items()}

        # TODO: Change surface to the correct one
        for spec in adsorbate_species:
            gen = RPG(
                images=struct,
                adsorbate_species=spec,
                min_adsorbate_distance=1.5,
                surface="fcc111",
                heights=heights,
                # species_forbidden_sites={'CHOO': ['ontop','bridge']},
                trajectory=tmp_file,
            )
            gen.run(num_gen=repeat, action="add", num_act=5)
            atoms = ase_read(tmp_file)
            adsorb_structs.append(atoms)
            visualize(atoms)

    if limit_num_structures:
        custom_print(
            f"Limiting number of structures to  {limit_num_structures}", "debug"
        )

        limit_num_structures = np.min([limit_num_structures, len(adsorb_structs)])

        adsorb_structs = np.random.choice(
            adsorb_structs, limit_num_structures, replace=False
        )

    # Saving in database
    custom_print("Saving surfaces with adsorbates in dataframe...", "debug")
    for surface in adsorb_structs:
        db_obj._save_row(structure=surface)
    custom_print(f"Dataframe shape after saving: {db_obj.df.shape}.", "debug")

    # Deleting temporary file
    os.remove(tmp_file)

    return adsorb_structs


def fix_bottom_layers(structure: Structure, n_layers: int) -> Structure:
    """
    Fixes the bottom n layers of a pymatgen Structure by setting the selective
    dynamics to False for the atoms in those layers.

    Parameters
    ----------
    structure : pymatgen.core.Structure
        The structure to modify.
    n_layers : int
        The number of bottom layers to fix.

    Returns
    -------
    Structure
        A new Structure object with selective dynamics updated.
    """
    sorted_indices = sorted(range(len(structure)), key=lambda i: structure[i].z)
    bottom_indices = sorted_indices[:n_layers]

    # Set selective dynamics, fixing bottom layers
    selective_dynamics = [[True, True, True]] * len(structure)
    for idx in bottom_indices:
        selective_dynamics[idx] = [False, False, False]

    structure_with_constraints = structure.copy()
    structure_with_constraints.add_site_property(
        "selective_dynamics", selective_dynamics
    )

    print("structure_with_constraints: ", structure_with_constraints.site_properties)
    return structure_with_constraints
