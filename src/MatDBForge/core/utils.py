# General utilities
import json as js
import logging
import pathlib

import numpy as np
from ase import visualize
from dscribe.descriptors import SOAP
from dscribe.kernels import AverageKernel
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element, Species
from pymatgen.io import ase as pmg_ase
from pymatgen.io.ase import AseAtomsAdaptor
import tempfile

import MatDBForge.core.initial_db as mdb_indb
import MatDBForge.core.structure as mdb_struct


LINE_UP = "\033[1A"
LINE_CLEAR = "\x1b[2K"


def init_logger(source, log_path=None):
    logger = logging.getLogger("mdb")
    # logger.levels
    logger.setLevel(logging.DEBUG)

    # Console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter_con = logging.Formatter("%(message)s")
    ch.setFormatter(formatter_con)
    logger.addHandler(ch)

    filename = tempfile.NamedTemporaryFile(prefix=f"mdb_{source}_", suffix=".log").name

    if log_path:
        log_path_dir = pathlib.Path(log_path)
        log_filename = pathlib.Path(filename + ".log").stem
        filename = log_path_dir / log_filename

    fh = logging.FileHandler(filename=filename)
    fh.setLevel(logging.DEBUG)
    formatter_fil = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter_fil)
    logger.addHandler(fh)

    custom_print(f"Logging in '{filename}'", print_type="info")

    return logger


def custom_print(string: str, print_type: str = "default", end="\n"):
    """Prints a string using different formatting styles for
    easier debugging.

    Parameters
    ----------
    string : str
        Text to be printed
    print_type : str, optional, `default=info`
        Style to use when printing. Available styles are:
        - `info/default`: prefixes [i] before the string
        - `warning`: prefixes [!] before the string
        - `debug`: prefixes [...] before the string
        - `done`: prefixes [ ✔ ] before the string
    """
    normal = "\u001b[0m"

    if print_type in ["info", "default"]:
        prefix = "\u001b[38;5;33m [ i ]"
        # print(f"{prefix}{normal}\t{string}", end=end)
        logging.getLogger("mdb").info(f"{prefix}{normal}\t{string}")
    elif print_type in ["warn", "warning"]:
        prefix = "\u001b[38;5;220m [ ! ]"
        # print(f"{prefix}\t{string}{normal}", end=end)
        logging.getLogger("mdb").warning(f"{prefix}{normal}\t{string}")
    elif print_type in ["warn-soft", "warning-soft"]:
        prefix = "\u001b[38;5;220m [ ! ]"
        # print(f"{prefix}{normal}\t{string}", end=end)
        logging.getLogger("mdb").warning(f"{prefix}{normal}\t{string}")
    elif print_type in ["extra", "debug"]:
        prefix = "\u001b[38;5;8m [···]"
        # print(f"{prefix}\t{string}{normal}", end=end)
        logging.getLogger("mdb").debug(f"{prefix}{normal}\t{string}")
    elif print_type in ["done"]:
        prefix = "\u001b[38;5;46m [ ✔ ]"
        # print(f"{prefix}{normal}\t{string}", end=end)
        logging.getLogger("mdb").info(f"{prefix}{normal}\t{string}")
    if print_type in ["error", "problem"]:
        prefix = "\u001b[38;5;1m [ X ]"
        # print(f"{prefix}{normal}\t{string}", end=end)
        logging.getLogger("mdb").error(f"{prefix}{normal}\t{string}")


def clear_previous_print():
    print(LINE_UP, end=LINE_CLEAR)


def gather_secrets():
    """
    Gather Materials project API key from a secret.json file.

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
    initial_db_path = pathlib.Path(__file__).parent

    if pathlib.Path("secrets.json").exists():
        with open("secrets.json", "r") as f:
            secrets = js.load(f)

    elif pathlib.Path(initial_db_path, "secrets.json").exists():
        path = pathlib.Path(initial_db_path, "secrets.json")
        with open(path, "r") as f:
            secrets = js.load(f)

    else:
        raise FileNotFoundError(
            "'secrets.json' not found!\nPlease, add a 'secrets.json' file in the"
            f" following directory: '{initial_db_path}'. "
        )
        secrets = None

    return secrets


def check_incorrect_ratios(df, curr_phase_diag):
    for id, row in df.iterrows():
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
            assert (
                tot_cu + tot_zn == tot_atoms
            ), f"""Total count does not match.
            tot_cu: {tot_cu}, tot_zn: {tot_zn}, total: {tot_atoms}.
            Species: {set(strct.species)}"""

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
    Display all of the given structures using the ase gui

    Parameters
    ----------
    structures : list
        List of structures
    data : dict, optional
        Dictionary containing additional data, by default None
    """
    atoms_obj_list = []
    for structure in structures:
        struct = pmg_ase.AseAtomsAdaptor().get_atoms(structure)
        atoms_obj_list.append(struct)
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


def similarity_check_list(db_obj, replaced_structures, save_in_db=True):
    custom_print(
        f"Checking replacements for {len(replaced_structures)} structures.", "debug"
    )

    # Checking for similarity after replacement
    uuid_list = _check_repeat_struct_list(replaced_structures)

    # Deleting equivalent structures
    replaced_structures = _del_structure_list_by_uuid(replaced_structures, uuid_list)
    custom_print(
        f"{len(replaced_structures)} structures after duplicate check", "debug"
    )

    if save_in_db:
        custom_print("Saving to db...", "debug")
        for idx, cluster in enumerate(replaced_structures):
            db_obj._save_row(structure=cluster)

    return replaced_structures


def gauss_perturb(structure: Structure, center: float = 0.04):
    struct_types = (
        mdb_struct.Structure,
        mdb_struct.Bulk,
        mdb_struct.Cluster,
        mdb_struct.Surface,
    )

    if isinstance(structure, struct_types):
        structure_obj = structure
        structure = structure.structure

    new_structure = structure.copy()
    new_structure.perturb(distance=0.08, min_distance=0.02)

    if isinstance(structure, struct_types):
        structure_obj.structure = new_structure
        new_structure = structure_obj

    return new_structure


def _check_repeat_struct_list(structure_list):
    # Setting SOAP related parameters
    r_cut = 6
    r_cut = 6
    n_max = 8
    l_max = 6

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
    for struct_idx, struct in enumerate(structure_list):
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
