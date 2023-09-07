import copy
import os
from strenum import StrEnum

import aiida_tim.utils as ut
import ase.data as ad
import numpy as np
from aiida import load_profile
from aiida_tim import MODULEROOT as AT_MODULEROOT
from aiida_vasp.utils.aiida_utils import get_data_node
from pymatgen.io.vasp import Poscar

DATAPATH = f"{AT_MODULEROOT}/tests/input_files"
VDW_DATA_PATH = "/home/psanz/Documents/phd-iciq/Projects/P2-Cu/forpol/vdw-data"

# Loading default aiida user profile
load_profile()

PARSER_DICT = {
    "parser_settings": {
        "add_misc": [
            "notifications",
            "run_status",
            "run_stats",
            "version",
        ],
        "add_kpoints": False,
        "add_structure": False,
        "add_poscar-structure": False,
        "add_trajectory": False,
        "add_forces": False,
        "add_stress": False,
        "add_bands": False,
        "add_dos": False,
        "add_energies": False,
        "add_projectors": False,
        "add_born_charges": False,
        "add_dielectrics": False,
        "add_hessian": False,
        "add_dynmat": False,
        "add_charge_density": False,
        "add_wavecar": False,
        "add_site_magnetization": False,
        "add_maximum_force": False,
        "add_maximum_stress": False,
        "add_total_energies": False,
    }
}

# DB_NAME = "aiida/test_database/test.db"
# DB_NAME = f"{os.getcwd()}/small_fragment_adsorption.db"
SEL_DYNAMICS = None

# INCAR equivalent
# Set input parameters
INCAR_SP = {
    "incar": {
        ## general:
        "istart": 0,
        "icharg": 2,
        "gga": "Pe",
        "ispin": 1,
        # "lorbit": 11,
        ## electronic steps:
        "encut": 450,
        "ediff": 1e-6,
        "ismear": 0,
        "sigma": 0.03,
        "algo": "Fast",
        "lreal": "Auto",
        "nelm": 60,
        ## ionic steps:
        "ibrion": -1,
        "nsw": 2,
        "ediffg": -0.03,
        "isif": 2,
        "potim": 0.3,
        ## files to write:
        "lwave": False,
        "lcharg": False,
        ## parallelization:
        "ncore": 4,
        # "kpar": 4,
        ## dipole correction
        "lelf": False,
        ## van der Waals:
        "ivdw": 11,
        ## surface
        "idipol":3,
        "ldipol":True,
    }
}


INCAR_RELAX = {
    "incar": {
        ## general:
        "istart": 0,
        "icharg": 2,
        "gga": "Pe",
        "ispin": 1,
        # "lorbit": 11,
        ## electronic steps:
        "encut": 450,
        "ediff": 1e-6,
        "ismear": 0,
        "sigma": 0.03,
        "algo": "Fast",
        "lreal": "Auto",
        "nelm": 60,
        ## ionic steps:
        "ibrion": 2,
        "nsw": 350,
        "ediffg": -0.03,
        "isif": 3,
        "potim": 0.3,
        ## files to write:
        "lwave": False,
        "lcharg": False,
        ## parallelization:
        "ncore": 4,
        # "kpar": 4,
        ## dipole correction
        "lelf": False,
        ## van der Waals:
        "ivdw": 11,
    }
}


# Default k-spacing values for every phase to be
# included in the INCAR.
# Ideally would be overwritten by the user.
KSPACING_DEFAULT = {
    "alpha": 0.133203528512207,
    "m1": 0.100530964914873,
    "beta-prime": 0.100530964914873,
    "m2": 0.100530964914873,
    "gamma": 0.141371669411541,
    "m3": 0.166504410640259,
    "epsilon": 0.153309721495182,
    "eta": 0.0948760981384118,
    "m4": 0.0948760981384118,
}

# TODO: Convert the subtypes into actual clases
class CalcType(StrEnum):
    """
    Class representing the available calculation types for vasp
    Here, the term relaxation signifies ionic relaxation and can be
    called by either relaxation or relax.

    """

    relax = "relax"
    relaxation = "relax"
    single_point = "sp"
    single_point_surface = "sp_surface"
    sp = "sp"
    static = "sp"


def choose_queue(node_type: int, tot_procs: int = None):
    """
    Choose the scheduler and aiida's computer and code options.

    Parameters
    ----------
    node_type : int
        Whether to use 12, 24 or 28 core nodes in tekla.
    tot_procs : int
        How many cores to be used. Using more than the total number of cores in
        a single node will make use of more nodes.

    Returns
    -------
    dict
        The OPTIONS dict is aiida jobfile equivalent
    str
        The CODE_STRING str is aiida code identifier.
    """
    # Jobfile equivalent
    # In OPTIONS, we typically set scheduler options. See:
    # https://aiida.readthedocs.io/projects/aiida-core/en/latest/scheduler/index.html
    OPTIONS = {}
    OPTIONS["account"] = ""
    OPTIONS["qos"] = ""
    OPTIONS["max_wallclock_seconds"] = 117280000
    OPTIONS["max_memory_kb"] = 102400000

    if not tot_procs:
        # If tot_procs is not given, use the number of cores
        # on a node, in order to use a single node.
        tot_procs = node_type

    # Code_string is chosen among the list given by 'verdi code list'
    if node_type == 28:
        CODE_STRING = "vasp-5.4.4_28core@tekla2"
        OPTIONS["resources"] = {
            "parallel_env": "c28m128ib_mpi",
            "tot_num_mpiprocs": tot_procs,
        }

    elif node_type == 24:
        CODE_STRING = "vasp-5.4.4_24core@tekla2"
        OPTIONS["resources"] = {
            "parallel_env": "c24m128ib_mpi",
            "tot_num_mpiprocs": tot_procs,
        }

    elif node_type == 12:
        CODE_STRING = "vasp-5.4.4@tekla2"
        OPTIONS["resources"] = {
            "parallel_env": "c12m48ib_mpi",
            "tot_num_mpiprocs": tot_procs,
        }

    return OPTIONS, CODE_STRING


def choose_queue_from_struct(structure, assign_dict: dict):
    """
    Choose the scheduler and aiida's computer and code options according
    to the size of the structure.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
        Structure
    assign_dict : dict
        Dictionary specifying which queue gets assigned to every atom
        interval.
    Returns
    -------
    dict
        The OPTIONS dict is aiida jobfile equivalent
    str
        The CODE_STRING str is aiida code identifier.
    """

    # Getting the number of atoms
    num_atom = len(structure.sites)

    # Code_string is chosen among the list given by 'verdi code list'
    keys_list = list(assign_dict.keys())
    sort_keys_list = keys_list.copy()
    sort_keys_list.append(num_atom)
    sort_keys_list = sorted(sort_keys_list)
    num_atom_posc = sort_keys_list.index(num_atom)

    # If our target structure is larger than the maximum
    if num_atom_posc == (len(sort_keys_list) - 1):
        next_val = keys_list[num_atom_posc - 1]

    # Every other case
    elif num_atom_posc == 0:
        next_val = keys_list[0]
    else:
        next_val = keys_list[num_atom_posc - 1]

    # Getting our data for the the largest queue
    queue_data = assign_dict.get(next_val, keys_list[-1])

    # Jobfile equivalent
    # In OPTIONS, we typically set scheduler options. See:
    # https://aiida.readthedocs.io/projects/aiida-core/en/latest/scheduler/index.html
    OPTIONS = {}
    OPTIONS["account"] = ""
    OPTIONS["qos"] = queue_data.get("qos", None)
    OPTIONS["max_wallclock_seconds"] = queue_data.get("max_wallclock_seconds", None)
    OPTIONS["max_memory_kb"] = queue_data.get("max_memory_kb", None)

    # Getting code string
    CODE_STRING = queue_data["code_string"]
    OPTIONS["resources"] = queue_data["options_resources"]

    # Getting options
    node_cpus = queue_data["node_cpus"]

    mult_nodes = queue_data["multiple"]

    # Specific setting for slurm scheduler
    if queue_data.get("type") == "slurm":
        OPTIONS["resources"]["num_cores_per_machine"] = node_cpus

    OPTIONS["resources"]["tot_num_mpiprocs"] = node_cpus * mult_nodes

    return OPTIONS, CODE_STRING, mult_nodes

def kpoint_mesh_from_density(structure, kspacing):
    """Returns kpoint mesh (3x3) from kpoint array,
    intended for surfaces.
    """

    # Read POSCAR
    poscar = Poscar(structure)

    kpt_dens_arr = np.repeat(kspacing, 3)

    # Getting lattice vectors
    l_mat = poscar.structure.lattice.matrix

    # Getting volume of the reciprocal cell
    v_mat = np.dot(np.cross(l_mat[0, :], l_mat[1, :]), l_mat[2, :])

    # Computing values for each axis
    a_rcpr = np.linalg.norm((np.cross(l_mat[1, :], l_mat[2, :])) / v_mat)
    b_rcpr = np.linalg.norm((np.cross(l_mat[0, :], l_mat[2, :])) / v_mat)
    c_rcpr = np.linalg.norm((np.cross(l_mat[0, :], l_mat[1, :])) / v_mat)

    arr_kpt_run = 1/(kpt_dens_arr / np.array((a_rcpr, b_rcpr, c_rcpr)))
    arr_kpt_run = np.around(arr_kpt_run) 
    arr_kpt_run[2] = 1

    return arr_kpt_run
 

def select_kspacing(curr_structure, incar: dict, phase: str, kspacing: dict, calc_type):

    if "surface" in calc_type:
        kspacing_calc = kpoint_mesh_from_density(structure=curr_structure,kspacing=kspacing[phase])
    else:
        kspacing_calc = kspacing[phase]

    return kspacing_calc


def sort_chunk_size(chunk):
    # Creating list for storing the number of atoms
    size_list = []

    # Gathering the number of atoms for every structure
    # and adding it to the size_list
    for it, row in chunk.iterrows():
        size_list.append(len(row.structure.sites))

    # Creating a new column for the atom number
    chunk["num_atoms"] = size_list

    # Sorting the chunk using the atom number, from
    # small to large.
    chunk.sort_values(by=["num_atoms"], inplace=True)

    return chunk


def generate_incar(structure, phase: str, calc_type: str, kspacing: dict = KSPACING_DEFAULT):
    """
    Generate an incar file using depending on the calculation type.
    This incar includes a kspacing variable that depends on the phase.

    Parameters
    ----------
    structure: pymatgen.core.structure.Structure
        Current structure
    phase : str
        Phase of the current structure
    calc_type : str
        Calculation type, can be either 'relax' for relaxation
        or 'sp' for single point

    Returns
    -------
    dict
        dictionary representation of the INCAR
    """

    if "relax" in calc_type:
        ut.custom_print("Selecting relaxation INCAR...", 'debug')
        incar = INCAR_RELAX

    if "sp" in calc_type:
        ut.custom_print("Selecting single point INCAR...", 'debug')
        incar = INCAR_SP

    kspacing = select_kspacing(structure, incar, phase, kspacing, calc_type)

    return incar, kspacing


def generate_potential_mapping() -> dict:
    """
    Generate a dictionary specifying the potential mapping for vasp.
    As of now, this function only assigns the default potential for every
    atom.

    Inputs
    ------
    The function itself requires no input when called, but it will
    attempt to read the contents of a 'potential_mapping' file on
    the same folder where the code is being executed (CWD).
    There will be an example on the github repo. Also, an example
    follows:


    ::

        # Header lines that will be ignored.
        # The next two lines should not have any comment marks, '#'.
        Ag=Ag_gw
        Au=Au_gw


    Returns
    -------
    dict:
        Dictionary containing the potential assignation for each atom of
        the periodic table, with the shape:
        ``{'H': 'H', 'He': 'He', ...}``.
    """

    # Attempting to read the file containing the user-defined
    # potential mappings. Defaults to none if no potential is given

    # Attempting to open the potential mapping file
    try:
        atom_array = np.loadtxt(f"{os.getcwd()}/potential_mapping", dtype=str)

        # Checking for 0D arrays which appear when only 1 potential is
        # specified
        if atom_array.ndim == 0:
            atom_list = [str(atom_array)]
        else:
            atom_list = list(atom_array)

        # Creating a dictionary with the custom potentials
        atom_dict = {}
        for udp in atom_list:
            sym, pot = udp.split("=")
            atom_dict[sym.title()] = pot

    except FileNotFoundError:
        ut.custom_print(
            "No potential mapping file found. Using default potentials.", "warning"
        )
        atom_dict = {}

    # Creating empty dict for the potential mapping
    potential_mapping = {}

    # Mapping every symbol on the periodic table to itself
    # unless it appears on the dictionary
    for symbol in ad.chemical_symbols[1:]:
        user_defined_potential = atom_dict.get(symbol)

        if user_defined_potential:
            potential_mapping[symbol] = user_defined_potential
        else:
            potential_mapping[symbol] = symbol

    return potential_mapping


def default_array(name, array):
    """Used to set ArrayData for spec.input."""
    array_cls = get_data_node("array")
    array_cls.set_array(name, array)

    return array_cls


def get_vdw_params(structure, incar):
    elements = structure.symbol_set
    new_incar = copy.deepcopy(incar)

    c6_ele_list = []
    r0_ele_list = []

    try:
        for element in elements:
            with open(VDW_DATA_PATH + "/" + element) as f:
                param_file = f.readlines()
            c6_ele_list.append(float(param_file[-2].strip()))
            r0_ele_list.append(float(param_file[-1].strip()))

        # if len(c6_ele_list) == 1:
        #     new_incar["vdw_c6"] = c6_ele_list[0]
        #     new_incar["vdw_r0"] = r0_ele_list[0]
        # else:
        new_incar["incar"]["vdw_c6"] = c6_ele_list
        new_incar["incar"]["vdw_r0"] = r0_ele_list

        ut.custom_print(f"Gathered vdW info found for element '{element}'.", "info")

        return new_incar

    except FileNotFoundError:
        ut.custom_print(f"No vdW info found for element '{element}', ignoring.", "warn")
        return incar


if __name__ == "__main__":
    ut.custom_print("This file is not intented to be run as a script.", "error")