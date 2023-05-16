import copy
import os

import aiida_tim.utils as ut
import ase.data as ad
import numpy as np

# import pymatgen.io.vasp as pyvasp
from aiida import load_profile
from aiida.engine import submit
from aiida.orm import Bool, Code, Dict, Int, Str, StructureData
from aiida.plugins import WorkflowFactory
from aiida_tim import MODULEROOT as AT_MODULEROOT
from aiida_vasp.utils.aiida_utils import get_data_node


DATAPATH = f"{AT_MODULEROOT}/tests/input_files"
struct_files = [file for file in os.listdir(DATAPATH)]
VDW_DATA_PATH = "/home/psanz/Documents/phd-iciq/Projects/P2-Cu/forpol/vdw-data"

# Loading default aiida user profile
load_profile()


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
        #"lorbit": 11,
        ## electronic steps:
        "encut": 450,
        "ediff": 1e-6,
        "ismear": 0,
        "sigma": 0.03,
        # "algo": "Fast",
        "algo": "Normal",
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
        "kpar": 4,
        ## dipole correction
        "lelf": False,
        ## van der Waals:
        "ivdw": 11,
    }
}


INCAR_RELAX = {
    "incar": {
        ## general:
        "istart": 0,
        "icharg": 2,
        "gga": "Pe",
        "ispin": 1,
        #"lorbit": 11,
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
        "kpar": 4,
        ## dipole correction
        "lelf": False,
        ## van der Waals:
        "ivdw": 11,
    }
}
# CONVERGE = {
#     # The plane-wave cutoff to be used during ce tests in electron volts.
#     "pwcutoff": Float(450.0),
#     # The k-point grid to be used during ce tests. needs an array.
#     "kgrid": default_array("array", np.array([11.0, 11.0, 11.0])),
#     # The plane-wave cutoff in electron volts.
#     "pwcutoff_start": Float(450.0),
#     # The plane-wave cutoff step (increment) in electron volts. Float
#     "pwcutoff_step": Float(50.0),
#     # The number of plane-wave cutoff samples. int
#     "pwcutoff_samples": Int(3),
#     # The target k-point stepping at the densest grid in inverse AA.
#     #  default: 'float', 0.07
#     "k_dense": Float(0.07),
#     # The target k-point stepping at the coursest grid in inverse AA.
#     # default: 'float', 0.35
#     "k_course": Float(0.35),
#     # The default k-point spacing in inverse AA. default: 'float', 0.1
#     "k_spacing": Float(0.1),
#     # The number of k-point samples. default: ('int', 10)
#     "k_samples": Int(10),
#     # The cutoff_type to check convergence against: energy, gap and forces.
#     # default: 'str', 'energy'
#     "cutoff_type": Str("forces"),
#     # If the diff. between calculations are within this value for cutoff_type,
#     # then it is converged. default: 'float', 0.01
#     "cutoff_value": Float(0.01),
#     # in this case the cutoff value is the difference between 'cutoff_type' for the
#     # input structure and an atomic displacement or a compression of the
#     # unitcell. default: 'float', 0.01
#     # "cutoff_value_r": 1,
#     # If True, a convergence test of the compressed structure is also performed.
#     # default : 'bool', False
#     # "compress": 1,
#     # If True, a convergence test of the displaced structure is also performed
#     # default: 'bool', False
#     # "displace": 1,
#     # The displacement unit vector for the displacement test.
#     # Sets the direction of displacement. default: 'array', np.array([1.0, 1.0, 1.0])
#     # "displacement_vector": 1,
#     # The displacement distance (L2 norm) for the displacement test in AA.
#     # ('float', 0.2)
#     # "displacement_distance": 1,
#     # Which atom to displace? Index starts from 1 and follows the sequence for the
#     # sites in the Aiida ``structure`` object. ('int', 1)
#     # "displacement_atom": 1,
#     # The volume change in direct coordinates for each lattice vector.
#     # default: 'array', np.array([1.05, 1.05, 1.05])
#     # "volume_change": 1,
#     # If True, we relax for each convergence test.
#     # default: 'bool', False
#     "relax": Bool(True),
#     # The energy type that is used when ``cutoff_type`` is set to `energy`.
#     # Default: 'str', 'energy_extrapolated'
#     # "total_energy_type": 1,
#     # If True, we assume testing to be performed (e.g. dummy calculations).
#     # default: 'bool', False
#     # "testing": 1,
# }

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

    num_atom = len(structure.sites)
    # Jobfile equivalent
    # In OPTIONS, we typically set scheduler options. See:
    # https://aiida.readthedocs.io/projects/aiida-core/en/latest/scheduler/index.html
    OPTIONS = {}
    OPTIONS["account"] = ""
    OPTIONS["qos"] = ""
    OPTIONS["max_wallclock_seconds"] = 117280000
    OPTIONS["max_memory_kb"] = 102400000

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

    # print('next_val: ', next_val)

    # Getting code string
    CODE_STRING = queue_data["code_string"]
    OPTIONS["resources"] = queue_data["options_resources"]

    # Getting options
    node_cpus = queue_data["node_cpus"]
    mult_nodes = queue_data["multiple"]
    OPTIONS["resources"]["tot_num_mpiprocs"] = node_cpus * mult_nodes

    return OPTIONS, CODE_STRING, mult_nodes


def select_kspacing(incar: dict, phase: str, kspacing: dict):
    incar["incar"]["kspacing"] = kspacing[phase]

    return incar


def generate_incar(phase: str, calc_type: str, kspacing: dict = KSPACING_DEFAULT):
    """
    Generate an incar file using depending on the calculation type.
    This incar includes a kspacing variable that depends on the phase.

    Parameters
    ----------
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

    if calc_type.lower() == "relax":
        incar = INCAR_RELAX

    elif calc_type.lower() == "sp":
        incar = INCAR_SP

    incar = select_kspacing(incar, phase, kspacing)

    return incar


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
    # # TODO: Prepare a test dataframe with one structure from alpha and other from beta-prime
    # # TODO: Read the structures from the dataframe.
    # # TODO: Prepare INCAR. It should use the correct parameters. KSPACING inside.
    # # TODO: Prepare dft-d3.

    # # Iterating over the target structures and launching a separate
    # # vasp workchain for all of them.
    # for it, target_structure in enumerate(target_structures):
    #     # TODO: Set phase correctly
    #     phase = "Cu"

    #     # Appending VDW parameters to INCAR
    #     INCAR = get_vdw_params(target_structure, INCAR)

    #     # Dictionary containing metadata for the calculation
    #     metadata_dict = {
    #         "label": f"Cu2-{it}-relaxation",
    #         "description": "Testing convergence workchain using two Cu atoms.",
    #     }

    #     # Getting structure as a pymatgen structure
    #     structure = StructureData(pymatgen=target_structure)

    #     # Defining the vasp.relax workchain object
    #     workchain = WorkflowFactory("vasp.converge")

    #     # Preparing a builder object to be able to submit the workchain
    #     # and pass inputs to it
    #     builder = workchain.get_builder()

    #     # Checking if the current phase is one that needs relaxation
    #     if phase in ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"]:
    #         CONVERGE["relax"] = Bool(True)
    #     else:
    #         CONVERGE["relax"] = Bool(False)

    #     # Passing the all inputs to the builder object
    #     builder["code"] = Code.get_from_string(CODE_STRING)
    #     builder["converge"] = CONVERGE
    #     # builder["dynamics"] = SEL_DYNAMICS
    #     builder["options"] = Dict(OPTIONS)
    #     builder["parameters"] = Dict(INCAR)
    #     builder["potential_family"] = Str(POTENTIAL_FAMILY)
    #     builder["potential_mapping"] = Dict(POTENTIAL_MAPPING)
    #     builder["structure"] = structure
    #     builder["metadata"] = metadata_dict
    #     builder["max_iterations"] = Int(500)
    #     builder["verbose"] = Bool(True)
    #     # builder["kpoints"] = KMESH

    #     # builder["settings"]

    #     # # Passing the relax inputs to the builder object
    #     # for key in relax_dict.keys():
    #     #     builder[key] = relax_dict[key]

    #     # Submitting the calculation.
    #     # Aiida should handle the scheduler, ssh connection and result
    #     # retrieval if everything is configured
    #     node = submit(builder)

    #     ut.custom_print(
    #         f"Launched workchain for Cu2-{it} structure - {node.id}", "debug"
    #     )
