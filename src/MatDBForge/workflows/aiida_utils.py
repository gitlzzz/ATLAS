"""Utility functions for running several through aiida for active learning loops."""

import copy
import os
import pathlib as pl

# import queue
# import threading
import time
from enum import Enum

import ase.data as ad
import numpy as np
import pymatgen.core.structure as pymg_struct
from aiida import load_profile, orm
from aiida.engine import submit
from aiida.orm import Bool, Dict, Group, Int, List, Str, StructureData
from aiida.orm.nodes.data.array.kpoints import KpointsData
from aiida.plugins import WorkflowFactory
from aiida_vasp.utils.aiida_utils import get_data_node
from pymatgen.core.surface import Slab
from pymatgen.io.vasp import Poscar

from MatDBForge.active_learning import conversion as mdb_conv
from MatDBForge.core import MDB_DATA_DIR
from MatDBForge.core import initial_db as mdb_indb
from MatDBForge.core import utils as mdb_ut
from MatDBForge.core.clusters import center_structure

VDW_DATA_PATH = pl.Path(MDB_DATA_DIR / "vdw-data")

# Loading default aiida user profile
try:
    load_profile()
except Exception as e:
    mdb_ut.custom_print(f"Error loading aiida profile: '{e}'", "error")


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
        "critical_notifications": {
            "add_edddav_zhegv": True,
            "add_eddrmm_zhegv": True,
        },
    }
}

# DB_NAME = "aiida/test_database/test.db"
# DB_NAME = f"{os.getcwd()}/small_fragment_adsorption.db"
SEL_DYNAMICS = None

# INCAR equivalent
# Set input parameters
# TODO: Move to /data
INCAR_SP = {
    "incar": {
        # general:
        "istart": 0,
        "icharg": 2,
        "gga": "Pe",
        "ispin": 1,
        # "lorbit": 11,
        # electronic steps:
        "encut": 450,
        "ediff": 1e-6,
        "ismear": 0,
        "sigma": 0.03,
        "algo": "Fast",
        "lreal": "Auto",
        "nelm": 60,
        # ionic steps:
        "ibrion": -1,
        "nsw": 1,
        "ediffg": -0.03,
        "isif": 2,
        "potim": 0.3,
        # files to write:
        "lwave": False,
        "lcharg": False,
        # parallelization:
        "ncore": 4,
        # "kpar": 4,
        # dipole correction
        "lelf": False,
        # van der Waals:
        "ivdw": 11,
        # surface
        "idipol": 3,
        "ldipol": True,
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
# INCAR units (the value includes 2pi, but 2pi not in units)
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
class CalcType(Enum):
    """
    Class representing the available calculation types for vasp.

    Several enumeration members are included to represent different
    calculation possibilities.
    Of those, the term relaxation signifies ionic relaxation and can be
    called by either relaxation or relax.
    Several ways of calling a single point calculation are available.

    """

    relax = "relax"
    relaxation = "relax"
    single_point_bulk = "sp_bulk"
    single_point_surface = "sp_surface"
    single_point_cluster = "sp_cluster"
    sp = "sp_bulk"
    static = "sp_bulk"


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


def choose_queue_from_struct(queue_data: dict):
    """
    Choose scheduler options for aiida.

    Choose the scheduler and aiida's computer and code options according
    to the size of the structure.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
        Structure
    queue_data : dict
        Dictionary specifying which queue gets assigned to every atom
        interval.

    Returns
    -------
    dict
        The OPTIONS dict is aiida jobfile equivalent
    str
        The CODE_STRING str is aiida code identifier.
    int
        The number of nodes that will be used.
    """
    # Jobfile equivalent
    # In OPTIONS, we typically set scheduler options. See:
    # https://aiida.readthedocs.io/projects/aiida-core/en/latest/scheduler/index.html
    options = {}
    options["account"] = ""
    options["qos"] = queue_data.get("qos")
    options["max_wallclock_seconds"] = queue_data.get("max_wallclock_seconds")
    options["max_memory_kb"] = queue_data.get("max_memory_kb")
    options["custom_scheduler_commands"] = queue_data.get("custom_scheduler_commands")

    # Getting code string
    code_string = queue_data["code_string"]
    options["resources"] = queue_data["options_resources"]

    # Getting options
    node_cpus = queue_data["node_cpus"]

    mult_nodes = queue_data["multiple"]

    # Specific settings for slurm scheduler
    if queue_data.get("type") == "slurm":
        options["resources"]["num_cores_per_machine"] = node_cpus

    options["resources"]["tot_num_mpiprocs"] = node_cpus * mult_nodes

    return options, code_string, mult_nodes


def submit_aiida_vasp_calculation(
    index,
    target_structure,
    phase,
    material_name,
    unique_id,
    kspacing_dict,
    incar_settings_dict,
    calc_type,
    queue_dict,
    potential_family,
    potential_mapping,
    dry_run,
    return_builder,
    group,
):
    mdb_ut.custom_print(f"Row {index}.", "debug")

    struct_formula = target_structure.formula.replace(" ", "")
    kspacing = kspacing_dict[phase]

    if incar_settings_dict:
        # TODO: Implement manual INCAR submission.
        # incar = incar_settings_dict["incar"]
        incar = incar_settings_dict
        kspacing_vec = select_kspacing(
            target_structure, incar, phase, kspacing_dict, calc_type
        )
    else:
        # Generate INCAR with correct kspacing
        incar, kspacing_vec = generate_incar(
            structure=target_structure,
            phase=phase,
            calc_type=calc_type,
            kspacing=kspacing_dict,
        )

    # Changing all keys to uppercase to match the VASP INCAR format
    upper_incar = {"incar": {}}
    for key, value in incar.items():
        upper_incar["incar"][key.upper()] = value

    # Dictionary containing metadata for the calculation
    struct_name = f"{material_name}-{struct_formula}-{index}_{calc_type}"
    metadata_dict = {
        "label": struct_name,
        "description": (
            f"Relaxation for {struct_formula} in CuZn initial database."
            # "options": {"dry_run": True},
        ),
    }

    # Slab structures cannot be converted to aiida StructureData
    # directly, they must be converted to pymatgen structures
    # first.
    if isinstance(target_structure, Slab):
        target_structure = _convert_Slab_to_Structure(target_structure)

    # Centering the structure if it is a cluster
    if calc_type == "cluster":
        target_structure = center_structure(target_structure)

    # Getting structure as an aiida structure from pymatgen.
    structure = StructureData(pymatgen=target_structure)

    mdb_ut.custom_print(f"Calculation type: {calc_type}", "debug")

    # Get kpoints for aiida
    kpoints_data = generate_kpoints_data(
        structure=structure,
        kspacing=kspacing,
        kspacing_vec=kspacing_vec,
        calc_type=calc_type,
    )

    # Get selective dynamics
    selective_dynamics = None
    if target_structure.site_properties.get("selective_dynamics"):
        dynamics_list = target_structure.site_properties.get("selective_dynamics")
        selective_dynamics = {"positions_dof": List(dynamics_list)}

    # Jobfile equivalent
    # In options, we typically set scheduler options. See:
    # https://aiida.readthedocs.io/projects/aiida-core/en/latest/scheduler/index.html
    options, code_string, mult = choose_queue_from_struct(
        # structure=target_structure, assign_dict=queue_dict
        queue_data=queue_dict
    )

    # Defining the vasp.relax workchain object
    workchain = WorkflowFactory("vasp.relax")

    # Preparing a builder object to be able to submit the workchain
    # and pass inputs to it
    builder = workchain.get_builder()

    # Passing the all inputs to the builder object
    builder["code"] = orm.load_code(code_string)
    # builder["converge"] = CONVERGE

    if selective_dynamics:
        builder["dynamics"] = selective_dynamics

    # Assembling the builder object
    # Entries available for the options dict:
    # https://aiida.readthedocs.io/projects/aiida-core/en/latest/topics/calculations/usage.html?highlight=options#options
    builder["options"] = Dict(options)
    # builder["parameters"] = Dict(incar)
    builder["parameters"] = Dict(upper_incar)
    print('builder["parameters"]: ', builder["parameters"])
    builder["potential_family"] = Str(potential_family)
    builder["potential_mapping"] = Dict(potential_mapping)
    builder["structure"] = structure
    builder["metadata"] = metadata_dict
    builder["max_iterations"] = Int(2)
    builder["verbose"] = Bool(True)
    builder["kpoints"] = kpoints_data

    if dry_run:
        options["metadata"] = {}
        options["metadata"]["dry_run"] = True

    # Setting parser options
    builder["settings"] = Dict(PARSER_DICT)

    if calc_type.lower() == "sp":
        builder["perform_static"] = Bool(True)
        builder["relax"]["perform"] = Bool(False)

    elif calc_type.lower() == "relax":
        builder["perform_static"] = Bool(False)
        builder["relax"]["perform"] = Bool(True)

    if dry_run:
        # TODO: Generate a fake node and return it
        mdb_ut.custom_print("Dry run: nothing generated.", "debug")

    if return_builder:
        mdb_ut.custom_print("Returning builder.", "debug")
        return builder

    else:
        # Submitting the calculation.
        # Aiida should handle the scheduler, ssh connection and result
        # retrieval if everything is configured correctly
        node = submit(builder)
        mdb_ut.custom_print(
            (
                f"Launched workchain for structure {index}:"
                f" '{struct_formula}' ({phase}) - node id: {node.id}"
            ),
            "debug",
        )

        # Storing each calculation's unique_id (uuid) with the node, so once the
        # calculation is done, it can be traced back to its database entry by
        # getting the hex of the uuid in the database and searching which
        # RelaxWorkChain has the same hex, and then, the calculation results
        # of its last children node will be gathered and added to the DB.
        # node.base.extras.set("mdb_calc_uuid", target_row.unique_id.hex)
        node.base.extras.set("mdb_calc_uuid", unique_id.hex)
        node.base.extras.set("mdb_struct_type", calc_type.lower())
        node.base.extras.set("struct_name", struct_name)

        if group:
            group.add_nodes(node)

    return node


def run_dataframe_vasp_simulations_aiida(
    sel_struct_df,
    group_name: str,
    calc_type: CalcType,
    kspacing_dict: dict,
    max_batch: int,
    start_on: int,
    potential_mapping: dict,
    potential_family: str,
    initial_db: "mdb_indb.InitialDatabase",
    queue_dict: dict,
    incar_dict: dict = None,
    dry_run: bool = False,
):
    # Getting current time
    ctime = time.strftime("%Y%m%dT%H%M%S")

    # Print for dry runs
    if dry_run:
        mdb_ut.custom_print(
            (
                "This is a dry run: calculations will not be submitted. "
                "No aiida group will be created"
            ),
            "warning",
        )
    else:
        # Creating a new aiida group for the calculations.
        # It will provide an ID for the entire batch
        group_label = f"{group_name}_{calc_type}_batch_{ctime}"
        group = Group(label=group_label)
        group.store()
        mdb_ut.custom_print(f'Group identifier: "{group.uuid}"', "info")
        mdb_ut.custom_print(f"Group label: {group_label}", "info")

    # Starting calculation index

    # Splitting the initial database in chunks of size
    # max_batch
    num_chunks = len(sel_struct_df) // max_batch
    if num_chunks == 0:
        num_chunks = 1

    mdb_ut.custom_print(
        (
            f"Splitting database with {len(sel_struct_df)}"
            f" entries into {num_chunks} chunks."
        ),
        "info",
    )

    # Iterating over every chunk.
    for chunk_id, chunk in enumerate(np.array_split(sel_struct_df, num_chunks)):
        # Skipping unwanted chunks
        if chunk_id < start_on:
            continue

        mdb_ut.custom_print(f"Working on chunk {chunk_id}...", "info")

        # Sorting chunk so smallest structures are run first
        sort_chunk_size(chunk)

        # Creating list for storing the nodes once submitted
        chunk_node_list = []

        # Iterating over the chunk and launching a separate
        # vasp workchain for every structure contained.
        for it, target_row in chunk.iterrows():
            # Gathering calculation information and
            # submitting the calculation through AiiDA.
            (
                curr_structure,
                curr_material_name,
                curr_unique_id,
                curr_phase,
            ) = gather_calc_data_from_row(target_row)

            node = submit_aiida_vasp_calculation(
                index=it,
                target_structure=curr_structure,
                phase=curr_phase,
                material_name=curr_material_name,
                unique_id=curr_unique_id,
                initial_db=initial_db,
                kspacing_dict=kspacing_dict,
                incar_dict=incar_dict,
                calc_type=calc_type,
                queue_dict=queue_dict,
                potential_family=potential_family,
                potential_mapping=potential_mapping,
                dry_run=dry_run,
                group=group,
            )

            chunk_node_list.append(node)

        # Waiting until the current chunk's calculations are done.
        chunk_finished = False

        # Initialization of chunk node list as all False.
        node_status_list = np.full(len(chunk_node_list), False)

        while not chunk_finished:
            # Skipping the wait if dry run is selected
            if dry_run:
                chunk_finished = True

            mdb_ut.custom_print(
                (
                    f"({time.strftime('%H:%M:%S')})"
                    f" - {np.count_nonzero(node_status_list)}"
                    f"/{len(node_status_list)} - Waiting for calculations"
                    f" from chunk {chunk_id} to be finished..."
                ),
                "info",
            )
            node_status_list = []
            for nod in chunk_node_list:
                # Some interesting options related to the node status:
                # 'is_excepted', 'is_failed', 'is_finished',
                # 'is_finished_ok', 'is_killed', 'is_sealed', 'is_stored',
                # 'process_status', exception, 'is_terminated',
                node_status_list.append(nod.is_finished)

            if all(node_status_list):
                chunk_finished = True

                for nod in chunk_node_list:
                    mdb_ut.custom_print(
                        (
                            f"VaspCalculation {nod.id} finished: "
                            f"{nod.exit_status} - {nod.exit_message}"
                        ),
                        "debug",
                    )

            else:
                time.sleep(500)

        mdb_ut.custom_print(f"Chunk {chunk_id} done!", "done")

    mdb_ut.custom_print("All calculations finished!", "done")
    mdb_ut.custom_print("Check 'verdi process list' for more information", "info.")


def gather_calc_data_from_row(target_row, curr_structure=None):
    # Gathering calculation information and
    # submitting the calculation through AiiDA.
    if not curr_structure:
        curr_structure = target_row.structure.get_sorted_structure()
    else:
        curr_structure = curr_structure.get_sorted_structure()

    curr_material_name = target_row.material_name
    curr_unique_id = target_row.unique_id

    try:
        curr_phase = target_row.phase.name
    except AttributeError:
        curr_phase = "alpha"

    return curr_structure, curr_material_name, curr_unique_id, curr_phase


def run_dataframe_vasp_aiida_queue(
    sel_struct_df,
    group_name: str,
    calc_type: CalcType,
    kspacing_dict: dict,
    max_batch: int,
    potential_mapping: dict,
    potential_family: str,
    initial_db: "mdb_indb.InitialDatabase",
    queue_dict: dict,
    start_on: int = 0,
    incar_dict: dict = None,
    dry_run: bool = False,
    queue_check_interval: int = 60,
):
    # Getting current time
    ctime = time.strftime("%Y%m%dT%H%M%S")

    # Print for dry runs
    if dry_run:
        mdb_ut.custom_print(
            (
                "This is a dry run: calculations will not be submitted. "
                "No aiida group will be created"
            ),
            "warning",
        )
        group = None
    else:
        # Creating a new aiida group for the calculations.
        # It will provide an ID for the entire batch
        group_label = f"{group_name}_{calc_type}_batch_{ctime}"
        group = Group(label=group_label)
        group.store()
        mdb_ut.custom_print(f'Group identifier: "{group.uuid}"', "info")
        mdb_ut.custom_print(f"Group label: {group_label}", "info")

    # Starting calculation queue
    # calc_queue = queue.Queue(maxsize=max_batch)

    mdb_ut.custom_print(
        (
            "Starting queue for running database with"
            f" {len(sel_struct_df)} structures..."
        ),
        "info",
    )

    current_row_index = start_on
    calcs_remaining = True
    first_step = True

    queue = []
    total_loops = 1

    # Repeat while there are calculations to finish
    while calcs_remaining:
        mdb_ut.custom_print(
            (
                f"Step {total_loops} - {queue_check_interval*total_loops} s "
                "- Checking queue..."
            ),
            "info",
        )
        # Check all calculations and remove any that are finished.
        chunk_status = []
        for nod in queue:
            chunk_status.append(nod.is_finished)
        chunk_status_arr = np.array(chunk_status)

        # Removing the finished ones from the queue and current chunk
        if not first_step:
            queue = list(np.array(queue)[~chunk_status_arr])
            chunk_status_arr = chunk_status_arr[~chunk_status_arr]

        # Update the queue while the length of the queue is smaller
        # than the maximum allowed number of calculations.
        while len(queue) < max_batch and total_loops < len(sel_struct_df):
            # Getting current row
            target_row = sel_struct_df.iloc[current_row_index]

            # Creating list for storing the nodes once submitted

            # Gathering row information
            (
                curr_structure,
                curr_material_name,
                curr_unique_id,
                curr_phase,
            ) = gather_calc_data_from_row(target_row)

            # Iterating over the chunk and launching a separate
            # vasp workchain for every structure contained.
            # for it, target_row in chunk.iterrows():
            # Gathering calculation information and
            # submitting the calculation through AiiDA.
            node = submit_aiida_vasp_calculation(
                index=current_row_index,
                target_structure=curr_structure,
                phase=curr_phase,
                material_name=curr_material_name,
                unique_id=curr_unique_id,
                initial_db=initial_db,
                kspacing_dict=kspacing_dict,
                incar_dict=incar_dict,
                calc_type=calc_type,
                queue_dict=queue_dict,
                potential_family=potential_family,
                potential_mapping=potential_mapping,
                dry_run=dry_run,
                group=group,
            )

            current_row_index += 1
            queue.append(node)
            mdb_ut.custom_print(f"Queue length: {len(queue)}/{max_batch}", "debug")

        time.sleep(queue_check_interval)

        first_step = False
        total_loops += 1

        if current_row_index >= len(sel_struct_df) and len(queue) == 0:
            calcs_remaining = False

    mdb_ut.custom_print("All calculations finished!", "done")
    mdb_ut.custom_print("Check 'verdi process list' for more information", "info.")


def kpoint_mesh_from_density(structure, kspacing):
    """Return kpoint mesh (3x3) from kpoint array,
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
    arr_kpt_run = 1 / (kpt_dens_arr / np.array((a_rcpr, b_rcpr, c_rcpr)))
    arr_kpt_run = np.around(arr_kpt_run)
    arr_kpt_run[2] = 1

    return arr_kpt_run


def select_kspacing(
    curr_structure, incar: dict, phase: str, kspacing_dict: dict, calc_type
):
    if "surface" in calc_type:
        kspacing_calc = kpoint_mesh_from_density(
            structure=curr_structure, kspacing=kspacing_dict[phase]
        )
        # if np.all(np.equal(kspacing_calc, 1)):
        #     kspacing_calc = kpoint_mesh_from_density(
        #         structure=curr_structure, kspacing=kspacing_dict[max(kspacing_dict)]
        #     )
    else:
        kspacing_calc = kspacing_dict[phase]

    return kspacing_calc


def _convert_Slab_to_Structure(target_structure):
    conv_structure = pymg_struct.Structure(
        species=target_structure.species,
        coords=target_structure.cart_coords,
        coords_are_cartesian=True,
        lattice=target_structure.lattice,
    )
    return conv_structure


def sort_chunk_size(chunk):
    # Creating list for storing the number of atoms
    size_list = []

    # Gathering the number of atoms for every structure
    # and adding it to the size_list
    for _it, row in chunk.iterrows():
        size_list.append(len(row.structure.sites))

    # Creating a new column for the atom number
    chunk["num_atoms"] = size_list

    # Sorting the chunk using the atom number, from
    # small to large.
    chunk.sort_values(by=["num_atoms"], inplace=True)

    return chunk


def generate_incar(
    structure, phase: str, calc_type: str, kspacing: dict = KSPACING_DEFAULT
):
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
        mdb_ut.custom_print("Selecting relaxation INCAR...", "debug")
        incar = INCAR_RELAX
    elif calc_type in ["surface", "cluster", "bulk"]:
        mdb_ut.custom_print("Selecting single point INCAR...", "debug")
        incar = INCAR_SP

    kspacing = select_kspacing(structure, incar, phase, kspacing, calc_type)

    if "cluster" in calc_type:
        # Setting the center of the cell in direct lattice coordinates
        # with respect to which the total dipole-moment in the cell
        # is calculated.
        incar["incar"]["DIPOL"] = [0.5, 0.5, 0.5]
        incar["incar"]["IDIPOL"] = 4

    # Removing kpar for multinode calculations
    # incar["incar"]["kpar"] = 4
    # if mult > 1:
    #     incar["incar"].pop("kpar")

    return incar, kspacing


def generate_kpoints_data(structure, calc_type, kspacing=None, kspacing_vec=None):
    # Get kpoints for aiida:
    # kpoints_data = DataFactory("core.array.kpoints")
    kpoints_data = KpointsData()

    if not isinstance(structure, StructureData):
        structure = StructureData(pymatgen=structure)

    kpoints_data.set_cell_from_structure(structuredata=structure)

    # TODO: This conditional could be done with isinstance, if the calc
    # types were defined differently. Check if the current StrEnum
    # can be used with isinstance.
    # Bulks
    if "bulk" in calc_type:
        kpoints_data.set_kpoints_mesh_from_density(distance=kspacing)
        mdb_ut.custom_print(
            f"Generated kpoints for bulk: {kpoints_data.get_kpoints_mesh()}", "debug"
        )

    # Surfaces
    elif "surface" in calc_type:
        # kpoints_data.set_kpoints_mesh(kspacing_vec)

        kpoints_data.set_kpoints_mesh_from_density(distance=kspacing)
        kpoint_mesh = kpoints_data.get_kpoints_mesh()[0]

        # This will return a tuple with the kmesh and displacement

        # As the surfaces will be slabs with a long z axis,
        # 1 kpoint on the z-axis wll be employed
        kpoint_mesh[2] = 1
        kpoints_data.set_kpoints_mesh(mesh=kpoint_mesh)

        mdb_ut.custom_print(
            (
                f"Generated kpoints for surface using kspacing ({kspacing:.4f}): "
                f"{kpoints_data.get_kpoints_mesh()[0]} (z-axis forced to 1)"
            ),
            "debug",
        )

    # Clusters
    elif "cluster" in calc_type:
        kpoints_data.set_cell_from_structure(structuredata=structure)
        kpoints_data.set_kpoints_mesh([1, 1, 1])
        mdb_ut.custom_print(
            f"Generated kpoints for cluster: {kpoints_data.get_kpoints_mesh()}",
            "debug",
        )

    return kpoints_data


def add_aiida_group_to_db(db_obj: str, group_identifier, copy=False):
    # Loading InitialDatabase from a given path
    database = mdb_indb.get_database(db_obj)
    database_df = database.df

    db_uuids = database_df.unique_id
    db_uuids_hex = [uuid_str.hex for uuid_str in db_uuids]

    # Loading aiida calculation group from a group identifier
    group = orm.load_group(identifier=group_identifier)

    for node in group.nodes:
        node_calc_uuid = node.extras.get("mdb_calc_uuid", None)

        called_nodes = []
        for called in node.called_descendants:
            if (
                called.class_node_type == "process.calculation.calcjob.CalcJobNode."
                and called.exit_status == 0
            ):
                called_nodes.append(called)

        if node_calc_uuid and len(called_nodes) > 0:
            # Finding the position of the current index
            # in the database
            hex_index = db_uuids_hex.index(node_calc_uuid)

            # Getting the database entry using the position
            # matching_db_entry = database_df.iloc[hex_index]

            # Getting calculation data from the node
            data_dict = mdb_conv.gather_calc_data_from_node(called_nodes[-1])

            # Inserting information into the database entry
            vasprun = data_dict["atoms_obj"]
            database_df.at[hex_index, "calc_performed"] = True
            database_df.at[hex_index, "calc_type"] = vasprun.calc
            database_df.at[hex_index, "calc_energy"] = vasprun.get_potential_energy()
            database_df.at[hex_index, "calc_energy_toten"] = vasprun.get_total_energy()
            database_df.at[hex_index, "calc_energy_per_atom"] = (
                vasprun.get_potential_energy() / len(vasprun)
            )
            database_df.at[hex_index, "calc_output"] = vasprun
        else:
            pass

    # Overwriting database with the new information
    database.df = database_df

    if copy:
        if isinstance(db_obj, str):
            db_path = pl.Path(db_obj)
            database.save_database(path=db_path.parent, suffix="copy")
        else:
            raise NotImplementedError


def generate_potential_mapping() -> dict:
    """
    Generate a dictionary specifying the potential mapping for vasp.

    This function only assigns the default potential for every
    atom. The user can specify different mappings if needed.

    Inputs
    ------
    The function itself requires no input when called, but it will
    attempt to read the contents of a 'potential_mapping' file on
    the same folder where the code is being executed (CWD).
    There will be an example on the github repo. Also, an example
    follows:


    ::

        # Header lines for comments that will be ignored.
        # The lines following the headers should not have any comment marks, i.e. '#'.
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
        atom_list = [str(atom_array)] if atom_array.ndim == 0 else list(atom_array)

        # Creating a dictionary with the custom potentials
        atom_dict = {}
        for udp in atom_list:
            sym, pot = udp.split("=")
            atom_dict[sym.title()] = pot

    except FileNotFoundError:
        mdb_ut.custom_print(
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
            with open(VDW_DATA_PATH / element) as f:
                param_file = f.readlines()
            c6_ele_list.append(float(param_file[-2].strip()))
            r0_ele_list.append(float(param_file[-1].strip()))

        # if len(c6_ele_list) == 1:
        #     new_incar["vdw_c6"] = c6_ele_list[0]
        #     new_incar["vdw_r0"] = r0_ele_list[0]
        # else:
        new_incar["incar"]["vdw_c6"] = c6_ele_list
        new_incar["incar"]["vdw_r0"] = r0_ele_list

        mdb_ut.custom_print(f"Gathered vdW info found for element '{element}'.", "info")

        return new_incar

    except FileNotFoundError:
        mdb_ut.custom_print(
            f"No vdW info found for element '{element}', ignoring.", "warn"
        )
        return incar


if __name__ == "__main__":
    mdb_ut.custom_print("This file is not intented to be run as a script.", "error")
