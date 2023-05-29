import logging
import time

import numpy as np
import pandas as pd
from aiida.engine import submit
from aiida.orm import Bool, Code, Dict, Group, Int, Str, StructureData
from aiida.orm.nodes.data.array.kpoints import KpointsData
from aiida.plugins import WorkflowFactory
from MatDBForge.core import utils as ut
from MatDBForge.workflows import aiida_utils as aut

# k-spacing values for every phase to be
# included in the INCAR
KSPACING = {
    "alpha": 0.135088484104361,
    "m1": 0.100530964914873,
    "beta-prime": 0.102415920507027,
    "m2": 0.100530964914873,
    "gamma": 0.141371669411541,
    "m3": 0.166504410640259,
    "epsilon": 0.105557513160617,
    "eta": 0.0993371597065093,
    "m4": 0.0948760981384118,
    "delta": 0.0994491889005363,
}

parser_dict = {
    "add_misc": False,
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

# Dictionary containing settings for selecting aiida's computer and code.
# Example:
#
# QUEUE_DICT = {
#    20: {
#     "node_cpus": 48,
#     "code_string": "vasp-5.4.4_mn@marenostrum1",
#     "qos": "class_a",
#     "max_wallclock_seconds": 3600,
#     "max_memory_kb": 96000000,
#     "options_resources": {
#         "parallel_env": "class_a",
#         "tot_num_mpiprocs": 48,
#     },
#     "multiple": 1,
# },
#
QUEUE_DICT = {
    10: {
        "type": "sge",
        "node_cpus": 12,
        "code_string": "vasp-std-5.4.4@tekla2",
        "options_resources": {
            "parallel_env": "c12m48ib_mpi",
            "tot_num_mpiprocs": 12,
        },
        "multiple": 1,
    },
    20: {
        "type": "slurm",
        "node_cpus": 48,
        "code_string": "vasp-5.4.4_mn@marenostrum1",
        "qos": "class_a",
        "max_wallclock_seconds": 7800,
        "max_memory_kb": 96000000,
        "options_resources": {
            "tot_num_mpiprocs": 48,
        },
        "multiple": 1,
    },
    80: {
        "type": "slurm",
        "node_cpus": 48,
        "code_string": "vasp-5.4.4_mn@marenostrum1",
        "qos": "class_a",
        "max_wallclock_seconds": 24600,
        "max_memory_kb": 96000000,
        "options_resources": {
            "tot_num_mpiprocs": 48,
        },
        "multiple": 1,
    },
}

# POTCAR equivalent
# Potential_family is chosen among the list given by
# 'verdi data vasp-potcar listfamilies'
POTENTIAL_FAMILY = "vasp-5.4-PBE-2023"

# The potential mapping selects which potential to use
# This could for instance be {'Si': 'Si_GW'} to use the GW ready potential
# We use a specific function which just uses the default potential for
# every atom.
POTENTIAL_MAPPING = aut.generate_potential_mapping()

# Paths for the source and target dataframe.
SOURCE_DF = "/home/psanz/teklahome/projects/p2-CuZn/relaxed_structures_initialdb/initial_db/initial-database_19052023-165058_final.pkl"
TARGET_DF = "/WAREHOUSE/sp_database.pkl"

# Which calculation to run.
# Refer to the CalcType class for more information.
# CALC_TYPE = "SP"
CALC_TYPE = aut.CalcType.single_point

# Maximum size of each calculation batch.
# A maximum of MAX_BATCH calculations will run at once.
MAX_BATCH = 300

# Skip all chunks from 0 up to the the chunk number set in START_ON.
START_ON = 6


if __name__ == "__main__":
    # Getting current time
    ctime = time.strftime("%Y%m%dT%H%M%S")

    # Configuring logger
    logging.basicConfig(
        filename=f"/tmp/run_{ctime}.log",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
    )

    ut.custom_print(f"Started logging on '/tmp/run_{ctime}.log'", "info")

    # Creating a new aiida group for the calculations.
    # It will provide an ID for the entire batch
    group_label = f"{CALC_TYPE}_batch_{ctime}"
    group = Group(label=group_label)
    group.store()
    ut.custom_print(f'Group identifier: "{group.uuid}"', "info")
    ut.custom_print(f"Group label: {group_label}", "info")

    # Loading the initial structures dataframe
    src_df = pd.read_pickle(SOURCE_DF)

    # Starting calculation index
    curr_ind = 0

    # Splitting the initial database in chunks of size
    # MAX_BATCH
    num_chunks = len(src_df) // MAX_BATCH

    ut.custom_print(
        f"Splitting database with {len(src_df)} entries into {num_chunks} chunks.",
        "info",
    )

    # Iterating over every chunk.
    for chunk_id, chunk in enumerate(np.array_split(src_df, num_chunks)):
        # Skipping unwanted chunks
        if chunk_id < START_ON:
            continue

        ut.custom_print(f"Working on chunk {chunk_id}...", "info")

        # Sorting chunk so smallest structures are run first
        aut.sort_chunk_size(chunk)

        # Creating list for storing the nodes once submitted
        chunk_node_list = []

        # Iterating over the chunk and launching a separate
        # vasp workchain for every structure contained.
        for it, target_row in chunk.iterrows():
            # Getting current structure, phase and formula.
            target_structure = target_row.structure.get_sorted_structure()

            # Getting the phase, formula and kspacing.
            phase = target_row.phase
            struct_formula = target_structure.formula.replace(" ", "")
            kspacing = KSPACING[phase]

            # Generate INCAR with correct kspacing
            incar = aut.generate_incar(
                phase=phase, calc_type=CALC_TYPE, kspacing=KSPACING
            )

            # Dictionary containing metadata for the calculation
            metadata_dict = {
                "label": f"{target_row.material_id}-{struct_formula}-{it}_{CALC_TYPE}",
                "description": f"Relaxation for {struct_formula} in CuZn initial database.",
            }

            # Getting structure as an aiida structure from pymatgen.
            structure = StructureData(pymatgen=target_structure)

            # Get kpoints for aiida:
            # kpoints_data = DataFactory("core.array.kpoints")
            kpoints_data = KpointsData()
            kpoints_data.set_cell_from_structure(structuredata=structure)
            kpoints_data.set_kpoints_mesh_from_density(distance=kspacing)

            # Jobfile equivalent
            # In options, we typically set scheduler options. See:
            # https://aiida.readthedocs.io/projects/aiida-core/en/latest/scheduler/index.html
            OPTIONS, CODE_STRING, mult = aut.choose_queue_from_struct(
                structure=target_structure, assign_dict=QUEUE_DICT
            )

            # Removing kpar for multinode calculations
            # incar["incar"]["kpar"] = 4
            # if mult > 1:
            #     incar["incar"].pop("kpar")

            # Defining the vasp.relax workchain object
            workchain = WorkflowFactory("vasp.relax")

            # Preparing a builder object to be able to submit the workchain
            # and pass inputs to it
            builder = workchain.get_builder()

            # Passing the all inputs to the builder object
            builder["code"] = Code.get_from_string(CODE_STRING)
            # builder["converge"] = CONVERGE
            # builder["dynamics"] = SEL_DYNAMICS
            builder["options"] = Dict(OPTIONS)
            builder["parameters"] = Dict(incar)
            builder["potential_family"] = Str(POTENTIAL_FAMILY)
            builder["potential_mapping"] = Dict(POTENTIAL_MAPPING)
            builder["structure"] = structure
            builder["metadata"] = metadata_dict
            builder["max_iterations"] = Int(2)
            builder["verbose"] = Bool(True)
            builder["kpoints"] = kpoints_data

            builder["monitors"] = {
                "monitor_1": Dict({"entry_point": "monitor.davwarning"}),
            }

            # Setting parser options
            builder["settings"] = Dict(parser_dict)

            if CALC_TYPE.lower() == "sp":
                builder["perform_static"] = Bool(True)
                builder["relax"]["perform"] = Bool(False)

            elif CALC_TYPE.lower() == "relax":
                builder["perform_static"] = Bool(False)
                builder["relax"]["perform"] = Bool(True)

            # Submitting the calculation.
            # Aiida should handle the scheduler, ssh connection and result
            # retrieval if everything is configured
            node = submit(builder)
            group.add_nodes(node)
            chunk_node_list.append(node)

            ut.custom_print(
                f"Launched workchain for structure {it}: '{struct_formula}' ({phase}) - node id: {node.id}",
                "debug",
            )

        # Waiting until the current chunk's calculations
        # are done.
        chunk_finished = False
        while not chunk_finished:
            ut.custom_print(
                f"({time.strftime('%H:%M:%S')}) Waiting for calculations from chunk {chunk_id} to be finished...",
                "info",
            )
            node_status_list = []
            for nod in chunk_node_list:
                # Some interesting options with dir(i):
                # 'is_excepted', 'is_failed', 'is_finished',
                # 'is_finished_ok', 'is_killed', 'is_sealed', 'is_stored', 'is_terminated',
                # 'process_status', exception
                node_status_list.append(nod.is_finished)

            if all(node_status_list):
                chunk_finished = True

                for nod in chunk_node_list:
                    ut.custom_print(
                        f"VaspCalculation {nod.id} finished: {nod.exit_status} - {nod.exit_message}",
                        "debug",
                    )

            else:
                time.sleep(500)

        ut.custom_print(f"Chunk {chunk_id} done!", "done")

    ut.custom_print("All calculations finished!", "done")
    ut.custom_print("Check 'verdi process list' for more information", "info.")
