import copy
import os
import uuid

import ase.data as ad
import numpy as np
import pandas as pd
from aiida.engine import submit
from aiida.orm import Bool, Code, Dict, Float, Int, Str, StructureData
from aiida.orm.nodes.data.array.kpoints import KpointsData
from aiida.plugins import DataFactory, WorkflowFactory
from MatDBForge.core import utils as ut
from MatDBForge.workflows import aiida_utils as aut

# quit()

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

QUEUE_DICT = {
    10: {
        "node_cpus": 12,
        "code_string": "vasp-std-5.4.4@tekla2",
        "options_resources": {
            "parallel_env": "c12m48ib_mpi",
            "tot_num_mpiprocs": 12,
        },
        "multiple": 1,
    },
    20: {
        "node_cpus": 28,
        "code_string": "vasp-5.4.4_28core@tekla2",
        "options_resources": {
            "parallel_env": "c28m128ib_mpi",
            "tot_num_mpiprocs": 28,
        },
        "multiple": 1,
    },
    80: {
        "node_cpus": 28,
        "code_string": "vasp-5.4.4_28core@tekla2",
        "options_resources": {
            "parallel_env": "c28m128ib_mpi",
            "tot_num_mpiprocs": 28,
        },
        "multiple": 2,
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
SOURCE_DF = "/home/psanz/teklahome/projects/p2-CuZn/relaxed_structures_initialdb/initial_db/initial-database_11052023-153657_final.pkl"
TARGET_DF = "/WAREHOUSE/sp_database.pkl"

# Which calculation to run.
# As of now, either "sp" or "relax".
CALC_TYPE = "SP"

if __name__ == "__main__":
    # ID for the entire batch
    batch_id = str(uuid.uuid4().hex)
    ut.custom_print(f'Batch identifier: "{batch_id}"', "info")
    src_df = pd.read_pickle(SOURCE_DF)

    # Iterating over the target structures and launching a separate
    # vasp workchain for all of them.
    for it, target_row in src_df.iterrows():
        # Getting current structure, phase and formula.
        target_structure = target_row.structure.get_sorted_structure()
        phase = target_row.phase
        struct_formula = target_structure.formula.replace(" ", "")
        kspacing = KSPACING[phase]

        # Generate INCAR with correct kspacing
        incar = aut.generate_incar(phase=phase, calc_type=CALC_TYPE, kspacing=KSPACING)

        # Dictionary containing metadata for the calculation
        metadata_dict = {
            "label": f"{phase}-{struct_formula}-{it}_{CALC_TYPE}-bb_{batch_id}",
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
        incar["incar"]["kpar"] = 4
        if mult > 1:
            incar["incar"].pop("kpar")

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

        if CALC_TYPE.lower() == "sp":
            builder["perform_static"] = Bool(True)
            builder["relax"]["perform"] = Bool(False)

        elif CALC_TYPE.lower() == "relax":
            builder["perform_static"] = Bool(False)
            builder["relax"]["perform"] = Bool(True)

        # builder["settings"]

        # # Passing the relax inputs to the builder object
        # for key in relax_dict.keys():
        #     builder[key] = relax_dict[key]

        # Submitting the calculation.
        # Aiida should handle the scheduler, ssh connection and result
        # retrieval if everything is configured
        node = submit(builder)

        ut.custom_print(
            f"Launched workchain for structure {it}: '{struct_formula}' ({phase}) - node id: {node.id}",
            "debug",
        )

    ut.custom_print("All calculations launched.", "done")
    ut.custom_print("Check 'verdi process list' for more information", "info.")
