"""Definition of an aiida workchain for MACE active learning loops using MD."""

import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pymatgen.io.ase as pmg_ase
import torch
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.engine import (
    BaseRestartWorkChain,
    CalcJob,
    WorkChain,
    append_,
    while_,
)
from aiida.orm import (
    Bool,
    Dict,
    Float,
    FolderData,
    Group,
    Int,
    List,
    SinglefileData,
    Str,
    load_code,
    load_group,
    load_node,
    to_aiida_type,
)
from aiida.parsers.parser import Parser
from aiida.plugins import CalculationFactory
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from mace import data as mace_data
from mace import tools as mace_tools
from mace.calculators import MACECalculator
from pymatgen.core import Structure
from pymatgen.core.trajectory import Trajectory
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.lammps.data import LammpsData

from MatDBForge.active_learning import active_learning_utils as mdb_al
from MatDBForge.core import DATA_DIR
from MatDBForge.training import conversion as mdb_conv


class TrainMACEModelCalculationParser(Parser):
    def parse(self, **kwargs):
        """Parse the retrieved files of the calculation job."""
        # str that represents the absolute filepath to the temporary folder
        retrieved_temporary_folder: Path = Path(kwargs["retrieved_temporary_folder"])

        for child_file in retrieved_temporary_folder.iterdir():
            # create singlefile data for the model
            if "swa.model" in child_file.name:
                model_file = SinglefileData(file=child_file)

            if "train.txt" in child_file.name:
                # TODO: gather rmse_e, rmse_f
                with open(child_file) as f:
                    for line in f:
                        line_dict = json.loads(line)
                        if "rmse_e" in line_dict.keys():
                            last_dict = line_dict

                rmse_e = float(last_dict["rmse_e_per_atom"]) * 1000  # meV / atom
                rmse_f = float(last_dict["rmse_f"]) * 1000  # meV / A

        # Return CalcJob outputs
        self.out("model_file", model_file)
        self.out("m_rmse_e", Float(rmse_e))
        self.out("m_rmse_f", Float(rmse_f))


class TrainMACEModelCalculation(CalcJob):
    """Implementation of CalcJob to perform a MACE training using a settings dir."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "mace_settings_dict",
            valid_type=Dict,
            help="Dictionary containing MACE training settings.",
        )
        spec.input(
            "mace_train_file_path",
            valid_type=Str,
            help=(
                "Path to the file containing the structures to be used for training, "
                "in the extxyz format."
            ),
            # non_db=True,
            serializer=to_aiida_type,
        )
        spec.input(
            "test_file",
            valid_type=SinglefileData,
            help=(
                "File containing the structures to be used for training, "
                "in the extxyz format."
            ),
            required=False,
            default=None,
        )

        spec.input(
            "model_name",
            valid_type=Str,
            help=("Name given to the model."),
            serializer=to_aiida_type,
        )

        spec.output(
            "model_file",
            valid_type=SinglefileData,
            help="Path of the trained MACE model.",
        )
        spec.output(
            "m_rmse_e",
            valid_type=Float,
            help="Validation RMSE for the energy, in meV / atom.",
        )
        spec.output(
            "m_rmse_f",
            valid_type=Float,
            help="Validation RMSE for the forces, in meV / Å.",
        )

    def prepare_for_submission(self, folder):
        """Write the input files that are required for the code to run.

        :param folder: an `~aiida.common.folders.Folder` to temporarily write files on disk
        :return: `~aiida.common.datastructures.CalcInfo` instance
        """
        # Parsing mace settings dict
        # TODO: Add a way of checking if validation_file was given.
        params_list = []
        for key, val in self.inputs.mace_settings_dict.items():
            # print("\nkey: ", key)
            # print("val: ", val)
            if key == "train_file":
                val = Path(val).resolve().name

            if isinstance(val, str):
                curr_key = f"--{key}={val}"
            elif isinstance(val, bool):
                if val:
                    curr_key = f"--{key}"
            else:
                curr_key = f"--{key}={val}"
            params_list.append(curr_key)

        # Adding random seed
        params_list.append(f"--seed={np.random.randint(1, 100000000)}")

        path = Path(self.inputs.mace_train_file_path.value)
        caller_uuid = mdb_al.process_call_root(self.node)
        final_db_path = path.parent / (str(path.stem) + f"_{caller_uuid}{path.suffix}")

        # train_db_length = len(ase_read(final_db_path, format="extxyz", index=":"))
        # self.report(f"Training database size: {train_db_length} configurations.")
        # print(
        #     "self.inputs.mace_train_file_path.value: ",
        #     self.inputs.mace_train_file_path.value,
        # )

        # Copying database to temporary folder
        folder.insert_path(
            src=final_db_path,
            dest_name=self.inputs.mace_settings_dict["train_file"],
        )

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = self.options.output_filename
        codeinfo.cmdline_params = params_list

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = []
        calcinfo.provenance_exclude_list = [
            self.inputs.mace_settings_dict["train_file"]
        ]
        calcinfo.remote_copy_list = []

        # Gathering files. They won't be added to the repository,
        # and instead kept into a temporary folder.
        # They can later be processed during the parse function
        # by accessing the temporary folder.
        calcinfo.retrieve_temporary_list = [
            self.metadata.options.output_filename,
            "./*.model",
            "./results/*",
        ]

        return calcinfo

    def _build_process_label(self):
        model_name = self.inputs.model_name.value
        label = f"Training MACE model - {model_name}"
        return label


class ActiveLearningWorkChain(WorkChain):
    """
    WorkChain to run an active learning loop for a MACE potential using MD
    simulations to generate training data.
    """

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)

        spec.input("init_db_path", valid_type=Str, serializer=to_aiida_type)
        spec.input("final_db_path", valid_type=Str, serializer=to_aiida_type)
        spec.input("data_path", valid_type=Str, serializer=to_aiida_type)
        spec.input("mace_settings_path", valid_type=Str, serializer=to_aiida_type)
        spec.input("al_loop_iteration", valid_type=Int, serializer=to_aiida_type)
        spec.input("seed_size_frac", valid_type=Float, serializer=to_aiida_type)
        spec.input("md_temperature_K", valid_type=Float, serializer=to_aiida_type)
        spec.input("md_num_steps", valid_type=Int, serializer=to_aiida_type)
        spec.input("commitee_num_models", valid_type=Int, serializer=to_aiida_type)
        spec.input(
            "md_timestep_duration_ps", valid_type=Float, serializer=to_aiida_type
        )
        spec.input(
            "al_keep_frame_interval_perc", valid_type=Float, serializer=to_aiida_type
        )
        spec.input(
            "current_train_seed_structs", valid_type=List, serializer=to_aiida_type
        )
        spec.input(
            "seed_gen_db",
            valid_type=List,
            non_db=True,
        )
        spec.input("database_training", valid_type=List, serializer=to_aiida_type)
        spec.input(
            "current_train_seed_structs_idx",
            valid_type=List,
            serializer=to_aiida_type,
        )
        spec.input(
            "train_seed_group",
            valid_type=Str,
            serializer=to_aiida_type,
        )
        spec.input("mace_train", valid_type=Dict)

        spec.outline(
            # Training the main mace model (M0) and the commitee models
            # using the training database (Dt).
            cls.train_mace_model,
            # Gathering results from mace training.
            cls.get_mace_train_output,
            # All of the structures in the seed will be run using the MD
            # code selected, using the main model (M0)
            cls.run_md_seed,
            # Structures and energy predictions will be gathered and prepared
            # into a dataframe
            cls.gather_m0_md_results,
            # The structures from M0 will be evaluated using M1, M2 and M3.
            cls.check_commitee_results,
            # According to the difference in error between the models either:
            # The original structure will be removed from D0, or
            # The problematic structure will be calcualated using DFT
            cls.send_calc_or_remove_structures,
            cls.return_seed_dft,
            # TODO:
            # If high error (define this) is found on a training seed,
            # do not change seed until the error is decreased
            # cls.choose_next_seed,
        )
        spec.output("dft_calcs", valid_type=List, required=False)
        spec.output("m0_model_file", valid_type=SinglefileData)
        spec.output("upd_seed_gen_db", valid_type=List)
        spec.output("stop_md_seed_no_disagreement", valid_type=Bool)

        # spec.exit_code(
        #     400,
        #     "ERROR_NEGATIVE_NUMBER",
        #     message="The result is a negative number.",
        # )

    def train_mace_model(self):
        """
        Setup and submit TrainMACEModelCalculation for MACE model training.

        This function generates a new training database file by appending the
        workchain UUID to the original database file name. It then proceeds to train
        a specified number of MACE models using settings defined in the workflow
        inputs. The function updates the workchain context with the futures of the
        submitted calculation jobs, allowing for the tracking of these jobs. The most
        accurate model from these trainings is selected in a later step to drive MD
        simulations, while the others serve as committee models for energy evaluation.

        Returns
        -------
        None
            The function does not return a value but updates the workchain context with
            futures of the submitted TrainMACEModelCalculation jobs for later reference.
        """
        self.report("Generating new training database file.")

        # Adding workchain uuid input.data filename to path
        caller_uuid = mdb_al.process_call_root(load_node(self.uuid))
        path = Path(self.inputs.final_db_path.value)
        updated_path = str(
            path.parent / (str(path.stem) + f"_{caller_uuid}{path.suffix}")
        )

        # Generate new training data file
        mdb_conv.gen_mace_train_structure_list(
            path=updated_path,
            structure_list=self.inputs.database_training,
        )

        # Train n models (M1-Mn)
        # The most accurate model (during validation) will be chosen as the main model,
        # and used to drive the MD simulations. The remaining models will act as
        # commitee models and will only be used to evaluate energies.
        self.report(
            f"Training M0 - M{self.inputs.commitee_num_models.value} using "
            "current iteration data."
        )

        for model_num in range(self.inputs.commitee_num_models.value + 1):
            model_name = mdb_al.generate_model_name()

            # TODO: Use a general toml configuration file that includes all settings
            # Load training settings from json.
            mace_train_settings: Dict = mdb_al.load_mace_settings_json(
                self.inputs.mace_settings_path,
                updated_path,
                curr_model=model_name,
                curr_iter=self.inputs.al_loop_iteration.value,
            )

            # Run training and save new model file
            mace_train = CalculationFactory("mace-train")
            mace_builder = mace_train.get_builder()

            mace_builder.model_name = model_name
            mace_builder.mace_settings_dict = Dict(mace_train_settings)
            mace_builder.mace_train_file_path = self.inputs.final_db_path.value

            # TODO: Add as an input (Str)
            mace_builder.code = load_code("mace_run_train_gpu@tekla2-new-test")

            # TODO: Add as an input (Dict)
            mace_builder.metadata.options.resources = {
                "parallel_env": "c128m1024ib_mpi_32slotsbis",
                "tot_num_mpiprocs": 32,
            }
            mace_builder.metadata.options.parser_name = "mace-training-parser"

            mace_builder.metadata.options.queue_name = "c128m1024ibgpu4.q"
            mace_builder.metadata.options.max_wallclock_seconds = 117280000
            mace_builder.metadata.options.max_memory_kb = 102400000
            mace_builder.metadata.options.account = ""
            mace_builder.metadata.options.qos = ""
            # mace_builder.metadata.options.withmpi = True
            mace_builder.metadata.options.output_filename = (
                f"train_{model_name}_iter-{self.inputs.al_loop_iteration.value}"
            )
            # TODO: This should be set in the code?
            mace_builder.metadata.options.custom_scheduler_commands = "#$ -l gpu=1"

            future = self.submit(mace_builder)
            self.to_context(mace_training_results=append_(future))

    def get_mace_train_output(self):
        """
        Retrieve and process MACE training output for model selection.

        This function evaluates MACE training results to identify the model with the
        best performance based on a weighted sum of RMSE values for energy (E) and
        forces (F). It selects the model with the lowest weighted E+F, considers
        the importance of forces via a weighting factor, and processes the best model
        to create a LAMMPS-compatible potential file. Information about the selected
        model and committee models is updated in the context for further use.

        Returns
        -------
        None
            The function updates the workchain context with the best model's RMSE values, the
            LAMMPS potential file, and the committee models' information but does not
            return any value directly.
        """
        mace_training_results = self.ctx.mace_training_results

        model_name_list = []
        weighted_E_F_sum_list = []

        # Iterate over all models and get weighted sum of E and F
        for calc in mace_training_results:
            # Loading calculation node
            curr_calc = load_node(calc.uuid)

            # Getting model name
            model_name_list.append(curr_calc.inputs.model_name.value)

            # Adding E + F multiplied by a weight value, in order to consider
            # forces when deciding which model to keep
            force_weight = self.inputs.mace_train.get("result_force_weight", 0.1)
            weighted_E_F_sum = curr_calc.outputs.m_rmse_e.value + (
                force_weight * curr_calc.outputs.m_rmse_f
            )
            weighted_E_F_sum_list.append(weighted_E_F_sum)

        # Get the most accurate model name during the validation by checking
        # the lowest E+F*weight.
        best_model_name = model_name_list[np.argmin(weighted_E_F_sum_list)]

        commitee_models_tupl_name_uuid = []
        for idx, calc in enumerate(mace_training_results):
            # Loading calculation node
            curr_calc = load_node(calc.uuid)

            # Loading model name
            model_name = curr_calc.inputs.model_name.value

            # Getting model file
            model_file = curr_calc.outputs.model_file

            # Use checking if the currentt case is the best model.
            # Saving model results
            if model_name == best_model_name:
                # Overwriting m0_rmse values with actual training values
                self.ctx.m0_rmse_e = curr_calc.outputs.m_rmse_e
                self.ctx.m0_rmse_f = curr_calc.outputs.m_rmse_f

                # Convert model to LAMMPS compatible format
                # and return it to workchain context
                self.ctx.lammps_potential_file = mdb_al.create_mace_lammps_model(
                    model_file
                )

                self.report(
                    f"Generated LAMMPS potential using'{model_name}' as M0."
                    f"RMSE E: {self.ctx.m0_rmse_e.value:.3f} meV / at"
                    f"RMSE F: {self.ctx.m0_rmse_f.value:.3f} meV / Å"
                )
                self.out("m0_model_file", model_file)
            else:
                self.report(
                    f"Trained commitee model '{model_name}' - "
                    f"RMSE E: {curr_calc.outputs.m_rmse_e.value:.3f} meV / at, "
                    f"RMSE F: {curr_calc.outputs.m_rmse_f.value:.3f} meV / Å"
                )
                commitee_models_tupl_name_uuid.append((model_name, model_file.uuid))

        # Sending commitee model paths to current context
        self.ctx.commitee_models_tupl_name_uuid = commitee_models_tupl_name_uuid

    def gen_md_input(
        self,
        structure: Structure,
        potential_path: str,
    ) -> str:
        """
        Generate a MACE-LAMMPS input file for MD simulations using a template.

        This function creates a customized MACE-LAMMPS input file for molecular dynamics
        simulations by reading and modifying a template file. The modifications include
        setting the pair style to a MACE-based style, defining pair coefficients using
        the provided potential file, specifying simulation parameters such as timestep
        size, temperature, initial velocities, and the number of timesteps. The function
        utilizes the structure's composition to dynamically adjust the input file's
        content, catering to the specifics of the simulation's atomic species and the
        applied potential.

        Parameters
        ----------
        structure : pymatgen.core.structure.Structure
            The structure object containing composition information used to
            set species-specific parameters in the LAMMPS input file.
        potential_path : str
            Path to the potential file to be used in the pair_coeff directive.

        Returns
        -------
        str
            The modified LAMMPS input file content as a string, ready for use in
            simulation.

        Notes
        -----
        - The template file is expected to be in the `DATA_DIR/input_files` directory.
        - This function replaces placeholders in the template with actual values from
        the function's inputs and the structure's composition.
        - Future versions may include more dynamic options based on LAMMPS's extensive
        configurability.
        """
        with open(f"{DATA_DIR}/input_files/input.lammps", "r") as f:
            lammps_template = f.read()

        lammps_template = lammps_template.replace(
            "$MACESTYLE", "mace no_domain_decomposition"
        )

        species = structure.composition.elements

        # Setting MACE potential as the potential to use
        pair_coeff_str = "* * "
        pair_coeff_str += f"{Path(potential_path).name} "

        # Adding species from given structure
        for spec in species:
            pair_coeff_str += f"{spec} "
        lammps_template = lammps_template.replace("$PAIRCOEFF", pair_coeff_str)

        # Setting elements
        elem_str = ""
        for elem in species:
            elem_str += f"{elem} "

        lammps_template = lammps_template.replace("$ELEMS", elem_str)

        # Setting timestep size
        timestep_val = self.inputs.md_timestep_duration_ps.value
        lammps_template = lammps_template.replace("$TSTEP_SIZE", str(timestep_val))

        # Setting temperature
        temp_val = self.inputs.md_temperature_K.value
        temp_arr = f"{temp_val} {temp_val} {100 * timestep_val}"
        lammps_template = lammps_template.replace("$TEMPARR", temp_arr)

        # Setting intial velocities.
        seed = np.random.randint(low=1, high=1000000)
        vel_str = f"{temp_val} {seed}"
        lammps_template = lammps_template.replace("$VELOCITY", vel_str)

        # Setting number of timesteps
        num_tstep_str = str(self.inputs.md_num_steps.value)
        lammps_template = lammps_template.replace("$NSTEPS", num_tstep_str)

        return lammps_template

    def run_md_seed(self):
        """
        Run MD simulations for all structures in the current training seed using M0.

        This function initiates molecular dynamics simulations for each structure
        within the current training seed, utilizing a predefined main model (M0).
        It generates and configures MACE-LAMMPS calculation jobs, sets up the necessary
        input files and parameters, and submits these jobs for execution.
        Calculation nodes are stored and managed within the workflow's context for
        later retrieval and analysis.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The function does not return a value but updates the workflow context with
            futures of the submitted MD simulation jobs, facilitating tracking and
            subsequent analysis of these simulations.

        Notes
        -----
        - This function assumes the availability of a trained MACE-LAMMPS potential file within
        the workflow's context.
        - Submitted calculation jobs are added to an AiiDA group for organization and
        are tagged with additional information to link them back to their respective
        positions in the database.
        """
        self.report("Running MD (using M0) for all structures in the current seed...")

        # Creating a list in the context to store the nodes
        self.ctx.current_train_seed = []

        # this string with the label used in the code setup.
        code = load_code("mace-lammps@localhost-mpirun.mpich")
        builder = CalculationFactory("lammps.raw").get_builder()
        builder.code = code

        # Getting the lammps potential file in a temporary foler
        with self.ctx.lammps_potential_file.as_path() as lmp_pot_path:
            lmp_pot_filename = Path(lmp_pot_path).name
            lmp_pot_path = str(lmp_pot_path)

            # Setting the trajectory to be retrieved and the
            # potential file to be copied into the calculation folder
            builder_settings = {
                "additional_retrieve_list": ["structure.lammpstrj"],
                "local_copy_list": [
                    (
                        self.ctx.lammps_potential_file.uuid,
                        lmp_pot_path,
                        lmp_pot_filename,
                    )
                ],
            }

            builder.settings = Dict(builder_settings)

            for idx, curr_structure in enumerate(
                self.inputs.current_train_seed_structs
            ):
                # Structures are stored as a dict in order to be json-serializable
                for key in ["pbc", "cell", "numbers", "positions", "forces"]:
                    curr_structure[key] = np.array(curr_structure[key])

                curr_structure = Atoms.fromdict(curr_structure)

                # Converting structure to pymatgen
                curr_structure = pmg_ase.AseAtomsAdaptor.get_structure(curr_structure)
                struct_properties = curr_structure.properties

                curr_input = self.gen_md_input(
                    structure=curr_structure,
                    potential_path=lmp_pot_filename,
                )

                script = SinglefileData(io.StringIO(curr_input))
                builder.script = script

                lammps_struct_str = LammpsData.from_structure(
                    curr_structure, atom_style="atomic"
                ).get_str()

                data = SinglefileData(io.StringIO(lammps_struct_str))
                builder.files = {
                    "data": data,
                    "mace_potential": self.ctx.lammps_potential_file,
                }
                builder.filenames = {
                    "data": "structure.lammps",
                    "mace_potential": lmp_pot_filename,
                }

                index_in_db = self.inputs.current_train_seed_structs_idx[idx]

                # TODO: Add this as initial parameters (TOML)
                # HACK: During debugging, run the calculation on 1 CPU and kill it
                # if it runs longer than 1800 seconds.
                builder.metadata.options = {
                    "resources": {
                        "num_machines": 1,
                        "num_mpiprocs_per_machine": 1,
                        "num_cores_per_mpiproc": 2,
                    },
                    "max_wallclock_seconds": 1800,
                    "withmpi": True,
                }

                # Submitting current calculation
                future = self.submit(builder)

                # Add calculation to the workchain's aiida group.
                self.ctx.current_train_seed.append(future)
                curr_group = load_group(uuid=self.inputs.train_seed_group.value)
                curr_group.add_nodes(future)

                # Writing extra information that helps associating the calculation
                # with its position on the database.
                for key, val in struct_properties.items():
                    future.base.extras.set(key, val)
                future.base.extras.set("index_in_db", index_in_db)

                # Telling the work chain to wait for the md to finish
                # before continuing the workflow.
                # We append the future to a list of workflows.
                self.to_context(md_seed_workchains=append_(future))

    def gather_m0_md_results(self):
        """
        Gather MD simulation results for all structures in the current seed.

        This function collects the results from MD simulations of each structure within
        the current training seed.
        It extracts trajectories, energies, and forces from each simulation's output and
        aggregates these into a structured format. The collected data is then organized
        into a pandas DataFrame, which is stored in the workflow's context.
        """
        self.report("Gathering M0 MD results for the current seed...")
        new_rows = []

        # Gathering all results
        for workchain in self.ctx.md_seed_workchains:
            workchain_results = workchain.outputs.retrieved
            steps_E_F_arr = self.gather_energies_from_workchain(workchain_results)
            traj, forces = self.gather_traj_from_workchain(workchain_results)

            new_rows.append(
                {
                    "trajectory": traj,
                    "energy": {"m0": steps_E_F_arr[:, 1]},
                    "forces": {"m0": forces},
                    "al_step": self.inputs.al_loop_iteration.value,
                    "index_in_db": workchain.base.extras.all["index_in_db"],
                    "mdb_struct_type": workchain.base.extras.all["mdb_struct_type"],
                    "material_name": workchain.base.extras.all["struct_name"],
                    "unique_id": workchain.base.extras.all["aiida_uuid"],
                }
            )

        # Creating a DataFrame with all the results
        self.ctx.md_seed_results_df = pd.DataFrame(new_rows)

    def gather_energies_from_workchain(self, workchain_results):
        """
        Extracts energy, force, and step information from a 'lammps.out' output file
        contained within a workchain result.

        This function parses the 'lammps.out' output file from a given set of workchain
        results. It specifically looks for lines starting with 'thermo ', then extracts
        step number, energy, and force values from these lines. The extracted values are
        compiled into a numpy array with each column representing steps, energies, and
        forces, respectively.

        Parameters
        ----------
        workchain_results : object
            An object containing the results of a workchain. It is expected to have
            a method `get_object_content` which can retrieve the content of 'lammps.out'.

        Returns
        -------
        numpy.ndarray
            A 2D numpy array where each row corresponds to a step in the workchain.
            The columns represent step number, energy, and force,respectively.

        Notes
        -----
        - The function assumes that the 'lammps.out' file has a specific format where
          relevant data is prefixed with 'thermo '.
        - The function starts processing from the second occurrence of lines starting
          with 'thermo '.
        - The function assumes that the step number is at index 1, energy at index 4,
          and force at index 6 of each relevant line.

        Examples
        --------
        >>> for workchain in self.ctx.md_seed_workchains:
        >>>     workchain_results = workchain.outputs.retrieved
        >>>     step_E_F_arr = gather_energies_from_workchain(workchain_results)
        >>>     print(step_E_F_arr)
        [[1, energy1, force1],
        [2, energy2, force2],
        ...]

        """
        output = workchain_results.get_object_content("lammps.out")
        steps = [line for line in output.splitlines() if line.startswith("thermo ")]

        energy_array = []
        force_array = []
        step_array = []

        for step in steps:
            split = step.split()
            step_array.append(int(split[1]))
            energy_array.append(float(split[4]))
            force_array.append(float(split[7]))

        step_E_F_arr = np.stack((step_array, energy_array, force_array), axis=1)
        return step_E_F_arr

    def gather_traj_from_workchain(self, workchain_results: FolderData) -> Trajectory:
        """
        Extracts trajectory data from a LammpsRawCalculation as pymatgen Trajectory.

        This function parses 'structure.lammpstrj' from the given workchain results to
        extract atomic coordinates and lattice information. It then constructs a
        sequence of pymatgen Structure objects, representing each frame of the
        trajectory, which are combined into a pymatgen Trajectory object.

        Parameters
        ----------
        workchain_results : object
            An object containing the results of a workchain, expected to have a method
            `get_object_content` to retrieve the content of 'structure.lammpstrj'.

        Returns
        -------
        Trajectory
            A pymatgen Trajectory object representing the sequence of structures over time.
        np.array
            A (n_frames x n x 3) numpy array containing the forces of every atom.

        Notes
        -----
        The function currently assumes a constant lattice across all frames and extracts
        the number of atoms and frames from the trajectory file. The time step duration is
        taken from `self.inputs.md_timestep_duration_ps.value`.
        """
        # Get trajectory file from aiida repo node
        traj_data = workchain_results.get_object_content("structure.lammpstrj")

        # Separate the file the file into lines
        lines = traj_data.splitlines()

        num_atoms = 0
        num_frames = 0
        offset = 9

        # Get the number of atoms and the total number of frames
        for posc, line in enumerate(lines):
            if "ITEM: NUMBER OF ATOMS" in line:
                num_atoms = int(lines[posc + 1])
            elif "ITEM: TIMESTEP" in line:
                num_frames = int(lines[posc + 1])

        # assembling a pymatgen structure
        struct_list = []
        forces_list = []
        line_posc = 0
        for curr_struct in range(0, num_frames + 1):
            line_posc_ini = line_posc
            line_posc += num_atoms + offset
            curr_struct_list = lines[line_posc_ini:line_posc]
            curr_struct_info = np.array([line.split() for line in curr_struct_list[9:]])

            curr_struct_coords = curr_struct_info[:, :5]
            curr_struct_forces = curr_struct_info[:, 5:].astype(np.float32)

            # Lammps box bounds snapshot format
            # xlo_bound xhi_bound xy
            # ylo_bound yhi_bound xz
            # zlo_bound zhi_bound yz
            lattice = np.zeros([3, 3])
            lattice_vals = np.array(
                [[vec.split()[1]] for vec in lines[5:8]], dtype=float
            )
            cnt = 0
            for posc, row in enumerate(lattice):
                row[cnt] = lattice_vals[posc]
                cnt += 1

            # species = [ele.split()[1] for ele in curr_struct_coords]
            species = curr_struct_coords[:, 1]

            # coord_array = np.array(
            #     [ele.split()[2:] for ele in curr_struct_coords], dtype=float
            # )
            coord_array = curr_struct_coords[:, 2:5].astype(np.float32)

            curr_struct = Structure(
                lattice=lattice,
                species=species,
                coords=coord_array,
                coords_are_cartesian=True,
            )
            struct_list.append(curr_struct)
            forces_list.append(curr_struct_forces)

        # TODO: Add a way to change constant lattice.
        traj = Trajectory.from_structures(
            struct_list,
            constant_lattice=True,
            time_step=self.inputs.md_timestep_duration_ps.value,
        )

        return traj, np.array(forces_list)

    # TODO: Implement a CalcJob for this?
    def check_commitee_results(self):
        self.report("Evaluating trajectories with models...")

        for row in self.ctx.md_seed_results_df.iterrows():
            # self.report(f"Checking struct {row[0]} results with all models...")
            curr_traj = row[1]["trajectory"]

            # Working with all models
            for model_name, model in self.ctx.commitee_models_tupl_name_uuid:
                # model_id += 1
                # model_path = str(Path(model.value).resolve())
                # model_name = "m" + str(model_id)

                # TODO: Use MACE settings input to set the settings.

                with load_node(model).as_path() as model_path:
                    device_type = "cuda"
                    # mace_tools.torch_tools.set_default_dtype("float32")
                    device = mace_tools.torch_tools.init_device(device_type)

                    # Load MACE model
                    model = torch.load(f=model_path, map_location=device_type)
                    model = model.to(device_type)

                    for param in model.parameters():
                        param.requires_grad = False

                    # Load data and prepare input
                    atoms_list = [
                        AseAtomsAdaptor().get_atoms(pym_struct)
                        for pym_struct in curr_traj
                    ]

                    configs = [
                        mace_data.config_from_atoms(atoms) for atoms in atoms_list
                    ]

                    z_table = mace_tools.utils.AtomicNumberTable(
                        [int(z) for z in model.atomic_numbers]
                    )

                    data_loader = mace_tools.torch_geometric.dataloader.DataLoader(
                        dataset=[
                            mace_data.AtomicData.from_config(
                                config, z_table=z_table, cutoff=float(model.r_max)
                            )
                            for config in configs
                        ],
                        batch_size=64,
                        shuffle=False,
                        drop_last=False,
                    )

                    # Collect data
                    energies_list = []
                    forces_collection = []

                    for batch in data_loader:
                        batch = batch.to(device)
                        output = model(batch.to_dict(), compute_stress=True)
                        # print("\n\noutput: ", output.keys())
                        energies_list.append(
                            mace_tools.torch_tools.to_numpy(output["energy"])
                        )

                        forces = np.split(
                            mace_tools.torch_tools.to_numpy(output["forces"]),
                            indices_or_sections=batch.ptr[1:],
                            axis=0,
                        )
                        forces_collection.append(forces[:-1])  # drop last as its empty

                    energies = np.concatenate(energies_list, axis=0)
                    forces_list = [
                        forces
                        for forces_list in forces_collection
                        for forces in forces_list
                    ]
                    assert len(atoms_list) == len(energies) == len(forces_list)

                    updated_ene_dict = row[1]["energy"]
                    updated_ene_dict.update({model_name: energies})
                    self.ctx.md_seed_results_df.at[row[0], "energy"] = updated_ene_dict

                    # TODO: Check if this is the proper way of gathering the forces
                    # forces_list = np.array(forces_list)
                    # forces_norm = np.linalg.norm(forces_list, axis=2)
                    # total_force_norm_per_frame = forces_norm.sum(axis=1)

                    self.ctx.md_seed_results_df.at[row[0], "forces"][
                        model_name
                    ] = forces_list  # total_force_norm_per_frame

    def send_calc_or_remove_structures(self):
        self.report("Deciding which structures to keep...")

        # TODO: Add as a input.
        # TODO: Set to 10.
        chem_acc = 30  # meV?
        chem_acc_multiplier = 10  # TESTING: 0.0001
        e_rmse = self.ctx.m0_rmse_e.value
        e_error_threshold = chem_acc_multiplier * e_rmse

        # REMOVE
        # e_error_threshold = chem_acc_multiplier

        f_rmse = self.ctx.m0_rmse_f.value
        f_error_threshold = chem_acc_multiplier * f_rmse

        # REMOVE
        # f_error_threshold = chem_acc_multiplier

        delete_indices = []
        dft_structures = []

        # Every row contains the results of MD for a single structure, which are:
        # trajectory, energies, forces, al_step, index_in_db, mdb_struct_type,
        # cluster, material_name, unique_id
        for idx, row in self.ctx.md_seed_results_df.iterrows():
            # Getting all energy predictions
            # TODO For E: Do variance
            model_energies_dict = row["energy"]
            energies_std = mdb_al.get_model_energies_std(model_energies_dict)

            # Any True value in this array is over the energy error threshold
            # and must be sent to calculate with DFT.
            error_e_structures = np.ma.make_mask(energies_std >= e_error_threshold)

            # TODO For F: Do variance
            model_forces_dict = row["forces"]
            forces_std = mdb_al.get_model_energies_std(model_forces_dict)
            forces_std_norm = np.linalg.norm(forces_std, axis=2)
            forces_std_norm_max = np.amax(forces_std_norm, axis=1)

            # Any True value in this array is over the force error threshold
            # and must be sent to calculate with DFT.
            error_f_structures = np.ma.make_mask(
                forces_std_norm_max >= f_error_threshold
            )

            # Joining both error masks to get a single True/False array marking structures
            # to be computed
            error_all_structures = np.ma.mask_or(error_e_structures, error_f_structures)

            # If all values in error_all_structures are false, delete the main
            # structure from D0.
            flag_no_error_structs = np.all(error_all_structures == 0)

            # True if the model is above chemical accuracy (bad performance)
            if (e_rmse > chem_acc) or (f_rmse > chem_acc):
                flag_above_chemical_acc = True
                self.report("Current model not reaching chemical accuracy.")

            # The index of the structure to delete will
            # be added to a list, which will be used as a mask to select
            # which structures to remove outside of the loop.
            if flag_no_error_structs and not flag_above_chemical_acc:
                delete_indices.append(row["unique_id"])

            # If there are some structures to submit or the model does not reach
            # chemical accuracy, select some of them and send them to DFT.
            elif not flag_no_error_structs or flag_above_chemical_acc:
                struct_arr = error_all_structures

                if isinstance(error_all_structures, np.bool_):
                    struct_arr = np.ones_like(energies_std)

                # Instead of keeping them all, select some of them (get 1 frame
                # every n frames)
                dft_structures = mdb_al.select_dft_structures(
                    struct_arr=struct_arr,
                    frame_interval=self.inputs.al_keep_frame_interval_perc,
                )

                dft_structures = [
                    row["trajectory"][int(struct)] for struct in dft_structures
                ]

                # REMOVE: For testing purposes. Remove this!
                # dft_structures = [row["trajectory"][5]]

                for calc_idx, dft_struct in enumerate(dft_structures):
                    builder = mdb_al.get_dft_calc_builder(
                        dft_struct,
                        row,
                        calc_idx,
                        self.inputs.train_seed_group.value,
                    )

                    # Submitting current calculation
                    future = self.submit(builder)
                    future.base.extras.set("mdb_calc_uuid", row["unique_id"])
                    self.to_context(dft_struct_seed_calcs=append_(future))

        self.report(
            f"Commitee decision: {len(dft_structures)} DFT - "
            f"{len(delete_indices)} delete"
        )

        # Deleting well represented structures from seed_gen_db (Ds), if any.
        if len(delete_indices) > 0:
            self.report(
                f"Deleting {len(delete_indices)} structures from seed"
                " generating DB (Ds)"
            )
            seed_gen_db = mdb_al.remove_structs_from_seed_gen_db(
                self.inputs.seed_gen_db, delete_indices
            )

            self.inputs.seed_gen_db = seed_gen_db
            self.out("upd_seed_gen_db", seed_gen_db)

        # If no structure is well represented, nothing will be deleted.
        else:
            self.report("Nothing removed from DB.")
            self.out("upd_seed_gen_db", self.inputs.seed_gen_db)

        # if isinstance(seed_gen_db, list):
        # self.inputs.seed_gen_db = List(self.inputs.seed_gen_db)

    def return_seed_dft(self):
        """
        Gather and output DFT calculations for the current seed structures.

        This function collects DFT calculations for the structures in the current seed,
        which are then returned as outputs in the workchain using the namespace
        `dft_calcs`. A check is performed to determine if the results agree using
        MACE models, and this check also outputted to the workchain using the
        namespace `stop_md_seed_no_disagreement`.
        """
        try:
            dft_calcs = len(self.ctx.dft_struct_seed_calcs)
            self.report(f"Gathered {dft_calcs} DFT calculations.")
        except AttributeError:
            self.ctx.dft_struct_seed_calcs = []

        return_list = mdb_al.gather_dft_calcs(
            [node.uuid for node in self.ctx.dft_struct_seed_calcs]
        )

        self.out("dft_calcs", return_list)
        self.out(
            "stop_md_seed_no_disagreement",
            mdb_al.check_md_seed_agreement(return_list),
        )


class ActiveLearningBaseWorkChain(BaseRestartWorkChain):
    _process_class = ActiveLearningWorkChain

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)

        ##########
        # FIXME  #
        ##########
        # TODO: There are some problems when exposing the inputs and outputs
        ##########
        spec.expose_inputs(
            ActiveLearningWorkChain,
            namespace="active_learning",
            exclude=[
                "current_train_seed_structs",
                "current_train_seed_structs_idx",
                "al_loop_iteration",
                "train_seed_group",
                "seed_gen_db",
                "database_training",
            ],
        )

        spec.expose_outputs(
            ActiveLearningWorkChain,
            exclude=[
                "final_training_db",
                "final_model_file",
            ],
        )

        spec.outline(
            # Load the initial database (D_ini), that will be used as the
            # training database (Dt) without changing the original database.
            # Additionally, create a copy of the database (seed_gen_db, Ds),
            # this will be used to generate the MD seeds.
            cls.get_database,
            # Create inputs for workchains and initialize iterative counter
            cls.setup,
            # This part will loop to complete the process
            # It will loop `self.ctx.inputs.max_al_iterations` times.
            # while_(cls.should_run_process)(
            while_(cls.check_al_loop_conditions)(
                # Get random structures from Ds to generate the MD seed.
                cls.get_training_seed,
                # TODO: Implement this function correctly.
                # Generate descriptors for the current seed.
                # cls.generate_descriptors,
                # Run training seed
                cls.run_process,
                # Check for correct results
                # cls.inspect_process,
                # Get results from workchain
                cls.results_loop,
                # Update Ds and Di to include results from DFT.
                # Update the inputs for the next workchain.
                cls.add_dft_results_to_db,
                cls.get_al_loop_break_conditions,
            ),
            cls.results_final,
        )
        spec.output(
            "final_training_db",
            valid_type=List,
        )
        spec.output("final_model_file", valid_type=SinglefileData)

    def get_database(self):
        """Loading initial database."""
        self.report("Reading database file...")

        # The training database (Dt) from which copies are made
        # for further processing.
        # New structures will be added here.
        self.ctx.database_training = ase_read(
            filename=self.inputs.active_learning.init_db_path.value,
            format="extxyz",
            index=":",
        )

        # If dtype=object is not used, numpy won't be able to create this jagged array.
        # We need an array to have an easier time indexing and using masks for item
        # removal.
        self.ctx.database_training = np.array(self.ctx.database_training, dtype=object)

        # A copy of the initial database, (Ds)
        # used specifically for generating training seeds and running the MDs.
        # New structures will be added and well represented configs removed from here.
        # TODO: Is this necessary?
        self.ctx.seed_gen_db = self.ctx.database_training.copy()

        self.report(
            f"Loaded initial database containing {len(self.ctx.seed_gen_db)} structures."
        )

    def results_loop(self):
        """Attach the outputs specified in the output specification from the last completed process."""
        node = self.ctx.children[self.ctx.iteration - 1]

        # TODO: Gather outputs manually, instead of using __attach_outputs
        self._attach_outputs(node)
        return None

    def add_dft_results_to_db(self):
        """
        Incorporate DFT calculation results into the training and seed generation databases.

        This method updates the training and seed generation databases with DFT calculation
        results. If any DFT calculations have been performed, their results are appended to
        both the training database and the seed generation database.
        """
        # Updating current training seed
        self.ctx.seed_gen_db = self.outputs["upd_seed_gen_db"]
        self.ctx.inputs.seed_gen_db = self.outputs["upd_seed_gen_db"]

        try:
            cnt_dft_calcs = len(self.outputs["dft_calcs"])
        except KeyError:
            cnt_dft_calcs = 0

        if cnt_dft_calcs > 0:
            self.report(f"Adding {cnt_dft_calcs} DFT calculations to DB.")

            seed_gen_db = self.ctx.seed_gen_db.get_list()
            # Adding calculations to training database and seed_generation database
            for dft_calc in self.outputs["dft_calcs"]:
                self.ctx.database_training = np.append(
                    self.ctx.database_training, dft_calc
                )
                seed_gen_db.append(dft_calc)

            self.ctx.seed_gen_db = List(seed_gen_db)
            self.ctx.seed_gen_db.store()
            self.ctx.inputs.seed_gen_db = self.ctx.seed_gen_db

        self.report(
            f"Iteration {self.ctx.iteration}: "
            f"seed_gen_db {len(self.ctx.seed_gen_db)}, "
            f"training_db: {len(self.ctx.database_training)} entries"
        )

        # Updating final database
        self.report("Updating database file...")

        database_training_all_ase = mdb_al.convert_database_to_ase_atoms(
            self.ctx.database_training, deserialize=True
        )

        path = Path(self.ctx.inputs.final_db_path.value)
        caller_uuid = mdb_al.process_call_root(self.node)
        final_db_path = path.parent / (str(path.stem) + f"_{caller_uuid}{path.suffix}")
        print("final_db_path in add_dft_results_to_db: ", final_db_path)

        ase_write(
            filename=final_db_path,
            images=database_training_all_ase,
            format="extxyz",
        )

        for idx, struct in enumerate(database_training_all_ase):
            database_training_all_ase[idx] = mdb_al.serialize_ase(struct)

        self.report("Database file updated.")

    def get_al_loop_break_conditions(self):
        """
        Evaluate and set conditions to potentially break the active learning loop.

        This function checks for specific conditions that might warrant terminating the
        active learning (AL) loop early:
        - Gathers `stop_md_seed_no_disagreement` from the outputs of the inner workchain
          and stores it in the workchain's context. If this is True, the workchain will stop.
        - Checks whether all structures have been removed from the seed generation database
          (indicating no further candidates for evaluation). If this is True,
          the workchain will stop.

        The results of these checks are stored in the workflow's context.
        """
        # Sending seed disagreement flag to context
        self.ctx.stop_md_seed_no_disagreement = self.outputs[
            "stop_md_seed_no_disagreement"
        ]

        # Sending empty seed_gen_db flag to context
        if len(self.ctx.inputs.seed_gen_db) == 0:
            self.ctx.seed_gen_db_all_structs_removed = Bool(True)
        else:
            self.ctx.seed_gen_db_all_structs_removed = Bool(False)

    def generate_descriptors(self):
        self.report("Generating descriptors...")

        # TODO: This does not work
        model_path = (
            self.inputs.active_learning.data_path.value + "/"
            # + self.inputs.active_learning.mace_potential_names[0]
        )
        calculator = MACECalculator(
            model_paths=model_path, device="cuda", default_dtype="float32"
        )
        descriptor_list = []

        # TODO: Check what to use: database_training or seed_gen_db?
        for struct in self.ctx.database_training:
            curr_struct_descriptors = calculator.get_descriptors(struct)
            descriptor_list.append(curr_struct_descriptors)

        np.vstack(descriptor_list)
        self.ctx.database_descriptors = descriptor_list

    def setup(self):
        """Call BaseRestartWorkChain setup and create inputs dict in self.ctx.inputs.

        This `self.ctx.inputs` dictionary will be used by the `BaseRestartWorkChain`
        to submit the process in the internal loop.
        """
        self.report("Starting Workchain setup.")
        super().setup()

        self.ctx.inputs = self.exposed_inputs(
            ActiveLearningWorkChain, "active_learning"
        )

        # Creating aiida group to store all calculations
        ctime = time.strftime("%Y%m%dT%H%M%S")
        # TESTING: Change this back to the string below
        seed_group = Group(label=f"remove_test_{ctime}")
        # seed_group = Group(label=f"train_md_seed_{ctime}")
        seed_group.store()
        self.ctx.inputs.train_seed_group = seed_group.uuid
        self.report(f"Created group: '{self.ctx.inputs.train_seed_group}'.")

        # Providing current iteration to children workchain.
        self.ctx.inputs.al_loop_iteration = self.ctx.iteration

        # Setting conditionals to always run the first iteration of the
        # active learning loop.
        self.ctx.stop_md_seed_no_disagreement = Bool(False)
        self.ctx.seed_gen_db_all_structs_removed = Bool(False)

        seed_db_serialized = []
        database_training_serialized = []
        for s in self.ctx.seed_gen_db:
            curr_s = mdb_al.serialize_ase(s)
            seed_db_serialized.append(curr_s)
            database_training_serialized.append(curr_s)

        self.ctx.inputs.seed_gen_db = List(seed_db_serialized)
        self.ctx.inputs.database_training = List(database_training_serialized)

        self.ctx.init_seed_gen_db_size = len(seed_db_serialized)
        self.report("Workchain setup finished.")

    def check_al_loop_conditions(self) -> bool:
        """
        Evaluate conditions to determine whether to continue the active learning loop.

        This method assesses multiple conditions to decide if the active learning (AL)
        loop should continue. It considers whether the maximum number of iterations
        has been reached, and the following two criteria:
         - No disagreement among predictions for an entire MD seed. (if True: stop)
         - Depletion of the seed generation database. (if True: stop)

        The function updates the workchain context with the iteration status and
        generates reports based on the evaluation of these conditions.

        Returns
        -------
        bool
            A boolean value indicating whether the AL loop should continue. Returns
            `True` if conditions are met for another iteration; otherwise,
            returns `False`.

        Notes
        -----
        - The method uses `self.ctx.is_finished`, `self.ctx.iteration`,
        `self.inputs.max_iterations.value`,
        `self.ctx.stop_md_seed_no_disagreement.value`,
        and `self.ctx.seed_gen_db_all_structs_removed.value`

        """
        max_iterations = self.inputs.max_iterations.value

        # This will be True if the workchain still needs to be running due
        # to the number of iterations.
        iterations_status_ok = (
            not self.ctx.is_finished and self.ctx.iteration < max_iterations
        )
        # If either stop_md_seed_no_disagreement or seed_gen_db_all_structs_removed are
        # True, the loop must be stopped
        # This will be True while the AL loop needs to be repeated
        continue_loop_conditions = (
            not self.ctx.stop_md_seed_no_disagreement.value
            and not self.ctx.seed_gen_db_all_structs_removed.value
        )

        # This will be True if the workchain can be repeated.
        continue_cond = continue_loop_conditions and iterations_status_ok

        if self.ctx.stop_md_seed_no_disagreement.value:
            self.report("Stopping AL Loop as all predictions agree for a MD seed.")
        elif self.ctx.seed_gen_db_all_structs_removed.value:
            self.report(
                "Stopping AL Loop as seed generating database has been depleted."
            )
        else:
            self.report(
                f"Proceeding with iteration-{self.ctx.iteration} of AL Loop "
                "as stopping conditions not met."
            )
            self.ctx.inputs.al_loop_iteration = self.ctx.iteration

        return continue_cond

    def get_training_seed(self):
        """
        Selects a random subset of structures from the seed generation database to
        create a training seed for the active learning loop.

        This function calculates the number of structures to be included in the training
        seed based on the specified fraction of the seed generation database's length.
        It then randomly selects and populates the training seed with these structures.

        Returns
        -------
            None. The function updates self.ctx.current_train_seed_structs with the selected
            structures.
        """
        self.report(f"Starting AL Loop iteration {self.ctx.iteration}...")
        self.report("Getting training seed...")
        self.ctx.inputs.metadata.description = (
            "Perform MD simulations, evaluate and refine ML models. "
            f"Step: {self.ctx.inputs.al_loop_iteration}"
        )
        self.ctx.inputs.metadata.label = f"Step - {self.ctx.inputs.al_loop_iteration}"

        # Getting length of the seed generating database
        db_length = len(self.ctx.seed_gen_db)

        # Defining the current seed size as a function of the intial seed size
        seed_size = int(
            self.ctx.inputs.seed_size_frac.value * self.ctx.init_seed_gen_db_size
        )

        # This should avoid tring to select more structures than available
        if seed_size > db_length:
            seed_size = db_length

        # Choosing structures at random to create the training seed
        selected_structs = np.random.choice(
            range(db_length),
            size=seed_size,
            replace=False,
        )

        self.ctx.inputs.current_train_seed_structs_idx = list(selected_structs)

        # The set of random structures selected from the seed generation
        # database to be used in training.
        self.ctx.current_train_seed_structs = []

        # Populating training seed with the selected random structures
        for idx in selected_structs:
            self.ctx.current_train_seed_structs.append(self.ctx.seed_gen_db[idx])

        self.report(
            f"Created training seed with {seed_size}"
            f" structures ({self.ctx.inputs.seed_size_frac.value*100}% of initial size)."
        )
        # Adding current train seed to the context
        current_train_seed_serialized = []
        for curr_s in self.ctx.current_train_seed_structs:
            curr_s = mdb_al.serialize_ase(curr_s)
            current_train_seed_serialized.append(curr_s)

        self.ctx.inputs.current_train_seed_structs = current_train_seed_serialized

    def results_final(self):
        """
        Finalize the results at the end of the workchain.

        This method is responsible for preparing and returning the final training
        database at the conclusion of the workchain. It serializes the structures
        within the training database to a format compatible with AiiDA storage and
        subsequent processing. The serialized structures are then used to prepare
        the final training database, which is outputted from the workchain. This
        signifies the completion of the workchain and the availability of the
        processed training data for further use.
        """
        self.report("Returning final results...")

        # Converting final training_db to aiida types
        struct_list_serialized = []
        for curr_s in list(self.ctx.database_training):
            curr_s = mdb_al.serialize_ase(curr_s)
            struct_list_serialized.append(curr_s)

        self.ctx.serialized_struct_list = struct_list_serialized

        train_db = mdb_al.prepare_output_final_training_db(
            self.ctx.serialized_struct_list
        )

        self.out("final_training_db", train_db)
        self.report("Workchain completed!")
