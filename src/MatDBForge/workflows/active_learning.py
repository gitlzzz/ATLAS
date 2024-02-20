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
    StructureData,
    load_code,
    load_group,
    load_node,
    to_aiida_type,
)
from aiida.parsers.parser import Parser
from aiida.plugins import CalculationFactory
from ase import Atoms
from ase.io import read as ase_read
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

        # TODO: Add a way of checking if validation_file was given.

        # Copying database to temporary folder
        folder.insert_path(
            src=self.inputs.mace_train_file_path.value,
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
        label = f"Training MACE model - {model_name.upper()}"
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
        # spec.input("mace_potential_names", valid_type=List, serializer=to_aiida_type)
        spec.input(
            "al_keep_frame_interval_perc", valid_type=Float, serializer=to_aiida_type
        )
        spec.input(
            "current_train_seed_structs", valid_type=List, serializer=to_aiida_type
        )
        spec.input("seed_gen_db", valid_type=List, serializer=to_aiida_type)
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
        # spec.input("md_code", valid_type=AbstractCode)
        # spec.input("mace_code", valid_type=AbstractCode)

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
            # If high error (define this) is found on a training seed,
            # do not change seed until the error is decreased
            # cls.choose_next_seed,
            ## When the loop ends this should run to finish the workchain.
            # Generate final database and model files. Store to given paths.
            # cls.return_final_db,
        )
        spec.output("dft_calcs", valid_type=List)
        spec.output("m0_model_file", valid_type=SinglefileData)
        spec.output("upd_seed_gen_db", valid_type=List)
        spec.output("stop_md_seed_no_disagreement", valid_type=Bool)

        # TODO: Implement these into the base workchain
        # spec.output("result_db_path", valid_type=Str)
        # spec.output("final_model_path", valid_type=Str)
        # spec.output("al_loop_results_dataframe", valid_type=Dict)

        # spec.exit_code(
        #     400,
        #     "ERROR_NEGATIVE_NUMBER",
        #     message="The result is a negative number.",
        # )

    # def _build_process_label(self):
    #     curr_iteration = self.inputs.al_loop_iteration.value
    #     label = f"Training and refining models. Step: {curr_iteration}"
    #     return label

    def train_mace_model(self):
        self.report("Generating new training database file.")

        # Generate new training data file
        mdb_conv.gen_mace_train_structure_list(
            path=self.inputs.final_db_path.value,
            structure_list=self.inputs.database_training,
        )

        # Train n models (M1-Mn), normally 5 in total (main M0 + committee M1-M4)
        self.report(
            f"Training M0 - M{self.inputs.commitee_num_models.value} using "
            "current iteration data."
        )
        # commitee_model_paths = []
        for model_num in range(self.inputs.commitee_num_models.value + 1):
            model_name = f"m{model_num}"

            # TODO: Use a general toml configuration file that includes all settings
            # Load training settings from json.
            mace_train_settings: Dict = mdb_al.load_mace_settings_json(
                self.inputs.mace_settings_path,
                self.inputs.final_db_path,
                curr_model=model_name,
                curr_iter=self.inputs.al_loop_iteration.value,
            )

            # Run training and save new model file
            # TODO: Launch TrainMACEModelCalculation here.
            # Check the CalcJob.options dict for the queue_name,
            # and CalcJob for the code input.
            # train_outputs: Dict = mdb_al.run_mace_train_custom(mace_train_settings)
            mace_train = CalculationFactory("mace-train")
            mace_builder = mace_train.get_builder()

            # database_file = SinglefileData(self.inputs.final_db_path.value).store()
            mace_builder.model_name = model_name
            mace_builder.mace_settings_dict = Dict(mace_train_settings)
            mace_builder.mace_train_file_path = self.inputs.final_db_path.value

            # TODO: Add as an input (Str)
            mace_builder.code = load_code("mace_run_train@tekla2")

            # TODO: Add as an input (Dict)
            mace_builder.metadata.options.resources = {
                "parallel_env": "c128m1024ib_mpi_32slots",
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
            mace_builder.metadata.options.custom_scheduler_commands = "#$ -l gpu=1"
            # mace_builder.metadata.options.prepend_text = "export CUDA_VISIBLE_DEVICES=GPU-e2fb8a48-c8e7-6c3c-e5e2-e252506896ae"
            # mace_builder.metadata.options.prepend_text = "module load Intel_OneAPI/2024\n"
            # mace_builder.metadata.options.prepend_text = "OMP_NUM_THREADS=32"

            future = self.submit(mace_builder)
            self.to_context(mace_training_results=append_(future))

    def get_mace_train_output(self):
        mace_training_results = self.ctx.mace_training_results

        commitee_model_paths = []
        for calc in mace_training_results:
            curr_calc = load_node(calc.uuid)

            model_name = curr_calc.inputs.model_name.value

            # Overwriting m0_rmse values with actual training values
            self.ctx.m0_rmse_e = curr_calc.outputs.m_rmse_e
            self.ctx.m0_rmse_f = curr_calc.outputs.m_rmse_f

            # Getting model file
            model_file = curr_calc.outputs.model_file

            # HACK: Maybe there is a more elegant ways of doing this?
            # Saving model results
            if model_name == "m0":
                # Convert model to LAMMPS compatible format
                # and return it to workchain context
                self.ctx.lammps_potential_file = mdb_al.create_mace_lammps_model(
                    model_file
                )

                self.report(f"Generated LAMMPS potential using '{model_name.upper()}'.")
                self.report(
                    f"Current iteration M0 RMSE E: {self.ctx.m0_rmse_e.value:.3f} meV / atom"
                )
                self.report(
                    f"Current iteration M0 RMSE F: {self.ctx.m0_rmse_f.value:.3f} meV / Å"
                )
                self.out("m0_model_file", model_file)
            else:
                self.report(f"Trained commitee model '{model_name.upper()}'.")
                commitee_model_paths.append(mace_training_results.m_path)

        # Sending commitee model paths to current context
        self.ctx.commitee_model_paths = commitee_model_paths

    def gen_md_input(self, structure, potential_path):
        with open(f"{DATA_DIR}/input_files/input.lammps", "r") as f:
            lammps_template = f.read()

        lammps_template = lammps_template.replace(
            "$MACESTYLE", "mace no_domain_decomposition"
        )

        species = structure.composition.elements

        pair_coeff_str = "* * "
        # potential_path = (
        #     self.inputs.data_path.value + "/" + self.inputs.mace_potential_names[0]
        # )
        pair_coeff_str += f"{Path(potential_path).name} "

        for spec in species:
            pair_coeff_str += f"{spec} "
        # print("pair_coeff_str: ", pair_coeff_str)
        lammps_template = lammps_template.replace("$PAIRCOEFF", pair_coeff_str)

        # timestep
        timestep_val = self.inputs.md_timestep_duration_ps.value
        lammps_template = lammps_template.replace("$TSTEP_SIZE", str(timestep_val))

        # temp
        temp_val = self.inputs.md_temperature_K.value
        temp_arr = f"{temp_val} {temp_val} {100 * timestep_val}"
        lammps_template = lammps_template.replace("$TEMPARR", temp_arr)

        elem_str = ""
        for elem in species:
            elem_str += f"{elem} "

        lammps_template = lammps_template.replace("$ELEMS", elem_str)

        # Velocity
        seed = np.random.randint(low=1, high=1000000)
        vel_str = f"{temp_val} {seed}"
        lammps_template = lammps_template.replace("$VELOCITY", vel_str)

        num_tstep_str = str(self.inputs.md_num_steps.value)
        lammps_template = lammps_template.replace("$NSTEPS", num_tstep_str)

        return lammps_template

    def run_md_seed(self):
        self.report("Running MD (using M0) for all structures in the current seed...")

        # print("\n\n###### AL self.inputs ######\n", self.inputs)

        # Creating a list in the context to store the nodes
        self.ctx.current_train_seed = []

        # this string with the label used in the code setup.
        code = load_code("mace-lammps-fix2@localhost")
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

                # Converting to pymatgen
                curr_structure = pmg_ase.AseAtomsAdaptor.get_structure(curr_structure)
                struct_properties = curr_structure.properties

                curr_input = self.gen_md_input(
                    structure=curr_structure,
                    # potential_path=lmp_pot_path,
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

                # TODO: Change this
                # HACK: Run the calculation on 1 CPU and kill it
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

                self.ctx.current_train_seed.append(future)
                curr_group = load_group(uuid=self.inputs.train_seed_group.value)
                curr_group.add_nodes(future)

                # Writing extra information
                for key, val in struct_properties.items():
                    future.base.extras.set(key, val)
                future.base.extras.set("index_in_db", index_in_db)

                # Telling the work chain to wait for the md to finish
                # before continuing the workflow.
                # We append the future to a list of workflows.
                self.to_context(md_seed_workchains=append_(future))

                # return ToContext(md=future)

    def gather_m0_md_results(self):
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

        # Increasing the current AL step number
        # self.ctx.al_loop_iteration += 1

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

    def check_commitee_results(self):
        for row in self.ctx.md_seed_results_df.iterrows():
            # self.report(f"Checking struct {row[0]} results with all models...")
            # print("row: ", row)
            curr_traj = row[1]["trajectory"]

            # Working with all models
            for model_id, model in enumerate(self.ctx.commitee_model_paths):
                model_id += 1

                model_path = str(Path(model.value).resolve())
                model_name = "m" + str(model_id)
                # self.report(f"Evaluating with model {model_name.upper()}...")

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
                    AseAtomsAdaptor().get_atoms(pym_struct) for pym_struct in curr_traj
                ]

                configs = [mace_data.config_from_atoms(atoms) for atoms in atoms_list]

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

        chem_acc_multiplier = 10  # TODO: Set to 10.
        e_rmse = self.ctx.m0_rmse_e.value
        e_error_threshold = chem_acc_multiplier * e_rmse

        f_rmse = self.ctx.m0_rmse_f.value
        f_error_threshold = chem_acc_multiplier * f_rmse

        delete_indices = []

        # self.report(f"Current iteration M0 RMSE E [meV / atom]: {e_rmse}")
        # self.report(f"Current iteration M0 RMSE F [meV / A]: {f_rmse}")

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

            if np.all(error_all_structures == 0):
                # If all values in error_all_structures are false, delete the main
                # structure from D0. The index of the structure to delete will
                # be added to a list, which will be used as a mask to select
                # which structures to remove outside of the loop.
                delete_indices.append(row["unique_id"])
                self.report(
                    f"{idx} - All models agree. No DFT calculations will be submitted."
                )
            else:
                # Else, select some of them and send them to DFT
                self.report("Models disagree. Sending structures to DFT.")

                # Instead of keeping them all, select some of them (get 1 frame
                # every n frames)
                dft_structures = mdb_al.select_dft_structures(
                    error_all_structures,
                    self.inputs.al_keep_frame_interval_perc,
                )

                dft_structures = [
                    row["trajectory"][int(struct)] for struct in dft_structures
                ]

                # REMOVE: For testing purposes. Remove this!
                dft_structures = [row["trajectory"][5]]

                self.report(f"Submitting {len(dft_structures)} DFT calculations.")
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

        # Deleting marked entries.
        print(
            "\n\n###### before removal self.inputs.seed_gen_db: ",
            len(self.inputs.seed_gen_db),
        )

        print("len(delete_indices): ", len(delete_indices))
        # Deleting structures from seed_gen_db (Ds)
        if len(delete_indices) > 0:
            self.report(
                f"Deleting {len(delete_indices)} structures from seed"
                " generating DB (Ds)"
            )
            seed_gen_db = mdb_al.remove_structs_from_seed_gen_db(
                self.inputs.seed_gen_db, delete_indices
            )

            self.inputs.seed_gen_db = seed_gen_db
            # seed_gen_db = np.array(self.inputs.seed_gen_db)
            # del_mask = np.ones(len(seed_gen_db), bool)
            # del_mask[delete_indices] = 0
            # self.inputs.seed_gen_db = list(seed_gen_db[del_mask])
            print(
                "###### after removal self.inputs.seed_gen_db: ",
                len(seed_gen_db),
            )
            self.out("upd_seed_gen_db", seed_gen_db)
        else:
            print(
                "###### nothing reomved from self.inputs.seed_gen_db",
            )
            self.report("Nothing removed from DB.")
            self.out("upd_seed_gen_db", self.inputs.seed_gen_db)

        # if isinstance(seed_gen_db, list):
        # self.inputs.seed_gen_db = List(self.inputs.seed_gen_db)

    def return_seed_dft(self):
        try:
            dft_calcs = len(self.ctx.dft_struct_seed_calcs)
            self.report(f"Gathered {dft_calcs} DFT calculations.")
        except AttributeError:
            self.ctx.dft_struct_seed_calcs = []

        # vasprun_list = []
        return_list = mdb_al.gather_dft_calcs(
            [node.uuid for node in self.ctx.dft_struct_seed_calcs]
        )
        # # Adding structures to the initial DB
        # for finished_dft_calc in self.ctx.dft_struct_seed_calcs:
        #     vasprun = mdb_conv._gather_mace_req_calc_data_from_node(finished_dft_calc)
        #     vasprun = vasprun.todict()
        #     vasprun["pbc"] = [bool(boo) for boo in vasprun["pbc"]]
        #     vasprun_list.append(vasprun)

        # return_list = List([val for val in vasprun_list])
        # print("return_list: ", return_list)
        # print("return_list: ", type(return_list))

        self.out("dft_calcs", return_list)

        self.out(
            "stop_md_seed_no_disagreement",
            mdb_al.check_md_seed_agreement(return_list),
        )

    def choose_next_seed(self):
        # Retrain the models and retry seed? Or choose a new seed?
        print("intial db after append: ", len(self.inputs.database_training))
        self.report("Choosing next seed...")

    def return_final_db(self):
        self.report("Returning final model and database...")
        self.ctx.string = mdb_al.generate_placeholder_text()
        self.ctx.string_2 = mdb_al.generate_placeholder_text()

        # training_df = prepare_output_dataframe(self.ctx.md_seed_results_df)

        # self.ctx.md_seed_results_df.index = self.ctx.md_seed_results_df.index.map(str)
        # training_df = Dict(self.ctx.md_seed_results_df.to_dict(orient="index"))

        self.out(
            "result_db_path",
            self.ctx.string,
        )

        self.out(
            "final_model_path",
            self.ctx.string_2,
        )

        # self.out(
        #     "al_loop_results_dataframe",
        #     training_df,
        # )


class ActiveLearningBaseWorkChain(BaseRestartWorkChain):
    _process_class = ActiveLearningWorkChain

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)

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
        spec.expose_outputs(ActiveLearningWorkChain)
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
                cls.results,
                # Update Ds and Di to include results from DFT.
                # Update the inputs for the next workchain.
                cls.add_dft_results_to_db,
                cls.get_al_loop_break_conditions,
            ),
            cls.results_final,
        )
        spec.output("final_training_db", valid_type=List)
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
        self.ctx.seed_gen_db = np.array(self.ctx.seed_gen_db, dtype=object)

        self.report(
            f"Loaded database containing {len(self.ctx.seed_gen_db)} structures."
        )

    def add_dft_results_to_db(self):
        # Updating current training seed
        self.ctx.seed_gen_db = self.outputs["upd_seed_gen_db"]
        self.ctx.inputs.seed_gen_db = self.outputs["upd_seed_gen_db"]

        try:
            cnt_dft_calcs = len(self.outputs["dft_calcs"])
        except KeyError:
            cnt_dft_calcs = 0

        if cnt_dft_calcs > 0:
            self.report("Adding {cnt_dft_calcs} DFT calculations to DB.")

            # Adding calculations to training database and seed_generation database
            for dft_calc in self.outputs["dft_calcs"]:
                self.ctx.seed_gen_db = np.append(self.ctx.seed_gen_db, dft_calc)
                self.ctx.database_training = np.append(
                    self.ctx.database_training, dft_calc
                )
            self.ctx.inputs.seed_gen_db = self.ctx.seed_gen_db

    def get_al_loop_break_conditions(self):
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

        self.ctx.inputs.seed_gen_db = seed_db_serialized
        self.ctx.inputs.database_training = database_training_serialized

        self.ctx.init_seed_gen_db_size = len(seed_db_serialized)
        self.report("Workchain setup finished.")

    def check_al_loop_conditions(self) -> bool:
        max_iterations = self.inputs.max_iterations.value

        # This will be True if the workchain still needs to be running due
        # to the number of iterations.
        iterations_status_ok = (
            not self.ctx.is_finished and self.ctx.iteration < max_iterations
        )

        # stop_md_seed_no_disagreement has to be True to stop the loop
        # seed_gen_db_all_structs_removed has to be True to stop the loop
        # If any of the above two is True, stop the loop.

        # This will be True while the AL loop needs to be repeated
        continue_loop_conditions = (
            not self.ctx.stop_md_seed_no_disagreement.value
            and not self.ctx.seed_gen_db_all_structs_removed.value
        )

        # This will be True if the workchain can be repeated.
        continue_cond = continue_loop_conditions and iterations_status_ok

        if self.ctx.stop_md_seed_no_disagreement.value:
            self.report("Stopping AL Loop as all predictions agree for a MD seed.")

        if self.ctx.seed_gen_db_all_structs_removed.value:
            self.report(
                "Stopping AL Loop as seed generating database has been depleted."
            )

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
        self.report(
            f"Starting AL Loop iteration {self.ctx.inputs.al_loop_iteration}..."
        )
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
        self.report("Returning final results...")
        # Converting training_db to aiida types
        struct_list_serialized = []
        for curr_s in list(self.ctx.database_training):
            curr_s = mdb_al.serialize_ase(curr_s)
            struct_list_serialized.append(curr_s)

        self.ctx.serialized_struct_list = struct_list_serialized

        train_db = mdb_al.prepare_output_final_training_db(
            self.ctx.serialized_struct_list
        )

        self.out("final_training_db", train_db)

        # Return model file as output
        # mace_training_results = self.ctx.mace_training_results
        # print("mace_training_results: ", mace_training_results)
        # for calc in mace_training_results:
        #     curr_calc = load_node(calc.uuid)
        #     print("curr_calc: ", curr_calc)

        #     model_name = curr_calc.inputs.model_name.value
        #     if model_name == "m0":
        #         model_file = curr_calc.outputs.model_file
        #         print("model_file: ", model_file)
        #         self.out("final_model_file", model_file)

        # super().results()
        self.report("Workchain completed!")
