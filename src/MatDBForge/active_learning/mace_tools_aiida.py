#!/usr/bin/env python
"""AiiDA plugin for MACE calculations."""

import json
import shutil
import time
from pathlib import Path

import numpy as np
from aiida import orm
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.folders import Folder
from aiida.engine import CalcJob
from aiida.parsers.parser import Parser
from aiida_lammps.calculations.raw import LammpsRawCalculation
from aiida_lammps.parsers.parse_raw import parse_outputfile
from ase.io import read as ase_read

from MatDBForge.active_learning import active_learning_utils as mdb_al_ut


class TrainMACEModelCalculationParser(Parser):
    """Parser for the retrieved files from a MACE training calculation job."""

    def parse(self, **kwargs):
        """Parse the retrieved files of the calculation job."""
        # str that represents the absolute filepath to the temporary folder
        retrieved_temporary_folder: Path = Path(kwargs["retrieved_temporary_folder"])

        model_file = None
        rmse_e = None
        rmse_f = None

        for child_file in retrieved_temporary_folder.rglob("*"):
            # Create singlefile data for the model

            # If swa was used, get the swa model preferentially
            if "swa.model" in child_file.name:
                model_file = orm.SinglefileData(file=child_file)
                continue

            # If swa was not used, get the non-compiled model, as it can be
            # used to get the descriptors.
            if ".model" in child_file.name and "compiled" not in child_file.name:
                model_file = orm.SinglefileData(file=child_file)
                continue

            # Get train statistics from the training output
            if "train.txt" in child_file.name:
                # TODO: gather rmse_e, rmse_f
                with open(child_file) as f:
                    for line in f:
                        line_dict = json.loads(line)
                        if "rmse_e" in line_dict:
                            last_dict = line_dict

                rmse_e = float(last_dict["rmse_e_per_atom"]) * 1000  # meV / atom
                rmse_f = float(last_dict["rmse_f"]) * 1000  # meV / A

        # Return failed code
        if not rmse_e or not rmse_f or not model_file:
            return self.exit_codes.ERROR_INVALID_OUTPUT

        # Return CalcJob outputs
        self.out("model_file", model_file)
        self.out("m_rmse_e", orm.Float(rmse_e))
        self.out("m_rmse_f", orm.Float(rmse_f))


class TrainMACEModelCalculation(CalcJob):
    """Implementation of a CalcJob to perform a MACE training using a settings dir.

    Inputs
    ------

    mace_settings_dict : orm.Dict
        Dictionary containing MACE settings.
    mace_train_file_path : orm.Str
        Local machine path to the structures to evaluate in extxyz format.
    test_file : orm.SinglefileData
        Local machine path to the structures for testing in extxyz format.
    mace_train_file_path : orm.Str
        Path to the configurations to evaluate in extxyz format.
    model_name : orm.Str
        Name given to the model.

    Outputs
    -------
    model_file : orm.SinglefileData
        Path of the trained MACE model
    m_rmse_e : orm.Float
        Validation RMSE for the energy, in meV / atom.
    m_rmse_f : orm.Float
        Validation RMSE for the forces, in meV / Å.

    Exit Codes
    ----------
    420 : ERROR_INVALID_OUTPUT
        Training calculation could not run.
    """

    @classmethod
    def define(cls, spec):
        """Define the input and output specifications for the CalcJob."""
        super().define(spec)
        spec.input(
            "mace_settings_dict",
            valid_type=orm.Dict,
            help="Dictionary containing MACE training settings.",
        )
        spec.input(
            "mace_train_file_path",
            valid_type=orm.Str,
            help=(
                "Path to the file containing the structures to be used for training, "
                "in the extxyz format."
            ),
            # non_db=True,
            serializer=orm.to_aiida_type,
        )
        spec.input(
            "test_file",
            valid_type=orm.SinglefileData,
            help=(
                "File containing the structures to be used for testing during training,"
                "in the extxyz format."
            ),
            required=False,
            non_db=True,
            default=None,
        )

        spec.input(
            "model_name",
            valid_type=orm.Str,
            help=("Name given to the model."),
            serializer=orm.to_aiida_type,
        )

        spec.output(
            "model_file",
            valid_type=orm.SinglefileData,
            help="Path of the trained MACE model.",
        )
        spec.output(
            "m_rmse_e",
            valid_type=orm.Float,
            help="Validation RMSE for the energy, in meV / atom.",
        )
        spec.output(
            "m_rmse_f",
            valid_type=orm.Float,
            help="Validation RMSE for the forces, in meV / Å.",
        )
        spec.exit_code(
            420, "ERROR_INVALID_OUTPUT", "training calculation could not run"
        )

    def prepare_for_submission(self, folder):
        """Write the input files that are required for the code to run.

        :param folder: an `Folder` to temporarily write files on disk
        :return: `CalcInfo` instance
        """
        # Parsing mace settings dict
        # TODO: Add a way of checking if validation_file was given.
        params_list = []

        # Adding cli parameters to list
        prepare_cli_args_mace(params_list, self.inputs.mace_settings_dict)

        # Adding random seed
        params_list.append(f"--seed={np.random.randint(1, 100000000)}")

        # (for MACE v0.3.7) Disabling multiheads finetuning
        # params_list.append("--multiheads_finetuning=False")

        # Copying database to temporary folder
        final_db_path = self.inputs.mace_train_file_path.value
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


# mace-descriptors-parser
class GetMACEDescriptorsCalculationParser(Parser):
    """Parser for the retrieved files from a MACE descriptors job."""

    def parse(self, **kwargs):
        """Parse the retrieved files of the calculation job."""
        # str that represents the absolute filepath to the temporary folder
        retrieved_temporary_folder: Path = Path(kwargs["retrieved_temporary_folder"])

        descriptor_arr_file = None
        descr_max_arr = None
        descr_min_arr = None

        for child_file in retrieved_temporary_folder.iterdir():
            # create singlefile data for the descriptors
            if "curr_it_db_descriptors.pkl" in child_file.name:
                descriptor_arr_file = orm.SinglefileData(file=child_file)

            if "curr_it_db_max.npy" in child_file.name:
                descr_max_arr = orm.ArrayData(np.load(child_file))

            if "curr_it_db_min.npy" in child_file.name:
                descr_min_arr = orm.ArrayData(np.load(child_file))

        # Return failed code if output files not found
        if not all((descriptor_arr_file, descr_max_arr, descr_min_arr)):
            return self.exit_codes.ERROR_INVALID_OUTPUT

        # Return CalcJob outputs
        self.out("descriptors_file", descriptor_arr_file)
        self.out("descriptors_max_array", descr_max_arr)
        self.out("descriptors_min_array", descr_min_arr)


# mace-get-descriptors
class GetMACEDescriptorsCalculation(CalcJob):
    """Calculation to obtain descriptors for a structure database from MACE."""

    @classmethod
    def define(cls, spec):  # noqa: D102
        super().define(spec)
        spec.input(
            "model_file",
            valid_type=orm.SinglefileData,
            help="Trained MACE model.",
            non_db=True,
        )
        spec.input(
            "mace_train_file_path",
            help=(
                "Path to the file containing the structures to be used for training, "
                "in the extxyz format."
            ),
            serializer=orm.to_aiida_type,
            non_db=True,
        )

        spec.output(
            "descriptors_file",
            valid_type=orm.SinglefileData,
            help="Path of the file containing the MACE descriptors array.",
        )
        spec.output(
            "descriptors_max_array",
            valid_type=orm.ArrayData,
            help="Array containing the maximum values for the MACE descriptors, "
            "shaped according to the model size.",
        )
        spec.output(
            "descriptors_min_array",
            valid_type=orm.ArrayData,
            help="Array containing the minimum values for the MACE descriptors, "
            "shaped according to the model size.",
        )

    def prepare_for_submission(self, folder):
        """Write the input files that are required for the code to run.

        :param folder: a `Folder` to temporarily write files on disk
        :return: `CalcInfo` instance
        """
        # Copying database to temporary folder
        if isinstance(self.inputs.mace_train_file_path, orm.Str):
            final_db_path = self.inputs.mace_train_file_path.value
            folder.insert_path(
                src=Path(final_db_path),
                dest_name="current_db.xyz",
            )
        elif isinstance(self.inputs.mace_train_file_path, orm.SinglefileData):
            with self.inputs.mace_train_file_path.as_path() as model_path:
                folder.insert_path(
                    src=model_path,
                    dest_name="current_db.xyz",
                )

        # Copying model to temporary folder
        model_file = self.inputs.model_file

        with model_file.as_path() as model_path:
            folder.insert_path(
                src=model_path,
                dest_name="current_model_mace.model",
            )

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = []
        calcinfo.provenance_exclude_list = []
        calcinfo.remote_copy_list = []

        # Gathering files. They won't be added to the repository,
        # and instead kept into a temporary folder.
        # They can later be processed during the parse function
        # by accessing the temporary folder.
        calcinfo.retrieve_temporary_list = [
            "./*.npy",
            "./*.pkl",
        ]

        return calcinfo


# entry-point: mace-eval
class EvaluateMACEConfigsCalculation(CalcJob):
    """CalcJob to evaluate E and F of structures using a MACE model."""

    @classmethod
    def define(cls, spec):  # noqa: D102
        super().define(spec)
        spec.input(
            "mace_settings_dict",
            valid_type=orm.Dict,
            help="Dictionary containing MACE settings.",
            serializer=orm.to_aiida_type,
        )
        spec.input(
            "model_file",
            valid_type=orm.SinglefileData,
            help="Path to the trained MACE model.",
        )
        spec.input(
            "configuration_to_evaluate",
            valid_type=orm.SinglefileData,
            help="Path to the configurations to evaluate in extxyz format.",
        )
        spec.output(
            "configuration_result_file",
            valid_type=orm.SinglefileData,
            help="File containing all configurations evaluated using MACE in"
            " the extxyz format.",
        )
        spec.output(
            "energy_result_list",
            valid_type=orm.List,
            help="List of values for the energy prediction.",
        )
        spec.output(
            "forces_result_list",
            valid_type=orm.List,
            help="List of array of values for the force prediction.",
        )
        spec.exit_code(
            420, "ERROR_INVALID_OUTPUT", "training calculation could not run"
        )

    def prepare_for_submission(self, folder):
        """Write the input files that are required for the code to run.

        :param folder: an `Folder` to temporarily write files on disk
        :return: `CalcInfo` instance
        """
        # Parsing mace settings dict
        params_list = []
        params_list.append("--model=current_mace_model.model")
        params_list.append("--configs=current_configuration.xyz")
        params_list.append("--output=results.out")

        # Adding cli parameters to list
        prepare_cli_args_mace(params_list, self.inputs.mace_settings_dict)

        params_list.append("--info_prefix=mdb_mace_eval_")

        # Remove duplicate entries
        params_list = list(set(params_list))

        # Copying configuration to temporary folder
        with self.inputs.model_file.as_path() as model_path:
            folder.insert_path(
                src=model_path,
                dest_name="current_mace_model.model",
            )

        curr_structure_file: orm.SinglefileData = self.inputs.configuration_to_evaluate

        with curr_structure_file.as_path() as path:
            folder.insert_path(
                src=path,
                dest_name="current_configuration.xyz",
            )

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = params_list

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = []
        calcinfo.provenance_exclude_list = []
        calcinfo.remote_copy_list = []

        # Gathering files.
        calcinfo.retrieve_list = [
            "results.out",
        ]

        # They won't be added to the repository,
        # and instead kept into a temporary folder.
        calcinfo.retrieve_temporary_list = [
            "results.out",
        ]

        return calcinfo


class EvaluateMACEConfigsCalculationParser(Parser):
    """Parser for MACE E and F evaluation calculation jobs."""

    def parse(self, **kwargs):
        """Parse the retrieved files of the calculation job."""
        # str that represents the absolute filepath to the temporary folder
        retrieved_temporary_folder: Path = Path(kwargs["retrieved_temporary_folder"])

        result_dict_list = []
        forces_dict_list = []
        result_structures = None
        result_dict = None

        for child_file in retrieved_temporary_folder.iterdir():
            # create singlefile data for the descriptors
            if child_file.name == "results.out":
                results_path = child_file.absolute()
                result_structures = ase_read(child_file, format="extxyz", index=":")
                for curr_structure in result_structures:
                    result_dict = mdb_al_ut.serialize_ase(curr_structure)
                    result_dict_list.append(result_dict)

                    forces_dict = np.vstack(result_dict["mdb_mace_eval_forces"])
                    forces_dict_list.append(forces_dict)

        energy_float_list = [
            ene_dict["info"]["mdb_mace_eval_energy"] for ene_dict in result_dict_list
        ]

        try:
            # Return failed code if output files not found
            if not result_structures or not result_dict:
                return self.exit_codes.ERROR_INVALID_OUTPUT
        except Exception:
            return self.exit_codes.ERROR_INVALID_OUTPUT

        # Return CalcJob outputs
        # self.out("configuration_result_list", orm.SinglefileData("result_dict_list"))
        self.out("configuration_result_file", orm.SinglefileData(results_path))
        self.out("energy_result_list", orm.List(energy_float_list))
        self.out("forces_result_list", orm.List(forces_dict_list))


class RunMDCalculationGPULAMMPSMACE(LammpsRawCalculation):
    """aiida-lammps raw calculation modified to run on gpu using kokkos."""

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        """Prepare the calculation for submission.

        :param folder: A temporary folder on the local file system.
        :returns: A :class:`aiida.common.datastructures.CalcInfo` instance.
        """
        filename_input = self.inputs.metadata.options.input_filename
        filename_output = self.inputs.metadata.options.output_filename
        filenames = (
            self.inputs["filenames"].get_dict() if "filenames" in self.inputs else {}
        )
        provenance_exclude_list = []

        with folder.open(filename_input, "w") as handle:
            handle.write(self.inputs.script.get_content())

        for key, node in self.inputs.get("files", {}).items():
            # The filename with which the file is written to the working directory
            # is defined by the `filenames` input namespace, falling back to the
            # filename of the `orm.SinglefileData` node if not defined.
            filename = filenames.get(key, node.filename)

            with folder.open(filename, "wb") as target, node.open(mode="rb") as source:
                shutil.copyfileobj(source, target)

            provenance_exclude_list.append(filename)

        codeinfo = CodeInfo()
        codeinfo.cmdline_params = [
            "-in",
            filename_input,
            "-k",
            "on",
            "g",
            "1",
            "-sf",
            "kk",
        ]
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = self.inputs.metadata.options.output_filename

        calcinfo = CalcInfo()
        calcinfo.provenance_exclude_list = provenance_exclude_list
        calcinfo.retrieve_list = [filename_output]
        if "settings" in self.inputs:
            calcinfo.retrieve_list += self.inputs.settings.get_dict().get(
                "additional_retrieve_list", []
            )
        calcinfo.codes_info = [codeinfo]

        return calcinfo


# entry-point: mace-committee-eval
class CheckMACECommitteeResultsCalculation(CalcJob):
    """CalcJob to check the E and F of structures using a committee of MACE models.

    Define the input and output specifications for the CalcJob.

    Parameters
    ----------
    spec : aiida.engine.processes.ports.PortNamespace
        The process specification to define the inputs, outputs, and exit codes.

    Inputs
    ------
    commitee_models : PortNamespace
        A namespace to hold an arbitrary number of committee MACE potentials.
    mace_settings_dict : aiida.orm.Dict
        Dictionary containing MACE settings.
    configurations_to_evaluate : aiida.orm.orm.SinglefileData
        Path to the configurations to evaluate in extxyz format.

    Outputs
    -------
    energy_result_dict : aiida.orm.Dict
        Dictionary of values for the energy prediction.
        The dict has the following format:
        `{"model_1": [E1, E2...], "model_2": [E1, E2...], ..."}`
    forces_result_dict : aiida.orm.Dict
        Dictionary of arrays of values for the force prediction.
        The dict has the following format:
        `{"model_1": <ndarray shape n_at, 3, n_frames>, "model_2": ..."}`
    num_threads : aiida.orm.Int
        Number of OpenMP threads to use for the evaluation.

    Exit Codes
    ----------
    420 : ERROR_OUT_OF_VRAM
        CUDA out of GPU memory.
    421 : ERROR_OUTPUT_NOT_FOUND
        Missing output file.
    """

    @classmethod
    def define(cls, spec):  # noqa: D102
        super().define(spec)

        # Namespace that will hold an arbitrary number of committee MACE potentials
        spec.input_namespace(
            "commitee_models",
            dynamic=True,
            valid_type=orm.SinglefileData,
            non_db=True,
        )

        spec.input(
            "mace_settings_dict",
            valid_type=orm.Dict,
            help="Dictionary containing MACE settings.",
            serializer=orm.to_aiida_type,
        )
        spec.input(
            "num_threads",
            valid_type=orm.Int,
            help="Number of OpenMP threads to use for the evaluation.",
            serializer=orm.to_aiida_type,
        )
        spec.input(
            "configurations_to_evaluate",
            valid_type=orm.SinglefileData,
            help="Path to the configurations to evaluate in extxyz format.",
        )
        spec.output(
            "energy_result_dict",
            valid_type=orm.Dict,
            help="Dicvt of values for the energy prediction.",
        )
        spec.output(
            "forces_result_dict",
            valid_type=orm.Dict,
            help="Dict of array of values for the force prediction.",
        )
        spec.exit_code(420, "ERROR_OUT_OF_VRAM", "CUDA out of GPU memory.")
        spec.exit_code(421, "ERROR_OUTPUT_NOT_FOUND", "Missing output file.")

    def prepare_for_submission(self, folder):
        """Write the input files that are required for the code to run.

        :param folder: an `Folder` to temporarily write files on disk
        :return: `CalcInfo` instance
        """
        # Parsing mace settings dict
        params_list = []
        params_list.append("--configs=configurations_to_evaluate.xyz")

        # Adding cli parameters to list
        prepare_cli_args_mace(params_list, self.inputs.mace_settings_dict)
        params_list.append("--info_prefix=mdb_mace_eval_")

        # Adding n_threads to the list
        params_list.append(f"--num_threads={self.inputs.num_threads.value}")

        # Remove duplicate entries
        params_list = list(set(params_list))

        # Copying configuration to temporary folder
        # print('self.commitee_models: ', self.commitee_models)
        for model_str, model_singlefile in self.inputs.commitee_models.items():
            with model_singlefile.as_path() as model_path:
                folder.insert_path(
                    src=model_path,
                    dest_name=f"{model_str}.model",
                )

        curr_structure_file: orm.SinglefileData = self.inputs.configurations_to_evaluate

        with curr_structure_file.as_path() as path:
            folder.insert_path(
                src=path,
                dest_name="configurations_to_evaluate.xyz",
            )

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = params_list

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = []
        calcinfo.provenance_exclude_list = []
        calcinfo.remote_copy_list = []

        # Gathering files.
        calcinfo.retrieve_list = [
            "results.out",
        ]

        # They won't be added to the repository,
        # and instead kept into a temporary folder.
        calcinfo.retrieve_temporary_list = [
            "*_output.out",
        ]

        return calcinfo


def prepare_cli_args_mace(params_list: list, settings_dict: dict):
    """Prepare the command line arguments for the MACE calculation."""
    for key, val in settings_dict.items():
        if key == "train_file":
            val = Path(val).resolve().name

        if key == "dtype":
            key = "default_dtype"

        if isinstance(val, str):
            curr_key = f"--{key}={val}"
        elif isinstance(val, bool):
            if key == "multiheads_finetuning":
                curr_key = f"--{key}={val}"
            else:
                if val:
                    curr_key = f"--{key}"
        else:
            curr_key = f"--{key}={val}"

        params_list.append(curr_key)


class CheckMACECommiteeResultsCalculationParser(Parser):
    """
    Parser for processing the retrieved files from a MACE committee results job.

    Methods
    -------
    parse(**kwargs)
        Parses the retrieved files and extracts the predicted energies and forces
        for each committee model. Outputs are stored in AiiDA Dict objects.
    """

    def parse(self, **kwargs):
        """
        Parse the retrieved files of the calculation job.

        Returns
        -------
        421 - ERROR_OUTPUT_NOT_FOUND
            Returns a 421 error exit code if the required output files are not found.

        Outputs
        -------
        energy_result_dict : aiida.orm.Dict
            Dictionary containing the energies predicted by each committee model.
        forces_result_dict : aiida.orm.Dict
            Dictionary containing the forces predicted by each committee model.
        """
        # str that represents the absolute filepath to the temporary folder
        retrieved_temporary_folder: Path = Path(kwargs["retrieved_temporary_folder"])

        result_model_forces = {}
        result_model_energies = {}

        for child_file in retrieved_temporary_folder.iterdir():
            # Gathering results from the output for each committee model
            if "_output.out" in child_file.name:
                model_name = child_file.name.replace("_output.out", "")
                curr_model_forces_dict_list = []
                curr_model_energy_float_list = []

                result_structures = ase_read(child_file, format="extxyz", index=":")

                # Iterating over every structure to get predicted energies and forces
                for structure in result_structures:
                    forces_dict = np.vstack(structure.arrays["mdb_mace_eval_forces"])

                    curr_model_forces_dict_list.append(forces_dict)

                    curr_model_energy_float_list.append(
                        structure.info["mdb_mace_eval_energy"]
                    )

                result_model_forces[model_name] = curr_model_forces_dict_list
                result_model_energies[model_name] = curr_model_energy_float_list

        # Return failed code if result lists are not populated
        if len(result_model_energies) == 0 or len(result_model_forces) == 0:
            return self.exit_codes.ERROR_OUTPUT_NOT_FOUND

        # Return CalcJob outputs
        result_model_forces = orm.Dict(result_model_forces)
        result_model_energies = orm.Dict(result_model_energies)

        self.out("energy_result_dict", orm.Dict(result_model_energies))
        self.out("forces_result_dict", orm.Dict(result_model_forces))


class LAMMPSMACERawParser(Parser):
    """Base parser for LAMMPS output."""

    def parse(self, **kwargs):
        """Parse the output files stored in the `retrieved` output node."""
        retrieved = self.retrieved
        retrieved_filenames = retrieved.base.repository.list_object_names()
        filename_out = LammpsRawCalculation.FILENAME_OUTPUT

        if filename_out not in retrieved_filenames:
            return self.exit_codes.ERROR_OUTFILE_MISSING

        parsed_data = parse_outputfile(
            file_contents=retrieved.base.repository.get_object_content(filename_out)
        )
        if parsed_data is None:
            return self.exit_codes.ERROR_PARSING_OUTFILE

        if parsed_data["global"]:
            if (
                "max_neighbors_atom" not in parsed_data["global"]
                or "units_style" not in parsed_data["global"]
            ):
                return self.exit_codes.ERROR_PARSING_OUTFILE
        else:
            return self.exit_codes.ERROR_PARSING_OUTFILE

        if parsed_data["global"]["errors"]:
            # Output the data for checking what was parsed
            self.out("results", orm.Dict({"compute_variables": parsed_data["global"]}))
            for entry in parsed_data["global"]["errors"]:
                self.logger.error(f"LAMMPS emitted the error {entry}")
                return self.exit_codes.ERROR_PARSER_DETECTED_LAMMPS_RUN_ERROR.format(
                    error=entry
                )

        global_data = parsed_data["global"]
        results = {"compute_variables": global_data}

        if "total_wall_time" in global_data:
            try:
                parsed_time = time.strptime(global_data["total_wall_time"], "%H:%M:%S")
            except ValueError:
                pass
            else:
                total_wall_time_seconds = (
                    parsed_time.tm_hour * 3600
                    + parsed_time.tm_min * 60
                    + parsed_time.tm_sec
                )
                global_data["total_wall_time_seconds"] = total_wall_time_seconds

        self.out("results", orm.Dict(results))

        return None
