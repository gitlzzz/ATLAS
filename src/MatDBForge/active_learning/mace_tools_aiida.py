import json
import shutil
from pathlib import Path

import numpy as np
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.folders import Folder
from aiida.engine import CalcJob
from aiida.orm import (
    ArrayData,
    Dict,
    Float,
    List,
    SinglefileData,
    Str,
    to_aiida_type,
)
from aiida.parsers.parser import Parser
from aiida_lammps.calculations.raw import LammpsRawCalculation
from ase.io import read as ase_read
from MatDBForge.active_learning import active_learning_utils as mdb_al_ut


class TrainMACEModelCalculationParser(Parser):
    def parse(self, **kwargs):
        """Parse the retrieved files of the calculation job."""
        # str that represents the absolute filepath to the temporary folder
        retrieved_temporary_folder: Path = Path(kwargs["retrieved_temporary_folder"])

        model_file = None
        rmse_e = None
        rmse_f = None

        for child_file in retrieved_temporary_folder.iterdir():
            # create singlefile data for the model
            if "swa.model" in child_file.name:
                model_file = SinglefileData(file=child_file)

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
        spec.exit_code(
            420, "ERROR_INVALID_OUTPUT", "training calculation could not run"
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


class GetMACEDescriptorsCalculationParser(Parser):
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
                descriptor_arr_file = SinglefileData(file=child_file)

            if "curr_it_db_max.npy" in child_file.name:
                descr_max_arr = ArrayData(np.load(child_file))

            if "curr_it_db_min.npy" in child_file.name:
                descr_min_arr = ArrayData(np.load(child_file))

        # Return failed code if output files not found
        if not descriptor_arr_file or not descr_max_arr or not descr_min_arr:
            return self.exit_codes.ERROR_INVALID_OUTPUT

        # Return CalcJob outputs
        self.out("descriptors_file", descriptor_arr_file)
        self.out("descriptors_max_array", descr_max_arr)
        self.out("descriptors_min_array", descr_min_arr)


class GetMACEDescriptorsCalculation(CalcJob):
    """Implementation of CalcJob to perform a MACE training using a settings dir."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "model_file",
            valid_type=SinglefileData,
            help="Path of the trained MACE model.",
        )
        spec.input(
            "mace_train_file_path",
            help=(
                "Path to the file containing the structures to be used for training, "
                "in the extxyz format."
            ),
            serializer=to_aiida_type,
        )

        spec.output(
            "descriptors_file",
            valid_type=SinglefileData,
            help="Path of the file containing the MACE descriptors array.",
        )
        spec.output(
            "descriptors_max_array",
            valid_type=ArrayData,
            help="Array containing the maximum values for the MACE descriptors, "
            "shaped according to the model size.",
        )
        spec.output(
            "descriptors_min_array",
            valid_type=ArrayData,
            help="Array containing the minimum values for the MACE descriptors, "
            "shaped according to the model size.",
        )

    def prepare_for_submission(self, folder):
        """Write the input files that are required for the code to run.

        :param folder: an `~aiida.common.folders.Folder` to temporarily write files on disk
        :return: `~aiida.common.datastructures.CalcInfo` instance
        """
        # Copying database to temporary folder
        if isinstance(self.inputs.mace_train_file_path, Str):
            final_db_path = self.inputs.mace_train_file_path.value
            folder.insert_path(
                src=Path(final_db_path),
                dest_name="current_db.xyz",
            )
        elif isinstance(self.inputs.mace_train_file_path, SinglefileData):
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
            "*.npy",
            "*.pkl",
        ]

        return calcinfo


class EvaluateMACEConfigsCalculation(CalcJob):
    """CalcJob to evaluate E and F of structures using a MACE model."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "mace_settings_dict",
            valid_type=Dict,
            help="Dictionary containing MACE settings.",
            serializer=to_aiida_type,
        )
        spec.input(
            "model_file",
            valid_type=SinglefileData,
            help="Path of the trained MACE model.",
        )
        spec.input(
            "configuration_to_evaluate",
            valid_type=SinglefileData,
            help="Path of the trained MACE model.",
        )
        spec.output(
            "configuration_result_list",
            valid_type=List,
            help="List of dicts representation of the predicted configuration using MACE.",
        )
        spec.output(
            "energy_result_list",
            valid_type=List,
            help="List of values for the energy prediction.",
        )
        spec.output(
            "forces_result_list",
            valid_type=List,
            help="List of array of values for the force prediction.",
        )
        spec.exit_code(
            420, "ERROR_INVALID_OUTPUT", "training calculation could not run"
        )

    def prepare_for_submission(self, folder):
        """Write the input files that are required for the code to run.

        :param folder: an `~aiida.common.folders.Folder` to temporarily write files on disk
        :return: `~aiida.common.datastructures.CalcInfo` instance
        """
        # Parsing mace settings dict
        params_list = []
        params_list.append("--model=current_mace_model.model")
        params_list.append("--configs=current_configuration.xyz")
        params_list.append("--output=results.out")

        for key, val in self.inputs.mace_settings_dict.items():
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

        params_list.append("--info_prefix=mdb_mace_eval_")

        # Remove duplicate entries
        params_list = list(set(params_list))

        # Copying configuration to temporary folder
        with self.inputs.model_file.as_path() as model_path:
            folder.insert_path(
                src=model_path,
                dest_name="current_mace_model.model",
            )

        curr_structure_file: SinglefileData = self.inputs.configuration_to_evaluate

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
        self.out("configuration_result_list", List(result_dict_list))
        self.out("energy_result_list", List(energy_float_list))
        self.out("forces_result_list", List(forces_dict_list))


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
            # The filename with which the file is written to the working directory is defined by the ``filenames`` input
            # namespace, falling back to the filename of the ``SinglefileData`` node if not defined.
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
