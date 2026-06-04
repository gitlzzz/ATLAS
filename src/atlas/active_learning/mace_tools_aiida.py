#!/usr/bin/env python
"""AiiDA plugin for MACE calculations."""

import json
import pickle
import shutil
import tempfile
import time
import tomllib
from pathlib import Path

import numpy as np
import yaml
from aiida import orm
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.folders import Folder
from aiida.engine import CalcJob
from aiida.parsers.parser import Parser
from aiida_lammps.calculations.raw import LammpsRawCalculation
from aiida_lammps.parsers.parse_raw import parse_outputfile
from ase.io import read as ase_read
from tomlkit import dumps as tomlkit_dumps

from atlas.active_learning import active_learning_utils as atl_al_ut
from atlas.workflows.datatypes import image_types as atl_img


# atl-process-md-seed-struct
class ProcessMDSeedStructCalculation(CalcJob):
    """
    Launch a calculation to process a structure in an AL Loop step.

    This CalcJob will process the output of a structure calculation
    by doing a MD simulation using the user provided settings, followed
    by checking all frames for extrapolation using several possible methods.

    Parameters
    ----------
    md_structure : orm.SinglefileData
        File containing the structure to be used for the MD, in the extxyz format.
    commitee_models : PortNamespace
        A namespace to hold an arbitrary number of committee MACE potentials.
    autoencoder_model : orm.SinglefileData, optional
        File containing the autoencoder model.
    m_rmse_e : orm.Float
        Validation RMSE of the best model for the energy, in meV / atom.
    m_rmse_f : orm.Float
        Validation RMSE of the best model for the forces, in meV / Å.
    concave_hull : orm.ArrayData, optional
        Array containing the concave hull to be used for the extrapolation check.
    desc_max_arr : orm.ArrayData
        Array containing the maximum values for the descriptors.
    desc_min_arr : orm.ArrayData
        Array containing the minimum values for the descriptors.
    settings_file_pth : orm.Str
        Path to the ATL settings file in the .toml format.

    Outputs
    -------
    extrapolating_structures : orm.SinglefileData
        File containing all structures that were found to be extrapolating.
        Uses the extxyz format.
    extrapolation_plot: atl_img.ImagePNGData
        File containing a visualization of the extrapolation check and latent
        space boundaries

    Exit Codes
    ----------
    420 : ERROR_INVALID_OUTPUT
        Structure could not be processed.
    """

    @classmethod
    def define(cls, spec):
        """Define the input and output specifications for the CalcJob."""
        super().define(spec)
        # Namespace that will hold an arbitrary number of committee MACE potentials
        spec.input_namespace(
            'commitee_models',
            dynamic=True,
            valid_type=orm.SinglefileData,
            non_db=True,
        )
        spec.input(
            'best_model_name',
            valid_type=orm.Str,
            help='Name of the best model.',
            required=True,
        )
        spec.input(
            'md_structure',
            valid_type=orm.SinglefileData,
            help=(
                'File containing the structure to be used for the MD,'
                'in the extxyz format.'
            ),
            required=True,
            # non_db=True,
        )
        spec.input(
            'autoencoder_model',
            valid_type=orm.SinglefileData,
            help='File containing the autoencoder model.',
            required=False,
            # non_db=True,
            default=None,
            serializer=orm.to_aiida_type,
        )

        spec.input(
            'm_rmse_e',
            valid_type=orm.Float,
            help='Validation RMSE of the best model for the energy, in meV / atom.',
        )

        spec.input(
            'm_rmse_f',
            valid_type=orm.Float,
            help='Validation RMSE of the best model for the forces, in meV / Å.',
            serializer=orm.to_aiida_type,
        )

        spec.input(
            'concave_hull',
            valid_type=orm.List,
            help=('List containing several concave hulls as lists of tuples.'),
            required=False,
            # non_db=True,
            default=None,
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'desc_max_arr',
            valid_type=orm.ArrayData,
            help=('Array containing the maximum values for the descriptors'),
            required=True,
            # non_db=True,
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'desc_min_arr',
            valid_type=orm.ArrayData,
            help=('Array containing the minimum values for the descriptors.'),
            required=True,
            # non_db=True,
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'settings_file_pth',
            valid_type=orm.Str,
            help='Path to the ATL settings file in the .toml format.',
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'curr_active_learning_step',
            valid_type=orm.Int,
            help='Current active learning iteration step.',
        )
        spec.output(
            'extrapolating_structures',
            valid_type=orm.SinglefileData,
            help=(
                'File containing all structures that were found to be extrapolating. '
                'Uses the the extxyz format.'
            ),
        )
        spec.output(
            'extrapolation_plot',
            valid_type=(atl_img.ImagePNGData, None),
            help=('File containing a figure showing the extrapolation results.'),
            required=False,
        )
        spec.exit_code(
            420,
            'ERROR_INVALID_OUTPUT',
            "Structure '{node_id}' could not be processed.",
        )

    def prepare_for_submission(self, folder):
        """Write the input files that are required for the code to run.

        :param folder: an `Folder` to temporarily write files on disk
        :return: `CalcInfo` instance
        """
        # Create E and F RMSE array
        rmse_arr = np.array([self.inputs.m_rmse_e.value, self.inputs.m_rmse_f.value])
        # Create a named temporary file using tmpfile
        with tempfile.NamedTemporaryFile(
            mode='w', delete=True, suffix='.npy', prefix='atl_process_md-'
        ) as f:
            np.save(f.name, rmse_arr)
            folder.insert_path(
                src=f.name,
                dest_name='rmse_arr.npy',
            )

            # Remove the file after insertion
            f.close()
            Path(f.name).unlink(missing_ok=True)

        # Copying structure to use for the MD
        md_structure: orm.SinglefileData = self.inputs.md_structure
        with md_structure.as_path() as md_struct_path:
            folder.insert_path(
                src=md_struct_path,
                dest_name='curr_structure.xyz',
            )

        # Loading settings to include current active learning step
        toml_settings = self.inputs.settings_file_pth.value
        with open(toml_settings, 'rb') as f:
            loaded_toml_settings = tomllib.load(f)

        # Adding current active learning step to monitor stage
        loaded_toml_settings['active_learning']['current_iteration'] = (
            self.inputs.curr_active_learning_step.value
        )

        # Saving modified settings to a temporary file and copying the
        # modified toml into the calculation folder
        with tempfile.NamedTemporaryFile(
            mode='wb', delete=True, suffix='.toml', prefix='atl_process_md-'
        ) as f_toml:
            f_toml.write(tomlkit_dumps(loaded_toml_settings).encode())
            toml_settings = f_toml.name

            # Copying settings file
            folder.insert_path(
                src=toml_settings,
                dest_name='settings.toml',
            )

            # Remove the file after insertion
            f_toml.close()
            Path(f_toml.name).unlink(missing_ok=True)

        # Copying concave hull for extrapolation
        if hasattr(self.inputs, 'concave_hull') and isinstance(
            self.inputs.concave_hull, (np.ndarray, orm.ArrayData)
        ):
            concave_hull = self.inputs.concave_hull.get_array()
            with tempfile.NamedTemporaryFile(
                mode='w',
                delete=True,
                suffix='.npy',
                prefix='atl_process_md-',
            ) as f:
                np.save(f.name, concave_hull)
                folder.insert_path(
                    src=f.name,
                    dest_name='concave_hull.npy',
                )

            # Remove the file after insertion
            f.close()
            Path(f.name).unlink(missing_ok=True)

        elif hasattr(self.inputs, 'concave_hull') and isinstance(
            self.inputs.concave_hull, (list, orm.List)
        ):
            with tempfile.NamedTemporaryFile(
                mode='wb',
                delete=True,
                suffix='.pkl',
                prefix='atl_process_md-',
            ) as f:
                pickle.dump(self.inputs.concave_hull.get_list(), f)

                # Ensure data is written to disk
                f.flush()
                folder.insert_path(
                    src=f.name,
                    dest_name='concave_hulls.pkl',
                )

        # Copying concave hull for extrapolation
        if hasattr(self.inputs, 'autoencoder_model') and isinstance(
            self.inputs.autoencoder_model,
            orm.SinglefileData,
        ):
            with self.inputs.autoencoder_model.as_path() as autoencoder_path:
                folder.insert_path(
                    src=autoencoder_path,
                    dest_name='autoencoder_model.pth',
                )

        # Copying descriptors max and min
        desc_max_arr: orm.ArrayData = self.inputs.desc_max_arr.get_array()
        with tempfile.NamedTemporaryFile(
            mode='w', delete=True, suffix='.npy', prefix='atl_process_md-'
        ) as f:
            np.save(f.name, desc_max_arr)
            folder.insert_path(
                src=f.name,
                dest_name='curr_it_db_max.npy',
            )

            # Remove the file after insertion
            f.close()
            Path(f.name).unlink(missing_ok=True)

        desc_min_arr = self.inputs.desc_min_arr.get_array()
        with tempfile.NamedTemporaryFile(
            mode='w', delete=True, suffix='.npy', prefix='atl_process_md-'
        ) as f:
            np.save(f.name, desc_min_arr)
            folder.insert_path(
                src=f.name,
                dest_name='curr_it_db_min.npy',
            )

            # Remove the file after insertion
            f.close()
            Path(f.name).unlink(missing_ok=True)

        # Copying configuration to temporary folder
        best_model_name = self.inputs.best_model_name.value.replace('-', '_')
        for model_str, model_singlefile in self.inputs.commitee_models.items():
            # If the best model is in the name, use it as the current model
            if model_str in best_model_name:
                with model_singlefile.as_path() as model_path:
                    folder.insert_path(
                        src=model_path,
                        dest_name='curr_model.model',
                    )
            else:
                with model_singlefile.as_path() as model_path:
                    folder.insert_path(
                        src=model_path,
                        dest_name=f'{model_str}.model',
                    )

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        # codeinfo.stdout_name = self.options.output_filename

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = []
        # calcinfo.provenance_exclude_list = [
        #     self.inputs.mace_settings_dict["train_file"]
        # ]
        calcinfo.remote_copy_list = []

        # Gathering files. They won't be added to the repository,
        # and instead kept into a temporary folder.
        # They can later be processed during the parse function
        # by accessing the temporary folder.
        calcinfo.retrieve_temporary_list = [
            # self.metadata.options.output_filename,
            './results/*.xyz',
            './results/*.png',
            './logs/*',
        ]

        return calcinfo


class ProcessMDSeedStructCalculationParser(Parser):
    """Parser for the retrieved files from an active learning MD sampling job."""

    def parse(self, **kwargs):
        """Parse the retrieved files of the calculation job."""
        # str that represents the absolute filepath to the temporary folder
        retrieved_temporary_folder: Path = Path(kwargs['retrieved_temporary_folder'])

        extrapolating_structures = None
        extrapolation_plot = None

        for child_file in retrieved_temporary_folder.rglob('*'):
            if 'extrapolating_frames.xyz' in child_file.name:
                extrapolating_structures = orm.SinglefileData(file=child_file)
            if '.png' in child_file.name:
                extrapolation_plot = atl_img.ImagePNGData(filepath=child_file)

        # Return failed code
        if not extrapolating_structures:
            return self.exit_codes.ERROR_INVALID_OUTPUT.format(
                node_id=self.node.pk,
            )

        # TODO: extrapolating_plot can be None, but the output will result in error,
        # as the required=False is not having an effect? This is a workaround for that
        if not extrapolation_plot:
            with tempfile.NamedTemporaryFile(
                mode='ab+',
                delete=True,
                suffix='.txt',
                prefix='atl_extrapolation_plot_placeholder-',
            ) as f:
                f.write(b'')
                extrapolation_plot = orm.SinglefileData(file=f)

        self.out('extrapolating_structures', extrapolating_structures)
        self.out('extrapolation_plot', extrapolation_plot)


class TrainMACEModelCalculationParser(Parser):
    """Parser for the retrieved files from a MACE training calculation job."""

    def parse(self, **kwargs):
        """Parse the retrieved files of the calculation job."""
        # str that represents the absolute filepath to the temporary folder
        retrieved_temporary_folder: Path = Path(kwargs['retrieved_temporary_folder'])

        model_file = None
        train_file = None
        rmse_e = None
        rmse_f = None

        for child_file in retrieved_temporary_folder.rglob('*'):
            # Create singlefile data for the model

            # If swa was used, get the swa model preferentially
            if 'swa.model' in child_file.name:
                model_file = orm.SinglefileData(file=child_file)
                continue

            # If swa was not used, get the non-compiled model, as it can be
            # used to get the descriptors.
            if '.model' in child_file.name and 'compiled' not in child_file.name:
                model_file = orm.SinglefileData(file=child_file)
                continue

            # Get train statistics from the training output
            if 'train.txt' in child_file.name:
                with open(child_file) as f:
                    for line in f:
                        line_dict = json.loads(line)
                        if 'rmse_e' in line_dict:
                            last_dict = line_dict

                rmse_e = float(last_dict['rmse_e_per_atom']) * 1000  # meV / atom
                rmse_f = float(last_dict['rmse_f']) * 1000  # meV / A
                continue

            if child_file.name.startswith('train_'):
                train_file = orm.SinglefileData(file=child_file)

        # If there are some missing variables, return failed code
        if None in (rmse_e, rmse_f, model_file, train_file):
            return self.exit_codes.ERROR_INVALID_OUTPUT.format(
                node_id=self.node.pk,
            )

        # Really weird data will make the training results NaN
        # Training will run 'fine', but model can't be used like that.
        # Return an error
        if np.isnan(rmse_e) or np.isnan(rmse_f):
            return self.exit_codes.ERROR_NAN_TRAINING_RESULTS.format(
                node_id=self.node.pk,
            )

        # Return CalcJob outputs
        self.out('model_file', model_file)
        self.out('train_file', train_file)
        self.out('m_rmse_e', orm.Float(rmse_e))
        self.out('m_rmse_f', orm.Float(rmse_f))


# mace-train
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
    use_container : orm.Bool
        Use code in container. Default is False. Will be set automatically by the
        code if the containerized mode is enabled.

    Outputs
    -------
    model_file : orm.SinglefileData
        Trained MACE model.
    train_file : orm.SinglefileData
        Log file containing training information.
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
            'mace_settings_dict',
            valid_type=orm.Dict,
            help='Dictionary containing MACE training settings.',
        )
        spec.input(
            'mace_train_file_path',
            valid_type=orm.Str,
            help=(
                'Path to the file containing the structures to be used for training, '
                'in the extxyz format.'
            ),
            # non_db=True,
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'test_file',
            valid_type=orm.SinglefileData,
            help=(
                'File containing the structures to be used for testing during training,'
                'in the extxyz format.'
            ),
            required=False,
            non_db=True,
            default=None,
        )

        spec.input(
            'multihead_finetuning',
            valid_type=orm.Bool,
            help=('Whether to use multihead finetuning.'),
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'use_container',
            valid_type=orm.Bool,
            help=('Whether to use code in container.'),
            serializer=orm.to_aiida_type,
            required=False,
            default=orm.Bool(False),
        )
        spec.input(
            'model_name',
            valid_type=orm.Str,
            help=('Name given to the model.'),
            serializer=orm.to_aiida_type,
        )

        spec.output(
            'model_file',
            valid_type=orm.SinglefileData,
            help='Trained MACE model.',
        )
        spec.output(
            'train_file',
            valid_type=orm.SinglefileData,
            help='Log file containing training information.',
        )
        spec.output(
            'm_rmse_e',
            valid_type=orm.Float,
            help='Validation RMSE for the energy, in meV / atom.',
        )
        spec.output(
            'm_rmse_f',
            valid_type=orm.Float,
            help='Validation RMSE for the forces, in meV / Å.',
        )
        spec.exit_code(
            420,
            'ERROR_INVALID_OUTPUT',
            'Training calculation ({node_id}) could not run',
        )
        spec.exit_code(
            421,
            'ERROR_NAN_TRAINING_RESULTS',
            (
                'Error table after training contains NaN values. '
                'This is likely due to bad training data. Check the generated '
                'structures in {node_id}.'
            ),
        )

    def prepare_for_submission(self, folder):
        """Write the input files that are required for the code to run.

        :param folder: an `Folder` to temporarily write files on disk
        :return: `CalcInfo` instance
        """
        # Parsing mace settings dict
        params_dict = self.inputs.mace_settings_dict.get_dict()

        # Adding random seed
        seed = np.random.randint(1, 100000000)
        params_dict['seed'] = seed

        # Save cpu model. This avoids a torch bug where gpu models cannot
        # be loaded in CPU-only machines
        params_dict['save_cpu'] = True

        # (for mace-torch == v0.3.7) Enabling multiheads finetuning for 'mp'
        foundation_model = self.inputs.mace_settings_dict.get('foundation_model')

        # Copying database to temporary folder
        final_db_path = self.inputs.mace_train_file_path.value
        folder.insert_path(
            src=final_db_path,
            dest_name=str(Path(self.inputs.mace_settings_dict['train_file']).name),
        )

        # Copying foundation model to temporary folder
        if foundation_model is not None:
            ft_path = Path(foundation_model).resolve()

            if ft_path.exists():
                folder.insert_path(
                    src=ft_path.resolve(),
                    dest_name=self.inputs.mace_settings_dict['foundation_model'],
                )

        # Create a yaml file using the settings dict and pyyaml
        with tempfile.NamedTemporaryFile(
            mode='w', delete=True, suffix='.yaml', prefix='atl_mace_train-'
        ) as f:
            yaml.dump(params_dict, f)
            folder.insert_path(
                src=f.name,
                dest_name='settings.yaml',
            )

            # Remove the file after insertion
            f.close()
            Path(f.name).unlink(missing_ok=True)

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = self.options.output_filename

        # Prepending specific paths for the ATLAS container.
        # This shouldn't affect non-container runs.
        config_prepend = '.'
        if hasattr(self.inputs.code, 'engine_command') and (
            'atl_data' in self.inputs.code.engine_command
        ):
            config_prepend = '/atl_data'

        codeinfo.cmdline_params = [f'--config={config_prepend}/settings.yaml']

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = []
        calcinfo.provenance_exclude_list = [
            self.inputs.mace_settings_dict['train_file']
        ]
        calcinfo.remote_copy_list = []

        # Gathering files. They won't be added to the repository,
        # and instead kept into a temporary folder.
        # They can later be processed during the parse function
        # by accessing the temporary folder.
        calcinfo.retrieve_temporary_list = [
            self.metadata.options.output_filename,
            './*.model',
            './results/*',
            './train_*',
        ]

        return calcinfo


# mace-descriptors-parser
class GetMACEDescriptorsCalculationParser(Parser):
    """Parser for the retrieved files from a MACE descriptors job."""

    def parse(self, **kwargs):
        """Parse the retrieved files of the calculation job."""
        # str that represents the absolute filepath to the temporary folder
        retrieved_temporary_folder: Path = Path(kwargs['retrieved_temporary_folder'])

        descriptor_arr_file = None
        descr_max_arr = None
        descr_min_arr = None

        for child_file in retrieved_temporary_folder.iterdir():
            # create singlefile data for the descriptors
            if 'curr_it_db_descriptors.pkl' in child_file.name:
                descriptor_arr_file = orm.SinglefileData(file=child_file)

            if 'curr_it_db_max.npy' in child_file.name:
                descr_max_arr = orm.ArrayData(np.load(child_file))

            if 'curr_it_db_min.npy' in child_file.name:
                descr_min_arr = orm.ArrayData(np.load(child_file))

        # Return failed code if output files not found
        if not all((descriptor_arr_file, descr_max_arr, descr_min_arr)):
            return self.exit_codes.ERROR_INVALID_OUTPUT

        # Return CalcJob outputs
        self.out('descriptors_file', descriptor_arr_file)
        self.out('descriptors_max_array', descr_max_arr)
        self.out('descriptors_min_array', descr_min_arr)


# mace-get-descriptors
class GetMACEDescriptorsCalculation(CalcJob):
    """Calculation to obtain descriptors for a structure database from MACE."""

    @classmethod
    def define(cls, spec):  # noqa: D102
        super().define(spec)
        spec.input(
            'model_file',
            valid_type=orm.SinglefileData,
            help='Trained MACE model.',
            non_db=True,
        )
        spec.input(
            'mace_train_file_path',
            help=(
                'Path to the file containing the structures to be used for training, '
                'in the extxyz format.'
            ),
            serializer=orm.to_aiida_type,
            non_db=True,
        )

        spec.output(
            'descriptors_file',
            valid_type=orm.SinglefileData,
            help='Path of the file containing the MACE descriptors array.',
        )
        spec.output(
            'descriptors_max_array',
            valid_type=orm.ArrayData,
            help='Array containing the maximum values for the MACE descriptors, '
            'shaped according to the model size.',
        )
        spec.output(
            'descriptors_min_array',
            valid_type=orm.ArrayData,
            help='Array containing the minimum values for the MACE descriptors, '
            'shaped according to the model size.',
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
                dest_name='current_db.xyz',
            )
        elif isinstance(self.inputs.mace_train_file_path, orm.SinglefileData):
            with self.inputs.mace_train_file_path.as_path() as model_path:
                folder.insert_path(
                    src=model_path,
                    dest_name='current_db.xyz',
                )

        # Copying model to temporary folder
        model_file = self.inputs.model_file

        with model_file.as_path() as model_path:
            folder.insert_path(
                src=model_path,
                dest_name='current_model_mace.model',
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
            './*.npy',
            './*.pkl',
        ]

        return calcinfo


# entry-point: mace-eval
class EvaluateMACEConfigsCalculation(CalcJob):
    """CalcJob to evaluate E and F of structures using a MACE model."""

    @classmethod
    def define(cls, spec):  # noqa: D102
        super().define(spec)
        spec.input(
            'mace_settings_dict',
            valid_type=orm.Dict,
            help='Dictionary containing MACE settings.',
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'model_file',
            valid_type=orm.SinglefileData,
            help='Path to the trained MACE model.',
        )
        spec.input(
            'use_container',
            valid_type=orm.Bool,
            help=('Whether to use code in container.'),
            serializer=orm.to_aiida_type,
            required=False,
            default=orm.Bool(False),
        )
        spec.input(
            'configuration_to_evaluate',
            valid_type=orm.SinglefileData,
            help='Path to the configurations to evaluate in extxyz format.',
        )
        spec.output(
            'configuration_result_file',
            valid_type=orm.SinglefileData,
            help='File containing all configurations evaluated using MACE in'
            ' the extxyz format.',
        )
        spec.output(
            'energy_result_list',
            valid_type=orm.List,
            help='List of values for the energy prediction.',
        )
        spec.output(
            'forces_result_list',
            valid_type=orm.List,
            help='List of array of values for the force prediction.',
        )
        spec.exit_code(
            420, 'ERROR_INVALID_OUTPUT', 'MACE labelling calculation could not run'
        )
        spec.exit_code(
            421,
            'ERROR_MISSING_ELEMENT',
            'Configuration evaluation with MACE model failed. ({node_id})'
            "The model wasn't trained on data containing the element {missing_element}."
            ' Please retrain the model with the missing element, or remove the element'
            ' from the structure database.',
        )

    def prepare_for_submission(self, folder):
        """Write the input files that are required for the code to run.

        :param folder: an `Folder` to temporarily write files on disk
        :return: `CalcInfo` instance
        """
        # Parsing mace settings dict
        params_list = []

        model_path_str = Path('current_mace_model.model')
        configs_path_str = Path('current_configuration.xyz')
        results_out_path = Path('results.out')

        # Prepending paths with /atl_data if using container
        if self.inputs.use_container:
            model_path_str = Path('/atl_data') / model_path_str
            configs_path_str = Path('/atl_data') / configs_path_str
            results_out_path = Path('/atl_data') / results_out_path

        params_list.append(f'--model={model_path_str}')
        params_list.append(f'--configs={configs_path_str}')
        params_list.append(f'--output={results_out_path}')

        # Adding cli parameters to list
        prepare_cli_args_mace(params_list, self.inputs.mace_settings_dict)

        params_list.append('--info_prefix=REF_')

        # Remove duplicate entries
        params_list = list(set(params_list))

        # Copying configuration to temporary folder
        with self.inputs.model_file.as_path() as model_path:
            folder.insert_path(
                src=model_path,
                dest_name='current_mace_model.model',
            )

        curr_structure_file: orm.SinglefileData = self.inputs.configuration_to_evaluate

        with curr_structure_file.as_path() as path:
            folder.insert_path(
                src=path,
                dest_name='current_configuration.xyz',
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
            'results.out',
        ]

        # They won't be added to the repository,
        # and instead kept into a temporary folder.
        calcinfo.retrieve_temporary_list = [
            'results.out',
        ]

        return calcinfo


class EvaluateMACEConfigsCalculationParser(Parser):
    """Parser for MACE E and F evaluation calculation jobs."""

    def parse(self, **kwargs):
        """Parse the retrieved files of the calculation job."""
        # str that represents the absolute filepath to the temporary folder
        retrieved_temporary_folder: Path = Path(kwargs['retrieved_temporary_folder'])

        schederr_str = self.node.get_scheduler_stderr()
        import re

        missing_element_re = r'ValueError: \d* is not in list'
        missing_element_re = re.findall(missing_element_re, schederr_str, re.MULTILINE)
        if len(missing_element_re) > 0:
            missing_element = missing_element_re[0].split(' ')[1]
            return self.exit_codes.ERROR_MISSING_ELEMENT.format(
                missing_element=missing_element,
                node_id=self.node.pk,
            )

        result_dict_list = []
        forces_dict_list = []
        result_structures = None
        result_dict = None

        for child_file in retrieved_temporary_folder.iterdir():
            # create singlefile data for the descriptors
            if child_file.name == 'results.out':
                results_path = child_file.absolute()
                result_structures = ase_read(child_file, format='extxyz', index=':')
                for curr_structure in result_structures:
                    result_dict = atl_al_ut.serialize_ase(curr_structure)

                    # Get values from calculator
                    # if curr_structure.calc:
                    try:
                        # calc_energies = curr_structure.calc.get_potential_energy()
                        calc_forces = curr_structure.calc.get_forces()
                    except Exception:
                        # calc_energies = result_dict['atl_mace_eval_energy']
                        calc_forces = result_dict['REF_forces']

                    result_dict_list.append(result_dict)
                    forces_dict = np.vstack(calc_forces)
                    forces_dict_list.append(forces_dict)

        energy_float_list = [
            ene_dict['info']['REF_energy'] for ene_dict in result_dict_list
        ]

        try:
            # Return failed code if output files not found
            if not result_structures or not result_dict:
                return self.exit_codes.ERROR_INVALID_OUTPUT
        except Exception:
            return self.exit_codes.ERROR_INVALID_OUTPUT

        # Return CalcJob outputs
        # self.out("configuration_result_list", orm.SinglefileData("result_dict_list"))
        self.out('configuration_result_file', orm.SinglefileData(results_path))
        self.out('energy_result_list', orm.List(energy_float_list))
        self.out('forces_result_list', orm.List(forces_dict_list))


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
            self.inputs['filenames'].get_dict() if 'filenames' in self.inputs else {}
        )
        provenance_exclude_list = []

        with folder.open(filename_input, 'w') as handle:
            handle.write(self.inputs.script.get_content())

        for key, node in self.inputs.get('files', {}).items():
            # The filename with which the file is written to the working directory
            # is defined by the `filenames` input namespace, falling back to the
            # filename of the `orm.SinglefileData` node if not defined.
            filename = filenames.get(key, node.filename)

            with folder.open(filename, 'wb') as target, node.open(mode='rb') as source:
                shutil.copyfileobj(source, target)

            provenance_exclude_list.append(filename)

        codeinfo = CodeInfo()
        codeinfo.cmdline_params = [
            '-in',
            filename_input,
            '-k',
            'on',
            'g',
            '1',
            '-sf',
            'kk',
        ]
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = self.inputs.metadata.options.output_filename

        calcinfo = CalcInfo()
        calcinfo.provenance_exclude_list = provenance_exclude_list
        calcinfo.retrieve_list = [filename_output]
        if 'settings' in self.inputs:
            calcinfo.retrieve_list += self.inputs.settings.get_dict().get(
                'additional_retrieve_list', []
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
            'commitee_models',
            dynamic=True,
            valid_type=orm.SinglefileData,
            non_db=True,
        )

        spec.input(
            'mace_settings_dict',
            valid_type=orm.Dict,
            help='Dictionary containing MACE settings.',
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'num_threads',
            valid_type=orm.Int,
            help='Number of OpenMP threads to use for the evaluation.',
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'configurations_to_evaluate',
            valid_type=orm.SinglefileData,
            help='Path to the configurations to evaluate in extxyz format.',
        )
        spec.output(
            'energy_result_dict',
            valid_type=orm.Dict,
            help='Dicvt of values for the energy prediction.',
        )
        spec.output(
            'forces_result_dict',
            valid_type=orm.Dict,
            help='Dict of array of values for the force prediction.',
        )
        spec.exit_code(420, 'ERROR_OUT_OF_VRAM', 'CUDA out of GPU memory.')
        spec.exit_code(421, 'ERROR_OUTPUT_NOT_FOUND', 'Missing output file.')

    def prepare_for_submission(self, folder):
        """Write the input files that are required for the code to run.

        :param folder: an `Folder` to temporarily write files on disk
        :return: `CalcInfo` instance
        """
        # Parsing mace settings dict
        params_list = []
        params_list.append('--configs=configurations_to_evaluate.xyz')

        # Adding cli parameters to list
        prepare_cli_args_mace(params_list, self.inputs.mace_settings_dict)
        params_list.append('--info_prefix=atl_mace_eval_')

        # Adding n_threads to the list
        params_list.append(f'--num_threads={self.inputs.num_threads.value}')

        # Remove duplicate entries
        params_list = list(set(params_list))

        # Copying configuration to temporary folder
        for model_str, model_singlefile in self.inputs.commitee_models.items():
            with model_singlefile.as_path() as model_path:
                folder.insert_path(
                    src=model_path,
                    dest_name=f'{model_str}.model',
                )

        curr_structure_file: orm.SinglefileData = self.inputs.configurations_to_evaluate

        with curr_structure_file.as_path() as path:
            folder.insert_path(
                src=path,
                dest_name='configurations_to_evaluate.xyz',
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
            'results.out',
        ]

        # They won't be added to the repository,
        # and instead kept into a temporary folder.
        calcinfo.retrieve_temporary_list = [
            '*_output.out',
        ]

        return calcinfo


def prepare_cli_args_mace(
    params_list: list, settings_dict: dict, use_container: bool = False
):
    """Prepare the command line arguments for the MACE calculation."""
    for key, val in settings_dict.items():
        if key == 'train_file':
            val = Path(val).resolve().name

            # Prepending the /atl_data path to any path in the settings dict
            # this is to ensure that the cwd data can be accessed by the container
            # in the /atl_data folder.
            if use_container:
                val = Path('/atl_data') / val
                params_list.append('--work_dir=/atl_data')

        if key == 'dtype':
            key = 'default_dtype'

        if isinstance(val, str):
            curr_key = f'--{key}={val}'
        elif isinstance(val, bool):
            if key == 'multiheads_finetuning':
                curr_key = f'--{key}={val}'
            else:
                if val:
                    curr_key = f'--{key}'
        else:
            curr_key = f'--{key}={val}'

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
        retrieved_temporary_folder: Path = Path(kwargs['retrieved_temporary_folder'])

        result_model_forces = {}
        result_model_energies = {}

        for child_file in retrieved_temporary_folder.iterdir():
            # Gathering results from the output for each committee model
            if '_output.out' in child_file.name:
                model_name = child_file.name.replace('_output.out', '')
                curr_model_forces_dict_list = []
                curr_model_energy_float_list = []

                result_structures = ase_read(child_file, format='extxyz', index=':')

                # Iterating over every structure to get predicted energies and forces
                for structure in result_structures:
                    forces_dict = np.vstack(structure.arrays['atl_mace_eval_forces'])

                    curr_model_forces_dict_list.append(forces_dict)

                    curr_model_energy_float_list.append(
                        structure.info['atl_mace_eval_energy']
                    )

                result_model_forces[model_name] = curr_model_forces_dict_list
                result_model_energies[model_name] = curr_model_energy_float_list

        # Return failed code if result lists are not populated
        if len(result_model_energies) == 0 or len(result_model_forces) == 0:
            return self.exit_codes.ERROR_OUTPUT_NOT_FOUND

        # Return CalcJob outputs
        result_model_forces = orm.Dict(result_model_forces)
        result_model_energies = orm.Dict(result_model_energies)

        self.out('energy_result_dict', orm.Dict(result_model_energies))
        self.out('forces_result_dict', orm.Dict(result_model_forces))


# atl-descriptors-combined-parser
class GetDescriptorsCombinedParser(Parser):
    """
    Parser for a descriptor and extrapolation gathering job.

    Methods
    -------
    parse(**kwargs)
        Parses the temporarily retrieved files.
        Outputs are stored in AiiDA SinglefileData objects.
    """

    def parse(self, **kwargs):
        """
        Parse the retrieved files of the calculation job.

        Returns
        -------
        descriptor_max : aiida.orm.SinglefileData
            File containing the maximum values for the descriptors.
        descriptor_min : aiida.orm.SinglefileData
            File containing the minimum values for the descriptors.
        concave_hull : aiida.orm.SinglefileData, optional
            File containing the concave hull of the latent space as an array.
        latent_space : aiida.orm.SinglefileData, optional
            File containing the latent space represented as an array.
        """
        # str that represents the absolute filepath to the temporary folder
        retrieved_temporary_folder: Path = Path(kwargs['retrieved_temporary_folder'])

        descriptor_max = None
        descriptor_min = None
        concave_hull = None
        latent_space = None
        extrapolation_plot = None
        autoencoder_model = None
        concave_hull_data = None

        # Gathering results from the temporary folder
        # for child_file in retrieved_temporary_folder.iterdir():
        for child_file in retrieved_temporary_folder.rglob('*'):
            match child_file.name:
                # case "curr_it_db_descriptors.pkl":
                # descriptor_arr_file = orm.SinglefileData(file=child_file.absolute())
                case 'curr_it_db_max.npy':
                    descriptor_max = orm.ArrayData(
                        arrays=np.load(child_file.absolute())
                    )
                case 'curr_it_db_min.npy':
                    descriptor_min = orm.ArrayData(
                        arrays=np.load(child_file.absolute())
                    )
                case 'concave_hull.npy':
                    concave_hull = orm.ArrayData(arrays=np.load(child_file.absolute()))
                case 'concave_hulls_data.pkl':
                    with open(child_file.absolute(), 'rb') as f:
                        concave_hull_data = orm.SinglefileData(
                            file=child_file.absolute()
                        )
                case 'concave_hulls.pkl':
                    with open(child_file.absolute(), 'rb') as f:
                        concave_hull = orm.List(pickle.load(f))
                case 'latent_space.npy':
                    latent_space = orm.ArrayData(arrays=np.load(child_file.absolute()))
                case 'concave_hull.png':
                    extrapolation_plot = atl_img.ImagePNGData(
                        filepath=child_file.absolute()
                    )
                case 'autoencoder_model.pth':
                    autoencoder_model = orm.SinglefileData(file=child_file.absolute())

        # Return failed code if the mandatory outputs are missing
        if not all((descriptor_max, descriptor_min)):
            return self.exit_codes.ERROR_OUTPUT_NOT_FOUND

        # Return CalcJob outputs
        self.out('descriptor_max', descriptor_max)
        self.out('descriptor_min', descriptor_min)

        if latent_space:
            self.out('latent_space', latent_space)
        if concave_hull:
            self.out('concave_hull', concave_hull)
        if concave_hull_data:
            self.out('detailed_concave_hull', concave_hull_data)
        if extrapolation_plot:
            self.out('extrapolation_plot', extrapolation_plot)
        if autoencoder_model:
            self.out('autoencoder_model', autoencoder_model)


# atl-eval-test-parser
class EvalTestDatabaseCalculationParser(Parser):
    """
    Parser for a descriptor and extrapolation gathering job.

    Methods
    -------
    parse(**kwargs)
        Parses the temporarily retrieved files.
        Outputs are stored in AiiDA SinglefileData objects.
    """

    def parse(self, **kwargs):
        """
        Parse the retrieved files of the calculation job.

        Returns
        -------
        descriptor_max : aiida.orm.SinglefileData
            File containing the maximum values for the descriptors.
        descriptor_min : aiida.orm.SinglefileData
            File containing the minimum values for the descriptors.
        concave_hull : aiida.orm.SinglefileData, optional
            File containing the concave hull of the latent space as an array.
        latent_space : aiida.orm.SinglefileData, optional
            File containing the latent space represented as an array.
        """
        # str that represents the absolute filepath to the temporary folder
        retrieved_temporary_folder: Path = Path(kwargs['retrieved_temporary_folder'])

        rmse_e = None
        rmse_f = None
        mae_e = None
        mae_f = None
        eval_plot = None

        # Gathering results from the temporary folder
        # for child_file in retrieved_temporary_folder.iterdir():
        for child_file in retrieved_temporary_folder.rglob('*'):
            match child_file.name:
                # case "curr_it_db_descriptors.pkl":
                # descriptor_arr_file = orm.SinglefileData(file=child_file.absolute())
                case 'test_db_eval_results_updated.json':
                    with open(child_file.absolute()) as f:
                        results_dict = json.load(f)
                        current_iter = results_dict['current_iteration']
                        current_results = results_dict[f'step_{current_iter}']

                        # Get error values
                        rmse_e = orm.Float(current_results['rmse_e'])
                        rmse_f = orm.Float(current_results['rmse_f'])
                        mae_e = orm.Float(current_results['mae_e'])
                        mae_f = orm.Float(current_results['mae_f'])

                        # Save float outputs
                        self.out('rmse_e', rmse_e)
                        self.out('rmse_f', rmse_f)
                        self.out('mae_e', mae_e)
                        self.out('mae_f', mae_f)

                        # Save dict as output
                        self.out(
                            'test_db_eval_results',
                            orm.SinglefileData(file=child_file.absolute()),
                        )

                case 'test_db_eval_plots.png':
                    eval_plot = atl_img.ImagePNGData(filepath=child_file.absolute())
                    self.out('eval_plot', eval_plot)

        # Return failed code if the mandatory outputs are missing
        if not all((rmse_e, rmse_f, mae_e, mae_f)):
            return self.exit_codes.ERROR_OUTPUT_NOT_FOUND


# entry-point: atl-eval-test
class EvalTestDatabaseCalculation(CalcJob):
    """CalcJob to evaluate the test database using the sampler model (M0).

    This calculation predicts all energies and forces for the test database,
    computing the MAE and RMSE values for both properties at each step of the
    active learning iteration, saving the results in the workchain and preparing
    a report figure showing the evolution of the error over time.

    Parameters
    ----------
    spec : aiida.engine.processes.ports.PortNamespace
        The process specification to define the inputs, outputs, and exit codes.

    Inputs
    ------
    sampler_model : aiida.orm.SinglefileData
        File containing the MACE model to use for evaluation.
    current_iteration : aiida.orm.Int
        Current iteration number.
    settings_file_path : aiida.orm.Str
        Path to the ATL settings file in the toml format.
    test_database : aiida.orm.SinglefileData
        File containing the structures for testing the model.
    test_db_eval_results : aiida.orm.Dict
        Dictionary containing the evaluation results up until now.


    Outputs
    -------
    rmse_e : aiida.orm.Float
        Root mean square error for energy predictions.
    rmse_f : aiida.orm.Float
        Root mean square error for force predictions.
    mae_e : aiida.orm.Float
        Mean absolute error for energy predictions.
    mae_f : aiida.orm.Float
        Mean absolute error for force predictions.
    eval_plot : aiida.orm.SinglefileData
        File containing the evaluation plot.
    test_db_eval_results : aiida.orm.SinglefileData
        Dictionary containing the updated evaluation results.


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
        spec.input(
            'sampler_model',
            valid_type=orm.SinglefileData,
            non_db=True,
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'current_iteration',
            valid_type=orm.Int,
            help='Current iteration number.',
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'settings_file_path',
            valid_type=orm.Str,
            help='Path to the ATL settings file in the toml format.',
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'test_database',
            valid_type=orm.SinglefileData,
            non_db=True,
            help='File containing the structures for testing the model.',
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'test_db_eval_results',
            help='Dictionary containing the evaluation results up until now.',
            valid_type=orm.Dict,
            serializer=orm.to_aiida_type,
        )
        spec.output(
            'rmse_e',
            valid_type=orm.Float,
            help='Root mean square error for energy predictions.',
            required=False,
        )
        spec.output(
            'rmse_f',
            valid_type=orm.Float,
            help='Root mean square error for force predictions.',
            required=False,
        )
        spec.output(
            'mae_e',
            valid_type=orm.Float,
            help='Mean absolute error for energy predictions.',
            required=False,
        )
        spec.output(
            'mae_f',
            valid_type=orm.Float,
            help='Mean absolute error for force predictions.',
        )
        spec.output(
            'eval_plot',
            valid_type=atl_img.ImagePNGData,
            help='File containing the evaluation plot.',
            required=False,
        )
        spec.output(
            'test_db_eval_results',
            help='Dictionary containing the updated evaluation results.',
            valid_type=orm.SinglefileData,
        )

        spec.exit_code(420, 'ERROR_OUT_OF_VRAM', 'CUDA out of GPU memory.')
        spec.exit_code(421, 'ERROR_OUTPUT_NOT_FOUND', 'Missing output file.')

    def prepare_for_submission(self, folder):
        """Write the input files that are required for the code to run.

        :param folder: an `Folder` to temporarily write files on disk
        :return: `CalcInfo` instance
        """
        # Copying configuration to temporary folder
        with self.inputs.sampler_model.as_path() as model_path:
            folder.insert_path(
                src=model_path,
                dest_name='curr_iter_best.model',
            )

        # Copying settings file
        toml_settings = self.inputs.settings_file_path.value
        folder.insert_path(
            src=toml_settings,
            dest_name='settings.toml',
        )

        # Copying test database file
        with self.inputs.test_database.as_path() as test_db_path:
            folder.insert_path(
                src=test_db_path,
                dest_name='test_db.xyz',
            )

        test_db_eval_results = self.inputs.test_db_eval_results.get_dict()

        current_iteration = self.inputs.current_iteration.value
        test_db_eval_results['current_iteration'] = current_iteration

        with tempfile.NamedTemporaryFile(
            mode='w', delete=True, suffix='.json', prefix='atl_mace_eval-test-'
        ) as tmp_file:
            # Write the updated results dict to a temporary json file
            with open(tmp_file.name, 'w') as f:
                json_str = json.dumps(obj=test_db_eval_results)
                f.write(json_str)

            # Insert the temporary file into the folder
            folder.insert_path(
                src=tmp_file.name,
                dest_name='test_db_eval_results.json',
            )

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = []
        calcinfo.provenance_exclude_list = []
        calcinfo.remote_copy_list = []

        # Gathering files.
        calcinfo.retrieve_list = [
            # "results.out",
            # "./results/curr_it_db*",
            # "./results/*.png",
            # "./results/*.npy",
            './logs/*',
        ]

        # They won't be added to the repository,
        # and instead kept into a temporary folder.
        calcinfo.retrieve_temporary_list = [
            '*.png',
            '*.json',
        ]

        return calcinfo


# entry-point: atl-descriptors-combined
class GetDescriptorsCombinedCalculation(CalcJob):
    """CalcJob to gather the descriptors for the training database of an AL Loop.

    This calculation job computes the descriptors for all the configurations in
    the training database. Additionally, further extrapolation metrics are computed
    depending on the the selected extrapolation type.
    With min-max extrapolation enabled, the the minimum and maximum range for all
    descriptors in the training database is computed.
    With advanced extrapolation, the ranges plus the concave hull of the latent space
    for all configurations in the training database, are provided, along with a plot
    showing the configuration distribution in the latent space and the concave hull.

    Parameters
    ----------
    spec : aiida.engine.processes.ports.PortNamespace
        The process specification to define the inputs, outputs, and exit codes.

    Inputs
    ------
    commitee_models : PortNamespace
        A namespace to hold an arbitrary number of committee MACE potentials.
    settings_file_path : orm.Str
        Path to the ATL settings file in the .toml format.
    training_database_path : orm.Str
        Path to the configurations to evaluate, provided in the extxyz format.
    autoencoder_model : orm.SinglefileData, optional
        File containing the autoencoder model.
        If not provided, a new autoencoder is trained is computed.
    latent_space : orm.SinglefileData, optional
        File containing the latent space represented as an array.
        If not provided, the latent space is computed.

    Outputs
    -------
    descriptor_max : orm.ArrayData
        File containing the maximum values for the descriptors.
    descriptor_min : orm.ArrayData
        File containing the minimum values for the descriptors.
    latent_space : orm.ArrayData, optional
        File containing the latent space as an array.
    concave_hull : orm.ArrayData, optional
        File containing the concave hull of the latent space as an array.
    extrapolation_plot : atl_img.ImagePNGData, optional
        Figure showing the extrapolation for the current database.
    autoencoder_model : orm.SinglefileData, optional
        File containing the autoencoder model.

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
        spec.input(
            'best_model',
            valid_type=orm.SinglefileData,
            non_db=True,
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'settings_file_path',
            valid_type=orm.Str,
            help='Path to the ATL settings file in the toml format.',
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'training_database_path',
            valid_type=orm.Str,
            help='Path with the configurations to evaluate in extxyz format.',
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'latent_space',
            valid_type=orm.ArrayData,
            help='File containing the latent space as an array.',
            serializer=orm.to_aiida_type,
            required=False,
            default=None,
            non_db=True,
        )
        spec.input(
            'autoencoder_model',
            valid_type=orm.ArrayData,
            help='File containing the autoencoder model.',
            serializer=orm.to_aiida_type,
            required=False,
            default=None,
            non_db=True,
        )
        spec.output(
            'concave_hull',
            valid_type=orm.List,
            help='Array containing the concave hull of the latent space.',
            required=False,
        )
        spec.output(
            'detailed_concave_hull',
            valid_type=orm.SinglefileData,
            help='Pickled list of dicts containing quadtree and latents space info.',
            required=False,
        )
        spec.output(
            'latent_space',
            valid_type=orm.ArrayData,
            help='Array containing the latent space.',
            required=False,
        )
        spec.output(
            'extrapolation_plot',
            valid_type=atl_img.ImagePNGData,
            help='Figure showing the extrapolation for the current database.',
            required=False,
        )
        spec.output(
            'descriptor_max',
            valid_type=orm.ArrayData,
            help='File containing the maximum values for the descriptors.',
        )
        spec.output(
            'descriptor_min',
            valid_type=orm.ArrayData,
            help='File containing the minimum values for the descriptors.',
        )
        spec.output(
            'autoencoder_model',
            valid_type=orm.SinglefileData,
            help='File containing the autoencoder model.',
            required=False,
        )

        spec.exit_code(420, 'ERROR_OUT_OF_VRAM', 'CUDA out of GPU memory.')
        spec.exit_code(421, 'ERROR_OUTPUT_NOT_FOUND', 'Missing output file.')

    def prepare_for_submission(self, folder):
        """Write the input files that are required for the code to run.

        :param folder: an `Folder` to temporarily write files on disk
        :return: `CalcInfo` instance
        """
        # Copying configuration to temporary folder
        with self.inputs.best_model.as_path() as model_path:
            folder.insert_path(
                src=model_path,
                dest_name='curr_iter_best.model',
            )

        # Copying settings file
        toml_settings = self.inputs.settings_file_path.value
        folder.insert_path(
            src=toml_settings,
            dest_name='settings.toml',
        )

        # Copying database file
        train_db_path = self.inputs.training_database_path.value
        train_db_path = str(Path(train_db_path).resolve())
        folder.insert_path(
            src=train_db_path,
            dest_name='training_db.xyz',
        )

        # Copying configuration to temporary folder
        if self.inputs.latent_space:
            with self.inputs.latent_space.as_path() as latent_space_path:
                folder.insert_path(
                    src=latent_space_path,
                    dest_name='latent_space.npy',
                )
        if self.inputs.autoencoder_model:
            with self.inputs.autoencoder_model.as_path() as autoencoder_model_path:
                folder.insert_path(
                    src=autoencoder_model_path,
                    dest_name='autoencoder_model.pth',
                )

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = []
        calcinfo.provenance_exclude_list = []
        calcinfo.remote_copy_list = []

        # Gathering files.
        calcinfo.retrieve_list = [
            './logs/*',
        ]

        # They won't be added to the repository,
        # and instead kept into a temporary folder.
        calcinfo.retrieve_temporary_list = [
            # '*_output.out',
            # 'results/*.pkl',
            # 'results/curr_it_db*',
            # 'results/*.png',
            # 'results/*.npy',
            # 'results/concave_hull.npy',
            # 'results/latent_space.npy',
            # 'results/*.pth',
            'results/*',
            # '*.pth',
        ]

        return calcinfo


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

        if parsed_data['global']:
            if (
                'max_neighbors_atom' not in parsed_data['global']
                or 'units_style' not in parsed_data['global']
            ):
                return self.exit_codes.ERROR_PARSING_OUTFILE
        else:
            return self.exit_codes.ERROR_PARSING_OUTFILE

        if parsed_data['global']['errors']:
            # Output the data for checking what was parsed
            self.out('results', orm.Dict({'compute_variables': parsed_data['global']}))
            for entry in parsed_data['global']['errors']:
                self.logger.error(f'LAMMPS emitted the error {entry}')
                return self.exit_codes.ERROR_PARSER_DETECTED_LAMMPS_RUN_ERROR.format(
                    error=entry
                )

        global_data = parsed_data['global']
        results = {'compute_variables': global_data}

        if 'total_wall_time' in global_data:
            try:
                parsed_time = time.strptime(global_data['total_wall_time'], '%H:%M:%S')
            except ValueError:
                pass
            else:
                total_wall_time_seconds = (
                    parsed_time.tm_hour * 3600
                    + parsed_time.tm_min * 60
                    + parsed_time.tm_sec
                )
                global_data['total_wall_time_seconds'] = total_wall_time_seconds

        self.out('results', orm.Dict(results))

        return None
