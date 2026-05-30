"""AiiDA CalcJob and Parser for the autoencoder training and latent space."""

import json
from pathlib import Path

import numpy as np
from aiida import orm
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.engine import CalcJob
from aiida.parsers.parser import Parser

from atlas.workflows.datatypes import image_types as atl_img


# entry point: mdb-autoencoder-train-parser
class TrainAutoencoderCalculationParser(Parser):
    """Parser for the retrieved files from an Autoencoder training calculation job."""

    def parse(self, **kwargs):
        """Parse the retrieved files of the calculation job."""
        # str that represents the absolute filepath to the temporary folder
        retrieved_temporary_folder: Path = Path(kwargs['retrieved_temporary_folder'])

        model_file = None

        for child_file in retrieved_temporary_folder.rglob('*'):
            # Create singlefile data for the model
            if '.pt' in child_file.name:
                model_file = orm.SinglefileData(file=child_file)
                continue

            # Get train statistics from the training output
            if 'atl_train_autoencoder' in child_file.name:
                training_log = orm.SinglefileData(file=child_file)

        # Return failed code
        if not training_log or not model_file:
            return self.exit_codes.ERROR_INVALID_OUTPUT

        # Return CalcJob outputs
        self.out('model_file', model_file)
        self.out('training_log', training_log)


# entry point: mdb-autoencoder-train
class TrainAutoencoderCalculation(CalcJob):
    """Implementation of a CalcJob to perform an Autoencoder training using
    a settings dir.

    Inputs
    ------

    settings_dict : orm.Dict
        Dictionary containing training settings.
    descriptors_file_path : orm.Str
        Path to the descriptors to evaluate in npy format.

    Outputs
    -------
    model_file : orm.SinglefileData
        Path of the trained Autoencoder model.
    training_log : orm.SinglefileData
        Log file containing the training process.

    Exit Codes
    ----------
    420 : ERROR_INVALID_OUTPUT
        Autoencoder Training calculation could not run.
    """

    @classmethod
    def define(cls, spec):
        """Define the input and output specifications for the CalcJob."""
        super().define(spec)
        spec.input(
            'settings_dict',
            valid_type=orm.Dict,
            help='Dictionary containing Autoencoder training settings.',
        )
        spec.input(
            'descriptors_file_path',
            valid_type=orm.Str,
            help=('Path to the descriptors to evaluate in npy format.'),
            # non_db=True,
            serializer=orm.to_aiida_type,
        )
        spec.output(
            'model_file',
            valid_type=orm.SinglefileData,
            help='Path of the trained Autoencoder model.',
        )
        spec.output(
            'training_log',
            valid_type=orm.SinglefileData,
            help='Log file containing information of the training process',
        )
        spec.exit_code(
            420,
            'ERROR_INVALID_OUTPUT',
            'Autoencoder training calculation could not run',
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
        prepare_cli_args_autoencoder(params_list, self.inputs.settings_dict)

        dest_name = self.inputs.settings_dict.get_dict().get(
            'dataset', 'all_descriptors.npz'
        )

        # Copying database to temporary folder
        final_db_path = Path(self.inputs.descriptors_file_path.value).resolve()
        folder.insert_path(
            src=final_db_path,
            dest_name=dest_name,
        )

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        # codeinfo.stdout_name = self.options.output_filename
        codeinfo.stdout_name = 'calc_train_stdout.out'
        codeinfo.cmdline_params = params_list

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = []
        calcinfo.provenance_exclude_list = [dest_name]
        calcinfo.remote_copy_list = []

        # Gathering files. They won't be added to the repository,
        # and instead kept into a temporary folder.
        # They can later be processed during the parse function
        # by accessing the temporary folder.
        calcinfo.retrieve_temporary_list = [
            self.metadata.options.output_filename,
            './*.model',
            './*.pth',
            './*.pt',
            './results/*',
            './*.log',
        ]

        return calcinfo


# mdb-get-latent-space-parser
class GetLatentSpaceAutoencoderCalculationParser(Parser):
    """Parser for the retrieved files from a MACE descriptors job."""

    def parse(self, **kwargs):
        """Parse the retrieved files of the calculation job."""
        # str that represents the absolute filepath to the temporary folder
        retrieved_temporary_folder: Path = Path(kwargs['retrieved_temporary_folder'])

        latent_space = None
        autoencoder_model_file = None
        descr_max_arr = None
        descr_min_arr = None

        for child_file in retrieved_temporary_folder.iterdir():
            # Get output files in the appropriate format
            match child_file.name:
                case 'latent_space.npy':
                    latent_space = orm.ArrayData(np.load(child_file))
                case 'autoencoder_model.pth':
                    autoencoder_model_file = orm.SinglefileData(file=child_file)
                case 'curr_it_db_max.npy':
                    descr_max_arr = orm.ArrayData(np.load(child_file))
                case 'curr_it_db_min.npy':
                    descr_min_arr = orm.ArrayData(np.load(child_file))
                case 'curr_it_db_descriptors.pkl':
                    descriptor_arr_file = orm.SinglefileData(file=child_file)

        if not all(
            (latent_space, autoencoder_model_file, descr_max_arr, descr_min_arr)
        ):
            return self.exit_codes.ERROR_INVALID_OUTPUT

        # Return CalcJob outputs\
        self.out('latent_space', latent_space)
        self.out('autoencoder_model_file', autoencoder_model_file)
        self.out('descriptors_max_array', descr_max_arr)
        self.out('descriptors_min_array', descr_min_arr)
        self.out('descriptors_file', descriptor_arr_file)


# mdb-get-latent-space
class GetLatentSpaceAutoencoderCalculation(CalcJob):
    """Calculation to train an autoencoder and use it to get the latent space
    for the descriptors of a structure database.
    """

    @classmethod
    def define(cls, spec):  # noqa: D102
        super().define(spec)
        spec.input(
            'settings_dict',
            valid_type=orm.Dict,
            help='Dictionary containing autoencoder training settings.',
            required=False,
            default=None,
        )
        spec.input(
            'model_file',
            valid_type=orm.SinglefileData,
            help='Trained MACE model.',
            non_db=True,
        )
        spec.input(
            'mace_train_file_path',
            help=(
                'Path to the file containing the structures to be used for training,'
                ' in the extxyz format.'
            ),
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'trained_autoencoder_model',
            valid_type=orm.SinglefileData,
            help='Trained Autoencoder model.',
            non_db=True,
            required=False,
            default=None,
        )
        spec.output(
            'autoencoder_model_file',
            help='Trained Autoencoder model.',
            valid_type=orm.SinglefileData,
        )
        spec.output(
            'descriptors_file',
            valid_type=orm.SinglefileData,
            help=(
                'This file will contain a dict of length n_struct,'
                ' that will have two keys inside inside, `descriptors` and'
                ' `latent_space`. The `descriptors` will contain model_size'
                ' lists of descriptor values. The `latent_space` will contain'
                ' the 2D latent space representation of the descriptors.'
            ),
            required=False,
        )
        spec.output(
            'autoencoder_model_file',
            help='Trained Autoencoder model.',
            valid_type=orm.SinglefileData,
        )
        spec.output(
            'latent_space',
            help='Array containing the latent space descriptors array.',
            valid_type=orm.ArrayData,
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
        spec.exit_code(
            420,
            'ERROR_INVALID_OUTPUT',
            'calculation could not generate/gather all necessary outputs',
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
                src=Path(final_db_path).resolve(),
                dest_name='current_db.xyz',
            )
        elif isinstance(self.inputs.mace_train_file_path, orm.SinglefileData):
            with self.inputs.mace_train_file_path.as_path() as model_path:
                folder.insert_path(
                    src=model_path,
                    dest_name='current_db.xyz',
                )

        # If trained autoencoder model is given, copy it to the temporary folder
        # It will be used to generate the latent space instead of training a new model
        if self.inputs.trained_autoencoder_model:
            with self.inputs.trained_autoencoder_model.as_path() as auto_model_path:
                folder.insert_path(
                    src=auto_model_path,
                    dest_name='autoencoder_model.pth',
                )

        # Copying model to temporary folder
        model_file = self.inputs.model_file

        with model_file.as_path() as model_path:
            folder.insert_path(
                src=model_path,
                dest_name='current_model_mace.model',
            )

        # Writing settings dict to a json file
        if self.inputs.settings_dict:
            with folder.open('settings_dict.json', 'w') as f:
                json.dump(self.inputs.settings_dict.get_dict(), f)

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
            './*.pt*',
            './*.pkl*',
        ]

        return calcinfo


# mdb-get-concave-hull-parser
class GetConcaveHullCalculationParser(Parser):
    """Parser for the retrieved files from a MACE descriptors job."""

    def parse(self, **kwargs):
        """Parse the retrieved files of the calculation job."""
        # str that represents the absolute filepath to the temporary folder
        retrieved_temporary_folder: Path = Path(kwargs['retrieved_temporary_folder'])

        concave_hull_array = None
        concave_hull_plot = None

        for child_file in retrieved_temporary_folder.iterdir():
            # Get output files in the appropriate format
            match child_file.name:
                case 'concave_hull.npy':
                    concave_hull_array = orm.ArrayData(np.load(child_file))
                case 'concave_hull.png':
                    concave_hull_plot = atl_img.ImagePNGData(filepath=child_file)

        if not all((child_file, concave_hull_plot)):
            return self.exit_codes.ERROR_INVALID_OUTPUT

        # Return CalcJob outputs\
        self.out('concave_hull_array', concave_hull_array)
        self.out('concave_hull_plot', concave_hull_plot)


# mdb-get-concave-hull
class GetConcaveHullCalculation(CalcJob):
    """Calculation to get the concave hull of the latent space
    of a set of descriptors for a structure database.
    """

    @classmethod
    def define(cls, spec):  # noqa: D102
        super().define(spec)
        spec.input(
            'latent_space',
            valid_type=orm.ArrayData,
            help='Array containing the latent space of a set of descriptors.',
        )
        spec.output(
            'concave_hull_array',
            help='Array containing the computed concave hull.',
            valid_type=orm.ArrayData,
        )
        spec.output(
            'concave_hull_plot',
            valid_type=atl_img.ImagePNGData,
            help='Chart showing the 2D representation of the concave hull.',
        )
        spec.exit_code(
            420,
            'ERROR_INVALID_OUTPUT',
            'calculation could not generate/gather all necessary outputs',
        )

    def prepare_for_submission(self, folder):
        """Write the input files that are required for the code to run.

        :param folder: a `Folder` to temporarily write files on disk
        :return: `CalcInfo` instance
        """
        # Writing latent space a file
        with folder.open('latent_space.npy', 'wb') as f:
            np.save(f, self.inputs.latent_space.get_array())

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
            './*.png*',
        ]

        return calcinfo


def prepare_cli_args_autoencoder(params_list: list, settings_dict: dict):
    """Prepare the command line arguments for the Autoencoder training."""
    for key, val in settings_dict.items():
        if key == 'train_file':
            val = Path(val).resolve().name

        if key == 'dtype':
            key = 'default_dtype'

        if isinstance(val, str):
            curr_key = f'--{key}={val}'
        elif isinstance(val, bool):
            if val:
                curr_key = f'--{key}'
            else:
                continue
        else:
            curr_key = f'--{key}={val}'

        params_list.append(curr_key)
