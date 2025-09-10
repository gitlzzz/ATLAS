"""AiiDA workchain for NNP active learning loops using MD."""

import io
import logging
import os
import pickle
import shutil
import time
import uuid
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from aiida import orm
from aiida.engine import (
    BaseRestartWorkChain,
    WorkChain,
    append_,
    if_,
    while_,
)
from aiida.plugins import CalculationFactory
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.pretty import Pretty

from MatDBForge import MDB_ROOT_DIR
from MatDBForge.active_learning import active_learning_utils as mdb_al_ut
from MatDBForge.active_learning import conversion as mdb_conv
from MatDBForge.core.code_utils import LevelNameFilter, get_mdb_version_info


class SimpleActiveLearningWorkChain(WorkChain):
    """
    WorkChain to run an active learning loop for a MACE potential using MD
    simulations to generate training data.
    """

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)

        spec.input('al_start_mode', valid_type=orm.Str, serializer=orm.to_aiida_type)
        spec.input('init_db_path', valid_type=orm.Str, serializer=orm.to_aiida_type)
        spec.input('toml_file', valid_type=orm.Str, serializer=orm.to_aiida_type)
        spec.input('final_db_name', valid_type=orm.Str, serializer=orm.to_aiida_type)
        spec.input('run_name', valid_type=orm.Str, serializer=orm.to_aiida_type)
        spec.input(
            'load_init_models',
            valid_type=(orm.List, type(None)),
            serializer=orm.to_aiida_type,
            default=None,
        )
        spec.input('results_dir', valid_type=orm.Str, serializer=orm.to_aiida_type)
        spec.input(
            'al_loop_iteration', valid_type=orm.Int, serializer=orm.to_aiida_type
        )
        spec.input('container_settings', valid_type=orm.Dict)
        spec.input(
            'seed_min_num_structs',
            valid_type=orm.Int,
            serializer=orm.to_aiida_type,
            required=False,
        )
        spec.input('seed_size_frac', valid_type=orm.Float, serializer=orm.to_aiida_type)
        spec.input(
            'seed_max_num_structs', valid_type=orm.Int, serializer=orm.to_aiida_type
        )
        spec.input(
            'seed_select_settings', valid_type=orm.Dict, serializer=orm.to_aiida_type
        )
        spec.input(
            'delete_seed_structs', valid_type=orm.Bool, serializer=orm.to_aiida_type
        )
        spec.input(
            'committee_num_models', valid_type=orm.Int, serializer=orm.to_aiida_type
        )
        spec.input(
            'model_acc_multiplier', valid_type=orm.Float, serializer=orm.to_aiida_type
        )
        spec.input(
            'descriptor_settings', valid_type=orm.Dict, serializer=orm.to_aiida_type
        )
        spec.input(
            'md_temperature_list_K', valid_type=orm.List, serializer=orm.to_aiida_type
        )
        spec.input('md_num_steps', valid_type=orm.Int, serializer=orm.to_aiida_type)
        spec.input(
            'md_max_temp_multiplier', valid_type=orm.Float, serializer=orm.to_aiida_type
        )
        spec.input(
            'md_filters',
            valid_type=(orm.Dict, type(None)),
            serializer=orm.to_aiida_type,
            required=False,
            default=None,
        )
        spec.input(
            'md_timestep_duration_ps',
            valid_type=orm.Float,
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'al_keep_struct_every_n_ps',
            valid_type=orm.Float,
            serializer=orm.to_aiida_type,
        )

        spec.input(
            'current_md_seed_structs_path',
            valid_type=orm.Str,
            serializer=orm.to_aiida_type,
        )
        spec.input('seed_db_path', valid_type=orm.Str, serializer=orm.to_aiida_type)
        spec.input('training_db_path', valid_type=orm.Str, serializer=orm.to_aiida_type)
        spec.input(
            'current_md_seed_structs_idx',
            valid_type=orm.List,
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'train_seed_group',
            valid_type=orm.Str,
            serializer=orm.to_aiida_type,
        )
        spec.input(
            'mace_train',
            valid_type=orm.Dict,
            serializer=orm.to_aiida_type,
        )
        spec.input('md_parameters', valid_type=orm.Dict)
        spec.input('dft_method', valid_type=orm.Str, serializer=orm.to_aiida_type)
        spec.input(
            'dft_calc_limit',
            valid_type=orm.Int,
            serializer=orm.to_aiida_type,
            required=False,
            default=None,
        )
        spec.input('dft_settings', valid_type=orm.Dict)
        spec.input('committee_eval', valid_type=orm.Dict, serializer=orm.to_aiida_type)
        spec.input(
            'check_extrapolation_type',
            valid_type=orm.Str,
            serializer=orm.to_aiida_type,
            required=False,
            default=None,
        )
        spec.input(
            'gather_traj_cnt_lattice', valid_type=orm.Bool, serializer=orm.to_aiida_type
        )
        spec.input('use_kokkos', valid_type=orm.Bool, serializer=orm.to_aiida_type)

        # Data reduction specific inputs
        spec.input(
            'al_mode',
            valid_type=orm.Str,
            serializer=orm.to_aiida_type,
            required=False,
            default=lambda: orm.Str('md'),
        )
        spec.input(
            'data_reduction_settings',
            valid_type=orm.Dict,
            serializer=orm.to_aiida_type,
            required=False,
            default=None,
        )

        spec.outline(
            # Training the main mace model (M0) and the committee models
            # using the training database (Dt).
            cls.train_mace_model,
            # Gathering results from mace training.
            cls.get_mace_train_output,
            # For data reduction mode: select additional structures for training
            if_(cls.should_select_data_reduction_structures)(
                cls.select_data_reduction_structures,
            ),
            # This part of the workflow is only executed if the extrapolation
            # check is enabled.
            # It will get the descriptors for the entire database and use
            # the concave hull as the extrapolation mechanism.
            if_(cls.check_extrapolation_enabled)(
                # Generate descriptors for the current dataset
                # and apply dimensionality reduction if specified.
                # If advanced extrapolation is enabled, the latent space
                # will be used to generate the concave hull.
                cls.gen_descriptors_and_concave_hull,
                cls.get_descriptor_results,
            ),
            # All of the structures in the seed will be run using the MD
            # code selected, using the main model (M0).
            # Then, the trajectories produced with the M0 model will be evaluated
            # using the committee models.
            # Structures and energy predictions will be gathered and prepared
            # According to the difference in error between the models either:
            # The original structure will be removed from D0, or
            # The problematic structure will be calcualated using DFT
            cls.run_md_seed,
            # This uses extrapolation (if enabled), and an energy/force check.
            cls.send_calc_or_remove_structures,
            # Return the generated DFT calculations to the workchain as an output
            cls.return_seed_dft_and_model,
            # TODO: If high error (define this) is found on a training seed,
            # do not change seed until the error is decreased
            # cls.choose_next_seed,
        )
        spec.output('dft_calcs_path', valid_type=orm.Str, required=False)
        spec.output('m0_model_file', valid_type=orm.SinglefileData)
        spec.output('stop_md_seed_no_disagreement', valid_type=orm.Bool)

        spec.exit_code(
            420, 'ERROR_SCHEDULER_MACE', 'error when submitting a MACE calculation.'
        )

    def should_select_data_reduction_structures(self):
        """Check if we should select additional structures for data reduction mode."""
        return self.inputs.al_mode.value == 'data_reduction'

    def select_data_reduction_structures(self):
        """
        Select additional structures from the large database for training.

        This method is only called in data reduction mode and after the first iteration.
        It selects structures based on the iterative_selection_method and adds them
        to the training database while removing them from the seed database.
        """
        self.report('Selecting additional structures for data reduction...')

        # Get settings
        data_reduction_settings = self.inputs.data_reduction_settings.get_dict()
        structures_per_iteration = data_reduction_settings.get(
            'structures_per_iteration', 50
        )
        iterative_selection_method = data_reduction_settings.get(
            'iterative_selection_method', 'uncertainty'
        )

        # Load current databases
        seed_database = mdb_al_ut.load_database(self.ctx.seed_db_path)
        training_database = mdb_al_ut.load_database(self.ctx.training_db_path)

        if len(seed_database) == 0:
            msg = (
                'No more structures available in the large database. '
                'Stopping structure selection.'
            )
            self.report(msg)
            return

        # Prepare for structure selection
        descriptor_settings = None
        model_files = None

        if iterative_selection_method in ['fps', 'uncertainty']:
            descriptor_settings = self.inputs.descriptor_settings.get_dict()

        if iterative_selection_method == 'uncertainty':
            # Get committee model files from context
            if hasattr(self.ctx, 'commitee_models_tupl_name_uuid'):
                model_files = []
                for _, uuid_str in self.ctx.commitee_models_tupl_name_uuid:
                    # Load the model file from the calculation node
                    calc_node = orm.load_node(uuid_str)
                    if hasattr(calc_node.outputs, 'model_file'):
                        model_files.append(calc_node.outputs.model_file.filepath)

                if not model_files:
                    msg = (
                        'Warning: No model files found for uncertainty calculation. '
                        'Falling back to random selection.'
                    )
                    self.report(msg)
                    iterative_selection_method = 'random'
            else:
                msg = (
                    'Warning: No committee models found. '
                    'Falling back to random selection.'
                )
                self.report(msg)
                iterative_selection_method = 'random'

        # Limit selection to available structures
        n_to_select = min(structures_per_iteration, len(seed_database))

        # Select structures
        selected_structures = mdb_al_ut.select_structures_data_reduction(
            database=seed_database,
            n_structures=n_to_select,
            selection_method=iterative_selection_method,
            descriptor_settings=descriptor_settings,
            model_files=model_files,
        )

        breakpoint()

        # Remove selected structures from seed database
        selected_ids = {s.info['mdb_id'] for s in selected_structures}
        remaining_seed_database = [
            s for s in seed_database if s.info['mdb_id'] not in selected_ids
        ]

        # Add selected structures to training database
        training_database.extend(selected_structures)

        # Update databases on disk
        ase_write(
            filename=self.ctx.seed_db_path,
            format='extxyz',
            images=remaining_seed_database,
        )
        ase_write(
            filename=self.ctx.training_db_path,
            format='extxyz',
            images=training_database,
        )

        msg = (
            f'Selected {len(selected_structures)} additional structures using '
            f'{iterative_selection_method} method. Remaining in large database: '
            f'{len(remaining_seed_database)}, Total in training database: '
            f'{len(training_database)}'
        )
        self.report(msg)

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
        self.report('Generating new training database file.')

        # Adding workchain uuid input.data filename to path
        updated_path, _ = mdb_al_ut.get_final_db_path(
            result_dir_path=self.inputs.results_dir.value,
            final_db_name=self.inputs.final_db_name.value,
            node=mdb_al_ut.process_call_root(orm.load_node(self.uuid)),
        )

        database_training = mdb_al_ut.load_database(self.inputs.training_db_path.value)

        # Generate new training data file
        mdb_conv.gen_mace_train_structure_list(
            path=updated_path,
            structure_list=database_training,
        )

        # Determining if resume mode is activated
        self.ctx.resume_mode = self.inputs.al_start_mode.value == 'resume'

        # Train n models (M0-Mn)
        # The most accurate model (during validation) will be chosen as the main model,
        # and used to drive the MD simulations. The remaining models will act as
        # committee models and will only be used to evaluate energies.

        # Stop the calculation if initial models must be loaded
        if (
            self.inputs.load_init_models and self.inputs.al_loop_iteration.value == 0
        ) or (
            self.inputs.load_init_models
            and self.ctx.resume_mode
            and not hasattr(self.ctx, 'loaded_init_models')
        ):
            self.report(
                'Loading models from nodes: '
                f"'{self.inputs.load_init_models.get_list()}'."
            )
            return
        # Run the training
        else:
            self.report(
                f'Training {self.inputs.committee_num_models.value} models using '
                'current training dataset.'
            )

        calc_count = 0

        # Getting container settings
        containerized = False
        container_dict = self.inputs.container_settings.get_dict()
        if container_dict.get('use_container'):
            containerized = container_dict.get('use_container', False)
        if self.inputs.mace_train.get('ignore_container') is True:
            containerized = False

        for _ in range(self.inputs.committee_num_models.value):
            model_name = mdb_al_ut.generate_model_name()

            # Load training settings from inputs and update path and model names.
            mace_train_settings: orm.Dict = mdb_al_ut.update_mace_train_settings_dict(
                settings_dict=self.inputs.mace_train.get('train_settings'),
                train_data_path=str(updated_path),
                curr_model=model_name,
                curr_iter=self.inputs.al_loop_iteration.value,
                db_size=len(database_training),
            )

            # Run training and save new model file
            mace_train = CalculationFactory('mace-train')
            mace_builder = mace_train.get_builder()

            mace_builder.multihead_finetuning = self.inputs.mace_train.get(
                'multihead_finetuning', False
            )
            mace_builder.model_name = model_name
            mace_builder.mace_settings_dict = orm.Dict(mace_train_settings)

            # Set the use container flag
            mace_builder.use_container = orm.Bool(containerized)

            mace_train_file_path, _ = mdb_al_ut.get_final_db_path(
                result_dir_path=self.inputs.results_dir.value,
                final_db_name=self.inputs.final_db_name.value,
                node=self.node,
            )
            mace_builder.mace_train_file_path = str(mace_train_file_path)

            mace_train_calc_sched_options = self.inputs.mace_train.get_dict()[
                'metadata'
            ].get('options')

            if containerized:
                image_name = container_dict.get('image_name', '')
                engine_command = container_dict.get('engine_command', '')
                num_threads = mace_train_calc_sched_options.get('resources', {}).get(
                    'num_cores_per_mpiproc', os.cpu_count()
                )
                mace_train_prepend = (
                    self.inputs.mace_train.get_dict()
                    .get('metadata', {})
                    .get('prepend_text', '')
                )
                prepend_text = (
                    mace_train_prepend
                    + '\n'
                    + container_dict.get('prepend_text', '')
                    + f'\nexport OMP_NUM_THREADS={num_threads}'
                )
                computer = orm.load_computer(
                    self.inputs.mace_train.get_dict().get('computer', None)
                )
                code = orm.ContainerizedCode(
                    computer=computer,
                    image_name=image_name,
                    filepath_executable='mace_run_train',
                    prepend_text=prepend_text,
                    engine_command=engine_command,
                )
            else:
                code_str = self.inputs.mace_train.get_dict()['code']
                code = orm.load_code(code_str)
                computer = code.computer

            mace_builder.code = code

            mace_builder.metadata.options = mace_train_calc_sched_options

            # Manually setting parser. Default is 'mace-training-parser'
            # It might be interesting to allow the user to override in the future?
            mace_builder.metadata.options.parser_name = 'mace-training-parser'

            mace_builder.metadata.options.output_filename = (
                f'train_{model_name}_iter-{self.inputs.al_loop_iteration.value}'
            )
            mace_builder.metadata.label = model_name

            if not hasattr(mace_builder.metadata.options, 'prepend_text'):
                mace_builder.metadata.options.prepend_text = (
                    self.inputs.mace_train.get_dict()
                    .get('metadata', {})
                    .get('options', {})
                    .get('prepend_text', '')
                )

            # future = self.submit(mace_builder)
            # self.to_context(mace_training_results=append_(future))
            calc_limit = computer.metadata.get('mdb_calc_limit', 0)
            if calc_limit != 0:
                mdb_al_ut.aiida_wait_submit(
                    builder=mace_builder,
                    computer=computer,
                    calc_count=calc_count,
                )
            # Submit calculation
            future = self.submit(mace_builder)
            self.report(f'Submitted calculation {future.pk}.')
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
            The function updates the workchain context with the best model's RMSE
            values, the potential file, and the committee models' information
            but does not return any value directly.
        """
        # Get the current iteration and resume mode
        curr_iter = self.inputs.al_loop_iteration.value

        # Ensure that models are loaded only for the first resumed step
        if self.inputs.load_init_models:
            if curr_iter == 0 or (
                self.ctx.resume_mode and not hasattr(self.ctx, 'loaded_init_models')
            ):
                mace_training_results = [
                    orm.load_node(node) for node in self.inputs.load_init_models
                ]
                # Mark that models have been loaded
                self.ctx.loaded_init_models = True
            else:
                mace_training_results = self.ctx.mace_training_results
        else:
            mace_training_results = self.ctx.mace_training_results

        model_name_list = []
        weighted_E_F_sum_list = []

        # Iterate over all models and get weighted sum of E and F
        for calc in mace_training_results:
            # Loading calculation node
            curr_calc: orm.CalcJobNode = orm.load_node(calc.uuid)

            # Skipping model if training hasn't finished correctly.
            if curr_calc.exit_status != 0:
                continue

            # Getting model name
            model_name_list.append(curr_calc.inputs.model_name.value)

            # Adding E + F multiplied by a weight value, in order to consider
            # forces when deciding which model to keep
            force_weight = self.inputs.mace_train.get(
                'result_force_weight',
                0.1,
            )
            weighted_E_F_sum = curr_calc.outputs.m_rmse_e.value + (
                force_weight * curr_calc.outputs.m_rmse_f
            )
            weighted_E_F_sum_list.append(weighted_E_F_sum)

        # Get the most accurate model name during the validation by checking
        # the lowest E+F*weight.
        self.ctx.best_model_name = model_name_list[np.argmin(weighted_E_F_sum_list)]

        commitee_models_tupl_name_uuid = []
        for calc in mace_training_results:
            # Loading calculation node
            curr_calc: orm.CalcJobNode = orm.load_node(calc.uuid)

            # Skipping model if training hasn't finished correctly.
            if curr_calc.exit_status != 0:
                self.report(
                    f'Skipping MACE model that finished with errors (pk: {calc.pk}).'
                )
                continue

            # Loading model name
            model_name = curr_calc.inputs.model_name.value

            # Getting model file
            model_file = curr_calc.outputs.model_file

            # Use checking if the currentt case is the best model.
            # Saving model results
            if model_name == self.ctx.best_model_name:
                # Overwriting m0_rmse values with actual training values
                self.ctx.m0_rmse_e = curr_calc.outputs.m_rmse_e
                self.ctx.m0_rmse_f = curr_calc.outputs.m_rmse_f

                # Storing best model file in the context
                self.ctx.best_model_file = model_file

                # Adding best model name as an extra to the model file
                self.ctx.best_model_file.base.extras.set(
                    'model_name', self.ctx.best_model_name
                )

                # Convert model to LAMMPS compatible format
                # and return it to workchain context
                self.ctx.lammps_potential_file = mdb_al_ut.create_mace_lammps_model(
                    model_file
                )

                self.report(
                    f"Best model of current step '{model_name}' ({calc.pk}) as M0 - "
                    f'RMSE E: {self.ctx.m0_rmse_e.value:.3f} meV/at, '
                    f'RMSE F: {self.ctx.m0_rmse_f.value:.3f} meV/Å'
                )
                self.out('m0_model_file', model_file)
            else:
                self.report(
                    f"Trained committee model '{model_name}' ({calc.pk}) - "
                    f'RMSE E: {curr_calc.outputs.m_rmse_e.value:.3f} meV/at, '
                    f'RMSE F: {curr_calc.outputs.m_rmse_f.value:.3f} meV/Å'
                )
                commitee_models_tupl_name_uuid.append((model_name, model_file.uuid))

        # Sending committee model paths to current context
        self.ctx.commitee_models_tupl_name_uuid = commitee_models_tupl_name_uuid

    def check_extrapolation_enabled(self):
        """Check if the extrapolation check is enabled."""
        return bool(self.inputs.check_extrapolation_type.value)

    def gen_descriptors_and_concave_hull(self):
        self.report('Preparing descriptors calculation...')

        # Run training and save new model file
        desc_calc = CalculationFactory('mdb-descriptors-combined')
        desc_builder = desc_calc.get_builder()

        # Add current iteration to the process label
        desc_builder.metadata.label = (
            f'descriptors-combined_iter-{self.inputs.al_loop_iteration.value}'
        )

        # Getting the current seed structures
        mace_train_file_path, _ = mdb_al_ut.get_final_db_path(
            result_dir_path=self.inputs.results_dir.value,
            final_db_name=self.inputs.final_db_name.value,
            node=self.node,
        )
        desc_builder.training_database_path = str(mace_train_file_path)

        if not desc_builder.training_database_path.is_stored:
            desc_builder.training_database_path.store()

        # Getting the best model file
        desc_builder.best_model = self.ctx.best_model_file

        # Getting settings file path
        settings_file_pth = self.inputs.toml_file
        desc_builder.settings_file_path = settings_file_pth

        # Get the autoencoder model file. If not found, the calculation will be
        # submitted without providing the file and a new one will be trained
        # at runtime.
        if hasattr(self.ctx, 'autoencoder_model_file'):
            desc_builder.autoencoder_model = self.ctx.autoencoder_model_file

        # Get portable code
        descriptor_code_path = Path(
            f'{MDB_ROOT_DIR}/active_learning/mace_code/combined'
        )

        # Loading metadata settings
        computer = orm.load_computer(
            self.inputs.descriptor_settings['metadata']['computer']
        )
        desc_builder.metadata.computer = computer

        resources_dict = self.inputs.descriptor_settings['metadata']['options'][
            'resources'
        ]
        num_threads = resources_dict.get('num_cores_per_mpiproc', 2)
        ignore_container = self.inputs.descriptor_settings.get(
            'ignore_container', False
        )

        # Getting container settings
        containerized = False
        container_dict = self.inputs.container_settings.get_dict()
        if container_dict.get('use_container'):
            containerized = container_dict.get('use_container', False)
        if ignore_container is True:
            containerized = False

        if containerized:
            image_name = container_dict.get('image_name', '')
            engine_command = container_dict.get('engine_command', '')
            prepend_text = (
                self.inputs.descriptor_settings['metadata'].get('prepend_text', '')
                + '\n'
                + container_dict.get('prepend_text', '')
                + f'\nexport OMP_NUM_THREADS={num_threads}'
            )
            code = orm.ContainerizedCode(
                computer=desc_builder.metadata.computer,
                image_name=image_name,
                filepath_executable='mdb_check_descr_combined.py',
                prepend_text=prepend_text,
                engine_command=engine_command,
            )
        else:
            prepend_text = (
                self.inputs.descriptor_settings['metadata'].get('prepend_text', '')
                + '\nexport PATH=$PATH:.'
                + f'\nexport OMP_NUM_THREADS={num_threads}'
            )
            code = orm.PortableCode(
                label='mdb-descriptors-combined',
                filepath_files=descriptor_code_path,
                filepath_executable='mdb_check_descr_combined.py',
                prepend_text=prepend_text,
            )
        desc_builder.code = code

        # Loading AiiDA settings
        mace_eval_aiida_settings_dict = self.inputs.descriptor_settings['metadata'][
            'options'
        ]

        # Load scheduler and resources options
        desc_builder.metadata.options = mace_eval_aiida_settings_dict
        desc_builder.metadata.options.parser_name = 'mdb-descriptors-combined-parser'

        # Get the calculation limit, from the computer metadata set to 0
        # if not present.
        # `mdb_calc_limit` is a custom property set with:
        # computer.set_property(name='mdb_calc_limit', value=366)
        calc_limit = desc_builder.metadata.computer.metadata.get('mdb_calc_limit', 0)

        if calc_limit != 0:
            mdb_al_ut.aiida_wait_submit(
                builder=desc_builder,
                computer=computer,
                calc_count=0,
            )

        future = self.submit(desc_builder)
        if self.inputs.check_extrapolation_type.value == 'advanced':
            self.report(
                f'Submitted calculation ({future.pk}) for descriptors + concave hull.'
            )
        else:
            self.report(f'Submitted calculation ({future.pk}) for descriptors.')

        self.to_context(descrptor_results=append_(future))

    def get_descriptor_results(self):
        """Get descriptor results from the calculation and store them in the context."""
        # Get combined descriptor calculation node
        curr_calc: orm.CalcJobNode = self.ctx.descrptor_results[0]

        # Loading descriptor min and max into context as numpy arrays
        self.ctx.descriptors_max_array = curr_calc.outputs.descriptor_max
        self.ctx.descriptors_min_array = curr_calc.outputs.descriptor_min

        # Checking if outputs contains the concave hull and if so,
        # storing it in the context
        if hasattr(curr_calc.outputs, 'concave_hull'):
            self.ctx.concave_hull = curr_calc.outputs.concave_hull

        # Get the autoencoder model file
        if hasattr(curr_calc.outputs, 'autoencoder_model'):
            self.ctx.autoencoder_model_file = curr_calc.outputs.autoencoder_model

    def generate_descriptors(self):
        """Generate descriptors for the current dataset using the best model.

        According to the value of `dimensionality_reduction_method`, the descriptors
        will be generated using the autoencoder or directly from MACE.
        If no dimensionality reduction method is given, the descriptors will be
        obtained with `GetMACEDescriptorsCalculationParser` calculations, that
        use the MACE code to generate the descriptors.
        If the dimensionality reduction method is autoencoder, the descriptors
        will be obtained using the `GetLatentSpaceAutoencoderCalculationParser`
        calculations, that use the autoencoder code to generate the latent space
        of the descriptors.
        In both cases, descriptors will be available in the workchain context as
        `descriptor_results` for their use in the next steps.
        """
        dimensionality_reduction_method = self.inputs.descriptor_settings.get(
            'dimensionality_reduction_method'
        )
        if dimensionality_reduction_method == 'autoencoder':
            descr_calc = CalculationFactory('mdb-get-latent-space')
            self.report(
                f"Generating descriptors using model '{self.ctx.best_model_name}'"
                f' and latent space using autoencoder...'
            )
        else:
            # Prepare GetMACEDescriptorsCalculation
            descr_calc = CalculationFactory('mace-get-descriptors')
            self.report(
                f"Generating descriptors using model '{self.ctx.best_model_name}'..."
            )

        code_builder = descr_calc.get_builder()
        code_builder.model_file = self.ctx.best_model_file

        # Adding the CWD to the path in the script, so that the script can be
        # run from aiida.
        mace_train_file_path, _ = mdb_al_ut.get_final_db_path(
            result_dir_path=self.inputs.results_dir.value,
            final_db_name=self.inputs.final_db_name.value,
            node=self.node,
        )
        code_builder.mace_train_file_path = str(mace_train_file_path)

        if not code_builder.mace_train_file_path.is_stored:
            code_builder.mace_train_file_path.store()

        prepend_text = (
            self.inputs.descriptor_settings['metadata'].get('prepend_text', '')
            + '\nPATH=$PATH:.'
        )

        # Set metadata options
        code_builder.metadata.options = self.inputs.descriptor_settings['metadata'][
            'options'
        ]

        # Set the computer
        computer = orm.load_computer(
            self.inputs.descriptor_settings['metadata']['computer']
        )
        code_builder.metadata.computer = computer

        # Get latent space of the descriptors using the autoencoder.
        if dimensionality_reduction_method == 'autoencoder':
            train_settings = self.inputs.descriptor_settings['autoencoder'][
                'train_settings'
            ]
            # Preparing inputs
            # Add RNG seed if not present
            if not train_settings.get('rng_seed'):
                rng_seed = np.random.randint(1, int(1e15))
                train_settings['rng_seed'] = rng_seed

            # Overwriting `model_path` in the settings dict
            train_settings['model_path'] = 'autoencoder_model.pth'
            train_settings['dataset'] = 'all_descriptors.npz'

            # Set the settings dict
            code_builder.settings_dict = train_settings

            # Generate aiida code using the script in
            # the `descriptor_code_path` folder.
            descriptor_code_path = Path(
                f'{MDB_ROOT_DIR}/active_learning/extrapolation/autoencoder_scripts'
            )
            code = orm.PortableCode(
                label='mdb_get_latent_space',
                filepath_files=descriptor_code_path,
                filepath_executable='mdb_autoencoder_get_latent_space.py',
                prepend_text=prepend_text,
            )
            code_builder.metadata.options.parser_name = 'mdb-get-latent-space-parser'
            code_builder.metadata.label = self.ctx.best_model_name + '_latent_space'

        # Get descriptors coming directly from MACE.
        else:
            self.report(
                'No dimensionality reduction method specified.'
                ' Getting MACE descriptors...'
            )
            descriptor_code_path = Path(
                f'{MDB_ROOT_DIR}/active_learning/mace_code/descriptors'
            )
            code = orm.PortableCode(
                label='mace_get_descriptors',
                filepath_files=descriptor_code_path,
                filepath_executable='mdb_mace_get_descriptors.py',
                prepend_text=prepend_text,
            )
            code_builder.metadata.options.parser_name = 'mace-descriptors-parser'
            code_builder.metadata.label = self.ctx.best_model_name + '_descriptors'
            code_builder.metadata.options.output_filename = (
                f'descriptors_{self.ctx.best_model_name}_'
                f'iter-{self.inputs.al_loop_iteration.value}'
            )

        code_builder.code = code

        # Get the calculation limit, from the computer metadata set to 0
        # if not present.
        # `mdb_calc_limit` is a custom property set with:
        # computer.set_property(name='mdb_calc_limit', value=366)
        calc_limit = computer.metadata.get('mdb_calc_limit', 0)

        if calc_limit != 0:
            mdb_al_ut.aiida_wait_submit(
                builder=code_builder,
                computer=computer,
                calc_count=0,
            )

        future = self.submit(code_builder)
        self.to_context(descriptor_results=append_(future))

    def get_mace_descriptors_output(self):
        """Process the descriptors in the workchain context."""
        descriptor_results = self.ctx.descriptor_results
        for calc in descriptor_results:
            # Loading calculation node
            curr_calc = orm.load_node(calc.uuid)

            # Storing results in context
            self.ctx.descriptors_min_array = (
                curr_calc.outputs.descriptors_min_array.get_array()
            )

            self.ctx.descriptors_max_array = (
                curr_calc.outputs.descriptors_max_array.get_array()
            )

            # Getting latent space and autoencoder model from the
            # autoencoder calculation
            dimensionality_reduction_method = self.inputs.descriptor_settings.get(
                'dimensionality_reduction_method'
            )
            if dimensionality_reduction_method == 'autoencoder':
                with curr_calc.outputs.descriptors_file.open(mode='rb') as f:
                    db_descriptor_dict = pickle.load(f)

                db_latent_space = np.vstack(
                    [strc['latent_space'] for strc in db_descriptor_dict.values()]
                )
                self.ctx.latent_space = db_latent_space
                self.ctx.autoencoder_model_file = (
                    curr_calc.outputs.autoencoder_model_file
                )
                self.report(
                    'Gathered descriptor ranges, latent space, and autoencoder model'
                    ' for training database.'
                )
            else:
                self.report('Gathered descriptor ranges for training database.')

    def get_concave_hull(self):
        self.report('Getting concave hull...')
        descr_calc = CalculationFactory('mdb-get-concave-hull')

        code_builder = descr_calc.get_builder()

        # Provide the calculation with the latent space
        code_builder.latent_space = self.ctx.latent_space

        # Adding the CWD to the path in the script, so that the script can be
        # run from aiida.
        prepend_text = (
            self.inputs.descriptor_settings['metadata'].get('prepend_text', '')
            + '\nPATH=$PATH:.'
        )

        # Set metadata options
        code_builder.metadata.options = self.inputs.descriptor_settings['metadata'][
            'options'
        ]

        # Set the computer
        computer = orm.load_computer(
            self.inputs.descriptor_settings['metadata']['computer']
        )
        code_builder.metadata.computer = computer

        # Generate aiida code using the script in the `descriptor_code_path` folder.
        descriptor_code_path = Path(
            f'{MDB_ROOT_DIR}/active_learning/extrapolation/concave_hull_scripts'
        )

        code = orm.PortableCode(
            label='mdb_get_concave_hull',
            filepath_files=descriptor_code_path,
            filepath_executable='mdb_get_concave_hull.py',
            prepend_text=prepend_text,
        )
        code_builder.metadata.options.parser_name = 'mdb-get-concave-hull-parser'
        code_builder.metadata.label = self.ctx.best_model_name + '_concave_hull'

        code_builder.code = code

        # Get the calculation limit, from the computer metadata set to 0
        # if not present.
        # `mdb_calc_limit` is a custom property set using:
        # computer.set_property(name='mdb_calc_limit', value=366)
        calc_limit = code_builder.metadata.computer.metadata.get('mdb_calc_limit', 0)

        if calc_limit != 0:
            mdb_al_ut.aiida_wait_submit(
                builder=code_builder,
                computer=computer,
                calc_count=0,
            )

        future = self.submit(code_builder)
        self.to_context(concave_hull_results=append_(future))

    def get_concave_hull_output(self):
        """Process the concave hull in the workchain context."""
        concave_hull_results = self.ctx.concave_hull_results
        for calc in concave_hull_results:
            # Loading calculation node
            curr_calc = orm.load_node(calc.uuid)

            # Storing results in context
            self.ctx.concave_hull_array = (
                curr_calc.outputs.concave_hull_array.get_array()
            )

            self.report('Gathered concave hull for training database.')

    def run_md_seed(self):
        with open(self.inputs.current_md_seed_structs_path.value, 'rb') as f:
            current_md_seed_structs = pickle.load(f)

        self.report(
            f'Starting submission of {len(current_md_seed_structs)} '
            'structures to process...'
        )
        calc_count = 0
        for _, curr_structure in enumerate(current_md_seed_structs):
            # Run training and save new model file
            proc_seed = CalculationFactory('mdb-process-md-seed-struct')
            proc_seed_builder = proc_seed.get_builder()

            # Input committee models to `commitee_models` namespace as a dict like:
            # {"model_1": "/path/to/model_1/", "model_2": ...}
            # mace_builder.commitee_models = commitee_dict

            # Converting to ase.Atoms()
            # TODO: Find a way of not having to hardcode the keys
            for key in [
                'pbc',
                'cell',
                'numbers',
                'positions',
                'forces',
                'REF_forces',
                'MACE_forces',
                'momenta',
                'initial_magmoms',
                'bulk_equivalent',
                'bulk_wyckoff',
                'spacegroup_kinds',
                'mdb_mace_eval_forces',
                'curr_model_forces',
            ]:
                if curr_structure.get(key):
                    curr_structure[key] = np.array(curr_structure[key])

            if isinstance(curr_structure, dict):
                curr_structure = Atoms.fromdict(curr_structure)

            # Write xyz file into a string captured in the stdout,
            # write it to a temporary file.
            f = io.StringIO()
            with redirect_stdout(f):
                ase_write(
                    filename='-',
                    format='extxyz',
                    images=curr_structure,
                )
            xyz_string = f.getvalue()

            # Generating tmp file
            md_xyz_file = orm.SinglefileData(
                file=io.BytesIO(str.encode(xyz_string)),
                filename='md_db.xyz',
            )
            md_xyz_file.store()

            # Add inputs to builder
            proc_seed_builder.md_structure = md_xyz_file
            proc_seed_builder.best_model_name = self.ctx.best_model_name
            proc_seed_builder.m_rmse_e = self.ctx.m0_rmse_e
            proc_seed_builder.m_rmse_f = self.ctx.m0_rmse_f

            # Optional input, advanced extrapolation might not be enabled.
            if hasattr(self.ctx, 'concave_hull'):
                proc_seed_builder.concave_hull = self.ctx.concave_hull
            if hasattr(self.ctx, 'autoencoder_model_file'):
                proc_seed_builder.autoencoder_model = self.ctx.autoencoder_model_file

            proc_seed_builder.desc_max_arr = self.ctx.descriptors_max_array
            proc_seed_builder.desc_min_arr = self.ctx.descriptors_min_array
            proc_seed_builder.settings_file_pth = self.inputs.toml_file

            # Loading the settings file again to get updated settings
            settings_path = Path(self.inputs.toml_file.value)
            if settings_path.exists:
                current_settings = mdb_al_ut.read_toml_settings(
                    settings_file=self.inputs.toml_file.value
                )
            else:
                current_settings = None

            # Preparing comitee dict
            commitee_dict = {}
            for model_name, model in self.ctx.commitee_models_tupl_name_uuid:
                # Only alphanumeric and underscores are allowed as links
                model_name = model_name.replace('-', '_')

                model = orm.load_node(model)
                if not isinstance(model, orm.SinglefileData):
                    commitee_dict[model_name] = orm.SinglefileData(model)
                else:
                    commitee_dict[model_name] = model

            if not commitee_dict:
                self.report(
                    'Committee dict is empty: no committee models trained. '
                    'Using main model only. Check model training step for errors.'
                )

            # Adding the best model to the committee models
            # Only alphanumeric and underscores are allowed as links
            best_model_name_clean = self.ctx.best_model_name.replace('-', '_')
            commitee_dict[best_model_name_clean] = self.ctx.best_model_file

            proc_seed_builder.commitee_models = commitee_dict

            # Loading computer and removing it from the input dictionary
            if current_settings:
                ignore_container = current_settings.get('md', {}).get(
                    'ignore_container', False
                )
                metadata_dict = current_settings.get('md', {}).get('metadata', {})
                resc_dict = metadata_dict.get('options', {}).get('resources', {})
                num_threads = resc_dict.get('num_cores_per_mpiproc')

                # In case the number of threads is not set in the first try
                # try to get an additional key. If it does not work, set it to 1.
                if num_threads is None:
                    num_threads = resc_dict.get('tot_num_mpiprocs', 1)

                prepend_text_conf = metadata_dict.get('prepend_text', '')
            else:
                # TODO: The self input.descriptor_settings should be replaced with
                # the MD metadata section in the toml file, which should be read when
                # preparing the workchain inputs.

                metadata_dict = self.inputs.descriptor_settings['metadata']
                num_threads = self.inputs.descriptor_settings.get('num_cpus', 1)
                prepend_text_conf = self.inputs.descriptor_settings['metadata'].get(
                    'prepend_text', ''
                )

            options_dict = metadata_dict['options']
            computer = orm.load_computer(metadata_dict['computer'])
            proc_seed_builder.metadata.computer = computer
            proc_seed_builder.metadata.label = (
                f'process_{curr_structure.info.get("struct_name", "unknown")}'
            )
            proc_seed_builder.metadata.description = 'Processing structure using MD.'

            # Getting container settings
            containerized = False
            container_dict = self.inputs.container_settings.get_dict()
            if container_dict.get('use_container'):
                containerized = container_dict.get('use_container', False)
            if ignore_container is True:
                containerized = False

            if containerized:
                image_name = container_dict.get('image_name', '')
                engine_command = container_dict.get('engine_command', '')
                prepend_text = (
                    prepend_text_conf
                    + '\n'
                    + container_dict.get('prepend_text', '')
                    + f'\nexport OMP_NUM_THREADS={num_threads}'
                )
                code = orm.ContainerizedCode(
                    computer=computer,
                    image_name=image_name,
                    filepath_executable='mdb_process_structure.py',
                    prepend_text=prepend_text,
                    engine_command=engine_command,
                )
            else:
                # Get portable code
                code_path = Path(f'{MDB_ROOT_DIR}/active_learning/md')
                # TODO: This should not be `descriptor_settings`` after the simple
                # loop is introduced. A new section containing all settings
                # should be included, and this should be changed to the section
                # name.

                prepend_text = (
                    prepend_text_conf
                    + '\nexport PATH=$PATH:.'
                    + f'\nexport OMP_NUM_THREADS={num_threads}'
                )
                code = orm.PortableCode(
                    label='mdb_process_md_seed_struct',
                    filepath_files=code_path,
                    filepath_executable='mdb_process_structure.py',
                    prepend_text=prepend_text,
                )
            proc_seed_builder.code = code

            # options_dict.pop('computer', None)

            # Load scheduler and resources options
            proc_seed_builder.metadata.options = options_dict
            proc_seed_builder.metadata.options.parser_name = (
                'mdb-process-md-seed-struct-parser'
            )

            # Get the calculation limit, from the computer metadata set to 0
            # if not present.
            # `mdb_calc_limit` is a custom property set with:
            # computer.set_property(name='mdb_calc_limit', value=366)
            calc_limit = proc_seed_builder.metadata.computer.metadata.get(
                'mdb_calc_limit', 0
            )

            if calc_limit != 0:
                mdb_al_ut.aiida_wait_submit(
                    builder=proc_seed_builder,
                    computer=computer,
                    calc_count=calc_count,
                )
            future = self.submit(proc_seed_builder)

            if curr_structure.info.get('aiida_uuid'):
                future.base.extras.set(
                    'unique_id_old', curr_structure.info['aiida_uuid']
                )
            if curr_structure.info.get('mdb_id'):
                future.base.extras.set('unique_id', curr_structure.info['mdb_id'])

            future.base.extras.set('mdb_db_index', curr_structure.info['mdb_db_index'])
            # future.base.extras.set("md_temperature", temp_val)
            self.to_context(process_committee_results=append_(future))

        self.report(
            f'Submission done. Running MD for {len(self.ctx.process_committee_results)}'
            ' structures...'
        )

    def send_calc_or_remove_structures(self):
        """Decide which structures to keep and send to DFT or remove from db."""
        self.report(
            'Selecting which structures need performing DFT or removing from DB...'
        )

        # Get all of the processed structures
        processed_structures = self.ctx.process_committee_results

        mace_calcs_struct_list = []
        mace_calcs_idx_list = []
        calcs_to_submit = []
        delete_indices = []

        # Selecting which structures to send to DFT and
        # which to remove from the database.
        proc_calcjob: orm.CalcJobNode
        for proc_calcjob in processed_structures:
            # Skip calculation if it didn't finish correctly
            if proc_calcjob.exit_status != 0:
                self.report(
                    'Removing struct. processing that finished with errors '
                    f'(pk: {proc_calcjob.pk}).'
                )
                delete_indices.append(proc_calcjob.base.extras.all.get('unique_id'))
                continue

            # Getting unique_id from extras
            orig_str_uuid = proc_calcjob.base.extras.all.get('unique_id')

            # Loading extrapolating structures
            extrap_strc_file: orm.SinglefileData = (
                proc_calcjob.outputs.extrapolating_structures
            )
            with extrap_strc_file.as_path() as file_path:
                extrap_structs = ase_read(
                    filename=file_path, format='extxyz', index=':'
                )

            for struct in extrap_structs:
                struct.info['mdb_md_node'] = proc_calcjob.uuid
                struct.info['mdb_db_index'] = proc_calcjob.base.extras.get(
                    'mdb_db_index'
                )

            # If no extrapolating structures, mark for removal from database
            # using orig_str_uuid
            if len(extrap_structs) == 0:
                delete_indices.append(orig_str_uuid)
            # If extrapolating structures, send to DFT using the selected
            # method.
            else:
                calcs_to_submit.extend(extrap_structs)

        # Limit number of DFT calculations to be submitted using the
        # `dft_calc_limit` key in the dft settings section. Ommit if not present.
        if self.inputs.dft_calc_limit:
            dft_calc_limit = self.inputs.dft_calc_limit.value

            if len(calcs_to_submit) > dft_calc_limit:
                step = len(calcs_to_submit) / dft_calc_limit
                sampled_indices = [int(i * step) for i in range(dft_calc_limit)]
                calcs_to_submit = [calcs_to_submit[idx] for idx in sampled_indices]

        if len(calcs_to_submit) == 0:
            self.report(
                'No structures to send to DFT. '
                'Either all explored structures are well represented or '
                'there were problems during the processing step (MD).'
            )

        # Submitting calcs
        calc_count = 0
        for struct in calcs_to_submit:
            # Get row and index
            struct_uuid = struct.info.get('mdb_id')
            if not struct_uuid:
                struct_uuid = struct.info.get('aiida_uuid')
            calc_idx = struct.info.get('mdb_db_index')

            if self.inputs.dft_method == 'vasp':
                row = struct.info

                builder = mdb_al_ut.get_dft_calc_builder_vasp(
                    struct=struct,
                    row=row,
                    calc_idx=calc_idx,
                    group=None,
                    dft_settings=self.inputs.dft_settings.get_dict(),
                )

                # Get the code and computer from the builder, updated
                # to the current version of aiida-vasp.
                try:
                    curr_code = builder.code
                    curr_computer = curr_code.computer
                except AttributeError:
                    curr_code = builder.vasp.code
                    curr_computer = curr_code.computer

                # Get the calculation limit, from the computer metadata set to 0
                # if not present.
                # `mdb_calc_limit` is a custom property set with:
                # computer.set_property(name='mdb_calc_limit', value=366)
                calc_limit = 0
                try:
                    calc_limit = builder.metadata.computer.metadata.get(
                        'mdb_calc_limit', 0
                    )
                except AttributeError:
                    # The code namespace is in a different place in aiida-vasp 4.1.0
                    # This except block tries to get this new namespace and use
                    # it instead.
                    try:
                        calc_limit = builder.vasp.code.computer.metadata.get(
                            'mdb_calc_limit', 0
                        )
                    except Exception:
                        # Any exception in the alternative path
                        # will result in calc_limit being 0.
                        calc_limit = 0
                except Exception:
                    # If everything else fails, screw it it, 0 it is.
                    calc_limit = 0

                if calc_limit != 0:
                    mdb_al_ut.aiida_wait_submit(
                        builder=builder,
                        computer=curr_computer,
                        calc_count=calc_count,
                        code=curr_code,
                    )

                # Submitting current calculation
                future = self.submit(builder)

                unique_id = row.get('unique_id')
                if not unique_id:
                    unique_id = row.get('aiida_uuid')
                struct_name = row.get('material_name')
                if not struct_name:
                    struct_name = row.get('struct_name')

                future.base.extras.set('mdb_calc_uuid', unique_id)
                future.base.extras.set('mdb_struct_type', row['mdb_struct_type'])
                future.base.extras.set('struct_name', struct_name)
                self.to_context(dft_struct_seed_calcs=append_(future))

                if self.inputs.train_seed_group.value:
                    group = orm.load_group(self.inputs.train_seed_group.value)
                    group.add_nodes(future)

            elif self.inputs.dft_method == 'mace':
                mace_calcs_struct_list.append(struct)
                mace_calcs_idx_list.append(calc_idx)

        if self.inputs.dft_method == 'mace' and len(mace_calcs_struct_list) > 0:
            builder = mdb_al_ut.get_dft_calc_builder_mace_list(
                struct_list=mace_calcs_struct_list,
                row=struct.info,
                dft_settings=self.inputs.dft_settings.get_dict(),
                container_settings=self.inputs.container_settings.get_dict(),
            )

            # Get the calculation limit, from the computer metadata set to 0
            # if not present.
            # `mdb_calc_limit` is a custom property set with:
            # computer.set_property(name='mdb_calc_limit', value=366)
            calc_limit = builder.code.computer.metadata.get('mdb_calc_limit', 0)

            # Check if the calculation can be submitted
            if calc_limit != 0:
                mdb_al_ut.aiida_wait_submit(
                    builder=builder,
                    computer=builder.code.computer,
                    calc_count=calc_count,
                )

            # Submitting current calculation
            future = self.submit(builder)
            future.base.extras.set('mdb_calc_uuid', struct.info.get('aiida_uuid'))
            future.base.extras.set(
                'mdb_struct_type', struct.info.get('mdb_struct_type')
            )
            future.base.extras.set('struct_name', struct.info.get('struct_name'))
            future.base.extras.set('mdb_md_node', struct.info.get('mdb_md_node'))

            self.to_context(dft_struct_seed_calcs=append_(future))

            if self.inputs.train_seed_group.value:
                group = orm.load_group(self.inputs.train_seed_group.value)
                group.add_nodes(future)

        self.report(
            f'Committee decision: {len(calcs_to_submit)} get info / '
            f'{len(delete_indices)} delete.'
        )

        self.logger.log(15, f"Structures to delete: '{delete_indices}'")

        if set(delete_indices) == {'unknown'}:
            self.report("All structures to delete marked as 'unknown'. ")
            raise ChildProcessError

        # Deleting well represented structures from seed_gen_db (Ds), if
        # there are any and the seed deletion is enabled in the configuration
        # file.
        delete_seed_structs: bool = self.inputs.delete_seed_structs.value
        if len(delete_indices) > 0 and delete_seed_structs:
            self.report(
                f'Deleting {len(delete_indices)} structures from seed'
                ' generating DB (Ds)'
            )

            mdb_al_ut.remove_structs_from_seed_gen_db(
                self.inputs.seed_db_path, delete_indices
            )

        # If no structure is well represented, nothing will be deleted.
        else:
            self.report('Nothing removed from DB.')

    def return_seed_dft_and_model(self):
        """
        Gather and output the last NNP model and DFT calculations for the current seed.

        This function collects DFT calculations for the structures in the current seed,
        which are then returned as outputs in the workchain using the namespace
        `dft_calcs`. A check is performed to determine if the results agree using
        MACE models, and this check also outputted to the workchain using the
        namespace `stop_md_seed_no_disagreement`.
        """
        self.report('Gathering DFT calculations...')
        # Getting the results directory if not in the context
        if not hasattr(self.ctx, 'results_dir'):
            self.ctx.results_dir = mdb_al_ut.get_results_dir_path(
                result_dir_path=self.inputs.results_dir.value,
                node=self.node,
            )

        # Running VASP calulations if its the selected method
        if self.inputs.dft_method == 'vasp':
            try:
                dft_calcs = len(self.ctx.dft_struct_seed_calcs)

                # After setting
                dft_calcs_ok = [
                    node.uuid
                    for node in self.ctx.dft_struct_seed_calcs
                    if node.is_finished_ok
                    # if node.is_finished and node.exit_status in [0, 504, 503]
                ]
                if len(dft_calcs_ok) == 0:
                    self.report('No DFT calculations finished correctly.')
                    dft_calc_list = ''
                else:
                    self.report(
                        f'Gathering {len(dft_calcs_ok)} VASP DFT calculations '
                        'that finished correctly...'
                    )
                    dft_calc_list = mdb_al_ut.gather_dft_calcs_vasp(dft_calcs_ok)
                    self.report('Done gathering VASP DFT calculations!')

            except AttributeError:
                dft_calc_list = ''

        elif self.inputs.dft_method == 'mace':
            if hasattr(self.ctx, 'dft_struct_seed_calcs'):
                dft_calcs = self.ctx.dft_struct_seed_calcs
            else:
                dft_calcs = []
            try:
                self.report(
                    f'Gathered {len(dft_calcs)} MACE evaluation calculation jobs.'
                )

                dft_calcs_ok = [node.uuid for node in dft_calcs if node.is_finished_ok]
                if len(dft_calcs_ok) == 0:
                    self.report('No DFT calculations finished correctly.')
                    dft_calc_list = ''
                else:
                    # Gather all MACE evaluations, storing results into a file,
                    # stored in `result_list_path`.
                    # Results are filtered to remove outliers. Outliers are
                    # stored in a separate file in the same folder.
                    dft_calc_list: orm.List = mdb_al_ut.gather_dft_calcs_mace(
                        dft_calc_list=dft_calcs_ok,
                        results_dir=str(self.ctx.results_dir),
                        workchain=self.node.uuid,
                    )
            except AttributeError:
                dft_calc_list = ''

        if len(dft_calc_list) > 0:
            # Run filtering step based on NN vs DFT difference threshold
            # for both E and F
            filter_settings = self.inputs.dft_settings.get('filter', {})
            if filter_settings.get('filter_dft_calcs', False):
                threshold_E_meV = filter_settings.get('threshold_E_meV', 1e3)
                threshold_F_meV = filter_settings.get('threshold_F_meV', 1e4)
                self.report(
                    'Removing DFT structures with differences higher than: '
                    f'E - {threshold_E_meV} meV, F - {threshold_F_meV} meV'
                )
                dft_count = len(dft_calc_list)
                dft_calc_list = mdb_al_ut.filter_dft_calcs_threshold(
                    dft_calc_list=dft_calc_list,
                    threshold_E_meV=threshold_E_meV,
                    threshold_F_meV=threshold_F_meV,
                    workchain=self,
                )
                self.report(
                    f'Removed {abs(len(dft_calc_list) - dft_count)}'
                    ' DFT above thresholds.'
                )

            return_list_path, _ = mdb_al_ut.write_gathered_dft_calcs_to_file(
                dft_calc_list=dft_calc_list,
                results_dir=str(self.ctx.results_dir),
                workchain=self,
            )
        else:
            # If no DFT calculations were performed, return empty string
            return_list_path = ''

        # File containing structures
        if return_list_path:
            if isinstance(return_list_path, Path):
                return_list_path = orm.Str(return_list_path)
            self.out('dft_calcs_path', return_list_path)

        # orm.SinglefileData for the MACE model with the best performance
        self.out('m0_model_file', self.ctx.best_model_file)
        self.logger.log(
            15,
            f"Saved best model '{self.ctx.best_model_file.extras.get('model_name')}'"
            f' ({self.ctx.best_model_file.pk}) in workchain.',
        )

        self.out(
            'stop_md_seed_no_disagreement',
            mdb_al_ut.check_md_seed_agreement(return_list_path),
        )


class SimpleActiveLearningBaseWorkChain(BaseRestartWorkChain):
    """Base workchain for MDB active learning workflows.

    This workchain is used as a base for the `SimpleActiveLearningWorkChain` workchain.
    It handles setup of the workchain and the main loop, where the active learning
    steps are launched. After every step, the results are checked and added to the
    database, and the next step is prepared. The workchain will loop until the
    stopping conditions are met.

    It takes all the inputs of the `SimpleActiveLearningWorkChain` workchain, except for
    the inputs that are specific to the active learning loop.
    Additionally, the mandatory `log_path` and optional `resume_dict` can be provided.
    See the `define` method for more information on the inputs and outputs.

    Note
    ----
    This workchain can be restarted from a previous running workchain if its
    files are recovered. The optional `resume_dict` input is used for this.
    `resume_dict` must be a dictionary that contains the following keys:
     - `last_iteration`: The current iteration of the active learning loop.
     - `train_db_path`: The path to the last training database.
     - `seed_db_path`: The path to the last seed generation database.

    The loop will only be restarted from the beginning of the last step.

    Check `ActiveLearningWorkChain` for information on what is done in each step.
    """

    _process_class = SimpleActiveLearningWorkChain

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)

        spec.expose_inputs(
            SimpleActiveLearningWorkChain,
            namespace='active_learning',
            exclude=[
                'current_md_seed_structs_path',
                'current_md_seed_structs_idx',
                'al_loop_iteration',
                'train_seed_group',
                'seed_gen_db',
                'seed_size',
                'seed_db_path',
                'training_db_path',
                'database_training',
            ],
        )
        spec.input('log_path', valid_type=orm.Str, serializer=orm.to_aiida_type)

        # Optional input to resume the workchain from a previous state.
        spec.input(
            'resume_dict',
            valid_type=orm.Dict,
            serializer=orm.to_aiida_type,
            required=False,
            default=None,
        )

        spec.outline(
            # Add a filehandler to aiida logger
            cls.setup_textfile_logging,
            # Run version check,
            cls.log_mdb_version,
            # Load the initial database (D_ini), that will be used as the
            # training database (Dt) without changing the original database.
            # Additionally, create a copy of the database (seed_gen_db, Ds),
            # this will be used to generate the MD seeds.
            cls.get_database,
            # Copy toml file to results directory
            cls.copy_input_toml_file,
            # Create inputs for workchains and initialize iterative counter
            cls.setup,
            # Get a settings report for the active learning loop.
            cls.get_input_report,
            # This part will loop to complete the process
            # It will loop `self.ctx.inputs.max_al_iterations` times.
            # while_(cls.should_run_process)(
            while_(cls.check_al_loop_conditions)(
                # Check status of resume mode.
                cls.check_resume_mode,
                # Get random structures from Ds to generate the MD seed.
                # For data reduction mode, filter out structures already in training DB.
                cls.get_seed_structures,
                # Run training seed
                cls.run_process,
                # Check for correct results
                # cls.inspect_process,
                # Get results from workchain
                cls.get_results_loop,
                # Update Ds and Di to include results from DFT.
                # Update the inputs for the next workchain.
                cls.add_dft_results_to_db,
                cls.get_al_loop_break_conditions,
            ),
            cls.results_final,
        )
        spec.output(
            'final_training_db',
            valid_type=orm.SinglefileData,
        )
        spec.output('final_model_file', valid_type=orm.SinglefileData)

    def setup_textfile_logging(self):
        # Get log path
        log_path = self.inputs.log_path.value

        # Getting aiida logger
        aiida_logger = logging.getLogger('aiida')
        cli_handler = aiida_logger.handlers[0]
        aiida_logger.removeHandler(cli_handler)

        # Set the PARENT logger to INFO to silence framework debug messages
        aiida_logger.setLevel(15)

        # We only want to see our custom debug messages and our reports.
        log_filter = LevelNameFilter(
            levels_to_keep=[
                'MDB_DEBUG',
                'REPORT',
                '[ ✔ ]',
                '[ ! ]',
                '[ X ]',
            ]
        )

        # Create a file handle
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(1)
        file_handler.addFilter(log_filter)

        # Create a formatter and set it for the file handler
        file_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] - %(message)s',
            datefmt='%m/%d/%y %H:%M:%S',
        )
        file_handler.setFormatter(file_formatter)
        aiida_logger.addHandler(file_handler)

        # Adding console logger
        console = Console(color_system='truecolor')
        ch = RichHandler(
            markup=True,
            show_path=False,
            log_time_format='[%d/%m/%y %H:%M:%S]',
            omit_repeated_times=False,
            console=console,
        )
        ch.setLevel(23)
        formatter_con = logging.Formatter('%(message)s')
        ch.setFormatter(formatter_con)
        aiida_logger.addHandler(ch)

        self.report('Started Simple Active Learning!')
        self.report(f"Logging in '{log_path}'.")

    def log_mdb_version(self):
        curr_version, _, hash_str = get_mdb_version_info()
        self.report(f"Using MatDBForge version: '{curr_version}' ({hash_str[:7]}).")

    def get_database(self):
        """Loading initial database."""
        self.report('Reading database file...')

        # Check the AL mode. If no mode is specified, the default mode
        # is 'data_acquisition', where data is generated using a simulation
        # technique.
        if (
            hasattr(self.inputs.active_learning, 'al_mode')
            and self.inputs.active_learning.al_mode is not None
        ):
            self.ctx.al_mode = self.inputs.active_learning.al_mode.value
        else:
            self.ctx.al_mode = 'data_acquisition'

        self.report(
            f"Loading databases for active learning mode: '{self.ctx.al_mode}'."
        )

        if self.ctx.al_mode == 'data_reduction':
            self._setup_data_reduction_databases()
        else:
            self._setup_data_acquisition_databases()

    def _setup_data_reduction_databases(self):
        """Setup databases for data reduction mode."""
        self.report('Setting up databases for data reduction mode...')

        # Get data reduction settings
        if not self.inputs.active_learning.data_reduction_settings:
            raise ValueError(
                "data_reduction_settings must be provided when al_mode='data_reduction'"
            )

        data_reduction_settings = (
            self.inputs.active_learning.data_reduction_settings.get_dict()
        )
        large_db_path = data_reduction_settings['large_database_path']
        initial_selection_size = data_reduction_settings['initial_selection_size']
        initial_selection_method = data_reduction_settings['initial_selection_method']

        # Validate large database path
        if not Path(large_db_path).exists():
            raise FileNotFoundError(f'Large database file not found: {large_db_path}')

        # Load the large database that will serve as the seed database
        if self.inputs.resume_dict:
            # If resuming, load the seed database from the previous run
            large_database = ase_read(
                filename=self.inputs.resume_dict['seed_db_path'],
                format='extxyz',
                index=':',
            )
            # Load training database from previous run
            database_training = ase_read(
                filename=self.inputs.resume_dict['train_db_path'],
                format='extxyz',
                index=':',
            )
        else:
            # Load the large database
            large_database = ase_read(
                filename=large_db_path,
                format='extxyz',
                index=':',
            )
            # Start with empty training database
            database_training = []

        # Process database structures and add metadata
        for idx, struct in enumerate(large_database):
            if not struct.info.get('mdb_db_index'):
                struct.info['mdb_db_index'] = idx
            if not struct.info.get('mdb_al_step'):
                struct.info['mdb_al_step'] = 0
            if not struct.info.get('mdb_id'):
                struct.info['mdb_id'] = str(uuid.uuid4())
            large_database[idx] = struct

        # Process training database structures
        for idx, struct in enumerate(database_training):
            if not struct.info.get('mdb_db_index'):
                struct.info['mdb_db_index'] = idx
            if not struct.info.get('mdb_al_step'):
                struct.info['mdb_al_step'] = 0
            if not struct.info.get('mdb_id'):
                struct.info['mdb_id'] = str(uuid.uuid4())
            database_training[idx] = struct

        # Create result directories
        results_dir_path = Path(self.inputs.active_learning.results_dir.value)
        if not results_dir_path.exists():
            results_dir_path.mkdir()

        final_db_path, curr_run_results_dir = mdb_al_ut.get_final_db_path(
            result_dir_path=results_dir_path,
            final_db_name=self.inputs.active_learning.final_db_name.value,
            node=self.node,
        )
        self.ctx.curr_run_results_dir = curr_run_results_dir

        # Set up seed database (the large database)
        self.ctx.seed_db_path = curr_run_results_dir / 'mdb_seed_db.xyz'

        # Set up training database path
        self.ctx.training_db_path = final_db_path

        # If this is a new run and we have an empty training database,
        # perform initial selection
        if not self.inputs.resume_dict and len(database_training) == 0:
            msg = (
                f'Performing initial selection of {initial_selection_size} '
                f'structures using {initial_selection_method} method...'
            )
            self.report(msg)

            # Get descriptor settings for FPS if needed
            descriptor_settings = None
            if initial_selection_method == 'fps':
                descriptor_settings = (
                    self.inputs.active_learning.descriptor_settings.get_dict()
                )

            # Select initial structures
            selected_structures = mdb_al_ut.select_structures_data_reduction(
                database=large_database,
                n_structures=initial_selection_size,
                selection_method=initial_selection_method,
                descriptor_settings=descriptor_settings,
            )

            # Remove selected structures from large database
            selected_ids = {s.info['mdb_id'] for s in selected_structures}
            large_database = [
                s for s in large_database if s.info['mdb_id'] not in selected_ids
            ]

            # Add to training database
            database_training.extend(selected_structures)

            msg = (
                f'Selected {len(selected_structures)} structures '
                'for initial training database.'
            )
            self.report(msg)

        # Save databases
        ase_write(
            filename=self.ctx.seed_db_path, format='extxyz', images=large_database
        )
        ase_write(
            filename=self.ctx.training_db_path,
            format='extxyz',
            images=database_training,
        )

        # Set minimum seed size (for MD seed selection, not the large database)
        seed_inputs = self.inputs.active_learning
        if (
            hasattr(seed_inputs, 'seed_min_num_structs')
            and seed_inputs.seed_min_num_structs.value
        ):
            self.ctx.min_seed_size = int(seed_inputs.seed_min_num_structs.value)
        else:
            # For data reduction, we set a reasonable default
            self.ctx.min_seed_size = min(25, len(large_database))

        # For data reduction mode, seed size is the number of structures per iteration
        structures_per_iteration = data_reduction_settings.get(
            'structures_per_iteration', 50
        )
        self.ctx.seed_size = structures_per_iteration

        msg = (
            f'Data reduction setup complete. Initial seed database: '
            f'{len(large_database)} structures, '
            f'Training database: {len(database_training)} structures.'
        )
        self.report(msg)

    def _setup_data_acquisition_databases(self):
        """Setup databases for traditional MD-based active learning mode."""
        # The training database (Dt) from which copies are made
        # for further processing. New structures will be added here.
        if self.inputs.resume_dict:
            database_training = ase_read(
                filename=self.inputs.resume_dict['train_db_path'],
                format='extxyz',
                index=':',
            )
        else:
            database_training = ase_read(
                filename=self.inputs.active_learning.init_db_path.value,
                format='extxyz',
                index=':',
            )

        # Adding the database indexes to the info dict of the structures
        # and the current active learning loop step index (0).
        for idx, struct in enumerate(database_training):
            # Storing position in the database
            if not struct.info.get('mdb_db_index'):
                struct.info['mdb_db_index'] = idx

            # If workchain is resumed from a previous run, we should keep
            # the step index from the previous run.
            if self.inputs.resume_dict:
                # However, structures without step numbers will be set to 0.
                if not struct.info.get('mdb_al_step'):
                    struct.info['mdb_al_step'] = 0

            # On the other hand, if starting from scratch, we set the step index to 0,
            # as this is the first step of the active learning loop.
            else:
                # Adding step index
                struct.info['mdb_al_step'] = 0

            # Adding unique id to structures that don't have it.
            if not struct.info.get('mdb_id'):
                struct.info['mdb_id'] = str(uuid.uuid4())

            database_training[idx] = struct

        ase_write(
            filename=self.inputs.active_learning.init_db_path.value,
            format='extxyz',
            images=database_training,
        )
        # Create files for database_training and seed_gen_db
        results_dir_path = Path(self.inputs.active_learning.results_dir.value)
        if not results_dir_path.exists():
            results_dir_path.mkdir()

        final_db_path, curr_run_results_dir = mdb_al_ut.get_final_db_path(
            result_dir_path=results_dir_path,
            final_db_name=self.inputs.active_learning.final_db_name.value,
            node=self.node,
        )
        self.ctx.curr_run_results_dir = curr_run_results_dir

        # A copy of the initial database, (Ds)
        # used specifically for generating MD seeds and running the MDs.
        # New structures will be added and well represented configs removed from here.
        self.ctx.seed_db_path = curr_run_results_dir / 'mdb_seed_db.xyz'

        # Copying the seed database from the initial database if new run,
        # otherwise copy from the previous run.
        if self.inputs.resume_dict:
            shutil.copy(self.inputs.resume_dict['seed_db_path'], self.ctx.seed_db_path)
        else:
            shutil.copy(
                self.inputs.active_learning.init_db_path.value, self.ctx.seed_db_path
            )

        # Setting the minimum seed size for the md seed. Min size defaults to the
        # seed size fraction if not set.
        al_inputs = self.inputs.active_learning
        if hasattr(al_inputs, 'seed_size_frac') and al_inputs.seed_size_frac.value:
            self.ctx.min_seed_size = int(
                self.inputs.active_learning.seed_min_num_structs.value
            )
        else:
            self.ctx.min_seed_size = int(
                self.inputs.active_learning.seed_size_frac.value
                * len(database_training)
            )

        self.ctx.seed_size = int(
            self.inputs.active_learning.seed_size_frac.value * len(database_training)
        )

        # Create a copy of the training database for the current run.
        # If reading from a previous run, copy the database from the previous run.
        self.ctx.training_db_path = final_db_path
        if self.inputs.resume_dict:
            shutil.copy(
                self.inputs.resume_dict['train_db_path'], self.ctx.training_db_path
            )
            self.report(
                'Loaded initial database from previous run containing '
                f"'{len(database_training)}' structures."
            )
        else:
            shutil.copy(
                self.inputs.active_learning.init_db_path.value,
                self.ctx.training_db_path,
            )
            self.report(
                'Loaded initial database containing '
                f"'{len(database_training)}' structures."
            )

    def copy_input_toml_file(self):
        """Copy the input toml file to the results directory."""
        toml_file_path = Path(self.inputs.active_learning.toml_file.value)
        toml_file_name = toml_file_path.name
        toml_file_dest = self.ctx.curr_run_results_dir / toml_file_name

        shutil.copy(toml_file_path, toml_file_dest)

    def get_results_loop(self):
        """Attach the outputs specified in the spec from the last completed process."""
        self.logger.log(15, f'Getting results for iteration {self.ctx.iteration}.')

        # Mark last iteration differently for first resume step.
        first_resume_step = False
        is_resume_run = (
            hasattr(self.inputs, 'resume_dict') and self.inputs.resume_dict is not None
        )
        if is_resume_run:
            # If resuming, we need to check if this is the first step after resuming.
            # If it is, we need to get the last iteration from the resume_dict.
            # Otherwise, we can just use the current iteration.
            first_resume_step = (
                self.inputs.resume_dict['last_iteration'] == self.ctx.iteration - 1
            )

        if first_resume_step is True and is_resume_run is True:
            last_iteration = self.inputs.resume_dict['last_iteration']
            self.logger.log(
                15,
                f'Detected resume mode, first step. Last iteration:  {last_iteration}',
            )
            node = self.ctx.children[self.ctx.iteration - (last_iteration - 1) - 2]

        # If resuming, but not the first step
        elif first_resume_step is False and is_resume_run is True:
            self.logger.log(
                15,
                f'Detected resume mode but NOT first step. '
                f'Last iteration: {self.ctx.iteration}',
            )
            node = self.ctx.children[-1]
        else:
            node = self.ctx.children[self.ctx.iteration - 1]
            self.logger.log(
                15, f'Detected resume mode OFF. Last iteration:  {node.uuid}'
            )

        self.logger.log(
            15, f"ActiveLearningWorkChain to gather results from: '{node.uuid}'"
        )

        # TODO: Gather outputs manually, instead of using __attach_outputs
        # outputs = self._attach_outputs(node)

        # Sending seed disagreement flag to context
        if hasattr(node.outputs, 'stop_md_seed_no_disagreement'):
            self.ctx.stop_md_seed_no_disagreement = node.outputs[
                'stop_md_seed_no_disagreement'
            ]
        else:
            self.ctx.stop_al_loop_error = orm.Bool(True)

        self.ctx.last_workchain_completed = node
        self.logger.log(15, f'Done getting results for iteration {self.ctx.iteration}.')
        return None

    def add_dft_results_to_db(self):
        """
        Incorporate DFT calculation results into the training/seed generation databases.

        This method updates the training and seed generation databases with DFT
        calculation results. If any DFT calculations have been performed,
        their results are appended to both the training database and the seed
        generation database.
        """
        # Updating final and seed database.
        self.report('Checking if database files must be updated...')

        # Updating current training seed
        try:
            seed_gen_db = mdb_al_ut.load_database(self.ctx.seed_db_path)
            training_db = mdb_al_ut.load_database(self.ctx.training_db_path)
            last_wc = self.ctx.last_workchain_completed
            dft_calcs = ase_read(
                last_wc.outputs['dft_calcs_path'].value, format='extxyz', index=':'
            )

            cnt_dft_calcs = len(dft_calcs)

        except KeyError:
            self.report('No new structures available for DB.')
            cnt_dft_calcs = 0

        if cnt_dft_calcs > 0:
            self.report(f'Adding {cnt_dft_calcs} DFT calculations to DB.')

            # Adding calculations to training database and seed_generation database
            for dft_calc_idx, dft_calc in enumerate(dft_calcs):
                # Converting serialized structures to Atoms object.
                if isinstance(dft_calc, dict):
                    dft_calc: Atoms = mdb_al_ut.aiida_serialized_ase_dict_to_atoms(
                        dft_calc
                    )

                # Check if structure has `mdb_db_index` key. If not, add it using
                # the current length of the database + the current calc idx + 1.
                mdb_db_index = dft_calc.info.get(
                    'mdb_db_index', len(training_db) + dft_calc_idx + 1
                )
                dft_calc.info['mdb_db_index'] = mdb_db_index

                # Add information about the current AL step
                dft_calc.info['mdb_al_step'] = self.ctx.iteration

                # Adding the structure to the training database
                seed_gen_db.append(dft_calc)
                training_db.append(dft_calc)

            # Writing the updated databases to file
            ase_write(
                filename=self.ctx.training_db_path,
                images=training_db,
                format='extxyz',
            )
            ase_write(
                filename=self.ctx.seed_db_path,
                images=seed_gen_db,
                format='extxyz',
            )

            self.report('Database files updated.')

            tmp_folder_path: Path = self.ctx.results_dir / 'run_tmp_data'

            # Saving current step database
            curr_it_db_path = (
                tmp_folder_path / f'train_db_it_{self.ctx.iteration}.xyz.xz'
            )
            ase_write(
                filename=curr_it_db_path,
                images=seed_gen_db,
                format='extxyz',
            )

            # Removing teporary files from the AL loop step.
            # TODO: Can't use Path.walk() here as it's not available in Python 3.9
            # this will be added if the library is updated to Python 3.12
            for _, _, files in os.walk(top=tmp_folder_path):
                for file in files:
                    if 'md_seed_structures' not in file:
                        (tmp_folder_path / file).unlink(missing_ok=True)

        self.report(
            f'Iteration {self.ctx.iteration}: '
            f'seed_gen_db {len(seed_gen_db)}, '
            f'training_db: {len(training_db)} entries'
        )

    def get_al_loop_break_conditions(self):
        """
        Evaluate and set conditions to potentially break the active learning loop.

        This function checks for specific conditions that might warrant terminating the
        active learning (AL) loop early:

        - Gathers `stop_md_seed_no_disagreement` from the outputs of the inner workchain
          and stores it in the workchain's context. If this is True, the workchain will
          stop.
        - Checks whether all structures have been removed from the seed generation
          database (indicating no further candidates for evaluation). If this is True,
          the workchain will stop.

        The results of these checks are stored in the workflow's context.
        """
        # Sending empty seed_gen_db flag to context
        seed_gen_db = mdb_al_ut.load_database(self.ctx.inputs.seed_db_path)
        if len(seed_gen_db) == 0:
            self.ctx.seed_gen_db_all_structs_removed = orm.Bool(True)
        else:
            self.ctx.seed_gen_db_all_structs_removed = orm.Bool(False)

    def setup(self):
        """Call `BaseRestartWorkChain` setup and create input dict in self.ctx.inputs.

        This `self.ctx.inputs` dictionary will be used by the `BaseRestartWorkChain`
        to submit the process in the internal loop.
        """
        self.report('Starting Workchain setup.')
        super().setup()

        # Update the iteration counter if resuming from a previous run
        if self.inputs.resume_dict:
            self.report(
                'Resuming from previous run, stopped at iteration: '
                f"'{self.inputs.resume_dict['last_iteration']}'."
            )
            self.ctx.iteration = self.inputs.resume_dict['last_iteration']

        self.ctx.inputs = self.exposed_inputs(
            SimpleActiveLearningWorkChain, 'active_learning'
        )

        # Creating aiida orm.Group to store all calculations
        ctime = time.strftime('%Y%m%dT%H%M%S')
        seed_group = orm.Group(
            label=f'{self.inputs.active_learning.run_name.value}_train_md_seed_{ctime}'
        )
        seed_group.store()
        self.ctx.inputs.train_seed_group = seed_group.uuid
        self.report(f"Created group: '{self.ctx.inputs.train_seed_group}'.")

        # Providing current iteration to children workchain.
        self.ctx.inputs.al_loop_iteration = self.ctx.iteration

        # Setting conditionals to always run the first iteration of the
        # active learning loop.
        self.ctx.stop_md_seed_no_disagreement = orm.Bool(False)
        self.ctx.seed_gen_db_all_structs_removed = orm.Bool(False)

        # Adding database paths to inputs
        self.ctx.inputs.seed_db_path = str(self.ctx.seed_db_path)
        self.ctx.inputs.training_db_path = str(self.ctx.training_db_path)

        self.report('Workchain setup finished.')

    def check_resume_mode(self):
        # If resuming, we need to check if the last iteration is the same as the
        # current iteration. If it is, we keep resume mode for this first resume
        # iteration. Otherwise this means that resume mode has run for one step,
        # and we can disable it.
        self.ctx.resume_mode = self.ctx.inputs.al_start_mode.value == 'resume'
        if self.ctx.resume_mode:
            curr_iteration = self.ctx.iteration
            prev_iteration = self.inputs.resume_dict.get('last_iteration')
            is_first_resume_step = curr_iteration == prev_iteration

            if not is_first_resume_step:
                self.ctx.inputs.al_start_mode = orm.Str('normal')

    def get_input_report(self):
        console = Console(color_system='truecolor')

        input_dict = {}
        for i in self.ctx.inputs:
            if i != 'metadata':
                if self.ctx.inputs[i]:
                    if isinstance(self.ctx.inputs[i], orm.Dict):
                        input_dict[i] = {}
                        curr_dict = self.ctx.inputs[i].get_dict()
                        for key, val in curr_dict.items():
                            if key not in ['metadata', 'options', 'prepend_text']:
                                input_dict[i][key] = val
                    elif isinstance(self.ctx.inputs[i], orm.List):
                        input_dict[i] = self.ctx.inputs[i].get_list()
                    elif isinstance(
                        self.ctx.inputs[i], (orm.Int, orm.Bool, orm.Float, orm.Str)
                    ):
                        input_dict[i] = self.ctx.inputs[i].value
                    else:
                        input_dict[i] = self.ctx.inputs[i]
                else:
                    input_dict[i] = None

        # Hacky way of logging the inputs to both the console and the log file
        # in different formats.
        aiida_logger = logging.getLogger('aiida')
        ini_level = aiida_logger.level
        aiida_logger.level = 20
        aiida_logger.handlers[-1].setLevel(23)
        aiida_logger.log(msg=f'Active Learning Inputs: \n{input_dict}', level=20)
        aiida_logger.level = ini_level

        print()
        inputs = Panel(Pretty(input_dict), title='Active Learning Inputs')
        console.print(inputs)

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
        orm.Bool
            A boolean value indicating whether the AL loop should continue. Returns
            `True` if conditions are met for another iteration; otherwise,
            returns `False`.

        Notes
        -----
        The method uses `self.ctx.is_finished`, `self.ctx.iteration`,
        `self.inputs.active_learning.max_iterations.value`,
        `self.ctx.stop_md_seed_no_disagreement.value`,
        and `self.ctx.seed_gen_db_all_structs_removed.value`
        """
        if hasattr(self.ctx, 'stop_al_loop_error'):
            self.report(
                f'Last step ({self.ctx.last_workchain_completed.pk}) '
                'did not finish correctly. Stopping AL Loop.'
            )
            return False

        max_iterations = self.inputs.active_learning.max_iterations.value

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
            self.report('Stopping AL Loop as all predictions agree for a MD seed.')
        elif self.ctx.seed_gen_db_all_structs_removed.value:
            self.report(
                'Stopping AL Loop as seed generating database has been depleted.'
            )
        else:
            if self.ctx.iteration != 0:
                self.report(
                    f'Proceeding with iteration-{self.ctx.iteration + 1} of AL Loop '
                    'as stopping conditions not met.'
                )
            self.ctx.inputs.al_loop_iteration = self.ctx.iteration

        return continue_cond

    def get_seed_structures(self):
        """
        Dispatcher method to get seed structures based on the active learning mode.

        For MD mode: calls get_md_seed()
        For data_reduction mode: calls get_data_reduction_seed()
        """
        if self.ctx.al_mode == 'data_reduction':
            self.get_data_reduction_seed()
        else:
            self.get_md_seed()

    def get_data_reduction_seed(self):
        """
        Filter structures for data reduction mode.

        In data reduction mode, we need to filter out any structures that are
        already in the training database from the seed database. This ensures
        we don't re-select structures that have already been chosen.

        The actual structure selection for training will happen in the
        SimpleActiveLearningWorkChain.
        """
        self.report(
            f'Starting AL Loop iteration {self.ctx.iteration + 1}/'
            f'{self.inputs.active_learning.max_iterations.value}...'
        )
        self.report('Filtering seed database for data reduction mode...')

        # Load current seed database (large database)
        seed_gen_db: list = mdb_al_ut.load_database(self.ctx.seed_db_path)

        # Load current training database
        training_db: list = mdb_al_ut.load_database(self.ctx.training_db_path)

        # Get IDs of structures already in training database
        training_ids = {struct.info['mdb_id'] for struct in training_db}

        # Filter out structures that are already in training database
        filtered_seed_db = [
            struct
            for struct in seed_gen_db
            if struct.info['mdb_id'] not in training_ids
        ]

        # Remove structures with config_type == 'IsolatedAtom' if any
        filtered_seed_db = [
            struct
            for struct in filtered_seed_db
            if struct.info.get('config_type') != 'IsolatedAtom'
        ]

        # Update the seed database file with filtered structures
        ase_write(
            filename=self.ctx.seed_db_path,
            format='extxyz',
            images=filtered_seed_db,
        )

        # Update context metadata for the child workchain
        self.ctx.inputs.metadata.description = (
            'Data reduction: Select and train with most informative structures. '
            f'Step: {self.ctx.iteration + 1}'
        )
        self.ctx.inputs.metadata.label = f'DataReduction_Step_{self.ctx.iteration + 1}'

        self.report(
            f'Filtered seed database: {len(seed_gen_db)} -> {len(filtered_seed_db)} '
            f'structures (removed {len(seed_gen_db) - len(filtered_seed_db)} '
            'already in training database)'
        )

        # For data reduction mode, we don't set up MD seed structures here
        # The structure selection for training will happen in the
        # SimpleActiveLearningWorkChain based on the iterative_selection_method
        self.ctx.inputs.current_md_seed_structs_path = ''
        self.ctx.inputs.current_md_seed_structs_idx = []

    def get_md_seed(self):
        """
        Selects a random subset of structures from the seed generation database to
        create a MD seed for the active learning loop.

        This function calculates the number of structures to be included in the MD
        seed based on the specified fraction of the seed generation database's length.
        It then randomly selects and populates the training seed with these structures.

        Returns
        -------
            None. The function updates current_md_seed_structs with the
            selected structures.
        """
        self.report(
            f'Starting AL Loop iteration {self.ctx.iteration + 1}/'
            f'{self.inputs.active_learning.max_iterations.value}...'
        )
        self.report('Getting MD seed...')
        self.ctx.inputs.metadata.description = (
            'Perform MD simulations, evaluate and refine ML models. '
            f'Step: {self.ctx.iteration + 1}'
        )
        self.ctx.inputs.metadata.label = f'Step - {self.ctx.iteration + 1}'

        seed_gen_db: list = mdb_al_ut.load_database(self.ctx.seed_db_path)

        # Removing structures with config_type == 'IsolatedAtom'
        # from the seed generation database.
        seed_gen_db = [
            struct
            for struct in seed_gen_db
            if struct.info.get('config_type') != 'IsolatedAtom'
        ]

        # Getting length of the seed generating database
        db_length = len(seed_gen_db)

        # Defining the current md seed size as a function of the amount of structures
        # in the seed generation database
        seed_size = int(self.ctx.inputs.seed_size_frac.value * db_length)

        # Increasing the seed size if it is below the minimum seed size
        if seed_size < self.ctx.min_seed_size:
            seed_size = self.ctx.min_seed_size
            self.report(
                'Set MD seed size to minimum seed size as the calculated seed size was '
                f'below the minimum seed size: {self.ctx.min_seed_size}.'
            )

        # Limit the maximum number of structures to the seed size limit set on the input
        if seed_size > self.ctx.inputs.seed_max_num_structs.value:
            seed_size = int(self.ctx.inputs.seed_max_num_structs.value)
            self.report(f"MD seed size too large: limited to '{seed_size}'.")

        # This should avoid tring to select more structures than available
        if seed_size > db_length:
            seed_size = db_length
            self.report(
                'MD seed size larger than seed generation database. '
                f"Limited size to '{db_length}'."
            )

        # For small databases or percentages, the number of structures might be 0
        # if this happens, make it 1.
        if seed_size < 1:
            seed_size = 1
            self.report('MD seed size is 0. Set to 1.')

        # Get seed selection type
        seed_select_settings = self.ctx.inputs.seed_select_settings[
            'seed_select_settings'
        ]

        # Apply small-cell training-like seed selection if enabled.
        # In order to use small_first, the current iteration number must be
        # less than small_first_max_iter.
        if (
            seed_select_settings['seed_select_type'] == 'small_first'
            and self.ctx.iteration + 1 <= seed_select_settings['small_first_max_iter']
        ):
            self.report("Using 'small_first' seed selection type.")
            seed_select_type = 'small_first'
        else:
            seed_select_type = 'random'

        # Filter the structure database and only use the small structures.
        if seed_select_type == 'small_first':
            max_size = seed_select_settings['small_first_max_size']
            seed_gen_db_small = [s for s in seed_gen_db if len(s) <= max_size]
            if len(seed_gen_db_small) != 0:
                seed_gen_db = seed_gen_db_small
            else:
                self.logger.warn(
                    f'There are no structures with less than '
                    f'{max_size} atoms in the given database. '
                    "Please, remove the 'small_first' seed selection mode. "
                    'and try again.'
                )

        # Load algorithm for seed selection
        # Set score for structures in the seed generation database
        # and use it to select the structures for the MD seed.
        seed_ranking_algo_settings = self.ctx.inputs.seed_select_settings.get(
            'seed_ranking_algorithm', {}
        )
        seed_ranking_algorithm = seed_ranking_algo_settings.get(
            'seed_ranking_algorithm', 'random'
        )

        # If enabled, use Farthest Point Sampling (FPS) on the descriptors
        # starting from an initially selected structure to get new points.
        if seed_ranking_algorithm == 'descriptor_fps':
            descriptor_fps_settings = seed_ranking_algo_settings.get(
                'descriptor_fps', {}
            )
            self.report('Initializing farthest point sampling ranking...')
            # Select an initial structure depending on the seed ranking algorithm
            # `initial_structure` parameter.
            init_struct_type = descriptor_fps_settings.get(
                'initial_structure', 'random'
            )
            self.report(
                f"Selecting initial structure using '{init_struct_type}' method..."
            )

            # Sorting the seed generation database by the REF_energy so that the
            # lowest energy structure goes first
            if init_struct_type == 'lowest_energy':
                sorted_seed_db = sorted(
                    seed_gen_db, key=lambda x: float(x.info['REF_energy'])
                )
                init_structure_uuid = sorted_seed_db[0].info['mdb_id']
            elif init_struct_type == 'random':
                # If the initial structure is random, we will select a random
                # structure from the seed generation database.
                sorted_seed_db = seed_gen_db
                init_structure_uuid = np.random.choice(
                    [s.info['mdb_id'] for s in seed_gen_db]
                )

            # If the seed selection algorithm is descriptor_fps, we need to
            # calculate the fingerprints for the structures in the seed generation
            # database.
            self.report('Calculating descriptors for seed generation database...')
            descriptor_settings_dict = (
                self.inputs.active_learning.descriptor_settings.get_dict()
            )

            # Setting the average type for the descriptor to get global descriptors.
            if descriptor_settings_dict.get('descriptor', {}).get('average') is None:
                descriptor_settings_dict.get('descriptor', {})['average'] = 'inner'

            descr_dict, _ = mdb_al_ut.generate_descriptors(
                database=sorted_seed_db,
                descriptor_type=descriptor_fps_settings.get('descriptor_type', 'soap'),
                descriptor_settings=descriptor_fps_settings.get('descriptor', {}),
            )
            self.report('Done calculating descriptors!')

            # Compute FPS to get the scoring
            self.report('Ranking using farthest point sampling...')
            scores = mdb_al_ut.calculate_fps_scores_descriptor(
                init_structure_uuid=init_structure_uuid,
                descriptor_dict=descr_dict,
            )
            self.report('Ranking completed!')

            # Add scores to the structures in the seed generation database
            for struct in seed_gen_db:
                # If the structure is not in the scores, set the score to 0
                if struct.info['mdb_id'] not in scores:
                    scores[struct.info['mdb_id']] = 0.0

                # Add the score to the structure info
                struct.info['md_seed_ranking_score'] = scores[struct.info['mdb_id']]

            sorted_seed_db = sorted(
                seed_gen_db,
                key=lambda x: x.info['md_seed_ranking_score'],
                reverse=True,
            )

        elif seed_ranking_algorithm == 'random':
            # If the seed selection algorithm is random, we will select structures
            # at random from the seed generation database, and we will use a 1 as
            # the score for all structures.
            self.report("Using 'random' seed ranking algorithm.")
            scores = {s.info['mdb_id']: 1.0 for s in seed_gen_db}

            # Add scores to the structures in the seed generation database
            for struct in seed_gen_db:
                # If the structure is not in the scores, set the score to 0
                if struct.info['mdb_id'] not in scores:
                    scores[struct.info['mdb_id']] = 0.0

                # Add the score to the structure info
                struct.info['md_seed_ranking_score'] = scores[struct.info['mdb_id']]

            # Apply random shuffling to the seed generation database
            # TODO: Use numpy new random number generator class, not default one.
            sorted_seed_db = seed_gen_db.copy()
            np.random.shuffle(seed_gen_db)

        # Selecting first db_length structures from the seed generation database
        # to be used in the training seed.
        # Seed generation database is already sorted by the
        # seed ranking algorithm, so we can just take the first `seed_size` structures.
        selected_structs_idxs = range(seed_size)

        self.ctx.inputs.current_md_seed_structs_idx = list(selected_structs_idxs)

        # The set of random structures selected from the seed generation
        # database to be used in training.
        current_md_seed_structs = []

        # Populating MD seed with the selected random structures
        for idx in selected_structs_idxs:
            seed_struct = sorted_seed_db[idx]
            current_md_seed_structs.append(seed_struct)

        self.report(
            f'Created MD seed with {seed_size} structures '
            f'({(seed_size / db_length) * 100:.1f}% of current database size).'
        )

        # Adding current train seed to the context
        current_MD_seed_serialized = []
        for curr_s in current_md_seed_structs:
            curr_s = mdb_al_ut.serialize_ase(curr_s)
            current_MD_seed_serialized.append(curr_s)

        current_md_seed_structs = current_MD_seed_serialized

        # Saving the current md seed into the result directory as a file.
        self.ctx.results_dir = mdb_al_ut.get_results_dir_path(
            result_dir_path=self.inputs.active_learning.results_dir.value,
            node=self.node,
        )
        self.ctx.current_md_seed_structs_path = (
            f'{self.ctx.results_dir}/run_tmp_data/md_seed_structures-'
            f'{self.ctx.iteration}.pkl'
        )

        with open(self.ctx.current_md_seed_structs_path, 'wb') as seed_file:
            pickle.dump(current_md_seed_structs, seed_file)
        self.ctx.inputs.current_md_seed_structs_path = (
            self.ctx.current_md_seed_structs_path
        )

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
        self.report('Returning final results...')

        # Storing final database as a orm.SinglefileData object
        train_db = mdb_al_ut.prepare_output_final_training_db(
            training_db_path=self.ctx.inputs.training_db_path
        )

        self.out('final_training_db', train_db)

        if not hasattr(self.ctx, 'stop_al_loop_error'):
            # Returning final model as orm.SinglefileData object
            final_model_singlefile = self.ctx.last_workchain_completed.outputs[
                'm0_model_file'
            ]
            self.out(
                'final_model_file',
                self.ctx.last_workchain_completed.outputs['m0_model_file'],
            )

            target_file_name = (
                f'al_loop_{self.inputs.active_learning.run_name.value}.model'
            )
            target_file_path = self.ctx.curr_run_results_dir / target_file_name

            with (
                final_model_singlefile.open(mode='rb') as source,
                open(target_file_path, mode='wb') as target,
            ):
                shutil.copyfileobj(source, target)

            self.report('Workchain completed correctly!')
        else:
            self.report(f"Workchain '{self.node.pk}' exited with errors...")
