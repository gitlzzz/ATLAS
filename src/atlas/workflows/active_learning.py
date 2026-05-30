"""AiiDA workchain for NNP active learning loops using MD."""

import io
import logging
import os
import pickle
import shutil
import time
from contextlib import redirect_stdout, suppress
from pathlib import Path

import numpy as np
import pandas as pd
import pymatgen.io.ase as pmg_ase
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
from pymatgen.core import Structure
from pymatgen.core.trajectory import Trajectory
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.lammps.data import LammpsData
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.pretty import Pretty

import atlas.core.exceptions as atl_excp
from atlas import ATL_ROOT_DIR
from atlas.active_learning import active_learning_utils as atl_al_ut
from atlas.active_learning import conversion as atl_conv
from atlas.core import ATL_DATA_DIR
from atlas.core.code_utils import get_atl_version_info
from atlas.workflows.aiida_utils import can_submit_calculation


class ActiveLearningWorkChain(WorkChain):
    """
    WorkChain to run an active learning loop for a MACE potential using MD
    simulations to generate training data.
    """

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)

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
        spec.input('dft_method', valid_type=orm.Str, serializer=orm.to_aiida_type)
        spec.input(
            'dft_calc_limit',
            valid_type=orm.Int,
            serializer=orm.to_aiida_type,
            required=False,
            default=None,
        )
        spec.input('committee_eval', valid_type=orm.Dict, serializer=orm.to_aiida_type)
        spec.input(
            'check_extrapolation_type', valid_type=orm.Str, serializer=orm.to_aiida_type
        )
        spec.input(
            'gather_traj_cnt_lattice', valid_type=orm.Bool, serializer=orm.to_aiida_type
        )
        spec.input('use_kokkos', valid_type=orm.Bool, serializer=orm.to_aiida_type)

        spec.outline(
            # Training the main mace model (M0) and the committee models
            # using the training database (Dt).
            cls.train_mace_model,
            # Gathering results from mace training.
            cls.get_mace_train_output,
            # This part of the workflow is only executed if the extrapolation
            # check is enabled.
            # It will get the descriptors for the entire database and use
            # the concave hull as the extrapolation mechanism.
            if_(cls.check_extrapolation_enabled)(
                # Generate MACE descriptors for the current dataset.
                # Dimensionality reduction is used if specified,
                # returning an embedded/latent space.
                cls.generate_descriptors,
                # Gather the descriptors from the calcjob and store them
                # in the workchain context.
                cls.get_mace_descriptors_output,
                if_(cls.can_do_advanced_extrapolation)(
                    # Get the concave hull of the training database
                    cls.get_concave_hull,
                    # Gather the concave hull results
                    cls.get_concave_hull_output,
                ),
            ),
            # All of the structures in the seed will be run using the MD
            # code selected, using the main model (M0)
            cls.run_md_seed,
            # Structures and energy predictions will be gathered and prepared
            # into a dataframe
            cls.gather_m0_md_results,
            # The structures from M0 will be evaluated using the committee models.
            cls.check_committee_results_calcjob,
            cls.gather_committee_results,
            if_(cls.check_extrapolation_enabled)(
                # Getting MACE descriptors for the structures obtained with MD.
                cls.get_descriptors_from_md_results,
            ),
            # According to the difference in error between the models either:
            # The original structure will be removed from D0, or
            # The problematic structure will be calcualated using DFT
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
        updated_path, _ = atl_al_ut.get_final_db_path(
            result_dir_path=self.inputs.results_dir.value,
            final_db_name=self.inputs.final_db_name.value,
            node=atl_al_ut.process_call_root(orm.load_node(self.uuid)),
        )

        database_training = atl_al_ut.load_database(self.inputs.training_db_path.value)

        # Generate new training data file
        atl_conv.gen_mace_train_structure_list(
            path=updated_path,
            structure_list=database_training,
        )

        # Train n models (M0-Mn)
        # The most accurate model (during validation) will be chosen as the main model,
        # and used to drive the MD simulations. The remaining models will act as
        # committee models and will only be used to evaluate energies.
        self.report(
            f'Training {self.inputs.committee_num_models.value} models using '
            'current iteration data.'
        )

        # Stop the calculation if initial models must be loaded
        if self.inputs.load_init_models and self.inputs.al_loop_iteration.value == 0:
            self.report(
                'Loading models from nodes: '
                f"'{self.inputs.load_init_models.get_list()}'."
            )
            return

        for _ in range(self.inputs.committee_num_models.value):
            model_name = atl_al_ut.generate_model_name()

            # Load training settings from inputs and update path and model names.
            mace_train_settings: orm.Dict = atl_al_ut.update_mace_train_settings_dict(
                settings_dict=self.inputs.mace_train.get('train_settings'),
                train_data_path=str(updated_path),
                curr_model=model_name,
                curr_iter=self.inputs.al_loop_iteration.value,
                db_size=len(database_training),
            )

            # Run training and save new model file
            mace_train = CalculationFactory('mace-train')
            mace_builder = mace_train.get_builder()

            mace_builder.model_name = model_name
            mace_builder.mace_settings_dict = orm.Dict(mace_train_settings)

            mace_train_file_path, _ = atl_al_ut.get_final_db_path(
                result_dir_path=self.inputs.results_dir.value,
                final_db_name=self.inputs.final_db_name.value,
                node=self.node,
            )
            mace_builder.mace_train_file_path = str(mace_train_file_path)
            code_str = self.inputs.mace_train.get_dict()['code']
            code = orm.load_code(code_str)
            computer = code.computer
            mace_builder.code = code
            mace_builder.metadata.options.withmpi = True
            mace_builder.metadata.options = self.inputs.mace_train.get_dict()[
                'metadata'
            ].get('options')
            mace_builder.metadata.options.output_filename = (
                f'train_{model_name}_iter-{self.inputs.al_loop_iteration.value}'
            )
            mace_builder.metadata.label = model_name

            # Get the calculation limit, from the computer metadata set to 0
            # if not present.
            # `atl_calc_limit` is a custom property set with:
            # computer.set_property(name='atl_calc_limit', value=366)
            calc_limit = computer.metadata.get('atl_calc_limit', 0)

            # Check if the calculation can be submitted
            if calc_limit == 0:
                can_submit = True
            else:
                can_submit = can_submit_calculation(
                    code=code_str,
                    limit=calc_limit,
                )

            # If the calculation cannot be submitted, wait for a minute and check again
            while not can_submit:
                time.sleep(60)
                can_submit = can_submit_calculation(
                    code=code_str,
                    limit=calc_limit,
                )

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
            The function updates the workchain context with the best model's RMSE
            values, the LAMMPS potential file, and the committee models' information
            but does not return any value directly.
        """
        curr_iter = self.inputs.al_loop_iteration.value
        if (not self.inputs.load_init_models) or (
            self.inputs.load_init_models and curr_iter != 0
        ):
            mace_training_results = self.ctx.mace_training_results
        else:
            mace_training_results = [
                orm.load_node(node) for node in self.inputs.load_init_models
            ]

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

                # Convert model to LAMMPS compatible format
                # and return it to workchain context
                self.ctx.lammps_potential_file = atl_al_ut.create_mace_lammps_model(
                    model_file
                )

                self.report(
                    f"Generated LAMMPS potential using '{model_name}' as M0 - "
                    f'RMSE E: {self.ctx.m0_rmse_e.value:.3f} meV/at, '
                    f'RMSE F: {self.ctx.m0_rmse_f.value:.3f} meV/Å'
                )
                self.out('m0_model_file', model_file)
            else:
                self.report(
                    f"Trained committee model '{model_name}' - "
                    f'RMSE E: {curr_calc.outputs.m_rmse_e.value:.3f} meV/at, '
                    f'RMSE F: {curr_calc.outputs.m_rmse_f.value:.3f} meV/Å'
                )
                commitee_models_tupl_name_uuid.append((model_name, model_file.uuid))

        # Sending committee model paths to current context
        self.ctx.commitee_models_tupl_name_uuid = commitee_models_tupl_name_uuid

    def check_extrapolation_enabled(self):
        """Check if the extrapolation check is enabled."""
        return bool(self.inputs.check_extrapolation_type.value)

    def is_advanced_extrapolation(self):
        """Check if the advanced extrapolation check is enabled."""
        return self.inputs.check_extrapolation_type.value == 'advanced'

    def has_latent_space(self):
        """Check if the latent space was computed for the current iteration."""
        return hasattr(self.ctx, 'latent_space')

    def can_do_advanced_extrapolation(self):
        """Check if the advanced extrapolation can be done."""
        return self.has_latent_space() and self.is_advanced_extrapolation()

    def generate_descriptors(self):
        """Generate descriptors for the current seed using the best model.

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
        mace_train_file_path, _ = atl_al_ut.get_final_db_path(
            result_dir_path=self.inputs.results_dir.value,
            final_db_name=self.inputs.final_db_name.value,
            node=self.node,
        )
        code_builder.mace_train_file_path = str(mace_train_file_path)

        if not code_builder.mace_train_file_path.is_stored:
            code_builder.mace_train_file_path.store()

        prepend_text = (
            self.inputs.descriptor_settings['metadata']['prepend_text']
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
                f'{ATL_ROOT_DIR}/active_learning/extrapolation/autoencoder_scripts'
            )
            code = orm.PortableCode(
                label='atl_get_latent_space',
                filepath_files=descriptor_code_path,
                filepath_executable='atl_autoencoder_get_latent_space.py',
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
                f'{ATL_ROOT_DIR}/active_learning/mace_code/descriptors'
            )
            code = orm.PortableCode(
                label='mace_get_descriptors',
                filepath_files=descriptor_code_path,
                filepath_executable='atl_mace_get_descriptors.py',
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
        # `atl_calc_limit` is a custom property set with:
        # computer.set_property(name='atl_calc_limit', value=366)
        calc_limit = computer.metadata.get('atl_calc_limit', 0)

        # Check if the calculation can be submitted
        if calc_limit == 0:
            can_submit = True
        else:
            can_submit = can_submit_calculation(
                computer=computer,
                code=code.label,
                limit=calc_limit,
            )
        # If the calculation cannot be submitted, wait for a minute and check again
        while not can_submit:
            time.sleep(60)
            can_submit = can_submit_calculation(
                computer=computer,
                code=code.label,
                limit=calc_limit,
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
            self.inputs.descriptor_settings['metadata']['prepend_text']
            + '\nPATH=$PATH:.'
        )

        # Set metadata options
        code_builder.metadata.options = self.inputs.descriptor_settings['metadata'][
            'options'
        ]

        # Set the computer
        code_builder.metadata.computer = orm.load_computer(
            self.inputs.descriptor_settings['metadata']['computer']
        )

        # Generate aiida code using the script in the `descriptor_code_path` folder.
        descriptor_code_path = Path(
            f'{ATL_ROOT_DIR}/active_learning/extrapolation/concave_hull_scripts'
        )

        code = orm.PortableCode(
            label='atl_get_concave_hull',
            filepath_files=descriptor_code_path,
            filepath_executable='atl_get_concave_hull.py',
            prepend_text=prepend_text,
        )
        code_builder.metadata.options.parser_name = 'mdb-get-concave-hull-parser'
        code_builder.metadata.label = self.ctx.best_model_name + '_concave_hull'

        code_builder.code = code

        # Get the calculation limit, from the computer metadata set to 0
        # if not present.
        # `atl_calc_limit` is a custom property set with:
        # computer.set_property(name='atl_calc_limit', value=366)
        calc_limit = code.computer.metadata.get('atl_calc_limit', 0)

        # Check if the calculation can be submitted
        if calc_limit == 0:
            can_submit = True
        else:
            can_submit = can_submit_calculation(
                code=code.label,
                limit=calc_limit,
            )

        # If the calculation cannot be submitted, wait for a minute and check again
        while not can_submit:
            time.sleep(60)
            can_submit = can_submit_calculation(
                code=code.label,
                limit=calc_limit,
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

    def gen_md_input(
        self,
        structure: Structure,
        potential_path: str,
        current_temp: float,
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
        current_temperature : float
            The temperature value (in Kelvin) to be used for setting initial velocities
            and maintaining simulation temperature

        Returns
        -------
        str
            The modified LAMMPS input file content as a string, ready for use in
            simulation.

        Notes
        -----
        - The template file is located in the `ATL_DATA_DIR/input_files` directory.
        - This function replaces placeholders in the template with actual values from
        the function's inputs and the structure's composition.
        - Future versions may include more dynamic options based on LAMMPS's extensive
        configurability.

        """
        with open(f'{ATL_DATA_DIR}/input_files/input.lammps') as f:
            lammps_template = f.read()

        if self.inputs.use_kokkos:
            lammps_template = 'newton on\n' + lammps_template

        lammps_template = lammps_template.replace(
            '$MACESTYLE', 'mace no_domain_decomposition'
        )

        species = structure.composition.elements

        # Setting MACE potential as the potential to use
        pair_coeff_str = '* * '
        pair_coeff_str += f'{Path(potential_path).name} '

        # Adding species from given structure
        for spec in species:
            pair_coeff_str += f'{spec} '
        lammps_template = lammps_template.replace('$PAIRCOEFF', pair_coeff_str)

        # Setting elements
        elem_str = ''
        for elem in species:
            elem_str += f'{elem} '

        lammps_template = lammps_template.replace('$ELEMS', elem_str)

        # Setting timestep size
        timestep_val = self.inputs.md_timestep_duration_ps.value
        lammps_template = lammps_template.replace('$TSTEP_SIZE', str(timestep_val))

        # Setting start and end temperature and the damping parameter.
        # The max T is calculated using a multiplier applied to the initial T.
        # The damping coefficient is computed as 100*dt as by the lammps docs,
        # see note in: https://docs.lammps.org/fix_nh.html#description
        temp_coeff = self.inputs.md_max_temp_multiplier.value
        temp_arr = f'{current_temp} {current_temp * temp_coeff} {100 * timestep_val}'
        lammps_template = lammps_template.replace('$TEMPARR', temp_arr)

        # Setting intial velocities.
        seed = np.random.randint(low=1, high=1000000)
        vel_str = f'{current_temp} {seed}'
        lammps_template = lammps_template.replace('$VELOCITY', vel_str)

        # Setting number of timesteps
        num_tstep_str = str(self.inputs.md_num_steps.value)
        lammps_template = lammps_template.replace('$NSTEPS', num_tstep_str)

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
        - This function assumes the availability of a trained MACE-LAMMPS potential file
        within the workflow's context.
        - Submitted calculation jobs are added to an AiiDA orm.Group for organization
        and are tagged with additional information to link them back to their respective
        positions in the database.
        """
        self.report('Running MD (using M0) for all structures in the current seed...')

        # Creating a orm.List in the context to store the nodes
        self.ctx.current_train_seed = []

        # Getting atom count cutoff value for structures to be considered large
        n_at_large = self.inputs.md_parameters.get('num_at_large_struct')

        # this string with the label used in the code setup.
        code_str = self.inputs.md_parameters.get('code')
        if self.inputs.use_kokkos:
            builder = CalculationFactory('mace-lammps-gpu-md').get_builder()
        else:
            builder = CalculationFactory('lammps.raw').get_builder()

        builder.code = orm.load_code(code_str)

        # Getting the lammps potential file in a temporary folder
        with self.ctx.lammps_potential_file.as_path() as lmp_pot_path:
            lmp_pot_filename = Path(lmp_pot_path).name
            lmp_pot_path = str(lmp_pot_path)

            # Setting the trajectory to be retrieved and the
            # potential file to be copied into the calculation folder
            builder_settings = {
                'additional_retrieve_list': ['structure.lammpstrj.gz'],
                'local_copy_list': [
                    (
                        self.ctx.lammps_potential_file.uuid,
                        lmp_pot_path,
                        lmp_pot_filename,
                    )
                ],
            }

            builder.settings = orm.Dict(builder_settings)

            with open(self.inputs.current_md_seed_structs_path.value, 'rb') as f:
                current_md_seed_structs = pickle.load(f)

            for idx, curr_structure in enumerate(current_md_seed_structs):
                # Structures are stored as a orm.Dict in order to be json-serializable
                for key in [
                    'pbc',
                    'cell',
                    'numbers',
                    'positions',
                    'forces',
                    'REF_forces',
                    'MACE_forces',
                ]:
                    if curr_structure.get(key):
                        curr_structure[key] = np.array(curr_structure[key])

                is_structure_large = False
                curr_structure = Atoms.fromdict(curr_structure)

                # Checking if the structure is considered 'large'
                if n_at_large and len(curr_structure) > n_at_large:
                    is_structure_large = True

                # Converting structure to pymatgen
                curr_structure = pmg_ase.AseAtomsAdaptor.get_structure(curr_structure)
                struct_properties = curr_structure.properties

                # Running a MD calculation for every T specified by the user
                for temp_val in self.inputs.md_temperature_list_K:
                    # TODO: Add to TOML. Check how to include parameters needed here as
                    # initial parameters (TOML)
                    curr_input = self.gen_md_input(
                        structure=curr_structure,
                        potential_path=lmp_pot_filename,
                        current_temp=temp_val,
                    )

                    script = orm.SinglefileData(io.StringIO(curr_input))
                    builder.script = script

                    lammps_struct_str = LammpsData.from_structure(
                        curr_structure, atom_style='atomic'
                    ).get_str()

                    data = orm.SinglefileData(io.StringIO(lammps_struct_str))
                    builder.files = {
                        'data': data,
                        'mace_potential': self.ctx.lammps_potential_file,
                    }
                    builder.filenames = {
                        'data': 'structure.lammps',
                        'mace_potential': lmp_pot_filename,
                    }

                    index_in_db = self.inputs.current_md_seed_structs_idx[idx]

                    # Loading metadata settings from workchain inputs
                    builder.metadata = self.inputs.md_parameters.get('metadata')
                    builder.metadata.label = (
                        f'struct_{index_in_db}_mace_lammps_md_{temp_val}_K'
                    )
                    builder.metadata.options.parser_name = 'mace-lammps-raw-parser'

                    # Changing the number of cores used for large structures
                    if is_structure_large:
                        n_cpus_large = self.inputs.md_parameters.get(
                            'num_cpus_large_struct'
                        )
                        if n_cpus_large is not None:
                            builder.metadata.options.resources[
                                'num_cores_per_mpiproc'
                            ] = int(n_cpus_large)

                    # Get the calculation limit, from the computer metadata set to 0
                    # if not present.
                    # `atl_calc_limit` is a custom property set with:
                    # computer.set_property(name='atl_calc_limit', value=366)
                    calc_limit = builder.code.computer.metadata.get('atl_calc_limit', 0)

                    # Check if the calculation can be submitted
                    if calc_limit == 0:
                        can_submit = True
                    else:
                        can_submit = can_submit_calculation(
                            code=builder.code.label,
                            limit=calc_limit,
                        )

                    # If the calculation cannot be submitted,
                    # wait for a minute and check again
                    while not can_submit:
                        time.sleep(60)
                        can_submit = can_submit_calculation(
                            code=builder.code.label,
                            limit=calc_limit,
                        )

                    # Submitting current calculation
                    future = self.submit(builder)

                    # Add calculation to the workchain's aiida orm.Group.
                    self.ctx.current_train_seed.append(future)
                    curr_group = orm.load_group(uuid=self.inputs.train_seed_group.value)
                    curr_group.add_nodes(future)

                    # Writing extra information that helps associating the calculation
                    # with its position on the database.
                    for key, val in struct_properties.items():
                        future.base.extras.set(key, val)
                    future.base.extras.set('index_in_db', index_in_db)
                    future.base.extras.set('md_temperature', temp_val)

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
        self.report('Gathering best model MD results for the current seed...')
        new_rows = []

        # Gathering all results
        for md_calc_wkch in self.ctx.md_seed_workchains:
            # Skipping MD calc if it hasn't finished correctly.
            if md_calc_wkch.exit_status != 0:
                continue

            md_wkch_res = md_calc_wkch.outputs.retrieved
            steps_E_F_arr = self.gather_energies_from_workchain(md_wkch_res)
            traj, forces = self.gather_traj_from_workchain(md_wkch_res)
            traj: Trajectory

            # Instead of keeping all frames, select some of them
            # Get 1 frame every n picoseconds of MD simulation
            traj, steps_E_F_arr, forces = atl_al_ut.select_md_frames_to_keep(
                frame_interval=self.inputs.al_keep_struct_every_n_ps,
                md_tstep_duration_ps=self.inputs.md_timestep_duration_ps.value,
                traj=traj,
                steps_E_F_arr=steps_E_F_arr,
                forces=forces,
            )

            energies = steps_E_F_arr[:, 1]

            # Use several filters to identify incorrect frames from MD trajectories.
            filters_structure_wrong_list = []

            # Using distance between layers to filter structures
            if self.inputs.md_filters and 'layer_distance' in self.inputs.md_filters:
                structure_wrong_list = []
                max_dist = self.inputs.md_filters['layer_distance'][
                    'max_layer_distance_ang'
                ]
                for frame in traj:
                    is_structure_wrong = atl_al_ut.apply_layer_distance_filter(
                        struct=frame, max_layer_distance_ang=max_dist
                    )
                    structure_wrong_list.append(is_structure_wrong)

                filters_structure_wrong_list.append(structure_wrong_list)

            # Checking if there are atoms with no neighbors (isolated atoms)
            if (
                self.inputs.md_filters
                and 'check_atoms_no_neighbor' in self.inputs.md_filters
            ):
                structure_wrong_list = []
                for frame in traj:
                    is_structure_wrong = atl_al_ut.apply_filter_no_neighbors(
                        struct=frame
                    )
                    structure_wrong_list.append(is_structure_wrong)

                filters_structure_wrong_list.append(structure_wrong_list)

            # Combine all filters to get the final list of wrong structures
            filters_structure_wrong_list = np.array(filters_structure_wrong_list)
            filters_structure_wrong_list = np.logical_or.reduce(
                filters_structure_wrong_list
            )

            # Removing incorrect structures from the trajectory by removing frames,
            # energies, and forces from the row to add to the DataFrame.
            if np.all(filters_structure_wrong_list):
                self.report(
                    'Completely removing trajectory as all frames are incorrect.'
                )
                continue
            if np.any(filters_structure_wrong_list):
                self.report(
                    f'Removing {len(np.nonzero(filters_structure_wrong_list)[0])} '
                    'incorrect MD frames.'
                )

                # Getting frames to keep
                traj_nones = [
                    1 if not is_wrong else None
                    for i, is_wrong in enumerate(filters_structure_wrong_list)
                ]
                frames_to_keep = np.nonzero(traj_nones)

                # Removing energies
                energies = [
                    energies[i] if not is_wrong else None
                    for i, is_wrong in enumerate(filters_structure_wrong_list)
                ]
                energies = list(np.array(energies)[frames_to_keep])

                # Removing forces
                forces = [
                    forces[i] if not is_wrong else np.full([len(traj[i]), 3], np.nan)
                    for i, is_wrong in enumerate(filters_structure_wrong_list)
                ]
                forces = list(np.array(forces)[frames_to_keep])

                # Removing frames from trajectory
                new_traj = [traj.get_structure(int(i)) for i in frames_to_keep[0]]
                traj = Trajectory.from_structures(new_traj)

            new_rows.append(
                {
                    'trajectory': traj,
                    'energy': {self.ctx.best_model_name: energies},
                    'forces': {self.ctx.best_model_name: forces},
                    'atl_al_step': self.inputs.al_loop_iteration.value,
                    'index_in_db': md_calc_wkch.base.extras.all['index_in_db'],
                    'atl_struct_type': md_calc_wkch.base.extras.all['atl_struct_type'],
                    'material_name': md_calc_wkch.base.extras.all['struct_name'],
                    'unique_id': md_calc_wkch.base.extras.all['aiida_uuid'],
                    'atl_md_node': str(md_calc_wkch.uuid),
                    'md_temperature': md_calc_wkch.base.extras.all['md_temperature'],
                    'extrapolation': np.nan,
                }
            )

        # Creating a DataFrame with all the results
        md_seed_results_df = pd.DataFrame(new_rows)

        self.ctx.results_dir = atl_al_ut.get_results_dir_path(
            result_dir_path=self.inputs.results_dir.value, node=self.node
        )
        # Saving the DataFrame into the result directory as a file.
        self.ctx.md_seed_results_df_path = (
            f'{self.ctx.results_dir}/run_tmp_data/md_seed_results_df_step-'
            f'{self.inputs.al_loop_iteration.value}.pkl'
        )
        md_seed_results_df.to_pickle(path=str(self.ctx.md_seed_results_df_path))

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
            a method `get_object_content` which can retrieve the content of 'lammps.out'

        Returns
        -------
        numpy.ndarray
            A 2D numpy array where each row corresponds to a step in the workchain.
            The columns represent step number, energy, and force, respectively.

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
        # Loading lammps output
        output = workchain_results.get_object_content('lammps.out')
        # Generating list with the logged steps data
        steps = [line for line in output.splitlines() if line.startswith('thermo ')]

        energy_array = []
        force_array = []
        step_array = []

        for step in steps:
            split = step.split()

            # Getting current step results.
            # If an IndexError is raised, probably one of the results is missing,
            # most likely because the program was stopped mid-step (maximum run time
            # exceeded), and thus the current step must not be gathered.
            # This allows for the AL # loop to keep running even if there are MD
            # calculations that don't run to completion.
            try:
                curr_step = int(split[1])
                curr_energy = float(split[4])
                curr_force = float(split[7])
            except IndexError as e:
                parent_calc = workchain_results.creator.pk
                self.report(
                    f'Error while gathering MD calculation results {parent_calc} - {e}.'
                )
                break

            step_array.append(curr_step)
            energy_array.append(curr_energy)
            force_array.append(curr_force)

        step_E_F_arr = np.stack((step_array, energy_array, force_array), axis=1)
        return step_E_F_arr

    def gather_traj_from_workchain(
        self, workchain_results: orm.FolderData
    ) -> Trajectory:
        """
        Extracts trajectory data from a `LammpsRawCalculation` as pymatgen Trajectory.

        This function parses `structure.lammpstrj.gz` from the given workchain
        results using ase.
        It then constructs a sequence of pymatgen Structure objects
        representing each frame of the trajectory which are combined
        into a pymatgen Trajectory object.

        Parameters
        ----------
        workchain_results : orm.FolderData
            A orm.FolderData containing the results of a workchain, expected to have
            a method `get_object_content` to retrieve the contents of
            `structure.lammpstrj.gz`.

        Returns
        -------
        Trajectory
            A pymatgen Trajectory object representing the structure over time.
        np.array
            A (n_frames x n x 3) numpy array containing the
            forces of every atom.

        Notes
        -----
        The user can change the `gather_traj_cnt_lattice` input to True/False to
        select if the lattice changes during the simulation.
        """
        # This flag selects if a constant lattice volume is assumed
        # for all frames.
        cnt_lat_setting = True

        # Get trajectory file from aiida repo node and
        # parse it file using ase
        with workchain_results.as_path() as results_path:
            ase_traj = ase_read(
                filename=Path(results_path) / 'structure.lammpstrj.gz',
                format='lammps-dump-text',
                index=':',
            )

        # Gather the forces
        forces_list = [atm.get_forces() for atm in ase_traj]

        # Convert ase atoms to pymatgen structures
        struct_list = [AseAtomsAdaptor().get_structure(atm) for atm in ase_traj]

        traj = Trajectory.from_structures(
            struct_list,
            constant_lattice=cnt_lat_setting,
            time_step=0.003,
        )

        return traj, np.array(forces_list)

    def check_committee_results_calcjob(self):
        """Gets predictions of all the committee models for the MD trajectories."""
        self.report('Evaluating trajectories with committee models...')

        # Gather all commitee models
        # Prepare all committee information in a dict
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
            best_model_name_clean = self.ctx.best_model_name.replace('-', '_')
            commitee_dict[best_model_name_clean] = self.ctx.best_model_file

        # Reading md seed results DataFrame
        md_seed_results_df: pd.DataFrame = pd.read_pickle(
            self.ctx.md_seed_results_df_path
        )

        # Checking committee predictions for every trajectory
        for row in md_seed_results_df.iterrows():
            curr_traj = row[1]['trajectory']

            # Run training and save new model file
            mace_train = CalculationFactory('mace-committee-eval')
            mace_builder = mace_train.get_builder()

            # Input committee models to `commitee_models` namespace as a dict like:
            # {"model_1": "/path/to/model_1/", "model_2": ...}
            mace_builder.commitee_models = commitee_dict

            curr_traj_frames_atoms = []

            # Converting all framesto ase.Atoms()
            for frame in curr_traj:
                curr_traj_frames_atoms.append(AseAtomsAdaptor.get_atoms(frame))

            # Write xyz file into a string captured in the stdout,
            # write it to a temporary file.
            f = io.StringIO()
            with redirect_stdout(f):
                ase_write(
                    filename='-',
                    format='extxyz',
                    images=curr_traj_frames_atoms,
                )
            xyz_string = f.getvalue()

            # Generating tmp file
            md_xyz_file = orm.SinglefileData(
                file=io.BytesIO(str.encode(xyz_string)),
                filename='md_db.xyz',
            )

            # Input configurations to evaluate
            mace_builder.configurations_to_evaluate = md_xyz_file

            # Input number of threads to use
            mace_builder.num_threads = int(self.inputs.committee_eval['openmp_threads'])

            # Gather mace evaluation settings
            mace_builder.mace_settings_dict = orm.Dict(
                self.inputs.committee_eval['mace']
            )

            # Get portable code
            descriptor_code_path = Path(
                f'{ATL_ROOT_DIR}/active_learning/mace_code/committee'
            )
            prepend_text = (
                self.inputs.descriptor_settings['metadata']['prepend_text']
                + '\nPATH=$PATH:.'
            )
            portable_code = orm.PortableCode(
                label='mace_get_descriptors',
                filepath_files=descriptor_code_path,
                filepath_executable='atl_mace_eval_committee_configs.py',
                prepend_text=prepend_text,
            )
            mace_builder.code = portable_code

            # Loading computer and removing it from the input dictionary
            mace_eval_aiida_settings_dict = self.inputs.committee_eval['metadata'][
                'options'
            ]
            computer = orm.load_computer(mace_eval_aiida_settings_dict['computer'])
            mace_builder.metadata.computer = computer
            mace_eval_aiida_settings_dict.pop('computer', None)

            # Load scheduler and resources options
            mace_builder.metadata.options = mace_eval_aiida_settings_dict

            # Get the calculation limit, from the computer metadata set to 0
            # if not present.
            # `atl_calc_limit` is a custom property set with:
            # computer.set_property(name='atl_calc_limit', value=366)
            calc_limit = mace_builder.metadata.computer.metadata.get(
                'atl_calc_limit', 0
            )

            # Check if the calculation can be submitted
            if calc_limit == 0:
                can_submit = True
            else:
                can_submit = can_submit_calculation(
                    computer=computer,
                    code=mace_builder.code.label,
                    limit=calc_limit,
                )

            # If the calculation cannot be submitted, wait for a minute and check again
            while not can_submit:
                time.sleep(60)
                can_submit = can_submit_calculation(
                    computer=computer,
                    code=mace_builder.code.label,
                    limit=calc_limit,
                )

            future = self.submit(mace_builder)
            future.base.extras.set('unique_id', row[1]['unique_id'])
            future.base.extras.set('md_temperature', row[1]['md_temperature'])
            self.to_context(committee_results=append_(future))

    def gather_committee_results(self):
        """Gather committee results for all trajectories."""
        self.report('Gathering committee E and F evaluation...')

        # # REMOVE: Testing only
        # md_seed_results_df.to_pickle("/tmp/md_seed_results_df")

        # Reading md seed results DataFrame
        md_seed_results_df: pd.DataFrame = pd.read_pickle(
            self.ctx.md_seed_results_df_path
        )

        for curr_calc in self.ctx.committee_results:
            # Skipping calculation if training hasn't finished correctly.
            if curr_calc.exit_status != 0:
                self.report(
                    f'Skipping calculation [{curr_calc.pk}]'
                    f' - exit status: {curr_calc.exit_status}'
                )
                continue

            # Gather extras to identify the current calc
            curr_unique_id = curr_calc.base.extras.all['unique_id']
            curr_md_temperature = curr_calc.base.extras.all['md_temperature']

            # Find row matching the calculation using curr_unique_id and
            # current_temperature
            row_index = md_seed_results_df[
                md_seed_results_df.unique_id == curr_unique_id
            ][md_seed_results_df.md_temperature == curr_md_temperature].index[0]

            # Collect energies from dict using model name
            energies_dict = curr_calc.outputs.energy_result_dict.get_dict()

            for model_name, energies_list in energies_dict.items():
                # Convering list of eneries into 1D ndarray
                energies = np.array(energies_list)

                # Updating current energy dict with the new results from every model
                md_seed_results_df.loc[[row_index], 'energy'][row_index][model_name] = (
                    energies
                )

            # Collect forces from dict using model name
            forces_collection = curr_calc.outputs.forces_result_dict.get_dict()
            for model_name, forces_model_list in forces_collection.items():
                forces_list = [forces for forces in forces_model_list]

                # Updating current forces dict with the new results from every model
                md_seed_results_df.loc[[row_index], 'forces'][row_index][model_name] = (
                    np.array(forces_list)
                )

        # Updating md seed results DataFrame
        md_seed_results_df.to_pickle(path=self.ctx.md_seed_results_df_path)

    def get_descriptors_from_md_results(self):
        """Get descriptors for the MD generated structures using the best model."""
        # Get the dimensionality reduction method
        dimensionality_reduction_method = self.inputs.descriptor_settings.get(
            'dimensionality_reduction_method'
        )

        if dimensionality_reduction_method == 'autoencoder':
            descr_calc = CalculationFactory('mdb-get-latent-space')
            self.report(
                'Getting latent space of descriptors for MD generated structures...'
            )
        else:
            # Prepare GetMACEDescriptorsCalculation
            descr_calc = CalculationFactory('mace-get-descriptors')
            self.report('Getting descriptors for MD generated structures...')

        code_builder = descr_calc.get_builder()
        code_builder.model_file = self.ctx.best_model_file

        # Reading md seed results DataFrame
        md_seed_results_df: pd.DataFrame = pd.read_pickle(
            self.ctx.md_seed_results_df_path
        )

        # Store all frames from the trajectory into a list
        for _, row in md_seed_results_df.iterrows():
            # all_frames_list = []
            curr_traj = row['trajectory']
            traj_frames = []

            for frame in curr_traj:
                curr_frame: Atoms = AseAtomsAdaptor.get_atoms(frame)
                curr_frame.info['aiida_uuid'] = row['unique_id']
                curr_frame.info['md_temperature'] = row['md_temperature']
                traj_frames.append(curr_frame)

            # Write xyz file into a string captured in the stdout,
            # write it to a temporary file.
            f = io.StringIO()
            with redirect_stdout(f):
                ase_write(
                    filename='-',
                    format='extxyz',
                    images=traj_frames,
                )
            xyz_string = f.getvalue()

            # Generating tmp file
            md_xyz_file = orm.SinglefileData(
                file=io.BytesIO(str.encode(xyz_string)),
                filename='md_db.xyz',
            )

            prepend_text = (
                self.inputs.descriptor_settings['metadata']['prepend_text']
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

            code_builder.model_file = self.ctx.best_model_file
            code_builder.mace_train_file_path = md_xyz_file
            code_builder.mace_train_file_path.store()

            # Get latent space of the descriptors using the autoencoder.
            if dimensionality_reduction_method == 'autoencoder':
                # Get autoencoder model
                code_builder.trained_autoencoder_model = self.ctx.autoencoder_model_file

                # Generate aiida code using the script in
                # the `descriptor_code_path` folder.
                descriptor_code_path = Path(
                    f'{ATL_ROOT_DIR}/active_learning/extrapolation/autoencoder_scripts'
                )
                code = orm.PortableCode(
                    label='atl_get_latent_space',
                    filepath_files=descriptor_code_path,
                    filepath_executable='atl_autoencoder_get_latent_space.py',
                    prepend_text=prepend_text,
                )
                code_builder.metadata.options.parser_name = (
                    'mdb-get-latent-space-parser'
                )
                code_builder.metadata.label = self.ctx.best_model_name + '_latent_space'

            else:
                # Prepare GetMACEDescriptorsCalculation
                descriptor_code_path = Path(
                    f'{ATL_ROOT_DIR}/active_learning/mace_code/descriptors'
                )

                code = orm.PortableCode(
                    label='mace_get_descriptors',
                    filepath_files=descriptor_code_path,
                    filepath_executable='atl_mace_get_descriptors.py',
                    prepend_text=prepend_text,
                )

                code_builder.metadata.options.parser_name = 'mace-descriptors-parser'

                code_builder.metadata.label = (
                    row['unique_id'][:8]
                    + '_md_descriptors_'
                    + f'{row["md_temperature"]}_K'
                )

                code_builder.metadata.options.output_filename = (
                    f'descriptors_{self.ctx.best_model_name}_iter'
                    f'-{self.inputs.al_loop_iteration.value}'
                )

            code_builder.code = code

            # Get the calculation limit, from the computer metadata set to 0
            # if not present.
            # `atl_calc_limit` is a custom property set with:
            # computer.set_property(name='atl_calc_limit', value=366)
            calc_limit = computer.metadata.get('atl_calc_limit', 0)

            # Check if the calculation can be submitted
            if calc_limit == 0:
                can_submit = True
            else:
                can_submit = can_submit_calculation(
                    computer=computer,
                    code=code.label,
                    limit=calc_limit,
                )

            # If the calculation cannot be submitted, wait for a minute and check again
            while not can_submit:
                time.sleep(60)
                can_submit = can_submit_calculation(
                    computer=computer,
                    code=code.label,
                    limit=calc_limit,
                )

            future = self.submit(code_builder)
            future.base.extras.set('unique_id', row['unique_id'])
            future.base.extras.set('md_temperature', row['md_temperature'])

            self.to_context(md_descriptor_results=append_(future))

    def send_calc_or_remove_structures(self):
        """Decide which structures to keep and send to DFT or remove from db."""
        self.report('Deciding which structures to keep...')

        model_acc_multiplier = self.inputs.model_acc_multiplier.value
        e_rmse = self.ctx.m0_rmse_e.value
        e_error_threshold = model_acc_multiplier * e_rmse

        f_rmse = self.ctx.m0_rmse_f.value
        f_error_threshold = model_acc_multiplier * f_rmse

        maximum_value_e = 1000  # meV
        maximum_value_f = 1000  # meV

        delete_indices = []
        dft_structures = []

        # Reading md seed results DataFrame
        md_seed_results_df: pd.DataFrame = pd.read_pickle(
            self.ctx.md_seed_results_df_path
        )

        # Gathering MD descriptor results and adding them to dataframe
        if self.inputs.check_extrapolation_type.value is not None:
            for curr_calc in self.ctx.md_descriptor_results:
                # Ignore failed calculations
                if not curr_calc.is_finished_ok:
                    continue

                curr_unique_id = curr_calc.base.extras.all['unique_id']
                curr_md_temperature = curr_calc.base.extras.all['md_temperature']

                dimensionality_reduction_method = self.inputs.descriptor_settings.get(
                    'dimensionality_reduction_method'
                )

                # Find row matching the calculation using curr_unique_id and
                # current_temperature
                row_index = md_seed_results_df[
                    md_seed_results_df.unique_id == curr_unique_id
                ][md_seed_results_df.md_temperature == curr_md_temperature].index[0]

                # Creating context manager to load descriptor result files
                # descr_file
                with (
                    curr_calc.outputs.descriptors_file.as_path() as md_descr_file_path,
                    open(md_descr_file_path, 'rb') as descr_file,  # noqa: E501
                ):
                    md_descr_dict: list[list[list]] = pickle.load(descr_file)

                # Gather latent space
                if dimensionality_reduction_method == 'autoencoder':
                    # Get latent space
                    desc_f_curr_row: dict = md_descr_dict[curr_unique_id]
                    latent_space = desc_f_curr_row['latent_space']

                    # Overwrite row in dataframe
                    md_seed_results_df.loc[[row_index], 'extrapolation'] = pd.Series(
                        [latent_space],
                        index=md_seed_results_df.index[[row_index]],
                    )

                # Gather descriptors
                else:
                    # Assign matching descriptors to the extrapolation column
                    desc_f_curr_row: list = md_descr_dict[curr_unique_id]

                    # Assign matching descriptors to the extrapolation column
                    # Overwrite row in dataframe
                    md_seed_results_df.loc[[row_index], 'extrapolation'] = pd.Series(
                        [desc_f_curr_row],
                        index=md_seed_results_df.index[[row_index]],
                    )

        else:
            self.report('Skipping extrapolation since check_extrapolation_type = None.')

        # Updating md seed results DataFrame
        md_seed_results_df.to_pickle(path=self.ctx.md_seed_results_df_path)

        # Every row contains the results of MD for a single structure, which are:
        # trajectory, energies, forces, al_step, index_in_db, atl_struct_type,
        # cluster, material_name, unique_id
        submitted_dft_cnt = 0
        for _, row in md_seed_results_df.iterrows():
            # Make len(traj) sized array filled with False.
            total_point_inside, total_point_outside = [], []
            extrapolating_frames = np.zeros(shape=len(row['trajectory']))

            # Use extrapolation based on descriptor ranges
            if self.inputs.check_extrapolation_type.value == 'basic':
                curr_struct_descr = row['extrapolation']

                # TODO: Check if this can be avoided
                # Safeguard check for filtered trajectories.
                # In some cases curr_struct_descr might be an np.nan value, which
                # will raise an error when trying to iterate over it.
                if isinstance(curr_struct_descr, float):
                    extrapolating_frames = np.zeros(shape=1)
                # If the current structure has no extrapolation data, fill the
                # extrapolating_frames array with zeros.
                else:
                    if len(extrapolating_frames) < len(curr_struct_descr):
                        extrapolating_frames = np.zeros(shape=len(row['extrapolation']))

                try:
                    # Checking if the frames for the current structure are extrapolating
                    for frame_idx, frame_descriptors in enumerate(curr_struct_descr):
                        below_min = frame_descriptors < self.ctx.descriptors_min_array
                        above_max = frame_descriptors > self.ctx.descriptors_max_array
                        is_frame_extrapolating = np.any(
                            np.logical_or(below_min, above_max)
                        )

                        # Change to True the ones that are extrapolating.
                        if is_frame_extrapolating:
                            extrapolating_frames[frame_idx] = 1
                except TypeError as e:
                    self.report(e)
                    pass

            # Use advanced extrapolation
            elif self.inputs.check_extrapolation_type.value == 'advanced':
                curr_struct_descr = row['extrapolation']

                for frame_idx, frame_descriptors in enumerate(curr_struct_descr):
                    # print("type frame_descriptors: ", type(frame_descriptors))
                    in_domain_check_Dict: orm.Dict = atl_al_ut.check_atom_in_domain(
                        concave_hull=self.ctx.concave_hull_array,
                        descriptors=frame_descriptors,
                    )
                    point_inside = in_domain_check_Dict.get_dict()['inside']
                    point_outside = in_domain_check_Dict.get_dict()['outside']
                    total_point_inside.extend(point_inside)
                    total_point_outside.extend(point_outside)

                    # If there are any points outside the domain, the frame is
                    # considered extrapolating.
                    is_frame_extrapolating = len(point_outside) > 0

                    # Change to True the ones that are extrapolating.
                    if is_frame_extrapolating:
                        extrapolating_frames[frame_idx] = 1

            self.report(
                f'Out of {len(extrapolating_frames)} frames, '
                f'{len(np.nonzero(extrapolating_frames)[0])} were found '
                'to be extrapolating.'
            )

            # Getting all energy predictions
            # TODO - Select std or variance
            model_energies_dict = row['energy']

            energies_stat = atl_al_ut.get_model_energies_std(model_energies_dict)

            # Checking if the energies are over the error threshold
            error_e_structures_sm = np.ma.make_mask(
                energies_stat >= e_error_threshold,
                shrink=False,
            )
            error_e_structures_bg = np.ma.make_mask(
                energies_stat < maximum_value_e,
                shrink=False,
            )

            # Any True value in this array is over the energy error threshold
            # and must be sent to calculate with DFT.
            error_e_structures = np.logical_and(
                error_e_structures_sm, error_e_structures_bg
            )

            model_forces_dict = row['forces']
            forces_std = atl_al_ut.get_model_forces_std(model_forces_dict)
            forces_std_norm = np.linalg.norm(forces_std, axis=2)
            forces_std_norm_max = np.amax(forces_std_norm, axis=1)

            # Checking if the forces are over the error threshold
            err_f_struct_sm = np.ma.make_mask(
                forces_std_norm_max >= f_error_threshold,
                shrink=False,
            )
            err_f_struct_bg = np.ma.make_mask(
                forces_std_norm_max < maximum_value_f,
                shrink=False,
            )

            # Any True value in this array is over the force error threshold
            # and must be sent to calculate with DFT.
            error_f_structures = np.logical_and(err_f_struct_sm, err_f_struct_bg)

            # Pad the extrapolating_frames array with zeros to match the
            # length of the error arrays
            # This prevents broadcasting errors when joining the arrays,
            # as they must have the same shape.
            # TODO: Find a safer way of doing this, probably once the final
            # extrapolation method is defined.
            if len(extrapolating_frames) < len(error_f_structures) or len(
                extrapolating_frames
            ) < len(error_e_structures):
                extrapolating_frames = np.pad(
                    extrapolating_frames,
                    (0, len(error_f_structures) - len(extrapolating_frames)),
                    'constant',
                    constant_values=(0),
                )

            # Joining both error masks to get a single True/False array marking
            # structures to be computed with True
            error_all_structures = np.logical_or(
                np.logical_or(error_e_structures, error_f_structures),
                extrapolating_frames,
            )

            # If all values in error_all_structures are false, delete the main
            # structure from D0.
            flag_no_error_structs = np.all(error_all_structures == 0)

            # The index of the structure to delete will
            # be added to a list, which will be used as a mask to select
            # which structures to remove outside of the loop.
            if flag_no_error_structs:
                delete_indices.append(row['unique_id'])
            elif not flag_no_error_structs:
                # If there are some structures to submit, select some of them and
                # mark them for DFT.
                struct_arr = error_all_structures

                if isinstance(error_all_structures, np.bool_):
                    struct_arr = np.ones_like(energies_stat)

                selected_high_error = np.nonzero(struct_arr)[0]

                dft_structures = []
                for struct in selected_high_error:
                    # Safeguard check for filtered trajectories. In some cases
                    # the selected structure might be out of bounds for the
                    # trajectory array if frames are removed.
                    with suppress(IndexError):
                        dft_structures.append(row['trajectory'][int(struct)])

                # REMOVE: For testing purposes.
                # TESTING
                # dft_structures = [row["trajectory"][0]]
                # print('dft_structures: ', dft_structures)

                # If dft_calc_limit is specified in the settings, use this value
                # as the limit for the number of calculations that can be submitted.
                dft_calc_limit = self.inputs.dft_calc_limit
                if dft_calc_limit:
                    dft_calc_limit = dft_calc_limit.value

                    # Use the limit to get a random selection of structures to submit
                    if len(dft_structures) > dft_calc_limit:
                        dft_structures = np.random.choice(
                            dft_structures, size=dft_calc_limit, replace=False
                        )

                mace_calcs_struct_list = []
                mace_calcs_idx_list = []

                for calc_idx, dft_struct in enumerate(dft_structures):
                    if self.inputs.dft_method == 'vasp':
                        builder = atl_al_ut.get_dft_calc_builder_vasp(
                            dft_struct,
                            row,
                            calc_idx,
                            self.inputs.train_seed_group.value,
                            dft_settings=self.inputs.dft_settings.get_dict(),
                        )

                        # Get the calculation limit, from the computer metadata set to 0
                        # if not present.
                        # `atl_calc_limit` is a custom property set with:
                        # computer.set_property(name='atl_calc_limit', value=366)
                        calc_limit = builder.metadata.computer.metadata.get(
                            'atl_calc_limit', 0
                        )

                        # Check if the calculation can be submitted
                        if calc_limit == 0:
                            can_submit = True
                        else:
                            can_submit = can_submit_calculation(
                                code=builder.code.label,
                                limit=calc_limit,
                            )

                        # If the calculation cannot be submitted,
                        # wait for a minute and check again
                        while not can_submit:
                            time.sleep(60)
                            can_submit = can_submit_calculation(
                                code=builder.code.label,
                                limit=calc_limit,
                            )

                        # Submitting current calculation
                        future = self.submit(builder)
                        future.base.extras.set('atl_calc_uuid', row['unique_id'])
                        future.base.extras.set(
                            'atl_struct_type', row['atl_struct_type']
                        )
                        future.base.extras.set('struct_name', row['material_name'])
                        self.to_context(dft_struct_seed_calcs=append_(future))

                        if self.inputs.train_seed_group.value:
                            group = orm.load_group(self.inputs.train_seed_group.value)
                            group.add_nodes(future)

                    elif self.inputs.dft_method == 'mace':
                        mace_calcs_struct_list.append(dft_struct)
                        mace_calcs_idx_list.append(calc_idx)

                    submitted_dft_cnt += 1

                if self.inputs.dft_method == 'mace' and len(mace_calcs_struct_list) > 0:
                    builder = atl_al_ut.get_dft_calc_builder_mace_list(
                        struct_list=mace_calcs_struct_list,
                        row=row,
                        dft_settings=self.inputs.dft_settings.get_dict(),
                    )

                    # Get the calculation limit, from the computer metadata set to 0
                    # if not present.
                    # `atl_calc_limit` is a custom property set with:
                    # computer.set_property(name='atl_calc_limit', value=366)
                    calc_limit = builder.code.computer.metadata.get('atl_calc_limit', 0)

                    # Check if the calculation can be submitted
                    if calc_limit == 0:
                        can_submit = True
                    else:
                        can_submit = can_submit_calculation(
                            computer=builder.code.computer,
                            code=builder.code.label,
                            limit=calc_limit,
                        )

                    # If the calculation cannot be submitted,
                    # wait for a minute and check again
                    while not can_submit:
                        time.sleep(60)
                        can_submit = can_submit_calculation(
                            computer=builder.code.computer,
                            code=builder.code.label,
                            limit=calc_limit,
                        )

                    # Submitting current calculation
                    future = self.submit(builder)
                    future.base.extras.set('atl_calc_uuid', row['unique_id'])
                    future.base.extras.set('atl_struct_type', row['atl_struct_type'])
                    future.base.extras.set('struct_name', row['material_name'])
                    future.base.extras.set('atl_md_node', row['atl_md_node'])

                    self.to_context(dft_struct_seed_calcs=append_(future))

                    if self.inputs.train_seed_group.value:
                        group = orm.load_group(self.inputs.train_seed_group.value)
                        group.add_nodes(future)

        if self.inputs.check_extrapolation_type.value == 'advanced':
            self.report('Plotting extrapolation check results...')
            atl_al_ut.plot_concave_hull(
                point_inside=np.array(total_point_inside),
                point_outside=np.array(total_point_outside),
                concave_hull=self.ctx.concave_hull_array,
                latent_space=self.ctx.latent_space,
            )

        self.report(
            f'Committee decision: {submitted_dft_cnt} get info / '
            f'{len(delete_indices)} delete.'
        )

        # Deleting well represented structures from seed_gen_db (Ds), if
        # there are any and the seed deletion is enabled in the configuration
        # file.
        delete_seed_structs: bool = self.inputs.delete_seed_structs.value
        if len(delete_indices) > 0 and delete_seed_structs:
            self.report(
                f'Deleting {len(delete_indices)} structures from seed'
                ' generating DB (Ds)'
            )

            atl_al_ut.remove_structs_from_seed_gen_db(
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
        # TODO: Adapt for VASP usage
        if self.inputs.dft_method == 'vasp':
            try:
                dft_calcs = len(self.ctx.dft_struct_seed_calcs)
                self.report(f'Gathered {dft_calcs} VASP DFT calculations.')

                atl_al_ut.gather_dft_calcs_vasp(
                    [node.uuid for node in self.ctx.dft_struct_seed_calcs]
                )

            except AttributeError:
                orm.List([])

        elif self.inputs.dft_method == 'mace':
            try:
                dft_calcs = len(self.ctx.dft_struct_seed_calcs)
                self.report(f'Gathered {dft_calcs} MACE evaluations.')

                calc_list = [node.uuid for node in self.ctx.dft_struct_seed_calcs]

                # Gather all MACE evaluations, storing results into a file,
                # stored in `result_list_path`.
                # Results are filtered to remove outliers. Outliers are
                # stored in a separate file in the same folder.
                return_list_path: str = atl_al_ut.gather_dft_calcs_mace(
                    dft_calc_list=calc_list,
                    results_dir=str(self.ctx.results_dir),
                    workchain=self.node.uuid,
                )

            except AttributeError:
                return_list_path = ''

        # File containing structures
        if return_list_path:
            self.out('dft_calcs_path', return_list_path)

        # orm.SinglefileData for the MACE model with the best performance
        self.out('m0_model_file', self.ctx.best_model_file)

        self.out(
            'stop_md_seed_no_disagreement',
            atl_al_ut.check_md_seed_agreement(return_list_path),
        )


class ActiveLearningBaseWorkChain(BaseRestartWorkChain):
    """Base workchain for ATL active learning workflows.

    This workchain is used as a base for the `ActiveLearningWorkChain` workchain.
    It handles setup of the workchain and the main loop, where the active learning
    steps are launched. After every step, the results are checked and added to the
    database, and the next step is prepared. The workchain will loop until the
    stopping conditions are met.

    It takes all the inputs of the `ActiveLearningWorkChain` workchain, except for
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

    _process_class = ActiveLearningWorkChain

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)

        spec.expose_inputs(
            ActiveLearningWorkChain,
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
            cls.log_atl_version,
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
                # Get random structures from Ds to generate the MD seed.
                cls.get_md_seed,
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

        # Create a file handle
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(1)
        aiida_logger.addHandler(file_handler)

        # Adding console logger
        console = Console(color_system='truecolor')
        ch = RichHandler(
            markup=True,
            show_path=False,
            log_time_format='[%m/%d/%y %H:%M:%S]',
            omit_repeated_times=False,
            console=console,
        )
        ch.setLevel(23)
        formatter_con = logging.Formatter('%(message)s')
        ch.setFormatter(formatter_con)
        aiida_logger.addHandler(ch)

        self.report(f"Logging in '{log_path}'")

    def log_atl_version(self):
        curr_version, _, hash_str = get_atl_version_info()
        self.report(f"Using ATLAS version: '{curr_version}' ({hash_str}).")

    def get_database(self):
        """Loading initial database."""
        self.report('Reading database file...')

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

        # Create files for database_training and seed_gen_db
        results_dir_path = Path(self.inputs.active_learning.results_dir.value)
        if not results_dir_path.exists():
            results_dir_path.mkdir()

        final_db_path, curr_run_results_dir = atl_al_ut.get_final_db_path(
            result_dir_path=results_dir_path,
            final_db_name=self.inputs.active_learning.final_db_name.value,
            node=self.node,
        )
        self.ctx.curr_run_results_dir = curr_run_results_dir

        # A copy of the initial database, (Ds)
        # used specifically for generating MD seeds and running the MDs.
        # New structures will be added and well represented configs removed from here.
        self.ctx.seed_db_path = curr_run_results_dir / 'atl_seed_db.xyz'

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
        if self.inputs.active_learning.seed_size_frac.value:
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
        if self.inputs.resume_dict:
            last_iteration = self.inputs.resume_dict['last_iteration']
            node = self.ctx.children[self.ctx.iteration - (last_iteration - 1) - 2]
        else:
            node = self.ctx.children[self.ctx.iteration - 1]

        # TODO: Gather outputs manually, instead of using __attach_outputs
        # outputs = self._attach_outputs(node)
        # Sending seed disagreement flag to context
        try:
            stop_seed_no_disagreement = node.outputs['stop_md_seed_no_disagreement']
        except Exception:
            # TODO: This happens when the workchain is not finished correctly.
            # We could add a flag here
            stop_seed_no_disagreement = orm.Bool(False)
            # self.ctx.loop_error = orm.Bool(True)

        self.ctx.stop_md_seed_no_disagreement = stop_seed_no_disagreement
        self.ctx.last_workchain_completed = node
        return None

    def add_dft_results_to_db(self):
        """
        Incorporate DFT calculation results into the training/seed generation databases.

        This method updates the training and seed generation databases with DFT
        calculation results. If any DFT calculations have been performed,
        their results are appended to both the training database and the seed
        generation database.
        """
        # Updating current training seed
        try:
            seed_gen_db = atl_al_ut.load_database(self.ctx.seed_db_path)
            training_db = atl_al_ut.load_database(self.ctx.training_db_path)
            last_wc = self.ctx.last_workchain_completed
            dft_calcs = ase_read(
                last_wc.outputs['dft_calcs_path'].value, format='extxyz', index=':'
            )

            cnt_dft_calcs = len(dft_calcs)

        except KeyError:
            cnt_dft_calcs = 0

        if cnt_dft_calcs > 0:
            self.report(f'Adding {cnt_dft_calcs} DFT calculations to DB.')

            # Adding calculations to training database and seed_generation database
            for dft_calc in dft_calcs:
                # Converting serialized structures to Atoms object.
                if isinstance(dft_calc, dict):
                    dft_calc = atl_al_ut.aiida_serialized_ase_dict_to_atoms(dft_calc)

                seed_gen_db.append(dft_calc)
                training_db.append(dft_calc)

            # Updating final and seed database.
            self.report('Updating database files...')

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
        seed_gen_db = atl_al_ut.load_database(self.ctx.inputs.seed_db_path)
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
            ActiveLearningWorkChain, 'active_learning'
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

        seed_gen_db = atl_al_ut.load_database(self.ctx.seed_db_path)

        # Get seed selection type
        seed_select_settings = self.ctx.inputs.seed_select_settings

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
            seed_gen_db = [s for s in seed_gen_db if len(s) <= max_size]
            if len(seed_gen_db) == 0:
                raise atl_excp.FilterError(
                    f'There are no structures with less than '
                    f'{max_size} atoms in the given database. '
                    "Please, remove the 'small_first' seed selection mode. "
                    'and try again.'
                )

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

        # Choosing structures at random to create the training seed
        selected_structs = np.random.choice(
            range(db_length),
            size=seed_size,
            replace=False,
        )

        self.ctx.inputs.current_md_seed_structs_idx = list(selected_structs)

        # The set of random structures selected from the seed generation
        # database to be used in training.
        current_md_seed_structs = []

        # Populating training seed with the selected random structures
        for idx in selected_structs:
            current_md_seed_structs.append(seed_gen_db[idx])

        self.report(
            f'Created MD seed with {seed_size}'
            f' structures ({(seed_size / db_length) * 100:.1f}% of '
            'current database size).'
        )

        # Adding current train seed to the context
        current_MD_seed_serialized = []
        for curr_s in current_md_seed_structs:
            curr_s = atl_al_ut.serialize_ase(curr_s)
            current_MD_seed_serialized.append(curr_s)

        current_md_seed_structs = current_MD_seed_serialized

        # Saving the current md seed into the result directory as a file.
        self.ctx.results_dir = atl_al_ut.get_results_dir_path(
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
        train_db = atl_al_ut.prepare_output_final_training_db(
            training_db_path=self.ctx.inputs.training_db_path
        )

        self.out('final_training_db', train_db)

        # Returning final model as orm.SinglefileData object
        final_model_singlefile = self.ctx.last_workchain_completed.outputs[
            'm0_model_file'
        ]
        self.out(
            'final_model_file',
            self.ctx.last_workchain_completed.outputs['m0_model_file'],
        )

        target_file_name = f'al_loop_{self.inputs.active_learning.run_name.value}.model'
        target_file_path = self.ctx.curr_run_results_dir / target_file_name

        with (
            final_model_singlefile.open(mode='rb') as source,
            open(target_file_path, mode='wb') as target,
        ):
            shutil.copyfileobj(source, target)

        self.report('Workchain completed!')
