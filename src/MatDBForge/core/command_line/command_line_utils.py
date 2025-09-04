#!/usr/bin/env python
"""Run an active-learning procedure based on ML-MD using aiida."""

import pathlib as pl
import sys
import tomllib
import warnings

import yaml

from MatDBForge.core import MDB_DATA_DIR
from MatDBForge.core import exceptions as mdb_exc
from MatDBForge.core.code_utils import custom_print

warnings.filterwarnings('ignore', module='paramiko')


MDB_LOGO = """
  __  __      _   ___  ___ ___
 |  \/  |__ _| |_|   \| _ ) __|__ _ _ __ _ ___
 | |\/| / _` |  _| |) | _ \ _/ _ \ '_/ _` / -_)
 |_|  |_\__,_|\__|___/|___/_|\___/_| \__, \___|
                                     |___/
"""


def parse_input_toml(toml_dict: dict, type: str):
    """
    Parses and validates the input TOML dictionary based on the specified type.

    Parameters
    ----------
    toml_dict : dict
        The input dictionary parsed from a TOML file.
    type : str
        The type of configuration to validate. Currently supports "active_learning".

    Raises
    ------
    MissingMandatoryParameterError
        If any mandatory keys are missing from the input TOML dictionary.
    """
    if type == 'active_learning':
        mandatory_keys_list = ['active_learning', 'md', 'committee_eval', 'dft']

    elif type == 'generate_database':
        mandatory_keys_list = ['database', 'phase_diagram', 'generation']

    for key in mandatory_keys_list:
        if key not in list(toml_dict.keys()):
            raise mdb_exc.MissingMandatoryParameterError(
                f'Input toml file missing mandatory key: {key}.'
            )


def validate_config_file(
    config_type,
    config_dict=None,
    config_path=None,
    allow_deprecated=False,
    run_mode='workflow',
):
    """
    Validate a TOML configuration file against the schema.

    Parameters
    ----------
        config_path: str
            Path to the TOML configuration file
        config_dict: dict
            The loaded configuration dictionary
        config_type: str
            Type of configuration.
            One of: database_generation, dft, active_learning)
        schema: dict
            The complete configuration schema
        allow_deprecated: bool
            If True, deprecated keys will only generate warnings
        run_mode: str
            Either 'workflow' or 'script'. Workflow is for function runs at
            the start of a mdb workflow, before proceeding. 'script' is for standalone
            verification script execution.

    Returns
    -------
        tuple: (is_valid, errors)
            where is_valid is bool and errors is list of strings
    """
    if config_dict is None and config_path is None:
        return False, ['Either config_dict or config_path must be provided']

    if config_path is not None:
        try:
            # Load the TOML file
            config_path = pl.Path(config_path)
            with open(config_path, 'rb') as f:
                config_data = tomllib.load(f)
        except FileNotFoundError:
            return False, [f'Configuration file not found: {config_path}']
        except Exception as e:
            return False, [f'Error reading TOML file: {e}']
    elif config_dict is not None:
        config_data = config_dict

    # Get the schema for this config type
    schema = get_schema()
    config_schema = schema.get(config_type)
    if not config_schema:
        return False, [f'Unknown configuration type: {config_type}']

    custom_print('Validating TOML input file...', print_type='info')

    # Check for deprecated keys and migrate them
    migrated_config, deprecation_warnings = check_deprecated_keys(
        config_data, config_schema
    )

    # Show deprecation warnings
    if deprecation_warnings:
        for warning in deprecation_warnings:
            custom_print(warning, print_type='warning')

    # Validate the (possibly migrated) configuration
    errors = validate_section_recursive(
        migrated_config, config_schema, root_config_data=migrated_config
    )

    # Add deprecation warnings as validation errors to fail validation
    if deprecation_warnings and not allow_deprecated:
        errors.extend(
            [
                'Configuration contains deprecated keys. '
                'Please update your file to use the new key names.'
            ]
        )

    if len(errors) == 0:
        custom_print('TOML input file is valid!', print_type='done')
        return len(errors) == 0, errors
    else:
        custom_print('TOML input file validation failed:', print_type='error')
        for error in errors:
            custom_print(f'  • {error}', print_type='error')

        if run_mode == 'workflow':
            custom_print(
                'Process terminated due to validation errors.', print_type='error'
            )
            sys.exit(1)


def check_deprecated_keys(config_data, schema_dict, path='', warnings=None):
    """
    Check for deprecated keys in configuration and provide migration suggestions.

    Args:
        config_data (dict): The configuration data to check
        schema_dict (dict): The schema definition
        path (str): Current path in the configuration
        warnings (list): List to accumulate deprecation warnings

    Returns
    -------
        dict: Migrated configuration data with deprecated keys mapped to new ones
        list: List of deprecation warnings
    """
    if warnings is None:
        warnings = []

    if not isinstance(config_data, dict):
        return config_data, warnings

    migrated_data = config_data.copy()

    # Create reverse lookup for deprecated aliases (section level)
    deprecated_section_map = {}
    for section_name, section_schema in schema_dict.items():
        if isinstance(section_schema, dict):
            aliases = section_schema.get('deprecated_aliases', [])
            for alias in aliases:
                deprecated_section_map[alias] = section_name

    # Check for deprecated section names
    for old_key, new_key in deprecated_section_map.items():
        if old_key in migrated_data:
            full_old_path = f'{path}.{old_key}' if path else old_key
            full_new_path = f'{path}.{new_key}' if path else new_key

            warnings.append(
                f"Deprecated section '{full_old_path}' found. "
                f"Please rename to '{full_new_path}'"
            )

            # Migrate the data
            if new_key not in migrated_data:
                migrated_data[new_key] = migrated_data[old_key]
            del migrated_data[old_key]

    # Check for deprecated parameter names within sections
    for section_name, section_data in migrated_data.items():
        if isinstance(section_data, dict) and section_name in schema_dict:
            section_schema = schema_dict[section_name]
            if isinstance(section_schema, dict) and 'type' not in section_schema:
                # This is a section, check its parameters
                section_path = f'{path}.{section_name}' if path else section_name
                migrated_section, section_warnings = check_deprecated_parameters(
                    section_data, section_schema, section_path
                )
                migrated_data[section_name] = migrated_section
                warnings.extend(section_warnings)

                # Recursively check nested sections
                migrated_nested, nested_warnings = check_deprecated_keys(
                    migrated_section, section_schema, section_path
                )
                migrated_data[section_name] = migrated_nested
                warnings.extend(nested_warnings)

    return migrated_data, warnings


def check_deprecated_parameters(config_data, schema_dict, path=''):
    """
    Check for deprecated parameter names within a section.

    Args:
        config_data (dict): The configuration data to check
        schema_dict (dict): The schema definition for this section
        path (str): Current path in the configuration

    Returns
    -------
        dict: Migrated configuration data
        list: List of deprecation warnings
    """
    warnings = []
    migrated_data = config_data.copy()

    # Create reverse lookup for deprecated parameter aliases
    deprecated_param_map = {}
    for param_name, param_schema in schema_dict.items():
        if isinstance(param_schema, dict) and 'type' in param_schema:
            aliases = param_schema.get('deprecated_aliases', [])
            for alias in aliases:
                deprecated_param_map[alias] = param_name

    # Check for deprecated parameter names
    for old_param, new_param in deprecated_param_map.items():
        if old_param in migrated_data:
            full_old_path = f'{path}.{old_param}'
            full_new_path = f'{path}.{new_param}'

            warnings.append(
                f"Deprecated parameter '{full_old_path}' found. "
                f"Please rename to '{full_new_path}'"
            )

            # Migrate the data
            if new_param not in migrated_data:
                migrated_data[new_param] = migrated_data[old_param]
            del migrated_data[old_param]

    return migrated_data, warnings


def should_validate_mandatory(schema_item, config_data, root_config_data):
    """
    Determine if a schema item should be validated as mandatory based on dependencies.

    Args:
        schema_item (dict): Schema definition for parameter or section
        config_data (dict): Current section's config data
        root_config_data (dict): Complete configuration data

    Returns
    -------
        bool: True if item should be validated as mandatory
    """
    if not schema_item.get('mandatory', True):
        return False

    depends_on = schema_item.get('depends_on')

    if depends_on:
        return evaluate_dependency(depends_on, config_data, root_config_data)

    return True


def evaluate_dependency(depends_on, config_data, root_config_data):
    """
    Evaluate whether a dependency condition is met.

    Args:
        depends_on (dict): The dependency specification from schema
        config_data (dict): Current section's config data
        root_config_data (dict): Complete configuration data for resolving paths

    Returns
    -------
        bool: True if dependency condition is met, False otherwise
    """
    if not depends_on:
        return True

    key_path = depends_on.get('key', '')
    expected_value = depends_on.get('value')

    if not key_path:
        return True

    # Navigate to the dependency value in config
    path_parts = key_path.split('.')
    current_data = root_config_data

    try:
        for part in path_parts:
            if isinstance(current_data, dict):
                current_data = current_data.get(part)
            else:
                return False

        # Check if the actual value matches the expected value
        return current_data == expected_value

    except (KeyError, TypeError, AttributeError):
        # If we can't find the dependency path, assume dependency is not met
        return False


def validate_parameter(value, param_key, param_schema, path, root_config_data=None):
    """
    Validate a single parameter against its schema definition.

    Args:
        value: The value to validate
        param_key (str): The parameter key name
        param_schema (dict): The parameter schema definition
        path (str): The current path in the configuration for error reporting
        root_config_data (dict): Complete configuration data for dependency resolution

    Returns
    -------
        list: List of validation error messages (empty if valid)
    """
    errors = []
    full_path = f'{path}.{param_key}' if path else param_key

    # Check if parameter is mandatory but missing
    if value is None:
        # Check if parameter is actually mandatory considering dependencies
        if should_validate_mandatory(param_schema, {}, root_config_data or {}):
            errors.append(f'Missing mandatory parameter: {full_path}')
        return errors

    # Type validation
    expected_type = param_schema.get('type')
    if expected_type:
        if expected_type == 'str' and not isinstance(value, str):
            errors.append(
                f'Parameter {full_path}: expected string, got {type(value).__name__}'
            )
        elif expected_type == 'int' and not isinstance(value, int):
            errors.append(
                f'Parameter {full_path}: expected integer, got {type(value).__name__}'
            )
        elif expected_type == 'float' and not isinstance(value, (int, float)):
            errors.append(
                f'Parameter {full_path}: expected float, got {type(value).__name__}'
            )
        elif expected_type == 'bool' and not isinstance(value, bool):
            errors.append(
                f'Parameter {full_path}: expected boolean, got {type(value).__name__}'
            )
        elif expected_type.startswith('list[') and not isinstance(value, list):
            errors.append(
                f'Parameter {full_path}: expected list, got {type(value).__name__}'
            )
        elif expected_type == 'dict' and not isinstance(value, dict):
            errors.append(
                f'Parameter {full_path}: expected dict, got {type(value).__name__}'
            )

    # Choice validation
    choices = param_schema.get('choices')
    if choices:
        if isinstance(value, list):
            # For lists, validate each element against choices
            for item in value:
                if item not in choices:
                    errors.append(
                        f"Parameter {full_path}: list item '{item}' not in "
                        f'allowed choices {choices}'
                    )
        else:
            # For non-list values, validate the value directly
            if value not in choices:
                errors.append(
                    f"Parameter {full_path}: value '{value}' not in "
                    f'allowed choices {choices}'
                )

    return errors


def validate_section_recursive(
    config_data,
    schema_dict,
    path='',
    errors=None,
    section_mandatory=True,
    root_config_data=None,
):
    """
    Recursively validate a configuration section against its schema.

    Args:
        config_data (dict): The configuration data to validate
        schema_dict (dict): The schema definition for this section
        path (str): Current path in the configuration
        errors (list): List to accumulate validation errors
        section_mandatory (bool): Whether the current section is mandatory
        root_config_data (dict): Complete configuration data for dependency resolution

    Returns
    -------
        list: List of validation error messages
    """
    if errors is None:
        errors = []

    # Set root_config_data to the current config_data if not provided
    if root_config_data is None:
        root_config_data = config_data

    if config_data is None:
        if section_mandatory:
            errors.append(f'Missing mandatory section: {path}')
        return errors

    # Separate parameters from nested sections
    params = {
        k: v for k, v in schema_dict.items() if isinstance(v, dict) and 'type' in v
    }
    sections = {
        k: v for k, v in schema_dict.items() if isinstance(v, dict) and 'type' not in v
    }

    # Validate parameters
    for param_key, param_schema in params.items():
        # Handle flattened parameters (dotted notation)
        if '.' in param_key:
            # This should be handled by the flattened section logic
            continue

        value = config_data.get(param_key)
        param_errors = validate_parameter(
            value, param_key, param_schema, path, root_config_data
        )
        errors.extend(param_errors)

    # Handle flattened sections
    for section_name, section_schema in sections.items():
        if section_schema.get('flatten'):
            # For flattened sections, look for dotted keys in config_data
            schema_content = section_schema.get('schema', {})
            for sub_key, sub_schema in schema_content.items():
                if isinstance(sub_schema, dict) and 'type' in sub_schema:
                    flattened_key = f'{section_name}.{sub_key}'
                    value = config_data.get(flattened_key)
                    param_errors = validate_parameter(
                        value, flattened_key, sub_schema, path, root_config_data
                    )
                    errors.extend(param_errors)
            continue

        # Handle dynamic key sections
        if section_schema.get('dynamic_keys'):
            # Look for keys that match the pattern
            section_prefix = f'{section_name}.'
            for config_key, config_value in config_data.items():
                if config_key.startswith(section_prefix) and isinstance(
                    config_value, dict
                ):
                    # This is a dynamic key entry
                    new_path = f'{path}.{config_key}' if path else config_key
                    is_section_mandatory = (
                        section_schema.get('mandatory', True) and section_mandatory
                    )
                    validate_section_recursive(
                        config_value,
                        section_schema.get('schema', {}),
                        new_path,
                        errors,
                        is_section_mandatory,
                        root_config_data,
                    )
            continue

        # Regular nested section
        section_data = config_data.get(section_name)
        new_path = f'{path}.{section_name}' if path else section_name

        # Check if this section should be mandatory considering dependencies
        is_section_mandatory = (
            should_validate_mandatory(section_schema, config_data, root_config_data)
            and section_mandatory
        )

        validate_section_recursive(
            section_data,
            section_schema,
            new_path,
            errors,
            is_section_mandatory,
            root_config_data,
        )

    return errors


def get_schema():
    # Load the master schema file
    schema_path = pl.Path(MDB_DATA_DIR) / 'config_schema.yaml'
    try:
        with open(schema_path) as f:
            schema = yaml.safe_load(f)
    except FileNotFoundError:
        custom_print(f'Schema file not found at {schema_path}', print_type='error')
        sys.exit(1)
    except yaml.YAMLError as e:
        custom_print(
            f'Failed to parse schema file {schema_path}: {e}', print_type='error'
        )
        sys.exit(1)

    return schema
