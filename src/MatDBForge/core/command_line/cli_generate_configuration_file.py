"""Generate default configuration files for MDB from a YAML schema."""

import argparse
import pathlib as pl
import sys
from argparse import RawTextHelpFormatter

import yaml

from MatDBForge.core import MDB_DATA_DIR
from MatDBForge.core.code_utils import custom_print, init_logger
from MatDBForge.core.command_line.command_line_utils import MDB_LOGO


def format_value(value):
    """Formats a Python value into a TOML-compatible string."""
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return f'[{", ".join(format_value(v) for v in value)}]'
    return str(value)


def generate_toml_recursive(
    schema_dict, path_parts, lines, prefix='', section_mandatory=True
):
    """
    Recursively traverses the schema dictionary to generate TOML content.

    Args:
        schema_dict (dict): The current section of the schema to process.
        path_parts (list): The list of keys representing the current path in the TOML
                           structure.
        lines (list): The list of lines to which the generated TOML content will
                      be appended.
        prefix (str): A prefix for keys when flattening a section.
        section_mandatory (bool): Whether the current section is mandatory.
    """
    # Separate parameters (leaves) from nested sections (branches)
    params = {
        k: v for k, v in schema_dict.items() if isinstance(v, dict) and 'type' in v
    }
    sections = {
        k: v for k, v in schema_dict.items() if isinstance(v, dict) and 'type' not in v
    }

    # Generate TOML for parameters at the current level
    if params:
        # Add a table header if we are in a nested section and not flattening
        if path_parts and not prefix:
            header = f'[{".".join(path_parts)}]'
            if not section_mandatory:
                header = f'# {header}'
            lines.append(f'\n{header}')

        for key, details in params.items():
            lines.append('')
            if 'description' in details:
                comment = f'# {details["description"]}'
                if not section_mandatory:
                    comment = f'# {comment}'
                lines.append(comment)
            if 'type' in details:
                comment = f'# type: {details["type"]}'
                if not section_mandatory:
                    comment = f'# {comment}'
                lines.append(comment)

            # Prepare the value to be written
            value_to_write = details.get('default')
            if value_to_write is None:
                value_to_write = details.get('example', '...')

            line = f'{prefix}{key} = {format_value(value_to_write)}'

            # Comment out the line if the parameter or section is not mandatory
            if not details.get('mandatory', False) or not section_mandatory:
                line = f'# {line}'

            lines.append(line)

    # Recursively process nested sections
    for name, content in sections.items():
        # Check if this section is mandatory
        is_section_mandatory = content.get('mandatory', True) and section_mandatory

        # If the section should be flattened, pass its name as a prefix
        if content.get('flatten'):
            generate_toml_recursive(
                content.get('schema', {}),
                path_parts,
                lines,
                prefix=f'{prefix}{name}.',
                section_mandatory=is_section_mandatory,
            )
            continue

        # Handle special cases with dynamic keys
        if content.get('dynamic_keys'):
            lines.append('\n' + '#' * 80)
            lines.append(f"# The following section '{name}' allows for dynamic keys.")
            lines.append(
                '# An example entry is provided below. You can add more '
                'sections like it.'
            )
            lines.append(
                "# Replace 'EXAMPLE_KEY' with a name of your choice "
                "(e.g., 'alpha_phase')."
            )
            lines.append('#' * 80)
            generate_toml_recursive(
                content.get('schema', {}),
                path_parts + [name, 'EXAMPLE_KEY'],
                lines,
                section_mandatory=is_section_mandatory,
            )
        else:
            if 'description' in content:
                lines.append('\n' + '#' * 80)
                desc_comment = f'# {content["description"]}'
                if not is_section_mandatory:
                    desc_comment = f'# {desc_comment}'
                lines.append(desc_comment)
                lines.append('#' * 80)
            generate_toml_recursive(
                content,
                path_parts + [name],
                lines,
                section_mandatory=is_section_mandatory,
            )


def generate_template(schema, config_type):
    """
    Generates a complete TOML template file as a string.

    Args:
        schema (dict): The entire configuration schema.
        config_type (str): The top-level key from the schema for which to generate
                           the template.

    Returns
    -------
        str: The generated TOML template as a string.
    """
    # Convert logo to commented lines for TOML
    logo_lines = [f'# {line}' if line.strip() else '#' for line in MDB_LOGO.split('\n')]

    lines = logo_lines + [
        '#',
        '# Template file generated by MatDBForge.',
        '# Please, fill in the values for the parameters below.',
    ]

    schema_section = schema.get(config_type, {})
    generate_toml_recursive(schema_section, [], lines)

    return '\n'.join(lines)


def gen_default_config():
    init_logger(source='gen_configuration_file')
    parser = argparse.ArgumentParser(
        prog='mdb_gen_configuration_file',
        description=(
            'Generate MDB default configuration files in the TOML format '
            'from a YAML schema.'
        ),
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        '-t',
        '--config_type',
        help=(
            'Type of the configuration file to be generated. Available types are:\n'
            '\t- database_generation: For initial database generation.\n'
            '\t- dft: For DFT calculations.\n'
            '\t- active_learning: For the active learning loop.\n'
        ),
        type=str,
        required=True,
        choices=['database_generation', 'dft', 'active_learning'],
        metavar='TYPE',
    )

    parser.add_argument(
        '-p',
        '--path',
        help=(
            'Path in which to store the file.\n'
            'Will use the CWD by default. Folders will be created if necessary.'
        ),
        type=pl.Path,
        default='.',
        metavar='PATH',
    )
    parser.add_argument(
        '-o',
        '--overwrite',
        help=('Whether to overwrite the destination file, if existent.'),
        action='store_const',
        const=True,
        default=False,
    )

    args = parser.parse_args()

    # Load the master schema file
    schema_path = pl.Path(MDB_DATA_DIR) / 'config_schema.yaml'
    try:
        with open(schema_path) as f:
            schema = yaml.safe_load(f)
    except FileNotFoundError:
        custom_print(
            f'ERROR: Schema file not found at {schema_path}', print_type='error'
        )
        sys.exit(1)
    except yaml.YAMLError as e:
        custom_print(
            f'ERROR: Failed to parse schema file {schema_path}: {e}', print_type='error'
        )
        sys.exit(1)

    # Map config type to output filename
    filename_map = {
        'database_generation': 'database_generation_settings.toml',
        'dft': 'dft_settings.toml',
        'active_learning': 'active_learning_settings.toml',
    }
    output_filename = filename_map[args.config_type]

    # Generate TOML content from schema
    toml_content = generate_template(schema, args.config_type)

    # Write the generated content to the destination file
    final_path = args.path.resolve() / output_filename
    final_path.parent.mkdir(parents=True, exist_ok=True)

    if not final_path.exists() or args.overwrite:
        with open(final_path, 'w') as f:
            f.write(toml_content)
        custom_print(
            f"Saved file '{output_filename}' in path '{final_path}'", print_type='done'
        )
    else:
        custom_print(
            f"File '{final_path}' already exists. Not overwriting as flag "
            '-o / --overwrite was not set.',
            print_type='error',
        )
        sys.exit(1)
