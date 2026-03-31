"""Generate default configuration files for MDB from a YAML schema."""

import argparse
import io
import pathlib as pl
import sys
from argparse import RawTextHelpFormatter

from rich.console import Console

from MatDBForge.core.code_utils import custom_print, init_logger
from MatDBForge.core.command_line.command_line_utils import (
    MDB_LOGO,
    get_schema,
    validate_config_file,
)


def wrap_text_with_rich(text: str, width: int = 80, comment: bool = False) -> str:
    """
    Wraps text using Rich's formatting engine and optionally prepends
    comment characters.

    Parameters
    ----------
    text : str
        The long text string to be wrapped.
    width : int, optional
        The target width for the wrapped text. Default is 50.
    comment : bool, optional
        If True, prepends '# ' to each line of the wrapped output.
        Default is False.

    Returns
    -------
    str
        The wrapped text.
    """
    console = Console(width=width, record=True, file=io.StringIO())
    console.print(text, highlight=False)

    # Export text and remove the single trailing newline from print()
    wrapped_text = console.export_text(styles=False).rstrip()

    if comment:
        # Split into lines to strictly apply the prefix to every row
        lines = wrapped_text.splitlines()
        # Re-join with the comment prefix
        return '\n'.join(f'# {line}' for line in lines)

    return wrapped_text


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
                comment = details['description']
                comment = wrap_text_with_rich(comment, comment=True)
                lines.append(comment)
            if 'type' in details:
                comment = f'# type: {details["type"]}'
                lines.append(comment)

            # Prepare the value to be written
            value_to_write = details.get('default')
            if value_to_write is None:
                value_to_write = details.get('example', None)

            line = f'{prefix}{key} = {format_value(value_to_write)}'

            # Comment out the line if the parameter or section is not mandatory
            if not details.get('mandatory', False) or not section_mandatory:
                line = f'# {line}'

            if value_to_write is None and details.get('dynamic_keys', False):
                line = ''
                for dyn_key, dyn_details in details.get(
                    'schema_under_dynamic_keys', {}
                ).items():
                    line += (
                        f'# {prefix}{key}.XXXX.{dyn_key} = '
                        f'{format_value(dyn_details.get("example"))}\n'
                    )
                line = line.rstrip()

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
                lines.append('\n' + '#' * 81)

                if content.get('name_pretty'):
                    lines.append(f'# Section: {content["name_pretty"]}\n')

                desc_comment = content['description']
                if not is_section_mandatory:
                    desc_comment = f'{desc_comment}'

                desc_comment = wrap_text_with_rich(desc_comment, comment=True)

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


class ValidationError(Exception):
    """Custom exception for configuration validation errors."""


def generate_command(args, schema):
    """Handle the generate subcommand."""
    # Map config type to output filename
    filename_map = {
        'database_generation': 'database_generation_settings.toml',
        'dft': 'dft_settings.toml',
        'active_learning': 'active_learning_settings.toml',
        'mlip_benchmarks': 'mdb_benchmark_settings.toml',
        'latent_space_analysis': 'latent_space_settings.toml',
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
        return True
    else:
        custom_print(
            f"File '{final_path}' already exists. Not overwriting as flag "
            '-o / --overwrite was not set.',
            print_type='error',
        )
        return False


def gen_default_config():
    init_logger(source='gen_configuration_file')

    # Main parser
    parser = argparse.ArgumentParser(
        prog='mdb_gen_configuration_file',
        description=(
            'Generate MDB default configuration files in the TOML format '
            'from a YAML schema, or validate existing configuration files.'
        ),
        formatter_class=RawTextHelpFormatter,
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Generate subcommand (default behavior)
    generate_parser = subparsers.add_parser(
        'generate',
        help='Generate template configuration files',
        formatter_class=RawTextHelpFormatter,
    )
    generate_parser.add_argument(
        '-t',
        '--config_type',
        help=(
            'Type of the configuration file to be generated. Available types are:\n'
            '\t- database_generation: For initial database generation.\n'
            '\t- dft: For DFT calculations.\n'
            '\t- active_learning: For the active learning loop.\n'
            '\t- latent_space_analysis: For latent space analysis of datasets.\n'
        ),
        type=str,
        required=True,
        choices=[
            'database_generation',
            'dft',
            'active_learning',
            'mlip_benchmarks',
            'latent_space_analysis',
        ],
        metavar='TYPE',
    )
    generate_parser.add_argument(
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
    generate_parser.add_argument(
        '-o',
        '--overwrite',
        help='Whether to overwrite the destination file, if existent.',
        action='store_true',
        default=False,
    )

    # Validate subcommand
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate an existing configuration file against the schema',
        formatter_class=RawTextHelpFormatter,
    )
    validate_parser.add_argument(
        '-i',
        '--input',
        help='Path to the TOML configuration file to validate',
        type=pl.Path,
        required=True,
        metavar='FILE',
    )
    validate_parser.add_argument(
        '-t',
        '--config_type',
        help=(
            'Type of the configuration file to validate:\n'
            '\t- database_generation: For initial database generation.\n'
            '\t- dft: For DFT calculations.\n'
            '\t- active_learning: For the active learning loop.\n'
        ),
        type=str,
        required=True,
        choices=['database_generation', 'dft', 'active_learning', 'mlip_benchmarks'],
        metavar='TYPE',
    )
    validate_parser.add_argument(
        '--allow-deprecated',
        help='Allow deprecated keys without failing validation (show warnings only)',
        action='store_true',
        default=False,
    )

    # Parse arguments
    args = parser.parse_args()

    # Handle backward compatibility (no subcommand = generate)
    if args.command is None:
        # If no subcommand is provided, parse as generate command
        # This maintains backward compatibility
        old_parser = argparse.ArgumentParser(
            prog='mdb_gen_configuration_file',
            description=(
                'Generate MDB default configuration files in the TOML format '
                'from a YAML schema.'
            ),
            formatter_class=RawTextHelpFormatter,
        )
        old_parser.add_argument(
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
        old_parser.add_argument(
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
        old_parser.add_argument(
            '-o',
            '--overwrite',
            help='Whether to overwrite the destination file, if existent.',
            action='store_true',
            default=False,
        )
        args = old_parser.parse_args()
        args.command = 'generate'

    schema = get_schema()

    # Execute the appropriate command
    success = False
    if args.command == 'generate':
        success = generate_command(args, schema)
    elif args.command == 'validate':
        success = validate_command(args, schema)
    else:
        parser.print_help()
        sys.exit(1)

    sys.exit(0 if success else 1)


def validate_command(args, schema):
    """Handle the validate subcommand."""
    config_path = args.input
    config_type = args.config_type

    allow_deprecated = getattr(args, 'allow_deprecated', False)
    validate_config_file(
        config_path=config_path,
        config_type=config_type,
        allow_deprecated=allow_deprecated,
        run_mode='script',
    )
