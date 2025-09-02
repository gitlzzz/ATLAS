"""Generate documentation for MDB configuration files from the YAML schema."""

import argparse
import pathlib as pl
import sys
from argparse import RawTextHelpFormatter

import yaml

from MatDBForge.core import MDB_DATA_DIR
from MatDBForge.core.code_utils import custom_print, init_logger


def format_parameter_line(key, details, level=0):
    """
    Format a parameter line according to the documentation style.

    Args:
        key (str): Parameter name
        details (dict): Parameter details from schema
        level (int): Nesting level for indentation

    Returns
    -------
        str: Formatted parameter line
    """
    indent = '  ' * level if level > 0 else ''
    param_type = details.get('type', 'unknown')
    description = details.get('description', 'No description available.')
    is_mandatory = details.get('mandatory', True)
    default_value = details.get('default')
    example_value = details.get('example')

    # Format type with optional indicator
    type_str = f'(optional, {param_type})' if not is_mandatory else f'({param_type})'

    # Start building the line
    line = f'{indent}- `{key}`: {type_str} {description}'

    # Add default value if present
    if default_value is not None:
        if isinstance(default_value, str):
            line += f" Default is '{default_value}'."
        else:
            line += f' Default is {default_value}.'

    # Add example if present and no default
    elif example_value is not None:
        if isinstance(example_value, str):
            line += f" Example: '{example_value}'."
        else:
            line += f' Example: {example_value}.'

    return line


def generate_section_docs(schema_dict, path_parts, lines, level=3):
    """
    Recursively generate documentation for a schema section.

    Args:
        schema_dict (dict): Current section of the schema
        path_parts (list): Current path in the schema (for section headers)
        lines (list): List to append generated lines to
        level (int): Header level for markdown (3 = ###)
    """
    # Separate parameters from nested sections
    params = {
        k: v for k, v in schema_dict.items() if isinstance(v, dict) and 'type' in v
    }
    sections = {
        k: v for k, v in schema_dict.items() if isinstance(v, dict) and 'type' not in v
    }

    # Generate documentation for parameters at current level
    if params:
        for key, details in params.items():
            lines.append('')
            lines.append(format_parameter_line(key, details))

    # Process nested sections
    for name, content in sections.items():
        if content.get('flatten'):
            # For flattened sections, document the flattened structure
            lines.append('')
            lines.append(f'Parameters using `{name}.` prefix:')
            lines.append('')
            schema_content = content.get('schema', {})
            for sub_key, sub_details in schema_content.items():
                if isinstance(sub_details, dict) and 'type' in sub_details:
                    flattened_key = f'{name}.{sub_key}'
                    lines.append(format_parameter_line(flattened_key, sub_details))
            continue

        if content.get('dynamic_keys'):
            # Handle dynamic key sections specially
            header_level = '#' * level
            section_path = '.'.join(path_parts + [name])

            lines.append('')
            description = content.get("description", f"{name.title()} Settings")
            lines.append(
                f'{header_level} {description} - '
                f'`[{section_path}.XXXXX]`'
            )
            lines.append('')

            lines.append(
                'This key describes settings for dynamic entries. '
                + 'Several entries can be added by using different key names.'
            )
            lines.append('')
            lines.append(
                'The key name (`XXXXX`) is used as the reference name. '
                + '**Replace XXXXX with a name of your choice.**'
            )
            lines.append('')

            # Add example if available
            schema_content = content.get('schema', {})
            if schema_content:
                lines.append('Example parameters for each entry:')
                lines.append('')
                generate_section_docs(
                    schema_content, path_parts + [name, 'XXXXX'], lines, level + 1
                )
            continue

        # Regular nested section
        if 'description' in content:
            header_level = '#' * level
            section_path = '.'.join(path_parts + [name])

            lines.append('')
            lines.append(
                f'{header_level} {content["description"]} - `[{section_path}]`'
            )
            lines.append('')

            # Check if section is optional
            if not content.get('mandatory', True):
                lines.append(':::{attention}')
                lines.append('This section is optional.')
                lines.append(':::')
                lines.append('')

            generate_section_docs(content, path_parts + [name], lines, level + 1)


def generate_tool_section(schema, tool_name, tool_config):
    """
    Generate documentation for a specific tool.

    Args:
        schema (dict): Full schema
        tool_name (str): Name of the tool (database_generation, dft, active_learning)
        tool_config (dict): Tool configuration from schema

    Returns
    -------
        list: Lines of generated documentation
    """
    lines = []

    # Tool header mapping
    tool_headers = {
        'database_generation': 'Database Generation',
        'dft': 'DFT Calculations',
        'active_learning': 'Active Learning Loop',
    }

    # Tool command mapping
    tool_commands = {
        'database_generation': 'database_generation',
        'dft': 'dft',
        'active_learning': 'active_learning',
    }

    header = tool_headers.get(tool_name, tool_name.replace('_', ' ').title())
    command = tool_commands.get(tool_name, tool_name)

    lines.append(f'## {header}')
    lines.append('')
    lines.append(
        f'Generate a {tool_name.replace("_", " ")} template file using '
        + f'`mdb_gen_configuration_file -t {command}`.'
    )
    lines.append('')
    lines.append(':::{attention}')
    lines.append('All keys are mandatory unless stated otherwise.')
    lines.append(':::')
    lines.append('')

    # Generate sections
    generate_section_docs(tool_config, [], lines, level=3)

    return lines


def generate_full_documentation(schema):
    """
    Generate the complete documentation file.

    Args:
        schema (dict): Complete schema from YAML file

    Returns
    -------
        str: Complete documentation as string
    """
    lines = []

    # Header and introduction
    lines.extend(
        [
            '# Input specification',
            '',
            (
                'The input format of MDB is [TOML](https://toml.io/en/). '
                'The syntax from TOML is unchanged. '
                'The available parameters are different depending on the selected tool.'
            ),
            '',
            (
                'Users are advised to use the `mdb_gen_configuration_file` utility '
                'to generate a template file which can be customized. However, '
                'the configuration files can be created from scratch using the '
                'sections from below and the appropiate TOML syntax '
                'as in the following example:'
            ),
            '',
            '```toml',
            '[database]',
            "database_name = 'test'",
            '...',
            '',
            '',
            '[database.plot_db]',
            'show = true',
            '',
            '...',
            '',
            '[generation]',
            "generate_type = ['bulk', 'surface', 'cluster']",
            '',
            '```',
            '',
            (
                "Please, check the tool's corresponding section to learn more "
                "about all the available options."
            ),
            '',
        ]
    )

    # Generate documentation for each tool
    for tool_name in ['database_generation', 'dft', 'active_learning']:
        if tool_name in schema:
            tool_lines = generate_tool_section(schema, tool_name, schema[tool_name])
            lines.extend(tool_lines)
            lines.append('')  # Extra space between tools

    return '\n'.join(lines)


def generate_docs():
    """Main function to generate documentation."""
    init_logger(source='gen_docs')

    parser = argparse.ArgumentParser(
        prog='mdb_gen_docs',
        description='Generate MDB documentation from the YAML schema.',
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        '-o',
        '--output',
        help='Output path for the generated documentation file.',
        type=pl.Path,
        default=pl.Path('docs/source/input.md'),
        metavar='PATH',
    )

    parser.add_argument(
        '--overwrite',
        help='Whether to overwrite the destination file, if existent.',
        action='store_true',
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

    # Generate documentation
    custom_print('Generating documentation from schema...', print_type='info')
    documentation = generate_full_documentation(schema)

    # Write the generated documentation
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not output_path.exists() or args.overwrite:
        with open(output_path, 'w') as f:
            f.write(documentation)
        custom_print(f"Documentation saved to '{output_path}'", print_type='done')
    else:
        custom_print(
            f"File '{output_path}' already exists. Use --overwrite to replace it.",
            print_type='error',
        )
        sys.exit(1)


if __name__ == '__main__':
    generate_docs()
