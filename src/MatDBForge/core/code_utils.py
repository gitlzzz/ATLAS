"""Utility functions for code manipulation."""

import functools
import logging
import os
import pathlib
import subprocess as sb
import tempfile
import warnings

from packaging.version import Version
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

from MatDBForge import MDB_ROOT_DIR, __version__


def init_logger(source, log_path=None):
    # Starting console
    console = Console(
        theme=Theme(
            {
                'logging.level.[ i ]': 'blue',
                'logging.level.[ ! ]': 'yellow',
                'logging.level.[...]': 'white',
                'logging.level.[ ✔ ]': 'green',
                'logging.level.[ X ]': 'red',
            }
        )
    )

    logger = logging.getLogger('mdb')
    logger.setLevel(logging.DEBUG)

    # TODO: Check if this is compatible with the rest of the code
    logger.propagate = False

    # Console logger
    ch = RichHandler(
        markup=True,
        show_path=False,
        log_time_format='[%m/%d/%y %H:%M:%S]',
        omit_repeated_times=False,
        console=console,
    )
    ch.setLevel(logging.INFO)
    formatter_con = logging.Formatter('%(message)s')
    ch.setFormatter(formatter_con)
    logger.addHandler(ch)

    filename = tempfile.NamedTemporaryFile(prefix=f'mdb_{source}_', suffix='.log').name

    if log_path:
        log_path_dir = pathlib.Path(log_path)
        log_filename = pathlib.Path(filename + '.log').stem
        filename = log_path_dir / log_filename

    fh = logging.FileHandler(filename=filename, mode='a+')
    fh.setLevel(logging.DEBUG)
    formatter_fil = logging.Formatter('%(asctime)s - %(levelname)s - %(shortmsg)s')
    fh.setFormatter(formatter_fil)
    logger.addHandler(fh)

    logging.addLevelName(10, '[...]')
    logging.addLevelName(19, '     ')
    logging.addLevelName(20, '[ i ]')
    logging.addLevelName(25, '[ ✔ ]')
    logging.addLevelName(30, '[ ! ]')
    logging.addLevelName(40, '[ X ]')

    custom_print(f"Logging in '{filename}'", print_type='info')

    return logger, filename


def custom_print(string: str, print_type: str = 'default', end='\n', extra_tab=False):
    """Prints a string using different formatting styles for easier debugging.

    Parameters
    ----------
    string : str
        Text to be printed
    print_type : str, optional, `default=info`
        Style to use when printing. Available styles are:
            - `info/default`: prefixes [i] before the string.
            - `warning/warn`: prefixes [!] before the string.
            - `debug/extra`: prefixes [...] before the string.
            - `done/ok`: prefixes [ ✔ ] before the string.
            - `error/problem`: prefixes [ X ] before the string.
    """
    # normal = "\u001b[0m"

    normal = ''
    prefix = ''
    extra_tab = '\t' if extra_tab else ''

    logger = logging.getLogger('mdb')

    if print_type in ['info', 'default']:
        # prefix = "\u001b[38;5;33m [ i ]"
        logger.log(
            level=20,
            msg=f'{prefix}{normal}{extra_tab}{string}',
            extra={'shortmsg': string},
        )
    elif print_type in ['warn', 'warning', 'warn-soft', 'warning-soft']:
        # prefix = "\u001b[38;5;220m [ ! ]"
        logger.log(
            level=30,
            msg=f'{prefix}{normal}{extra_tab}{string}',
            extra={'shortmsg': string},
        )
    elif print_type in ['extra', 'debug']:
        # prefix = "\u001b[38;5;8m [···]"
        logger.log(
            level=10,
            msg=f'{prefix}{normal}{extra_tab}{string}',
            extra={'shortmsg': string},
        )
    if print_type in ['none', 'clean', 'clear', 'empty']:
        # prefix = ""
        # logger.info(f'{prefix}{normal}{extra_tab}{string}',
        # extra={'shortmsg': string})
        logger.log(
            level=15,
            msg=f'{prefix}{normal}{extra_tab}{string}',
            extra={'shortmsg': string},
        )
    elif print_type in ['done', 'ok']:
        # prefix = "\u001b[38;5;46m [ ✔ ]"
        # logger.info(
        #     f"{prefix}{normal}{extra_tab}{string}", extra={"shortmsg": string}
        # )
        logger.log(
            level=25,
            msg=f'{prefix}{normal}{extra_tab}{string}',
            extra={'shortmsg': string},
        )
    if print_type in ['error', 'problem']:
        # prefix = "\u001b[38;5;1m [ X ]"
        logger.log(
            level=40,
            msg=f'{prefix}{normal}{extra_tab}{string}',
            extra={'shortmsg': string},
        )


def deprecated(reason, since_ver=None):
    """
    Decorator to mark a function as deprecated.

    Parameters
    ----------
    reason : str
        Reason to print for the deprecation of the old function

    Example
    -------
    Use it as a decorator:

    >>> @deprecated(reason="Use to_cluster instead.", since_ver="0.6.2")
    >>> def to_atoms():
    >>>     pass

    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warn_text = f'{func.__name__}() is deprecated: {reason}.'
            if since_ver:
                warn_text += f"\n(Deprecated since version: '{since_ver}')."
            warnings.warn(warn_text, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_config_path() -> pathlib.Path:
    """Get the path to MatDBForge's the configuration directory."""
    # Try to get XDG_CONFIG_HOME, if it doesn't exist, return None
    config_path = os.environ.get('XDG_CONFIG_HOME', None)

    # Check if $HOME/.config exists and if it does, return the path
    if not config_path:
        config_folder = pathlib.Path().home() / '.config'
        if config_folder.exists():
            config_path = config_folder

    return pathlib.Path(config_path)


def init_config_dir(config_dir):
    """Create the configuration directory and the secrets file template."""
    # Create a 'mdb' directory inside the config directory
    config_dir = config_dir / 'mdb'
    config_dir.mkdir(parents=True, exist_ok=True)

    # Create a 'secrets.json' file inside the 'mdb' directory
    try:
        file_path = config_dir / 'secrets.json'

        with open(file_path, 'x') as f:
            f.write('{\n' '"API_KEY": ""\n' '}')

        # Limiting the permissions of the secrets file
        # to the owner only
        os.chmod(file_path, 0o700)

        return config_dir

    except FileExistsError:
        return None


def get_cache_path() -> pathlib.Path:
    """Get the path to MatDBForge's the cacheuration directory."""
    # Try to get XDG_cache_HOME, if it doesn't exist, return None
    cache_path = os.environ.get('$XDG_CACHE_HOME', None)

    # Check if $HOME/.cache exists and if it does, return the path
    if not cache_path:
        cache_folder = pathlib.Path().home() / '.cache'
        if cache_folder.exists():
            cache_path = cache_folder

    return pathlib.Path(cache_path)


def init_cache_dir(cache_dir):
    """Create the mdb cache directory."""
    # Create an mdb folder inside the ~/.cache directory
    cache_dir = cache_dir / 'mdb'

    try:
        cache_dir.mkdir(parents=True, exist_ok=False)
        return cache_dir
    except FileExistsError:
        return None


def get_last_tagged_version():
    """
    Get the last tagged version from the repository.

    Returns
    -------
    str
        Last tagged version in the repository
    """
    # Save the current directory path
    cwd = os.getcwd()

    # Change to the MatDBForge root directory
    os.chdir(MDB_ROOT_DIR)

    # Run the git fetch to get the last tagged version
    _ = sb.check_output(['git', 'fetch', '--quiet'])

    # Getting a sorted tag list
    output = (
        sb.check_output(['git', 'tag', '--merged', 'master', '--sort=-creatordate'])
        .decode()
        .strip()
    )

    # Split the output into a list of tags
    tags = output.split('\n') if output else []

    # Get the newest tag (first in the sorted list)
    newest_tag = tags[0] if tags else None

    # Go back to the original directory
    os.chdir(cwd)
    return newest_tag


def get_mdb_version_info():
    # Check the current version of MatDBForge
    curr_version = Version(__version__)

    # Check the last tagged version in the repository
    last_tagged_version = Version(get_last_tagged_version())

    return curr_version, last_tagged_version


def check_mdb_version(logger=None):
    """
    Check and print if the current version of MatDBForge is up-to-date.

    Returns
    -------
    tuple[Version, Version]
        Current version of MatDBForge and the last tagged version in the repository.
    """
    curr_version, last_tagged_version = get_mdb_version_info()

    print()
    new_logger = False
    if not logger:
        new_logger = True
        logger, _ = init_logger('mdb_version_check')

    if curr_version < last_tagged_version:
        custom_print(
            f'Current version of MatDBForge ({curr_version}) is outdated. '
            f'Please update to the latest version ({last_tagged_version}).',
            'warn',
        )
    else:
        custom_print(
            f'Current version of MatDBForge ({curr_version}) is up-to-date.',
            'done',
        )

    print()

    # Clearing version check logger
    if new_logger:
        logger.handlers.clear()

    return curr_version, last_tagged_version
