"""Utility functions for code manipulation."""

import functools
import logging
import os
import pathlib
import subprocess as sb
import tempfile
import warnings
from logging import LogRecord

import qrcode
from packaging.version import Version
from packaging.version import parse as parse_version
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

from MatDBForge import MDB_ROOT_DIR, __repo__, __version__


def display_qr_in_cli(data: str):
    """Generates and displays a QR code in the terminal.

    Parameters
    ----------
    data : str
        The data to encode in the QR code (e.g., a URL).
    """
    qr = qrcode.QRCode(version=1, border=1)
    qr.add_data(data)
    qr.make(fit=True)
    qr.print_tty()


def save_qr_to_file(data: str, filename: str = "qr_code.png"):
    """Generates a QR code and saves it as an image file."""
    qr = qrcode.QRCode(version=4, border=1)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(filename)


def get_console_handler():
    # Starting console
    console = Console(
        theme=Theme(
            {
                "logging.level.[ i ]": "blue",
                "logging.level.[ ! ]": "yellow",
                "logging.level.[...]": "white",
                "logging.level.[ ✔ ]": "green",
                "logging.level.[ x ]": "red",
            }
        )
    )

    # Console logger
    ch = RichHandler(
        markup=True,
        show_path=False,
        log_time_format="[%m/%d/%y %H:%M:%S]",
        omit_repeated_times=False,
        console=console,
        # 11 is one level above DEBUG (10). Allows to show custom low-priority messages
        level=11,
    )
    formatter_con = logging.Formatter("%(message)s")
    ch.setFormatter(formatter_con)
    ch.set_name("mdb_rich_handler")
    ch.addFilter(create_handler_filters("console"))
    return ch, console


def logging_set_levels():
    logging.addLevelName(10, "[...]")
    logging.addLevelName(19, "     ")
    logging.addLevelName(15, "MDB_DEBUG")
    logging.addLevelName(20, "[ i ]")
    logging.addLevelName(25, "[ ✔ ]")
    logging.addLevelName(30, "[ ! ]")
    logging.addLevelName(40, "[ X ]")


def create_handler_filters(handler: str):
    def handler_filter(record: LogRecord):
        return not (hasattr(record, "block") and record.block == handler)

    return handler_filter


def init_logger(source, log_path=None, show_log_path=True):
    logger = logging.getLogger("mdb")
    logger.setLevel(logging.DEBUG)

    logger.propagate = False

    # Console logger
    ch, console = get_console_handler()
    logger.addHandler(ch)

    if not log_path:
        _, filename = tempfile.mkstemp(prefix=f"mdb_{source}_", suffix=".log")
    else:
        log_path_dir = pathlib.Path(log_path)
        _, filename = tempfile.mkstemp(
            prefix=f"mdb_{source}_", suffix=".log", dir=log_path_dir
        )

    fh = logging.FileHandler(filename=filename, mode="a+")
    fh.set_name("mdb_file_handler")
    fh.addFilter(create_handler_filters("file"))
    fh.setLevel(logging.DEBUG)
    formatter_fil = logging.Formatter("%(asctime)s - %(levelname)s - %(shortmsg)s")
    fh.setFormatter(formatter_fil)
    logger.addHandler(fh)

    logging_set_levels()

    if show_log_path:
        custom_print(f"Logging in '{filename}'", print_type="info")

    return logger, filename


class LevelNameFilter(logging.Filter):
    """Filters log records based on a list of allowed level names."""

    def __init__(self, levels_to_keep):
        super().__init__()
        self.levels_to_keep = set(levels_to_keep)

    def filter(self, record):
        return record.levelname in self.levels_to_keep


def custom_print(
    string: str,
    print_type: str = "default",
    end="\n",
    extra_tab=False,
    logger=None,
    extras: dict = None,
):
    r"""Prints a string using different formatting styles for easier debugging.

    Parameters
    ----------
    string : str
        Text to be printed
    print_type : str, optional, `default=info`
        Style to use when printing. Available styles are:
            - `info/default`: prefixes [ i ] before the string.
            - `warning/warn`: prefixes [ ! ] before the string.
            - `debug/extra`: prefixes [...] before the string.
            - `done/ok`: prefixes [ ✔ ] before the string.
            - `error/problem`: prefixes [ X ] before the string.
            - `none/clean/clear/empty`: leaves an empty space before the string.
    end : str, optional, `default=\n`
        String appended after the last value, default a newline.
    extra_tab : bool, optional, `default=False`
        If True, adds an extra tab before the string.
    logger : logging.Logger, optional, `default=None`
        Logger to use for printing. If None, a new logger named 'mdb' is created

    Returns
    -------
    logging.Logger
        Logger used for printing the string
    """
    # normal = "\u001b[0m"

    normal = ""
    prefix = ""
    extra_tab = "\t" if extra_tab else ""

    if not logger:
        logger = logging.getLogger("mdb")

    # Allows to use the custom print function without initializing
    # the logger first
    if logger.handlers == []:
        logger.setLevel(logging.DEBUG)
        ch, _ = get_console_handler()
        logger.addHandler(ch)
        logging_set_levels()

    if print_type in ["info", "default"]:
        # prefix = "\u001b[38;5;33m [ i ]"
        logger.log(
            level=20,
            msg=f"{prefix}{normal}{extra_tab}{string}",
            extra={"shortmsg": string, **(extras if extras else {})},
        )
    elif print_type in ["warn", "warning", "warn-soft", "warning-soft"]:
        # prefix = "\u001b[38;5;220m [ ! ]"
        logger.log(
            level=30,
            msg=f"{prefix}{normal}{extra_tab}{string}",
            extra={"shortmsg": string, **(extras if extras else {})},
        )
    elif print_type in ["extra", "debug"]:
        # prefix = "\u001b[38;5;8m [···]"
        logger.log(
            level=10,
            msg=f"{prefix}{normal}{extra_tab}{string}",
            extra={"shortmsg": string, **(extras if extras else {})},
        )
    if print_type in ["none", "clean", "clear", "empty"]:
        prefix = ""
        # logger.info(f'{prefix}{normal}{extra_tab}{string}',
        # extra={'shortmsg': string})
        logger.log(
            level=15,
            msg=f"{prefix}{normal}{extra_tab}{string}",
            extra={"shortmsg": string, **(extras if extras else {})},
        )
    elif print_type in ["done", "ok"]:
        # prefix = "\u001b[38;5;46m [ ✔ ]"
        # logger.info(
        #     f"{prefix}{normal}{extra_tab}{string}", extra={"shortmsg": string}
        # )
        logger.log(
            level=25,
            msg=f"{prefix}{normal}{extra_tab}{string}",
            extra={"shortmsg": string, **(extras if extras else {})},
        )
    if print_type in ["error", "problem"]:
        # prefix = "\u001b[38;5;1m [ X ]"
        logger.log(
            level=40,
            msg=f"{prefix}{normal}{extra_tab}{string}",
            extra={"shortmsg": string, **(extras if extras else {})},
        )
    return logger


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
            warn_text = f"{func.__name__}() is deprecated: {reason}."
            if since_ver:
                warn_text += f"\n(Deprecated since version: '{since_ver}')."
            warnings.warn(warn_text, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_config_path() -> pathlib.Path:
    """Get the path to MatDBForge's the configuration directory."""
    # Try to get XDG_CONFIG_HOME, if it doesn't exist, return None
    config_path = os.environ.get("XDG_CONFIG_HOME", None)

    # Check if $HOME/.config exists and if it does, return the path
    if not config_path:
        config_folder = pathlib.Path().home() / ".config"
        if config_folder.exists():
            config_path = config_folder

    return pathlib.Path(config_path)


def init_config_dir(config_dir, config_file: str):
    """Create the configuration directory and the secrets file template."""
    # Create a 'mdb' directory inside the config directory
    config_dir = config_dir / "mdb"
    config_dir.mkdir(parents=True, exist_ok=True)

    # Create a 'config_file' file inside the 'mdb' directory
    try:
        file_path = config_dir / config_file

        with open(file_path, "x") as f:
            f.write('{\n"API_KEY": ""\n}')

        # Limiting the permissions of the secrets file
        # to the owner only
        os.chmod(file_path, 0o700)

        return True, config_dir

    except FileExistsError:
        return False, config_dir


def get_cache_path() -> pathlib.Path:
    """Get the path to MatDBForge's the cacheuration directory."""
    # Try to get XDG_cache_HOME, if it doesn't exist, return None
    cache_path = os.environ.get("$XDG_CACHE_HOME", None)

    # Check if $HOME/.cache exists and if it does, return the path
    if not cache_path:
        cache_folder = pathlib.Path().home() / ".cache"
        if cache_folder.exists():
            cache_path = cache_folder

    return pathlib.Path(cache_path)


def init_cache_dir(cache_dir):
    """Create the mdb cache directory."""
    # Create an mdb folder inside the ~/.cache directory
    cache_dir = cache_dir / "mdb"

    try:
        cache_dir.mkdir(parents=True, exist_ok=False)
        return cache_dir
    except FileExistsError:
        return None


def get_last_tagged_version():
    """
    Get the last tagged version from the GitHub repository (via SSH).
    If SSH access fails, fall back to checking the local Git repository.

    Returns
    -------
    str
        Last tagged version in the repository.
    str
        Hash of the current commit.
    """
    try:
        # Try fetching the latest tag from GitHub via SSH
        output = (
            sb.check_output(["git", "ls-remote", "--tags", __repo__], stderr=sb.DEVNULL)
            .decode()
            .strip()
        )

        # Extract tag names from the output
        tags = [
            line.split("/")[-1].replace("^{}", "").replace("v", "")
            for line in output.split("\n")
        ]
        # Get the newest tag. Use versioning.version.parse to handle the version sorting
        # Return '0.0.0' if no tags are found
        newest_tag = (
            sorted(tags, reverse=True, key=lambda v: parse_version(v))[0]
            if tags
            else "0.0.0"
        )

        # Get the latest commit hash from the remote
        hash_str = (
            sb.check_output(["git", "ls-remote", __repo__, "HEAD"]).decode().split()[0]
        )

    except (sb.CalledProcessError, IndexError):
        # If SSH access fails, fall back to local repository
        newest_tag, hash_str = get_last_tagged_version_local()

    return newest_tag, hash_str


def get_last_tagged_version_local(repo_dir_path: str = None):
    """
    Get the last tagged version from the local Git repository.

    Returns
    -------
    str
        Last tagged version in the local repository.
    str
        Hash of the current commit.
    """
    # Save the current directory path
    cwd = os.getcwd()

    # Use the default MatDBForge root directory if no path is provided
    if not repo_dir_path or not pathlib.Path(repo_dir_path).exists():
        repo_dir_path = MDB_ROOT_DIR

    try:
        # Change to the MatDBForge root directory
        os.chdir(repo_dir_path)

        # Split the output into a list of tags
        tags = get_list_of_tags()

        # Sort tags in descending order
        tags = sorted(tags, reverse=True)

        # Get the newest tag (first in the sorted list)
        newest_tag = tags[0] if tags else "0.0.0"

        # Get the current commit hash
        current_hash = sb.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

    finally:
        # Ensure we switch back to the original directory
        os.chdir(cwd)

    return newest_tag, current_hash


def get_list_of_tags(repo_path: str = None) -> list[Version]:
    """
    Get a list of tags from the git repository in the CWD.

    Parameters
    ----------
    repo_path : str, optional
        Path to the git repository. If None, uses the current working directory.

    Returns
    -------
    list[str]
        List of tags in the repository.
    """
    # Use the given repository path if provided
    if repo_path:
        os.chdir(repo_path)

    # Run the git fetch to get the last tagged version
    try:
        _ = sb.check_output(["git", "fetch", "--tags", "--quiet"])
    except sb.CalledProcessError:
        return "0.0.0", "unknown"

    # Getting a sorted tag list
    output = (
        sb.check_output(["git", "tag", "--merged", "master", "--sort=-creatordate"])
        .decode()
        .strip()
    )

    # Split the output into a list of tags
    tags = output.split("\n") if output else []

    # Convert to version object
    tags = [Version(tag) for tag in tags]

    return tags


def get_mdb_version_info():
    """
    Get the current version of MatDBForge and the last tagged version in the repository.

    Returns
    -------
    Version
        Current version of MatDBForge
    Version
        Last tagged version in the repository
    str
        Hash of the current commit
    """
    # Check the current version of MatDBForge
    curr_version = Version(__version__)

    # Check the last tagged version in the repository
    ver, hash_str = get_last_tagged_version()
    last_tagged_version = Version(ver)

    return curr_version, last_tagged_version, hash_str


def check_mdb_version(logger=None):
    """
    Check and print if the current version of MatDBForge is up-to-date.

    Returns
    -------
    tuple[Version, Version]
        Current version of MatDBForge and the last tagged version in the repository.
    """
    curr_version, last_tagged_version, hash_str = get_mdb_version_info()

    print()
    new_logger = False
    if not logger:
        new_logger = True
        logger, _ = init_logger("mdb_version_check", show_log_path=False)

    if curr_version < last_tagged_version:
        custom_print(
            (
                f"Current version of MatDBForge ({curr_version}, {hash_str[:7]}) "
                "is outdated. Please update to the latest "
                f"version ({last_tagged_version})."
            ),
            "warn",
        )
    elif curr_version > last_tagged_version:
        custom_print(
            (
                "[bold yellow]Current version of MatDBForge "
                f"({curr_version}, {hash_str[:7]}) is an unrealeased version."
            ),
            "warn",
        )
    elif hash_str == "unknown":
        custom_print("Unable to fetch the latest version!", "error")
    else:
        custom_print(
            (
                f"Current version of MatDBForge ({curr_version}, {hash_str[:7]}) "
                "is up-to-date."
            ),
            "done",
        )

    print()

    # Clearing version check logger
    if new_logger:
        logger.handlers.clear()

    return curr_version, last_tagged_version
