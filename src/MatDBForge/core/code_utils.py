"""Utility functions for code manipulation."""

import functools
import logging
import os
import pathlib
import subprocess as sb
import tempfile
import warnings

from packaging.version import Version
from rich.logging import RichHandler

from MatDBForge import MDB_ROOT_DIR, __version__


def init_logger(source, log_path=None):
    logger = logging.getLogger("mdb")
    logger.setLevel(logging.DEBUG)

    # Console logger
    ch = RichHandler(
        markup=True,
        show_path=False,
        log_time_format="[%x %X]",
        omit_repeated_times=False,
    )
    ch.setLevel(logging.INFO)
    formatter_con = logging.Formatter("%(message)s")
    ch.setFormatter(formatter_con)
    logger.addHandler(ch)

    filename = tempfile.NamedTemporaryFile(prefix=f"mdb_{source}_", suffix=".log").name

    if log_path:
        log_path_dir = pathlib.Path(log_path)
        log_filename = pathlib.Path(filename + ".log").stem
        filename = log_path_dir / log_filename

    fh = logging.FileHandler(filename=filename, mode="a+")
    fh.setLevel(logging.DEBUG)
    formatter_fil = logging.Formatter("%(asctime)s - %(levelname)s - %(shortmsg)s")
    fh.setFormatter(formatter_fil)
    logger.addHandler(fh)

    custom_print(f"Logging in '{filename}'", print_type="info")
    logging.addLevelName(25, "DONE")

    return logger, filename


def custom_print(string: str, print_type: str = "default", end="\n", extra_tab=False):
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

    normal = ""
    prefix = ""
    extra_tab = "\t" if extra_tab else ""

    if print_type in ["info", "default"]:
        # prefix = "\u001b[38;5;33m [ i ]"
        logging.getLogger("mdb").info(
            f"{prefix}{normal}{extra_tab}{string}", extra={"shortmsg": string}
        )
    elif print_type in ["warn", "warning", "warn-soft", "warning-soft"]:
        # prefix = "\u001b[38;5;220m [ ! ]"
        logging.getLogger("mdb").warning(
            f"{prefix}{normal}{extra_tab}{string}", extra={"shortmsg": string}
        )
    elif print_type in ["extra", "debug"]:
        # prefix = "\u001b[38;5;8m [···]"
        logging.getLogger("mdb").debug(
            f"{prefix}{normal}{extra_tab}{string}", extra={"shortmsg": string}
        )
    elif print_type in ["done", "ok"]:
        # prefix = "\u001b[38;5;46m [ ✔ ]"
        # logging.getLogger("mdb").info(
        #     f"{prefix}{normal}{extra_tab}{string}", extra={"shortmsg": string}
        # )
        logging.getLogger("mdb").log(
            level=25,
            msg=f"{prefix}{normal}{extra_tab}{string}",
            extra={"shortmsg": string},
        )
    if print_type in ["error", "problem"]:
        # prefix = "\u001b[38;5;1m [ X ]"
        logging.getLogger("mdb").error(
            f"{prefix}{normal}{extra_tab}{string}", extra={"shortmsg": string}
        )
    if print_type in ["none", "clean", "clear"]:
        # prefix = ""
        logging.getLogger("mdb").info(
            f"{prefix}{normal}{extra_tab}{string}", extra={"shortmsg": string}
        )


def deprecated(reason, since_ver=None):
    """
    Decorator to mark a function as deprecated.

    Parameters
    ----------
    reason : str
        Reason to print for the deprecation of the old function

    Examples
    --------
    Use it as a decorator:
    >>> @deprecated(reason="Use to_cluster instead.", since_ver="0.6.2")
        def to_atoms():
            pass
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


def get_last_tagged_version():
    """
    Get the last tagged version from the repository.

    Returns
    -------
    str
        Last tagged version in the repository
    """
    # Run the git command to get the last tagged version
    cwd = os.getcwd()
    os.chdir(MDB_ROOT_DIR)
    sb.call(["git", "fetch"])
    output = (
        sb.check_output(["git", "describe", "--tags", "--abbrev=0"]).decode().strip()
    )
    os.chdir(cwd)
    return output


def check_mdb_version():
    # Check the current version of MatDBForge
    curr_version = Version(__version__)

    # Check the last tagged version in the repository
    last_tagged_version = Version(get_last_tagged_version())

    print()

    if curr_version < last_tagged_version:
        custom_print(
            f"Current version of MatDBForge ({curr_version}) is outdated. "
            f"Please update to the latest version ({last_tagged_version}).",
            "warn",
        )
    else:
        custom_print(
            f"Current version of MatDBForge ({curr_version}) is up-to-date.",
            "done",
        )

    print()
