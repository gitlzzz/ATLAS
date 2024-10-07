"""Utility functions for code manipulation."""

import functools
import os
import subprocess as sb
import warnings

from packaging.version import Version

from MatDBForge import MDB_ROOT_DIR, __version__
from MatDBForge.core.utils import custom_print


def deprecated(reason, since_ver=None):
    """
    Decorator to mark a function as deprecated.

    Parameters
    ----------
    reason : str
        Reason to print for the deprecation of the old function
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

    if curr_version < last_tagged_version:
        custom_print(
            f"Current version of MatDBForge ({curr_version}) is outdated. "
            f"Please update to the latest version ({last_tagged_version}).",
            "warn",
        )
    else:
        custom_print(
            f"Current version of MatDBForge ({curr_version}) is up-to-date.",
            "info",
        )

    print()
