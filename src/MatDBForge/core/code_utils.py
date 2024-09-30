"""Utility functions for code manipulation."""

import functools
import warnings


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
