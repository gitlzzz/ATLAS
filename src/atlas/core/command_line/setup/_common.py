"""Shared utilities for the ATLAS setup subcommands."""

from __future__ import annotations

from rich import print as rprint
from rich.panel import Panel

from atlas.core.code_utils import custom_print


def heading(title: str, subtitle: str = '') -> None:
    """Print a section heading."""
    body = subtitle or title
    rprint()
    rprint(Panel(body, title=title, border_style='cyan'))
    rprint()


def check_aiida() -> bool:
    """Return True if aiida-core is importable."""
    try:
        import aiida  # noqa: F401

        return True
    except ImportError:
        return False


def require_profile() -> bool:
    """Check that an AiiDA profile is loaded. Print an error if not."""
    if not check_aiida():
        custom_print(
            'AiiDA is not installed. Install it first or skip this step.',
            'error',
        )
        return False
    try:
        from aiida.manage.configuration import get_profile

        profile = get_profile()
        if profile is None:
            custom_print(
                'No AiiDA profile is loaded. Run "atl_init_setup profile" first.',
                'error',
            )
            return False
        return True
    except Exception as exc:
        custom_print(f'Failed to check AiiDA profile: {exc}', 'error')
        return False


def list_profiles() -> tuple[list[str], str | None]:
    """Return (profile_names, default_name) or ([], None)."""
    try:
        from aiida.manage.configuration import get_config

        config = get_config()
        return list(config.profile_names), config.default_profile_name
    except Exception:
        return [], None


def list_computers() -> list[str]:
    """Return labels of configured AiiDA computers."""
    try:
        from aiida import orm

        return [c.label for c in orm.QueryBuilder().append(orm.Computer).all(flat=True)]
    except Exception:
        return []


def list_codes() -> list[str]:
    """Return full labels of configured AiiDA codes."""
    try:
        from aiida import orm

        return [
            c.full_label for c in orm.QueryBuilder().append(orm.Code).all(flat=True)
        ]
    except Exception:
        return []
