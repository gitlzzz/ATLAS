"""Fetch AiiDA entity lists for schema-form dropdown suggestions."""

from __future__ import annotations


def fetch_aiida_suggestions() -> dict[str, list[str]]:
    """Return a dict of suggestion lists keyed by suggestion type.

    Safe to call even when AiiDA is not installed, returns empty lists.
    """
    suggestions: dict[str, list[str]] = {
        'aiida_profiles': [],
        'aiida_codes': [],
        'aiida_computers': [],
        'aiida_potential_families': [],
    }

    try:
        from aiida.manage.configuration import get_config

        config = get_config()
        suggestions['aiida_profiles'] = list(config.profile_names)
    except Exception:
        return suggestions

    try:
        from aiida.manage.configuration import get_profile

        if get_profile() is None:
            return suggestions
    except Exception:
        return suggestions

    try:
        from aiida import orm

        suggestions['aiida_codes'] = sorted(
            c.full_label for c in orm.QueryBuilder().append(orm.Code).all(flat=True)
        )
    except Exception:
        pass

    try:
        from aiida import orm

        suggestions['aiida_computers'] = sorted(
            c.label for c in orm.QueryBuilder().append(orm.Computer).all(flat=True)
        )
    except Exception:
        pass

    try:
        from aiida import orm

        suggestions['aiida_potential_families'] = sorted(
            g.label for g in orm.QueryBuilder().append(orm.Group).all(flat=True)
        )
    except Exception:
        pass

    return suggestions
