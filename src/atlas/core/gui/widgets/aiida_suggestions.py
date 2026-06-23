"""Fetch AiiDA entity lists for schema-form dropdown suggestions."""

from __future__ import annotations

import time
from collections.abc import Callable

from PySide6.QtCore import QThread, Signal

_cache: dict[str, list[str]] | None = None
_cache_time: float = 0.0
_CACHE_TTL: float = 30.0


def _fetch_aiida_suggestions_uncached() -> dict[str, list[str]]:
    """Query AiiDA for codes/computers/groups (no caching)."""
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


def fetch_aiida_suggestions() -> dict[str, list[str]]:
    """Return cached suggestions, or fetch synchronously if cache is stale."""
    global _cache, _cache_time
    now = time.monotonic()
    if _cache is not None and (now - _cache_time) < _CACHE_TTL:
        return _cache

    suggestions = _fetch_aiida_suggestions_uncached()
    _cache = suggestions
    _cache_time = now
    return suggestions


def invalidate_aiida_suggestions_cache() -> None:
    """Reset the cache so the next call re-queries AiiDA."""
    global _cache
    _cache = None


class _SuggestionWorker(QThread):
    """Background thread that fetches AiiDA suggestions."""

    finished = Signal(dict)

    def run(self) -> None:
        self.finished.emit(_fetch_aiida_suggestions_uncached())


_active_worker: _SuggestionWorker | None = None


def fetch_aiida_suggestions_async(
    callback: Callable[[dict[str, list[str]]], None],
) -> None:
    """Populate suggestions via *callback*, without blocking the UI.

    If the cache is warm the callback fires immediately (synchronously).
    Otherwise a background thread is started; the callback fires on the
    main thread when the query completes.
    """
    global _cache, _cache_time, _active_worker

    now = time.monotonic()
    if _cache is not None and (now - _cache_time) < _CACHE_TTL:
        callback(_cache)
        return

    if _active_worker is not None and _active_worker.isRunning():
        return

    def _on_done(result: dict[str, list[str]]) -> None:
        global _cache, _cache_time, _active_worker
        _cache = result
        _cache_time = time.monotonic()
        _active_worker = None
        callback(result)

    _active_worker = _SuggestionWorker()
    _active_worker.finished.connect(_on_done)
    _active_worker.start()
