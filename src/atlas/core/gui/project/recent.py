"""Persistence for the Hub's "Recent Projects" list.

A small JSON file under the user's ATLAS config directory. Capped at 10
entries; most-recent-first.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from atlas.core.code_utils import get_config_path

RECENT_FILE = 'gui_recent.json'
MAX_ENTRIES = 10


@dataclass
class RecentEntry:
    """One row in the Hub's recent-projects list."""

    path: str
    name: str
    last_opened_at: float

    @property
    def last_opened_when(self) -> str:
        """Human-readable relative time since last open."""
        delta = time.time() - self.last_opened_at
        if delta < 60:
            return 'just now'
        if delta < 3600:
            return f'{int(delta / 60)} min ago'
        if delta < 86400:
            return f'{int(delta / 3600)} h ago'
        return f'{int(delta / 86400)} d ago'


class RecentProjects:
    """File-backed recent-projects list."""

    def __init__(self, path: Path | None = None):
        if path is None:
            path = Path(get_config_path()) / RECENT_FILE
        self.path = Path(path)
        self.entries: list[RecentEntry] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            return
        required = {'path', 'name', 'last_opened_at'}
        self.entries = [
            RecentEntry(**item)
            for item in raw
            if isinstance(item, dict) and required <= item.keys()
        ]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(e) for e in self.entries]
        self.path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    def touch(self, project_path: Path, project_name: str) -> None:
        """Record an open / create of ``project_path`` as most recent."""
        canonical = str(Path(project_path).resolve())
        self.entries = [e for e in self.entries if e.path != canonical]
        self.entries.insert(
            0,
            RecentEntry(path=canonical, name=project_name, last_opened_at=time.time()),
        )
        self.entries = self.entries[:MAX_ENTRIES]
        self._save()

    def remove(self, project_path: Path) -> None:
        canonical = str(Path(project_path).resolve())
        before = len(self.entries)
        self.entries = [e for e in self.entries if e.path != canonical]
        if len(self.entries) != before:
            self._save()

    def prune_missing(self) -> int:
        """Drop entries whose project file no longer exists. Returns count dropped."""
        before = len(self.entries)
        self.entries = [e for e in self.entries if Path(e.path).exists()]
        dropped = before - len(self.entries)
        if dropped:
            self._save()
        return dropped
