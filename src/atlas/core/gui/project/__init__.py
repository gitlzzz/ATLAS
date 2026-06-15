"""ATLAS GUI project layer.

A `Project` bundles ATLAS configurations, structure databases, and run
summaries into a single addressable unit on disk.  See `project.py` for the
runtime API and `schema.py` for the SQLite schema.
"""

from atlas.core.gui.project.project import Project, ProjectError
from atlas.core.gui.project.recent import RecentProjects

__all__ = ['Project', 'ProjectError', 'RecentProjects']
