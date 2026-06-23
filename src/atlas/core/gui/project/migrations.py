"""Project schema migrations.

Each migration is a function `migrate_v{N}_to_v{N+1}(conn)`. `apply_migrations`
walks them in order from the project's current `schema_version` up to
``schema.SCHEMA_VERSION``.
"""

from __future__ import annotations

from atlas.core.gui.project import schema

# Register migrations as (from_version, function).
_MIGRATIONS: list[tuple[int, callable]] = [
    # Placeholders for future versions; add (N, migrate_vN_to_vN_plus_1) here.
]


def apply_migrations(conn) -> int:
    """Bring the database up to ``schema.SCHEMA_VERSION``. Returns the final version."""
    current = schema.read_schema_version(conn)
    if current == 0:
        # Fresh DB: tables were just created; record the current version.
        conn.execute(
            'INSERT OR REPLACE INTO project_meta (key, value) '
            "VALUES ('schema_version', ?)",
            (str(schema.SCHEMA_VERSION),),
        )
        conn.commit()
        return schema.SCHEMA_VERSION

    for from_version, migration in _MIGRATIONS:
        if from_version >= current and from_version < schema.SCHEMA_VERSION:
            migration(conn)
            current = from_version + 1
            conn.execute(
                "UPDATE project_meta SET value = ? WHERE key='schema_version'",
                (str(current),),
            )
            conn.commit()

    return current
