"""SQLite schema for ATLAS GUI project files.

The DB is a thin index / summary layer over the sibling `<project>.atlas/`
directory.  Heavy data (atom positions, model weights, plot images) is kept
on disk; only small queryable fields and pointers live here.
"""

from __future__ import annotations

SCHEMA_VERSION = 1


DDL = [
    """
    CREATE TABLE IF NOT EXISTS project_meta (
        key   TEXT PRIMARY KEY,
        value TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS config_snapshots (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        section     TEXT    NOT NULL,
        label       TEXT,
        toml        TEXT    NOT NULL,
        created_at  TEXT    NOT NULL,
        is_active   INTEGER NOT NULL DEFAULT 0
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_snapshots_section_active
        ON config_snapshots(section, is_active);
    """,
    """
    CREATE TABLE IF NOT EXISTS structures (
        atl_id          TEXT PRIMARY KEY,
        formula         TEXT,
        n_atoms         INTEGER,
        phase           TEXT,
        struct_type     TEXT,
        modifications   TEXT,
        calc_performed  INTEGER NOT NULL DEFAULT 0,
        calc_energy     REAL,
        created_at      TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS dft_runs (
        calc_uuid     TEXT PRIMARY KEY,
        aiida_pk      INTEGER,
        atl_id        TEXT,
        queue         TEXT,
        status        TEXT,
        submitted_at  TEXT,
        finished_at   TEXT,
        energy        REAL,
        notes         TEXT,
        FOREIGN KEY (atl_id) REFERENCES structures(atl_id)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_dft_runs_status
        ON dft_runs(status);
    """,
    """
    CREATE TABLE IF NOT EXISTS al_runs (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        base_workchain_pk   INTEGER,
        config_snapshot_id  INTEGER,
        started_at          TEXT,
        finished_at         TEXT,
        status              TEXT,
        FOREIGN KEY (config_snapshot_id) REFERENCES config_snapshots(id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS al_iterations (
        run_id                  INTEGER NOT NULL,
        iteration               INTEGER NOT NULL,
        db_size                 INTEGER,
        rmse_e                  REAL,
        rmse_f                  REAL,
        mae_e                   REAL,
        mae_f                   REAL,
        model_path              TEXT,
        latent_space_plot_path  TEXT,
        refreshed_at            TEXT,
        PRIMARY KEY (run_id, iteration),
        FOREIGN KEY (run_id) REFERENCES al_runs(id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS benchmark_runs (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        model_path    TEXT,
        benchmark     TEXT,
        metrics_json  TEXT,
        run_at        TEXT
    );
    """,
]


def initialise(conn) -> None:
    """Create every table on a fresh connection."""
    conn.execute('PRAGMA foreign_keys = ON')
    for statement in DDL:
        conn.execute(statement)
    conn.commit()


def read_schema_version(conn) -> int:
    cur = conn.execute("SELECT value FROM project_meta WHERE key='schema_version'")
    row = cur.fetchone()
    return int(row[0]) if row else 0
