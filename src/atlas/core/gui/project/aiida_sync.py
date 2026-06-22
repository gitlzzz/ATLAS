"""Query AiiDA for process states and sync them into the project DB.

All AiiDA imports are deferred so the module can be imported without
AiiDA installed, the functions raise ``RuntimeError`` at call time if
AiiDA is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _load_aiida():
    try:
        from aiida import orm
        from aiida.manage.configuration import get_profile

        profile = get_profile()
        if profile is None:
            raise RuntimeError(
                'No AiiDA profile is loaded. '
                'Set AIIDA_PROFILE or call `aiida.load_profile()` first.'
            )
        return orm
    except ImportError as exc:
        raise RuntimeError('AiiDA is not installed in this environment.') from exc


_AIIDA_STATE_MAP = {
    'created': 'submitted',
    'waiting': 'running',
    'running': 'running',
    'finished': None,
    'excepted': 'errored',
    'killed': 'stopped',
}


@dataclass
class SyncResult:
    """Summary of what changed during an AiiDA sync."""

    al_checked: int = 0
    al_updated: int = 0
    dft_checked: int = 0
    dft_updated: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        parts = []
        if self.al_checked:
            parts.append(f'AL: {self.al_updated}/{self.al_checked} updated')
        if self.dft_checked:
            parts.append(f'DFT: {self.dft_updated}/{self.dft_checked} updated')
        if self.errors:
            parts.append(f'{len(self.errors)} error(s)')
        return ' · '.join(parts) if parts else 'Nothing to sync.'


def _node_status(node) -> str:
    """Map an AiiDA node's process state to our status vocabulary."""
    state = str(node.process_state.value) if node.process_state else 'created'
    mapped = _AIIDA_STATE_MAP.get(state)
    if mapped is not None:
        return mapped
    if state == 'finished':
        if node.is_finished_ok:
            return 'completed'
        return 'errored'
    return 'submitted'


def sync_al_runs(project) -> SyncResult:
    """Refresh AL run statuses from AiiDA."""
    orm = _load_aiida()
    result = SyncResult()

    runs = project.list_al_runs()
    active_runs = [
        r
        for r in runs
        if r.get('base_workchain_pk') is not None
        and r.get('status') in ('submitted', 'running')
    ]
    result.al_checked = len(active_runs)

    for run in active_runs:
        pk = run['base_workchain_pk']
        run_id = run['id']
        try:
            node = orm.load_node(pk)
        except Exception as exc:
            result.errors.append(f'AL run #{run_id} (PK {pk}): {exc}')
            continue

        new_status = _node_status(node)
        old_status = run.get('status', '')
        if new_status != old_status:
            finished = new_status in ('completed', 'errored', 'stopped')
            project.update_al_run_status(run_id, new_status, finished=finished)
            result.al_updated += 1

    return result


def sync_dft_runs(project) -> SyncResult:
    """Refresh DFT run statuses from AiiDA."""
    orm = _load_aiida()
    result = SyncResult()

    runs = project.list_dft_runs()
    active_runs = [
        r
        for r in runs
        if r.get('aiida_pk') is not None and r.get('status') in ('submitted', 'running')
    ]
    result.dft_checked = len(active_runs)

    for run in active_runs:
        pk = run['aiida_pk']
        calc_uuid = run['calc_uuid']
        try:
            node = orm.load_node(pk)
        except Exception as exc:
            result.errors.append(f'DFT {calc_uuid[:8]}… (PK {pk}): {exc}')
            continue

        state = str(node.process_state.value) if node.process_state else 'created'
        mapped = _AIIDA_STATE_MAP.get(state)
        if mapped is not None:
            new_status = mapped
        elif state == 'finished':
            new_status = 'finished' if node.is_finished_ok else 'failed'
        else:
            new_status = 'submitted'

        old_status = run.get('status', '')
        if new_status != old_status:
            from datetime import UTC, datetime

            now = datetime.now(tz=UTC).isoformat()
            finished = new_status in ('finished', 'failed')
            with project.conn:
                if finished:
                    project.conn.execute(
                        'UPDATE dft_runs SET status = ?, finished_at = ? '
                        'WHERE calc_uuid = ?',
                        (new_status, now, calc_uuid),
                    )
                else:
                    project.conn.execute(
                        'UPDATE dft_runs SET status = ? WHERE calc_uuid = ?',
                        (new_status, calc_uuid),
                    )
            result.dft_updated += 1

    return result


def sync_all(project) -> SyncResult:
    """Sync both AL and DFT runs from AiiDA. Returns combined result."""
    combined = SyncResult()

    try:
        al = sync_al_runs(project)
        combined.al_checked = al.al_checked
        combined.al_updated = al.al_updated
        combined.errors.extend(al.errors)
    except RuntimeError as exc:
        combined.errors.append(f'AL sync: {exc}')

    try:
        dft = sync_dft_runs(project)
        combined.dft_checked = dft.dft_checked
        combined.dft_updated = dft.dft_updated
        combined.errors.extend(dft.errors)
    except RuntimeError as exc:
        combined.errors.append(f'DFT sync: {exc}')

    return combined
