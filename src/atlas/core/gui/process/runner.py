"""Background process runner for the ATLAS GUI.

`ProcessRunner` executes an external command in a `QThread`, streams its
combined stdout/stderr line-by-line to a Qt signal, optionally tees each
line into a log file, and reports the exit code when finished.  It is the
single point through which the GUI shells out to `atl_*` CLI tools.

When ``detached=True`` the child process is launched in a new session
(``setsid``) so it survives GUI shutdown.  A PID file is written next to
the log file, enabling the GUI to reconnect on next launch via
`DetachedProcessMonitor`.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from PySide6.QtCore import QThread, Signal


class ProcessRunner(QThread):
    """Run a command in a separate thread to keep the GUI responsive.

    Parameters
    ----------
    detached : bool
        If *True*, the child is started in a new session via ``setsid``
        so it survives GUI shutdown.  Output goes directly to
        *log_file* (required when detached); the thread tails the log
        to emit ``log_message`` signals.
    """

    log_message = Signal(str)
    process_finished = Signal(int)

    def __init__(
        self,
        command_args,
        parent=None,
        *,
        cwd: str | Path | None = None,
        log_file: str | Path | None = None,
        detached: bool = False,
    ):
        super().__init__(parent)
        self.command_args = command_args
        self.cwd = Path(cwd) if cwd is not None else None
        self.log_file = Path(log_file) if log_file is not None else None
        self.detached = detached
        self.process: subprocess.Popen | None = None

        if detached and self.log_file is None:
            raise ValueError('detached mode requires a log_file')

    @property
    def pid_file(self) -> Path | None:
        if self.log_file is None:
            return None
        return self.log_file.with_suffix('.pid')

    def run(self):
        if self.detached:
            self._run_detached()
        else:
            self._run_attached()

    # ---------------------------------------------------------- attached

    def _run_attached(self):
        log_handle = None
        try:
            if self.log_file is not None:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                log_handle = open(self.log_file, 'w', encoding='utf-8')  # noqa: SIM115

            header = f'🚀 Running command: {" ".join(self.command_args)}'
            if self.cwd is not None:
                header += f'  (cwd: {self.cwd})'
            self.log_message.emit(header)
            if log_handle is not None:
                log_handle.write(header + '\n')

            self.process = subprocess.Popen(
                self.command_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                bufsize=1,
                cwd=str(self.cwd) if self.cwd is not None else None,
            )
            for line in iter(self.process.stdout.readline, ''):
                stripped = line.rstrip('\n')
                self.log_message.emit(stripped)
                if log_handle is not None:
                    log_handle.write(line)
                    log_handle.flush()
            self.process.stdout.close()
            return_code = self.process.wait()
            self.process_finished.emit(return_code)
        except FileNotFoundError:
            self.log_message.emit(f"Error: Command not found '{self.command_args[0]}'.")
            self.process_finished.emit(-1)
        except Exception as e:
            self.log_message.emit(f'An unexpected error occurred: {e}')
            self.process_finished.emit(-1)
        finally:
            if log_handle is not None:
                log_handle.close()

    # ---------------------------------------------------------- detached

    def _run_detached(self):
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            log_handle = open(self.log_file, 'w', encoding='utf-8')  # noqa: SIM115

            header = f'🚀 Running (detached): {" ".join(self.command_args)}'
            if self.cwd is not None:
                header += f'  (cwd: {self.cwd})'
            log_handle.write(header + '\n')
            log_handle.flush()
            self.log_message.emit(header)

            self.process = subprocess.Popen(
                self.command_args,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                cwd=str(self.cwd) if self.cwd is not None else None,
                start_new_session=True,
            )
            log_handle.close()

            _write_pid_file(self.pid_file, self.process.pid, self.command_args)
            self.log_message.emit(
                f'📌 Detached process started (PID {self.process.pid}). '
                f'It will continue running if the GUI is closed.'
            )

            self._tail_until_done()

        except FileNotFoundError:
            self.log_message.emit(f"Error: Command not found '{self.command_args[0]}'.")
            self.process_finished.emit(-1)
        except Exception as e:
            self.log_message.emit(f'An unexpected error occurred: {e}')
            self.process_finished.emit(-1)

    def _tail_until_done(self):
        """Tail the log file while the detached process is alive."""
        pos = 0
        while self.process.poll() is None:
            pos = self._read_new_lines(pos)
            time.sleep(0.5)
        # Final drain
        time.sleep(0.2)
        self._read_new_lines(pos)

        return_code = self.process.returncode
        _remove_pid_file(self.pid_file)
        self.process_finished.emit(return_code)

    def _read_new_lines(self, pos: int) -> int:
        try:
            with open(self.log_file, encoding='utf-8') as f:
                f.seek(pos)
                for line in f:
                    self.log_message.emit(line.rstrip('\n'))
                return f.tell()
        except OSError:
            return pos

    # ---------------------------------------------------------- stop

    def stop(self):
        if self.process and self.process.poll() is None:
            if self.detached:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            else:
                self.process.terminate()
            self.log_message.emit('Process terminated by user.')


class DetachedProcessMonitor(QThread):
    """Reconnect to a detached process by tailing its log and watching its PID.

    Used when the GUI restarts and finds a PID file from a previous session.
    Emits the same signals as ``ProcessRunner`` so callers can treat it
    identically.
    """

    log_message = Signal(str)
    process_finished = Signal(int)

    def __init__(
        self,
        pid: int,
        log_file: Path,
        pid_file: Path,
        parent=None,
    ):
        super().__init__(parent)
        self.pid = pid
        self.log_file = log_file
        self.pid_file = pid_file
        self._stop_requested = False

    def run(self):
        self.log_message.emit(f'🔄 Reconnected to detached process (PID {self.pid})')
        pos = 0
        while not self._stop_requested:
            if not _pid_alive(self.pid):
                break
            pos = self._read_new_lines(pos)
            time.sleep(1.0)

        # Final drain
        time.sleep(0.3)
        self._read_new_lines(pos)

        if _pid_alive(self.pid):
            self.log_message.emit('Disconnected from detached process (still running).')
            self.process_finished.emit(-1)
        else:
            rc = _read_exit_code(self.pid_file)
            _remove_pid_file(self.pid_file)
            self.log_message.emit(f'Detached process finished (exit code: {rc}).')
            self.process_finished.emit(rc)

    def _read_new_lines(self, pos: int) -> int:
        try:
            with open(self.log_file, encoding='utf-8') as f:
                f.seek(pos)
                for line in f:
                    self.log_message.emit(line.rstrip('\n'))
                return f.tell()
        except OSError:
            return pos

    def stop(self):
        """Stop monitoring (does NOT kill the detached process)."""
        self._stop_requested = True

    def terminate_process(self):
        """Actually kill the detached process."""
        try:
            os.killpg(os.getpgid(self.pid), signal.SIGTERM)
            self.log_message.emit('Detached process terminated by user.')
        except ProcessLookupError:
            pass


# ================================================================ PID file helpers


def _write_pid_file(
    pid_file: Path,
    pid: int,
    command_args: list[str],
) -> None:
    pid_file.write_text(
        json.dumps(
            {
                'pid': pid,
                'command': command_args,
                'started_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            }
        ),
        encoding='utf-8',
    )


def _remove_pid_file(pid_file: Path | None) -> None:
    if pid_file is not None:
        with contextlib.suppress(OSError):
            pid_file.unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _read_exit_code(pid_file: Path) -> int:
    """Best-effort read of exit code; returns -1 if unknown."""
    # The PID file doesn't store exit codes, we only know the process
    # is gone.  In the future we could write a .rc file from a wrapper.
    return -1


def find_detached_process(log_dir: Path, command_prefix: str) -> dict | None:
    """Find a running detached process by scanning PID files.

    Returns ``{'pid': int, 'log_file': Path, 'pid_file': Path,
    'command': list[str]}`` if a live process is found, else ``None``.
    """
    if not log_dir.exists():
        return None
    for pid_file in log_dir.glob(f'{command_prefix}.*.pid'):
        try:
            info = json.loads(pid_file.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        pid = info.get('pid')
        if pid is not None and _pid_alive(pid):
            log_file = pid_file.with_suffix('.log')
            return {
                'pid': pid,
                'log_file': log_file,
                'pid_file': pid_file,
                'command': info.get('command', []),
            }
        else:
            _remove_pid_file(pid_file)
    return None
