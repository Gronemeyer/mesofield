"""Generic external-app subprocess supervisor (Qt-free, stdlib only).

Launches an arbitrary command, streams its stdout, fires ``on_ready`` when a
handshake token appears, and ``on_finished(exit_code)`` when the process exits.
``terminate()`` does a graceful ``terminate`` then a ``kill`` fallback.

This is the reusable engine behind :class:`mesofield.devices.stimulus_base
.SubprocessStimulusDevice` -- it is deliberately framework-agnostic (no Qt, no
mesofield config) so it can babysit any external stimulus app (MousePortal,
PsychoPy, ...).  See ``mesofield/devices/mouseportal_device.py`` for a concrete
subclass-driven user.
"""

from __future__ import annotations

import subprocess
import threading
from collections import deque
from typing import Callable, Mapping, Optional, Sequence

from mesofield.utils._logger import get_logger


class SubprocessSupervisor:
    """Launch and supervise an external subprocess with a stdout handshake.

    Parameters
    ----------
    command:
        Full argv to launch (e.g. ``[python_exe, "-m", "mouseportal", ...]``).
    ready_token:
        Substring printed by the child on stdout once it is ready. When seen,
        ``on_ready`` fires and :meth:`wait_ready` unblocks.
    cwd:
        Working directory for the child (so its relative asset paths resolve).
    env:
        Optional environment mapping; ``None`` inherits the parent's.
    on_ready / on_finished:
        Callbacks fired (from the reader thread) on the readiness handshake and
        on process exit, respectively. ``on_finished`` receives the exit code.
    name:
        Short label used in log lines and the reader thread name.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        ready_token: str,
        cwd: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
        on_ready: Optional[Callable[[], None]] = None,
        on_finished: Optional[Callable[[int], None]] = None,
        name: str = "subprocess",
    ) -> None:
        self.command = list(command)
        self.ready_token = ready_token
        self.cwd = cwd
        self.env = dict(env) if env is not None else None
        self._on_ready = on_ready
        self._on_finished = on_finished
        self.name = name

        self.logger = get_logger(f"{__name__}.SubprocessSupervisor[{name}]")
        self._proc: Optional[subprocess.Popen] = None
        self._reader: Optional[threading.Thread] = None
        self._ready = threading.Event()
        # Rolling tail of the child's merged stdout/stderr, so a failure
        # handler can show why it died before the readiness handshake.
        self._tail: deque[str] = deque(maxlen=200)

    # -- lifecycle ------------------------------------------------------
    def start(self) -> None:
        """Launch the subprocess and begin streaming its stdout."""
        self.logger.info(f"Launching: {' '.join(self.command)} (cwd={self.cwd})")
        self._proc = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._reader = threading.Thread(
            target=self._read_stdout, name=f"{self.name}-stdout", daemon=True
        )
        self._reader.start()

    def wait_ready(self, timeout: Optional[float] = None) -> bool:
        """Block until the readiness handshake fires (or ``timeout``)."""
        return self._ready.wait(timeout=timeout)

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def output_tail(self) -> str:
        """The child's last ~4000 chars of merged stdout/stderr."""
        return "\n".join(self._tail)[-4000:]

    def terminate(self, timeout: float = 5.0) -> None:
        """Stop the subprocess: ``terminate`` first, ``kill`` as a fallback."""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.logger.warning(f"{self.name} did not exit; killing.")
            proc.kill()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.logger.error(f"{self.name} could not be killed.")

    # -- internals ------------------------------------------------------
    def _read_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            for line in self._proc.stdout:
                line = line.rstrip("\n")
                self.logger.info(f"[{self.name}] {line}")
                self._tail.append(line)
                if self.ready_token in line and not self._ready.is_set():
                    self._ready.set()
                    if self._on_ready is not None:
                        try:
                            self._on_ready()
                        except Exception as exc:
                            self.logger.warning(f"on_ready callback failed: {exc}")
        finally:
            code = self._proc.wait()
            self.logger.info(f"{self.name} exited with code {code}")
            if self._on_finished is not None:
                try:
                    self._on_finished(code)
                except Exception as exc:
                    self.logger.warning(f"on_finished callback failed: {exc}")
