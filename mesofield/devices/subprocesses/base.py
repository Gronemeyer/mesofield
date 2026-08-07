"""Generic external-app subprocess supervisor (Qt-free, stdlib only).

Launches an arbitrary command, streams its stdout, fires ``on_ready`` when a
handshake token appears, and ``on_finished(exit_code)`` when the process exits.
``terminate()`` does a graceful ``terminate`` then a ``kill`` fallback.

**Whole-tree teardown.** Stimulus apps spawn their own helper processes
(PsychoPy launches a separate *iohub* process that binds a UDP port; Panda3D may
fork workers). Killing only the launched parent orphans those grandchildren,
which then linger and hold their sockets -- so the next launch fails with "only
one usage of each socket address". To prevent that, the child is launched into a
kill-on-close container and the *whole tree* is reaped:

* Windows -- a **Job Object** with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``. Every
  descendant joins the job; ``TerminateJobObject`` (or simply closing the last
  handle, e.g. when mesofield itself exits) kills the entire tree, even after the
  parent has already died on its own (a crash).
* POSIX -- a new **session/process group** (``start_new_session``); the group is
  signalled with ``killpg`` so descendants die with the leader.

The tree is also reaped the instant the parent exits (see :meth:`_read_stdout`),
so a stimulus that crashes on startup cannot leave a hung helper behind.

This is the reusable engine behind :class:`mesofield.devices.stimulus_base
.SubprocessStimulusDevice` -- it is deliberately framework-agnostic (no Qt, no
mesofield config) so it can babysit any external stimulus app (MousePortal,
PsychoPy, ...).  See ``mesofield/devices/mouseportal_device.py`` for a concrete
subclass-driven user.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from collections import deque
from typing import Callable, Mapping, Optional, Sequence

from mesofield.utils._logger import get_logger


class _WindowsJob:
    """A Windows Job Object that kills its whole process tree on close.

    Wraps the child in a job created with ``KILL_ON_JOB_CLOSE`` so that
    terminating the job (or merely dropping the last handle to it -- including
    when our own process exits) tears down every descendant the child spawned.
    All ``ctypes`` prototypes declare ``HANDLE`` argument types so 64-bit handles
    are not truncated on Win64. Construction is best-effort: any failure leaves
    :attr:`ok` False and the supervisor falls back to a plain process kill.

    Note: the parent is assigned to the job just after creation, so a descendant
    spawned in the microseconds before assignment can escape the job. Stimulus
    helpers we care about (PsychoPy's iohub) launch seconds later and are caught;
    the parent-exit reap (see :meth:`SubprocessSupervisor._read_stdout`) closes
    the rest of the gap. POSIX has no such race -- descendants inherit the
    process group at fork.
    """

    # JobObjectExtendedLimitInformation / JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE.
    _EXTENDED_LIMIT_INFO_CLASS = 9
    _KILL_ON_JOB_CLOSE = 0x00002000

    def __init__(self, proc: subprocess.Popen) -> None:
        self.ok = False
        self._handle = None
        self._k32 = None
        try:
            import ctypes
            from ctypes import wintypes

            class _BASIC_LIMIT(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD),
                ]

            class _IO_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("ReadOperationCount", ctypes.c_uint64),
                    ("WriteOperationCount", ctypes.c_uint64),
                    ("OtherOperationCount", ctypes.c_uint64),
                    ("ReadTransferCount", ctypes.c_uint64),
                    ("WriteTransferCount", ctypes.c_uint64),
                    ("OtherTransferCount", ctypes.c_uint64),
                ]

            class _EXTENDED_LIMIT(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", _BASIC_LIMIT),
                    ("IoInfo", _IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.CreateJobObjectW.restype = wintypes.HANDLE
            k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
            k32.SetInformationJobObject.restype = wintypes.BOOL
            k32.SetInformationJobObject.argtypes = [
                wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
            ]
            k32.AssignProcessToJobObject.restype = wintypes.BOOL
            k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            k32.TerminateJobObject.restype = wintypes.BOOL
            k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
            k32.CloseHandle.restype = wintypes.BOOL
            k32.CloseHandle.argtypes = [wintypes.HANDLE]

            handle = k32.CreateJobObjectW(None, None)
            if not handle:
                return
            info = _EXTENDED_LIMIT()
            info.BasicLimitInformation.LimitFlags = self._KILL_ON_JOB_CLOSE
            if not k32.SetInformationJobObject(
                handle, self._EXTENDED_LIMIT_INFO_CLASS,
                ctypes.byref(info), ctypes.sizeof(info),
            ):
                k32.CloseHandle(handle)
                return
            if not k32.AssignProcessToJobObject(handle, int(proc._handle)):  # type: ignore[attr-defined]
                k32.CloseHandle(handle)
                return
            self._k32 = k32
            self._handle = handle
            self.ok = True
        except Exception:
            self._handle = None
            self.ok = False

    def terminate_and_close(self) -> None:
        """Kill every process in the job and release the handle. Idempotent."""
        handle, self._handle = self._handle, None
        if handle is None or self._k32 is None:
            return
        try:
            self._k32.TerminateJobObject(handle, 1)
        finally:
            self._k32.CloseHandle(handle)


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
        # Whole-tree teardown handles (see module docstring). On POSIX the child
        # is its own session leader, so its pgid == pid; on Windows it is wrapped
        # in a kill-on-close Job Object.
        self._pgid: Optional[int] = None
        self._win_job: Optional[_WindowsJob] = None
        # Guards _reap_tree against the terminate() caller and the reader thread
        # racing to tear the tree down at once.
        self._reap_lock = threading.Lock()
        # Rolling tail of the child's merged stdout/stderr, so a failure
        # handler can show why it died before the readiness handshake.
        self._tail: deque[str] = deque(maxlen=200)

    # -- lifecycle ------------------------------------------------------
    def start(self) -> None:
        """Launch the subprocess (in its own group/job) and stream its stdout."""
        self.logger.debug(f"Launching: {' '.join(self.command)} (cwd={self.cwd})")
        kwargs: dict = dict(
            cwd=self.cwd,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        # New session/group so killpg reaps the whole tree on POSIX.
        if os.name == "posix":
            kwargs["start_new_session"] = True
        self._proc = subprocess.Popen(self.command, **kwargs)
        self.logger.info(f"{self.name} launched (pid {self._proc.pid}).")

        if os.name == "posix":
            # start_new_session makes the child the leader; pgid == pid, and it
            # stays valid for killpg while any group member is alive.
            self._pgid = self._proc.pid
        elif os.name == "nt":
            job = _WindowsJob(self._proc)
            if job.ok:
                self._win_job = job
            else:
                self.logger.warning(
                    f"{self.name}: could not create a kill-on-close Job Object; "
                    f"falling back to a single-process kill (helper processes the "
                    f"app spawns may survive)."
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
    def pid(self) -> Optional[int]:
        """PID of the launched parent process (``None`` before launch)."""
        return self._proc.pid if self._proc is not None else None

    @property
    def output_tail(self) -> str:
        """The child's last ~4000 chars of merged stdout/stderr."""
        return "\n".join(self._tail)[-4000:]

    def terminate(self, timeout: float = 5.0) -> None:
        """Stop the subprocess and its whole tree.

        Graceful ``terminate`` on the parent first, ``kill`` as a fallback, then
        an unconditional tree reap (Job Object / process group) so descendants
        the app spawned -- e.g. PsychoPy's iohub helper -- never linger, even
        when the parent has already exited on its own.
        """
        proc = self._proc
        try:
            if proc is not None and proc.poll() is None:
                if os.name == "posix":
                    self._signal_group(signal.SIGTERM)
                else:
                    proc.terminate()
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"{self.name} did not exit; killing.")
                    if os.name == "posix":
                        self._signal_group(signal.SIGKILL)
                    else:
                        proc.kill()
                    try:
                        proc.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        self.logger.error(f"{self.name} could not be killed.")
        finally:
            self._reap_tree()

    # -- internals ------------------------------------------------------
    def _signal_group(self, sig: int) -> None:
        """Send *sig* to the child's POSIX process group (best effort)."""
        if self._pgid is None:
            return
        try:
            os.killpg(self._pgid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def _reap_tree(self) -> None:
        """Kill any surviving descendants (orphaned helpers). Idempotent."""
        with self._reap_lock:
            if os.name == "posix":
                self._signal_group(signal.SIGKILL)
            elif self._win_job is not None:
                self._win_job.terminate_and_close()
                self._win_job = None

    def _read_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            for line in self._proc.stdout:
                line = line.rstrip("\n")
                self.logger.debug(f"[{self.name}] {line}")
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
            # The parent has exited (cleanly or by crashing). Reap the tree now
            # so a helper it spawned (PsychoPy's iohub, holding a UDP port)
            # cannot survive to block the next launch.
            self._reap_tree()
