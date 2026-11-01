"""Server jobs in isolated processes, with time and memory limits.

A public Hyperiad server runs each request in a child process of its own:

- the cost estimate refuses a job before it starts (the session's budget, as in the browser);
- a wall-clock limit stops a job that runs too long anyway (estimates are per builtin, and a
  notebook can loop);
- an address-space limit stops a job that would use too much memory;
- a semaphore caps how many jobs run at once; a request over the cap is turned away at once.

Children come from a fork server that has already imported Hyperiad, so a job starts in
milliseconds without a fresh interpreter. Only the server uses this module; the browser kernel
runs cells in the page.
"""

from __future__ import annotations

import multiprocessing
import resource
import threading
from dataclasses import dataclass
from typing import Any

from hyperiad.display import error_output


@dataclass(frozen=True)
class Limits:
    timeout: float = 60.0  # seconds of wall-clock time per job, replay included
    memory_mb: int = 1024  # memory a job may allocate
    max_jobs: int = 1  # jobs running at once in this server
    max_replay: int = 200  # earlier cells a request may ask to replay


def _child(conn, code: str, replay: list[str], budget: float | None, memory_mb: int, cpu_s: int) -> None:
    # The data limit counts what a job allocates (heap and anonymous mappings), not the address
    # space that NumPy's thread pools reserve without using.
    resource.setrlimit(resource.RLIMIT_DATA, (memory_mb * 2**20, memory_mb * 2**20))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))
    over = _stopped(f"This task went over the server's memory limit of {memory_mb} MB and was stopped.")
    try:
        from hyperiad.web_kernel import run_with_replay

        result = run_with_replay(code, replay, budget)
        # the kernel reports a failed allocation as an internal error; say what it was
        if any("(MemoryError)" in o.get("data", "") for o in result.get("outputs", []) if o.get("type") == "error"):
            result = over
    except MemoryError:
        result = over
    conn.send(result)
    conn.close()


def _stopped(message: str) -> dict[str, Any]:
    return {"status": "error", "outputs": [error_output(f"{message} Try a smaller instance.", kind="limit")]}


class Busy(Exception):
    """Every job slot is taken."""


class JobRunner:
    def __init__(self, budget: float | None, limits: Limits | None = None) -> None:
        limits = limits or Limits()
        self.budget, self.limits = budget, limits
        self._slots = threading.BoundedSemaphore(limits.max_jobs)
        self._ctx = multiprocessing.get_context("forkserver")
        self._ctx.set_forkserver_preload(["hyperiad", "hyperiad.web_kernel", "hyperiad.builtins"])

    def run(self, code: str, replay: list[str]) -> dict[str, Any]:
        if len(replay) > self.limits.max_replay:
            return _stopped(f"This notebook has more than {self.limits.max_replay} cells to replay on the server.")
        if not self._slots.acquire(blocking=False):
            raise Busy
        try:
            return self._run(code, replay)
        finally:
            self._slots.release()

    def _run(self, code: str, replay: list[str]) -> dict[str, Any]:
        lim = self.limits
        recv, send = self._ctx.Pipe(duplex=False)
        proc = self._ctx.Process(
            target=_child,
            args=(send, code, replay, self.budget, lim.memory_mb, int(lim.timeout) + 1),
            daemon=True,
        )
        proc.start()
        send.close()
        result, timed_out = None, False
        try:
            if recv.poll(lim.timeout):
                result = recv.recv()
            else:
                timed_out = True
        except EOFError:  # the child died without answering: it hit the memory or CPU limit
            pass
        finally:
            recv.close()
            if proc.is_alive():
                proc.kill()
            proc.join(5)
        if result is not None:
            return result
        if timed_out:
            return _stopped(f"This task ran past the server's time limit of {lim.timeout:g} s and was stopped.")
        return _stopped("This task went over the server's memory or time limits and was stopped.")
