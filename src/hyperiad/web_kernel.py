"""The entry points the notebook page calls, in the browser (Pyodide) or on a server.

Everything crosses the boundary as JSON text, so the page needs no knowledge of
Python objects.
"""

from __future__ import annotations

import json

from hyperiad.kernel import BUILTINS, Session

_session: Session | None = None


def start(budget: float | None = None) -> str:
    global _session
    _session = Session(budget=budget)
    return json.dumps({"ok": True})


def run(code: str) -> str:
    if _session is None:
        start()
    return json.dumps(_session.run(code))


def builtins() -> str:
    """Name and usage of every builtin, for the reference panel and highlighting."""
    import hyperiad.builtins  # noqa: F401 - registers them

    return json.dumps([{"name": n, "usage": BUILTINS[n].usage} for n in sorted(BUILTINS)])


def run_with_replay(code: str, replay: list[str], budget: float | None) -> dict:
    """Stateless evaluation for a server: rebuild state from earlier cells, then run ``code``."""
    session = Session(budget=budget)
    for earlier in replay:
        session.run(earlier)
    return session.run(code)
