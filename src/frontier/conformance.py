"""Tracking which of a reference implementation's test cases our code reproduces.

A ported test cites the cases it covers::

    @ported("hypergraph-paclet", "tests/ArrayAlgebraExamples.wls", 21, 22)
    def test_matrix_product(): ...

:func:`coverage` reads those citations from test sources as text (no import needed)
and compares them with the reference's test cases, so `frontier wl coverage` can
say how much of the reference's behaviour is pinned down here.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from frontier.wl import WLIndex

F = TypeVar("F", bound=Callable)

_PORTED_RE = re.compile(r'@ported\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*((?:,\s*\d+\s*)+),?\s*\)')


def ported(ref: str, path: str, *lines: int) -> Callable[[F], F]:
    """Mark a test as reproducing the reference's test cases at ``path:line``."""
    if not lines:
        raise ValueError("cite at least one line")

    def mark(fn: F) -> F:
        fn.__ported__ = [*getattr(fn, "__ported__", []), (ref, path, lines)]  # type: ignore[attr-defined]
        return fn

    return mark


@dataclass
class FileCoverage:
    path: str
    cases: list[int]
    ported: set[int] = field(default_factory=set)

    @property
    def missing(self) -> list[int]:
        return [n for n in self.cases if n not in self.ported]

    @property
    def unknown(self) -> list[int]:
        """Cited lines that are not test cases in the reference (stale citations)."""
        return sorted(self.ported - set(self.cases))


def citations(test_files: Iterable[Path], ref: str) -> dict[str, set[int]]:
    found: dict[str, set[int]] = {}
    for f in test_files:
        for r, path, nums in _PORTED_RE.findall(f.read_text(encoding="utf-8")):
            if r == ref:
                found.setdefault(path, set()).update(int(n) for n in re.findall(r"\d+", nums))
    return found


def coverage(index: WLIndex, ref: str, test_files: Iterable[Path]) -> list[FileCoverage]:
    cited = citations(test_files, ref)
    by_file: dict[str, list[int]] = {}
    for t in index.tests:
        by_file.setdefault(t.path, []).append(t.line)
    files = sorted(set(by_file) | set(cited))
    return [FileCoverage(p, sorted(by_file.get(p, [])), cited.get(p, set())) for p in files]
