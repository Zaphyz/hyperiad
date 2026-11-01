"""Shared notation table.

Each direction builds its own notation. The table keeps every symbol in one
place so that clashes are visible: two meanings for the same symbol inside a
direction is an error, across directions it is a warning (cross-use would be
ambiguous).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from frontier.keys import direction_of, validate_key


@dataclass(frozen=True)
class Notation:
    """One symbol and what it denotes.

    ``unicode`` is used in plain-text output and code docs, ``latex`` in the
    rendered interface, ``ascii`` (optional) for identifiers and terminals.
    """

    key: str
    unicode: str
    latex: str
    meaning: str
    ascii: str = ""

    def __post_init__(self) -> None:
        validate_key(self.key)

    @property
    def direction(self) -> str:
        return direction_of(self.key)


class NotationTable:
    def __init__(self) -> None:
        self._entries: dict[str, Notation] = {}

    def add(self, entry: Notation, *, replace: bool = False) -> Notation:
        if entry.key in self._entries and not replace and self._entries[entry.key] != entry:
            raise KeyError(f"notation {entry.key!r} already registered")
        self._entries[entry.key] = entry
        return entry

    def get(self, key: str) -> Notation:
        return self._entries[key]

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __iter__(self):
        return iter(sorted(self._entries.values(), key=lambda n: n.key))

    def __len__(self) -> int:
        return len(self._entries)

    def for_direction(self, direction: str) -> list[Notation]:
        return [n for n in self if n.direction == direction]

    def clashes(self) -> list[tuple[str, list[Notation]]]:
        """Symbols (unicode or LaTeX form) shared by more than one entry."""
        by_symbol: dict[str, list[Notation]] = defaultdict(list)
        for n in self:
            for form in {f"unicode:{n.unicode}", f"latex:{n.latex}"}:
                by_symbol[form].append(n)
        return [(sym, ns) for sym, ns in sorted(by_symbol.items()) if len(ns) > 1]
