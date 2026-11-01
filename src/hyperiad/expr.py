"""Hyperiad expressions: code is a tree.

Every program is an :class:`Expr` tree whose leaves are atoms: integers, floats,
strings, booleans, ``None`` (null) and :class:`Symbol`. The text syntax
(:mod:`hyperiad.syntax`) is one way of writing a tree; diagrammatic editors are
meant to be others. Operators are ordinary heads -- ``a + b`` is
``Expr(Symbol("Plus"), (a, b))`` -- so a tree can always be written back as text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

Atom = int | float | str | bool | None


@dataclass(frozen=True, slots=True)
class Symbol:
    name: str

    def __repr__(self) -> str:
        return f"Symbol({self.name!r})"


@dataclass(frozen=True, slots=True)
class Expr:
    head: Any  # a Symbol, or an Expr for curried calls such as f(x)(y)
    args: tuple[Any, ...] = ()

    def __repr__(self) -> str:
        return f"Expr({self.head!r}, {self.args!r})"

    @property
    def head_name(self) -> str | None:
        return self.head.name if isinstance(self.head, Symbol) else None


def S(name: str) -> Symbol:
    return Symbol(name)


def E(head: str | Symbol | Expr, *args: Any) -> Expr:
    return Expr(Symbol(head) if isinstance(head, str) else head, tuple(args))


def is_head(e: Any, name: str, arity: int | None = None) -> bool:
    return isinstance(e, Expr) and e.head_name == name and (arity is None or len(e.args) == arity)


def walk(e: Any):
    """Every node of the tree, depth first (heads included)."""
    yield e
    if isinstance(e, Expr):
        yield from walk(e.head)
        for a in e.args:
            yield from walk(a)
