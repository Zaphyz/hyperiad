"""Labels as patterns, and labels built out of other labels.

A rewrite rule reads the labels of its input three ways, as the reference paclet does
(``src/Rewriting.wl``):

* a **variable** -- a bare symbol, here a :class:`sympy.Symbol` (what an undefined name is in
  Hyperiad) -- matches any label and remembers it; every later occurrence must find the same;
* **no label** is a wildcard that remembers nothing;
* a **composite** label -- a :class:`Term` or an undefined function applied, such as
  ``comp(g, id(X))`` -- that contains variables is a **structured pattern**: it matches a composite
  label with the same head and as many arguments, argument by argument, binding its variables
  (:func:`match_label`). Matching is syntactic: no equations between labels are assumed;
* anything else is a **literal**, matching only itself. :class:`Verbatim` forces a symbol (or a
  whole composite) to be read literally.

The output's labels are expressions in the variables, written out with the bindings put in:
a rule may name what it builds after what it found. Composite labels such as the paclet's
``Row[{"(", x, y, ")"}]`` are :class:`Term` values, inert expressions that only hold their
parts; :func:`label_text` writes any label as the text a drawing shows.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import sympy
from sympy.core.function import AppliedUndef

CONTRACTED_MARK = "•"  # the vertex label of a contracted index (plex diagrams): drawn as a solid dark vertex


def _freeze(x: Any) -> Any:
    if isinstance(x, list | tuple):
        return tuple(_freeze(v) for v in x)
    return x


@dataclass(frozen=True)
class Term:
    """An inert expression ``head(args...)``, used for labels made of other labels."""

    head: str
    args: tuple = ()

    def __init__(self, head: str, args: Any = ()) -> None:
        object.__setattr__(self, "head", str(head))
        object.__setattr__(self, "args", _freeze(tuple(args)))

    def __str__(self) -> str:
        return label_text(self)

    def __repr__(self) -> str:
        return f"Term({self.head!r}, {self.args!r})"


@dataclass(frozen=True)
class Verbatim:
    """A label in a rule's input to be matched literally, even if it is a symbol."""

    value: Any

    def __init__(self, value: Any) -> None:
        object.__setattr__(self, "value", _freeze(value))


def is_label_variable(label: Any) -> bool:
    """Whether a label in a rule's input binds (a bare symbol) rather than being compared."""
    return isinstance(label, sympy.Symbol)


def _composite(label: Any) -> tuple[str, tuple] | None:
    """``(head, args)`` of a composite label: a :class:`Term` or an applied undefined function."""
    if isinstance(label, Term):
        return label.head, label.args
    if isinstance(label, AppliedUndef):
        return label.func.__name__, tuple(label.args)
    return None


def has_label_variables(label: Any) -> bool:
    """Whether a label in a rule's input binds anything: a variable, or a composite or tuple holding one."""
    if is_label_variable(label):
        return True
    if isinstance(label, Verbatim):
        return False
    parts = _composite(label)
    if parts is not None:
        return any(has_label_variables(a) for a in parts[1])
    if isinstance(label, tuple):
        return any(has_label_variables(a) for a in label)
    return False


def match_label(pattern: Any, host: Any, bindings: Mapping[Any, Any], *, patterns: bool = True) -> dict | None:
    """Match one label of a rule's input against a host label, extending ``bindings``; None if it fails.

    No label matches anything; a variable binds (and must find the same label again); a composite
    or tuple holding variables matches structurally; anything else must be equal. With
    ``patterns=False`` every label is literal (no label is still a wildcard)."""
    if pattern is None:
        return dict(bindings)
    if isinstance(pattern, Verbatim):
        return dict(bindings) if pattern.value == host else None
    if patterns and is_label_variable(pattern):
        if pattern in bindings:
            return dict(bindings) if bindings[pattern] == host else None
        return {**bindings, pattern: host}
    if patterns and has_label_variables(pattern):
        if isinstance(pattern, tuple):
            if not isinstance(host, tuple) or len(host) != len(pattern):
                return None
            pairs = zip(pattern, host, strict=True)
        else:
            mine, theirs = _composite(pattern), _composite(host)
            if theirs is None or mine[0] != theirs[0] or len(mine[1]) != len(theirs[1]):
                return None
            pairs = zip(mine[1], theirs[1], strict=True)
        found: dict | None = dict(bindings)
        for p, h in pairs:
            found = match_label(p, h, found, patterns=True)
            if found is None:
                return None
        return found
    return dict(bindings) if pattern == host else None


def substitute(label: Any, bindings: Mapping[Any, Any]) -> Any:
    """``label`` with every bound variable replaced by what it was bound to."""
    if not bindings or label is None:
        return label
    if is_label_variable(label):
        return bindings.get(label, label)
    if isinstance(label, Term):
        return Term(label.head, tuple(substitute(a, bindings) for a in label.args))
    if isinstance(label, tuple):
        return tuple(substitute(a, bindings) for a in label)
    if isinstance(label, Verbatim):
        return label
    if isinstance(label, AppliedUndef):
        args = tuple(substitute(a, bindings) for a in label.args)
        if all(isinstance(a, sympy.Basic) for a in args):
            return label.func(*args)
        return Term(label.func.__name__, args)  # a part is no longer symbolic: keep it as it is
    if isinstance(label, sympy.Basic):
        found = {k: v for k, v in bindings.items() if is_label_variable(k) and k in label.free_symbols}
        if all(isinstance(v, sympy.Basic | int) for v in found.values()):
            return label.xreplace({k: sympy.sympify(v) for k, v in found.items()})
    return label


# -- text -----------------------------------------------------------------------------------

_SUBSCRIPT = dict(zip("0123456789+-=()aehijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ", strict=True))


def _subscript(text: str) -> str:
    if text and all(c in _SUBSCRIPT for c in text):
        return "".join(_SUBSCRIPT[c] for c in text)
    return "_" + (text if len(text) == 1 else f"{{{text}}}")


def _compose(head: str, args: tuple) -> str:
    parts = [label_text(a) for a in args]
    if head == "Row":
        items = args[0] if len(args) == 1 and isinstance(args[0], tuple) else args
        return "".join(label_text(a) for a in items)
    if head == "Subscript" and len(parts) == 2:
        return parts[0] + _subscript(parts[1])
    if head == "Superscript" and len(parts) == 2:
        return f"{parts[0]}^{parts[1]}"
    if head == "SuperDagger" and len(parts) == 1:
        return parts[0] + "†"
    return f"{head}({', '.join(parts)})"


def label_text(label: Any) -> str:
    """The text a drawing shows for a label: ``Row`` concatenates, ``SuperDagger(x)`` is ``x†``."""
    if label is None:
        return ""
    if isinstance(label, str):
        return label
    if isinstance(label, Term):
        return _compose(label.head, label.args)
    if isinstance(label, AppliedUndef):
        return _compose(label.func.__name__, tuple(label.args))
    if isinstance(label, Verbatim):
        return label_text(label.value)
    if isinstance(label, tuple):
        return "(" + ", ".join(label_text(a) for a in label) + ")"
    return str(label)
