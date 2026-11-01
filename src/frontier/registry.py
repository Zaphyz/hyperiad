"""Registry of research objects.

Toolkits record what they introduce here:

* :class:`Concept` -- a definition (object, operation, structure, property),
  optionally tied to the Python code that implements it.
* :class:`Claim` -- a mathematical statement with a status and the evidence
  collected for it.
* :class:`~frontier.notation.Notation` -- symbols, kept in a shared table.

The registry is what agents query to see what exists, what
:mod:`frontier.present` renders for users, and the place where an informal
definition is linked to a formal counterpart, when one exists.
"""

from __future__ import annotations

import enum
import importlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, TypeVar

from frontier.keys import DIRECTIONS, direction_of, validate_key
from frontier.notation import Notation, NotationTable

T = TypeVar("T")

TOOLKIT_ENTRY_POINT = "frontier.toolkits"


class Maturity(enum.StrEnum):
    """How settled a concept is. Only STABLE concepts are shown to users by default."""

    SKETCH = "sketch"  # exploratory; may change or vanish at any time
    WORKING = "working"  # used in experiments; details still moving
    STABLE = "stable"  # settled; ready to present
    RETIRED = "retired"  # superseded; kept for history

    @property
    def rank(self) -> int:
        return _MATURITY_RANK[self]


_MATURITY_RANK = {Maturity.RETIRED: -1, Maturity.SKETCH: 0, Maturity.WORKING: 1, Maturity.STABLE: 2}


class ClaimStatus(enum.StrEnum):
    CONJECTURE = "conjecture"  # no evidence yet
    SUPPORTED = "supported"  # computational or heuristic evidence, no proof
    PROVED = "proved"  # informal proof accepted
    FORMALIZED = "formalized"  # Lean-checked, no sorry
    REFUTED = "refuted"  # counterexample or disproof recorded


class EvidenceKind(enum.StrEnum):
    NUMERICAL = "numerical"  # random or exhaustive computation
    SYMBOLIC = "symbolic"  # computer algebra
    LEAN = "lean"  # Lean / Mathlib check
    LITERATURE = "literature"  # known result, with citation
    ARGUMENT = "argument"  # informal proof or proof sketch


@dataclass(frozen=True)
class FormalRef:
    """Pointer to a Lean declaration.

    ``relation`` states how the formal object relates to the concept: it may be
    the same thing, or only a special case, a generalization or an analogue
    (common when the research notion has no Mathlib counterpart yet).
    """

    lean: str
    relation: str = "exact"  # exact | specializes | generalizes | analogue
    note: str = ""

    def __post_init__(self) -> None:
        if self.relation not in {"exact", "specializes", "generalizes", "analogue"}:
            raise ValueError(f"unknown relation {self.relation!r}")


@dataclass(frozen=True)
class SourceRef:
    """Pointer into an external reference listed in ``references/manifest.toml``.

    ``locator`` says where (see :mod:`frontier.refs`), e.g. ``symbol:ArrayMultiply``
    in a reference implementation or ``label:fishassoc`` in a paper. ``role`` says
    what the reference is to this object.
    """

    ref: str
    locator: str = ""
    role: str = "definition"  # definition | implementation | result | example | background
    note: str = ""

    def __post_init__(self) -> None:
        if self.role not in {"definition", "implementation", "result", "example", "background"}:
            raise ValueError(f"unknown role {self.role!r}")


@dataclass(frozen=True)
class Evidence:
    kind: EvidenceKind
    summary: str
    source: str = ""  # experiment path, Lean declaration, citation
    supports: bool = True  # False marks a counterexample / disproof


@dataclass
class Concept:
    key: str
    name: str
    definition: str  # informal; Markdown with $...$ LaTeX
    maturity: Maturity = Maturity.SKETCH
    notation: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    formal: FormalRef | None = None
    sources: tuple[SourceRef, ...] = ()
    tags: tuple[str, ...] = ()
    impl: Any = None  # the Python object implementing the concept, if any

    def __post_init__(self) -> None:
        validate_key(self.key)
        self.maturity = Maturity(self.maturity)

    @property
    def direction(self) -> str:
        return direction_of(self.key)

    @property
    def impl_path(self) -> str | None:
        if self.impl is None:
            return None
        module = getattr(self.impl, "__module__", None)
        qualname = getattr(self.impl, "__qualname__", None)
        return f"{module}.{qualname}" if module and qualname else None


@dataclass
class Claim:
    key: str
    statement: str  # informal; Markdown with $...$ LaTeX
    status: ClaimStatus = ClaimStatus.CONJECTURE
    about: tuple[str, ...] = ()  # concept keys
    evidence: list[Evidence] = field(default_factory=list)
    formal: FormalRef | None = None
    sources: tuple[SourceRef, ...] = ()
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_key(self.key)
        self.status = ClaimStatus(self.status)

    @property
    def direction(self) -> str:
        return direction_of(self.key)

    def add_evidence(self, evidence: Evidence) -> Claim:
        self.evidence.append(evidence)
        return self


@dataclass(frozen=True)
class Issue:
    level: str  # "error" | "warning"
    key: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.key}: {self.message}"


class RegistryError(KeyError):
    pass


class Registry:
    def __init__(self) -> None:
        self._concepts: dict[str, Concept] = {}
        self._claims: dict[str, Claim] = {}
        self.notation = NotationTable()

    # -- registration ---------------------------------------------------

    def add_concept(self, concept: Concept, *, replace: bool = False) -> Concept:
        existing = self._concepts.get(concept.key)
        if existing is not None and not replace and not _same_origin(existing, concept):
            raise RegistryError(f"concept {concept.key!r} already registered")
        if concept.key in self._claims:
            raise RegistryError(f"{concept.key!r} is already a claim key")
        self._concepts[concept.key] = concept
        return concept

    def add_claim(self, claim: Claim, *, replace: bool = False) -> Claim:
        existing = self._claims.get(claim.key)
        # Re-running a module (notebook reload) re-registers the same claim text.
        if existing is not None and not replace and existing.statement != claim.statement:
            raise RegistryError(f"claim {claim.key!r} already registered")
        if claim.key in self._concepts:
            raise RegistryError(f"{claim.key!r} is already a concept key")
        self._claims[claim.key] = claim
        return claim

    def add_notation(self, entry: Notation, *, replace: bool = False) -> Notation:
        return self.notation.add(entry, replace=replace)

    def define(self, key: str, name: str, definition: str, **fields: Any) -> Callable[[T], T]:
        """Decorator: register the decorated class/function as a concept's implementation."""

        def decorate(obj: T) -> T:
            concept = self.add_concept(Concept(key, name, definition, impl=obj, **fields))
            try:
                obj.__frontier_concept__ = concept.key  # type: ignore[attr-defined]
            except (AttributeError, TypeError):
                pass
            return obj

        return decorate

    # -- queries --------------------------------------------------------

    def concept(self, key: str) -> Concept:
        try:
            return self._concepts[key]
        except KeyError:
            raise RegistryError(f"no concept {key!r}") from None

    def claim(self, key: str) -> Claim:
        try:
            return self._claims[key]
        except KeyError:
            raise RegistryError(f"no claim {key!r}") from None

    def get(self, key: str) -> Concept | Claim | Notation:
        for table in (self._concepts, self._claims):
            if key in table:
                return table[key]
        if key in self.notation:
            return self.notation.get(key)
        raise RegistryError(f"nothing registered under {key!r}")

    def concepts(
        self,
        direction: str | None = None,
        min_maturity: Maturity | str | None = None,
        tag: str | None = None,
    ) -> list[Concept]:
        floor = Maturity(min_maturity).rank if min_maturity is not None else None
        out = [
            c
            for c in self._concepts.values()
            if (direction is None or c.direction == direction)
            and (floor is None or c.maturity.rank >= floor)
            and (tag is None or tag in c.tags)
        ]
        return sorted(out, key=lambda c: c.key)

    def claims(
        self,
        direction: str | None = None,
        status: ClaimStatus | str | None = None,
        about: str | None = None,
    ) -> list[Claim]:
        status = ClaimStatus(status) if status is not None else None
        out = [
            c
            for c in self._claims.values()
            if (direction is None or c.direction == direction)
            and (status is None or c.status == status)
            and (about is None or about in c.about)
        ]
        return sorted(out, key=lambda c: c.key)

    def directions(self) -> list[str]:
        seen = {c.direction for c in self._concepts.values()}
        seen |= {c.direction for c in self._claims.values()}
        seen |= {n.direction for n in self.notation}
        return [d for d in DIRECTIONS if d in seen] + sorted(seen - set(DIRECTIONS))

    def dependencies(self, key: str, *, transitive: bool = True) -> list[str]:
        direct = list(self.concept(key).depends_on)
        if not transitive:
            return direct
        seen: list[str] = []
        stack = direct[::-1]
        while stack:
            k = stack.pop()
            if k in seen or k not in self._concepts:
                continue
            seen.append(k)
            stack.extend(self._concepts[k].depends_on[::-1])
        return seen

    def dependents(self, key: str) -> list[str]:
        return sorted(c.key for c in self._concepts.values() if key in c.depends_on)

    def search(self, text: str) -> list[Concept | Claim]:
        """Case-insensitive substring search over keys, names, definitions and statements."""
        needle = text.lower()
        hits: list[Concept | Claim] = [
            c
            for c in self._concepts.values()
            if needle in f"{c.key} {c.name} {c.definition} {' '.join(c.tags)}".lower()
        ]
        hits += [c for c in self._claims.values() if needle in f"{c.key} {c.statement} {' '.join(c.tags)}".lower()]
        return sorted(hits, key=lambda c: c.key)

    # -- consistency ----------------------------------------------------

    def validate(self) -> list[Issue]:
        issues: list[Issue] = []
        add = lambda level, key, msg: issues.append(Issue(level, key, msg))  # noqa: E731

        for key in [*self._concepts, *self._claims, *(n.key for n in self.notation)]:
            if direction_of(key) not in DIRECTIONS:
                add("warning", key, f"unknown direction {direction_of(key)!r}")

        for c in self._concepts.values():
            for dep in c.depends_on:
                if dep not in self._concepts:
                    add("error", c.key, f"depends on unknown concept {dep!r}")
                elif c.maturity == Maturity.STABLE and self._concepts[dep].maturity.rank < Maturity.STABLE.rank:
                    add(
                        "warning",
                        c.key,
                        f"stable but depends on {self._concepts[dep].maturity.value} {dep!r}",
                    )
            for sym in c.notation:
                if sym not in self.notation:
                    add("error", c.key, f"uses unregistered notation {sym!r}")

        for key in self._find_cycles():
            add("error", key, "dependency cycle")

        for c in self._claims.values():
            for ref in c.about:
                if ref not in self._concepts:
                    add("error", c.key, f"about unknown concept {ref!r}")
            has_counter = any(not e.supports for e in c.evidence)
            if c.status == ClaimStatus.FORMALIZED and c.formal is None:
                add("error", c.key, "formalized but has no Lean reference")
            if c.status == ClaimStatus.REFUTED and not has_counter:
                add("error", c.key, "refuted but no counter-evidence recorded")
            if c.status == ClaimStatus.SUPPORTED and not c.evidence:
                add("error", c.key, "supported but no evidence recorded")
            if c.status in {ClaimStatus.SUPPORTED, ClaimStatus.PROVED, ClaimStatus.FORMALIZED} and has_counter:
                add("error", c.key, f"{c.status.value} but counter-evidence recorded")

        for symbol, entries in self.notation.clashes():
            directions = {n.direction for n in entries}
            keys = ", ".join(n.key for n in entries)
            level = "error" if len(directions) == 1 else "warning"
            add(level, entries[0].key, f"symbol `{symbol}` shared by {keys}")

        return issues

    def _find_cycles(self) -> list[str]:
        state: dict[str, int] = {}  # 1 = on stack, 2 = done
        in_cycle: set[str] = set()

        def visit(key: str, path: list[str]) -> None:
            state[key] = 1
            path.append(key)
            for dep in self._concepts[key].depends_on:
                if dep not in self._concepts:
                    continue
                if state.get(dep) == 1:
                    in_cycle.update(path[path.index(dep) :])
                elif dep not in state:
                    visit(dep, path)
            path.pop()
            state[key] = 2

        for key in self._concepts:
            if key not in state:
                visit(key, [])
        return sorted(in_cycle)

    # -- toolkits -------------------------------------------------------

    def load_toolkits(self, names: Iterable[str] | None = None) -> list[str]:
        """Import installed toolkits so their definitions register themselves.

        Toolkits advertise themselves under the ``frontier.toolkits`` entry point
        group. Pass ``names`` to load only some of them.
        """
        wanted = set(names) if names is not None else None
        loaded = []
        for ep in entry_points(group=TOOLKIT_ENTRY_POINT):
            if wanted is None or ep.name in wanted:
                importlib.import_module(ep.value)
                loaded.append(ep.name)
        return sorted(loaded)


def _same_origin(a: Concept, b: Concept) -> bool:
    """True when ``b`` is ``a`` re-registered by re-importing the same module."""
    return a.impl is not None and b.impl is not None and a.impl_path == b.impl_path


REGISTRY = Registry()


def define(key: str, name: str, definition: str, **fields: Any) -> Callable[[T], T]:
    """Decorator registering an implementation as a concept in the default registry."""
    return REGISTRY.define(key, name, definition, **fields)


def concept(key: str, name: str, definition: str, **fields: Any) -> Concept:
    """Register a concept with no Python implementation."""
    return REGISTRY.add_concept(Concept(key, name, definition, **fields))


def claim(key: str, statement: str, **fields: Any) -> Claim:
    return REGISTRY.add_claim(Claim(key, statement, **fields))


def symbol(key: str, unicode: str, latex: str, meaning: str, ascii: str = "") -> Notation:
    return REGISTRY.add_notation(Notation(key, unicode, latex, meaning, ascii))
