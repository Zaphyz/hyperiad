"""Hypergraph rewrite rules, ported from the reference paclet's ``src/Rewriting.wl``.

A rule is a pair of hypergraphs, input and output. Applying it to a hypergraph finds the
ways its input sits inside that hypergraph and, for each, replaces what was matched by the
output. Each such way is an :class:`Event`, which records not only the new hypergraph but
what was matched, created and destroyed: what the causal machinery is built from.

* **Matching.** The input's edges are laid over edges of the hypergraph with the same arity and
  symmetry type, in every vertex ordering the symmetry allows (one for a directed edge, the
  rotations of a cyclic one, every permutation of an unordered one). By default (the paclet's)
  they land on distinct edges and the input's vertices on distinct vertices; ``distinct_edges``
  and ``distinct_vertices`` lift either condition, which is what composition needs (one host
  edge may then fill several input edges, two input vertices may meet). A vertex of the input on
  no edge may sit on any vertex not already used, or on any vertex at all when vertices need not
  be distinct. Labels are matched as described in :mod:`frontier.hypergraphs.labels`, including
  structured patterns. ``restrict`` limits each input edge to given host edge positions (how
  composition keeps the factors apart). A match is one choice of host edges with one vertex map:
  an unordered input edge on an unordered host edge matches once per orientation.
* **Applying.** The matched edges go and the output's edges arrive. Vertices the output
  shares with the input keep the vertices they were matched to; vertices only the output has
  are created, named ``FormalV(1), FormalV(2), ...`` avoiding names already present; vertices
  only the input has are destroyed, but a destroyed vertex leaves the vertex list only when
  nothing left uses it (a rule rewrites what it matched, and nothing else). Where the output
  labels a vertex its label stands; where it does not, the hypergraph's is kept.

Positions (``match_edge_positions``, ``edge_origins``) are 0-based here; Hyperiad shows them
1-based, as the paclet does.
"""

from __future__ import annotations

import itertools
from collections.abc import Hashable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from frontier.hypergraphs.core import Edge, Hypergraph, HypergraphError, Vertex, order_key, symmetry_orderings
from frontier.hypergraphs.labels import match_label, substitute
from frontier.registry import SourceRef, define

_SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


@dataclass(frozen=True, order=True)
class FormalV:
    """The name of a vertex a rewrite created: the paclet's ``\\[FormalV][k]``."""

    index: int

    def __str__(self) -> str:
        return "v" + str(self.index).translate(_SUBSCRIPT_DIGITS)

    def __repr__(self) -> str:
        return f"FormalV({self.index})"

    def order_key(self) -> tuple:
        return (self.index,)


def _as_hypergraph(spec: Any, side: str) -> Hypergraph:
    if isinstance(spec, Hypergraph):
        return spec
    try:
        return Hypergraph(spec)
    except (HypergraphError, TypeError) as e:
        raise HypergraphError("HypergraphRule::hg", f"the {side} {spec!r} is not a hypergraph: {e}") from None


def _sorted_ids(ids) -> list[Hashable]:
    return sorted(ids, key=order_key)


class HypergraphRule:
    """``HypergraphRule(input, output)``: either side anything :class:`Hypergraph` accepts."""

    PROPERTIES = ("Input", "Output", "SharedVertices", "NewVertices", "DeletedVertices", "Signature")
    __slots__ = ("input", "output")

    def __init__(self, input: Any, output: Any) -> None:  # noqa: A002 - the paclet's name
        object.__setattr__(self, "input", _as_hypergraph(input, "input"))
        object.__setattr__(self, "output", _as_hypergraph(output, "output"))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("HypergraphRule is immutable")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, HypergraphRule) and (self.input, self.output) == (other.input, other.output)

    def __hash__(self) -> int:
        return hash(("HypergraphRule", self.input, self.output))

    def __repr__(self) -> str:
        return f"HypergraphRule({self.input!r}, {self.output!r})"

    @property
    def shared_vertices(self) -> list[Hashable]:
        return _sorted_ids(set(self.input.vertex_ids) & set(self.output.vertex_ids))

    @property
    def new_vertices(self) -> list[Hashable]:
        return _sorted_ids(set(self.output.vertex_ids) - set(self.input.vertex_ids))

    @property
    def deleted_vertices(self) -> list[Hashable]:
        return _sorted_ids(set(self.input.vertex_ids) - set(self.output.vertex_ids))

    def prop(self, name: str) -> Any:
        table = {
            "Input": lambda: self.input,
            "Output": lambda: self.output,
            "SharedVertices": lambda: self.shared_vertices,
            "NewVertices": lambda: self.new_vertices,
            "DeletedVertices": lambda: self.deleted_vertices,
            "Signature": lambda: [
                [len(self.input.edges), len(self.input.vertices)],
                [len(self.output.edges), len(self.output.vertices)],
            ],
            "Properties": lambda: list(self.PROPERTIES),
        }
        if name not in table:
            raise HypergraphError("HypergraphRule::prop", f"unknown property {name!r}; see rule['Properties']")
        return table[name]()

    def __getitem__(self, name: str) -> Any:
        return self.prop(name)

    def __call__(self, hg: Hypergraph, **options: Any) -> list[Event]:
        return apply_rule(self, hg, **options)


# -- matching -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Match:
    edge_positions: tuple[int, ...]  # host edge positions, one per input edge (in EdgeList order)
    vertex_map: dict  # rule vertex id -> host vertex id
    label_bindings: dict  # label variable -> label

    def __hash__(self) -> int:
        return hash((self.edge_positions, tuple(self.vertex_map.items()), tuple(self.label_bindings.items())))


def _symmetry_orderings(sym: str, k: int) -> list[tuple[int, ...]]:
    """The orderings of an edge's vertices that its symmetry type says are the same edge."""
    return symmetry_orderings(sym, k)


_FAILED = None


class _Matcher:
    def __init__(self, labeled: bool, label_patterns: bool, distinct: bool, distinct_edges: bool = True) -> None:
        self.labeled, self.patterns, self.distinct = labeled, label_patterns, distinct
        self.distinct_edges = distinct_edges

    def bind_label(self, state, patt: Any, host: Any):
        if state is _FAILED or not self.labeled or patt is None:
            return state
        vertices, labels = state
        found = match_label(patt, host, labels, patterns=self.patterns)
        return _FAILED if found is None else (vertices, found)

    def bind_vertex(self, state, pv: Vertex, hv: Vertex):
        if state is _FAILED:
            return state
        vertices, labels = state
        if pv.id not in vertices:
            if self.distinct and hv.id in vertices.values():
                return _FAILED
            state = ({**vertices, pv.id: hv.id}, labels)
        elif vertices[pv.id] != hv.id:
            return _FAILED
        return self.bind_label(state, pv.label, hv.label)

    def extend(self, state, pe: Edge, he: Edge) -> list:
        with_label = self.bind_label(state, pe.label, he.label)
        if with_label is _FAILED:
            return []
        out = []
        for perm in _symmetry_orderings(he.symmetry, he.arity):
            s = with_label
            for pv, k in zip(pe.vertices, perm, strict=True):
                s = self.bind_vertex(s, pv, he.vertices[k])
            if s is not _FAILED:
                out.append(s)
        return out

    def edges(self, pattern: Sequence[tuple[Edge, Any]], host: Sequence[Edge], used: tuple, state) -> Iterator[tuple]:
        if not pattern:
            yield used, state
            return
        (pe, allowed), rest = pattern[0], pattern[1:]
        for p, he in enumerate(host):
            if (self.distinct_edges and p in used) or pe.arity != he.arity or pe.symmetry != he.symmetry:
                continue
            if allowed is not None and p not in allowed:
                continue
            for s in self.extend(state, pe, he):
                yield from self.edges(rest, host, (*used, p), s)

    def free(self, state, free: Sequence[Vertex], host_vertices: Sequence[Vertex]) -> list:
        if not free:
            return [state]
        if self.distinct:
            taken = set(state[0].values())
            available = [v for v in host_vertices if v.id not in taken]
            choices = itertools.permutations(available, len(free))
        else:  # vertices need not be distinct: an isolated input vertex may sit anywhere
            choices = itertools.product(host_vertices, repeat=len(free))
        out = []
        for choice in choices:
            s = state
            for pv, hv in zip(free, choice, strict=True):
                s = self.bind_vertex(s, pv, hv)
            if s is not _FAILED:
                out.append(s)
        return out


def _check(rule: Any, hg: Any) -> None:
    if not isinstance(rule, HypergraphRule):
        raise HypergraphError("HypergraphRuleApply::rule", f"{rule!r} is not a rewrite rule")
    if not isinstance(hg, Hypergraph):
        raise HypergraphError("HypergraphRuleApply::hg", f"{hg!r} is not a hypergraph")


def rule_matches(
    rule: HypergraphRule,
    hg: Hypergraph,
    *,
    labeled: bool = True,
    label_patterns: bool = True,
    distinct_vertices: bool = True,
    distinct_edges: bool = True,
    restrict: Sequence[Any] | None = None,
    limit: int | None = None,
) -> list[Match]:
    """The ways the rule's input sits inside ``hg``, in the paclet's order, without rewriting.

    ``restrict``, if given, has one entry per input edge (in EdgeList order): the host edge positions
    that edge may land on, or None for any. ``limit`` stops after that many matches."""
    _check(rule, hg)
    m = _Matcher(labeled, label_patterns, distinct_vertices, distinct_edges)
    lhs = rule.input
    if restrict is not None and len(restrict) != len(lhs.edges):
        raise HypergraphError("HypergraphRuleMatches::restrict", "restrict needs one entry per input edge")
    allowed = [None if r is None else set(r) for r in (restrict or [None] * len(lhs.edges))]
    on_edges = {v.id for e in lhs.edges for v in e.vertices}
    free = [v for v in lhs.vertices if v.id not in on_edges]
    found: dict[Match, None] = {}
    for used, state in m.edges(list(zip(lhs.edges, allowed, strict=True)), list(hg.edges), (), ({}, {})):
        for vertices, labels in m.free(state, free, hg.vertices):
            found.setdefault(Match(used, vertices, labels))
            if limit is not None and len(found) >= limit:
                return list(found)
    return list(found)


# -- applying -------------------------------------------------------------------------------


@dataclass
class Event:
    """One application of a rule: the result and an account of what changed."""

    hypergraph: Hypergraph  # the result
    input: Hypergraph  # what the event acted on
    edge_origins: list[int | None]  # per result edge: the host position it carries over, or None if made
    match_vertices: list[Hashable]  # host vertices matched, in the order of the input's vertices
    match_edges: list[Edge]
    match_edge_positions: list[int]
    new_vertices: list[Hashable]
    new_edges: list[Edge]
    deleted_vertices: list[Hashable]
    rule_vertex_map: dict  # rule vertex -> host vertex, created vertices included
    label_bindings: dict
    rule: HypergraphRule | None = field(default=None, compare=False)

    KEYS = (
        "Hypergraph",
        "Input",
        "EdgeOrigins",
        "MatchVertices",
        "MatchEdges",
        "MatchEdgePositions",
        "NewVertices",
        "NewEdges",
        "DeletedVertices",
        "RuleVertexMap",
        "LabelBindings",
        "Rule",
    )

    def __getitem__(self, key: str) -> Any:
        """The paclet's keys (positions stay 0-based here)."""
        if key not in self.KEYS:
            raise KeyError(key)
        attr = "".join("_" + c.lower() if c.isupper() else c for c in key).lstrip("_")
        return getattr(self, attr)

    @property
    def created_edge_positions(self) -> list[int]:
        """Where the rule's new edges landed in the result: read off the origins, never by value."""
        return [i for i, o in enumerate(self.edge_origins) if o is None]


def _new_names(n: int, avoid: set) -> list[FormalV]:
    names, k = [], 0
    while len(names) < n:
        k += 1
        if FormalV(k) not in avoid:
            names.append(FormalV(k))
    return names


def apply_match(rule: HypergraphRule, hg: Hypergraph, match: Match) -> Event:
    lhs, rhs = rule.input, rule.output
    pos, vmap, bindings = list(match.edge_positions), match.vertex_map, match.label_bindings
    host_edges, host_label = list(hg.edges), {v.id: v.label for v in hg.vertices}

    new_ids = rule.new_vertices
    names = _new_names(len(new_ids), set(hg.vertex_ids) | set(rhs.vertex_ids))
    full = {**vmap, **dict(zip(new_ids, names, strict=True))}

    taken = set(pos)
    kept_positions = [p for p in range(len(host_edges)) if p not in taken]
    kept = [Edge(host_edges[p].ids, host_edges[p].symmetry, host_edges[p].label) for p in kept_positions]
    made = [Edge([full.get(i, i) for i in e.ids], e.symmetry, substitute(e.label, bindings)) for e in rhs.edges]

    output_vertices = []
    for v in rhs.vertices:
        vid = full.get(v.id, v.id)
        lab = substitute(v.label, bindings)
        output_vertices.append(Vertex(vid, host_label.get(vid) if lab is None else lab))
    output_ids = {v.id for v in output_vertices}

    deleted_host = [vmap[d] for d in rule.deleted_vertices]
    surviving = {i for e in kept + made for i in e.ids}
    kept_vertices = [
        Vertex(v.id, v.label)
        for v in hg.vertices
        if not ((v.id in deleted_host and v.id not in surviving) or v.id in output_ids)
    ]
    result = Hypergraph(output_vertices + kept_vertices, kept + made)

    # The constructor sorts the edges; redo its remapping and (stable) sort to recover where
    # each result edge came from. Two equal edges are still two elements of the multiset.
    canonical = {v.id: v for v in result.vertices}
    combined = [Edge([canonical[i] for i in e.ids], e.symmetry, e.label) for e in kept + made]
    order = sorted(range(len(combined)), key=lambda i: combined[i].sort_key())
    origins = [kept_positions[i] if i < len(kept) else None for i in order]

    return Event(
        hypergraph=result,
        input=hg,
        edge_origins=origins,
        match_vertices=[vmap[v.id] for v in lhs.vertices],
        match_edges=[host_edges[p] for p in pos],
        match_edge_positions=pos,
        new_vertices=names,
        new_edges=made,
        deleted_vertices=_sorted_ids(set(deleted_host) - surviving),
        rule_vertex_map=full,
        label_bindings=dict(bindings),
        rule=rule,
    )


@define(
    "common.hypergraph_rule",
    "Hypergraph rewrite rule",
    r"A rewrite rule $L \to R$ is a pair of hypergraphs. A **match** of $L$ in $H$ is a map of $L$'s "
    r"edges to edges of $H$ of the same arity and symmetry type (injective by default), with a vertex "
    r"map (injective by default) carrying each edge of $L$ onto its image up to the edge's symmetry, and "
    r"labels matched: a symbol in $L$ binds, a composite label holding symbols matches structurally "
    r"(same head, arguments matched in turn), no label is a wildcard, anything else is literal. The **event** "
    r"replaces the matched edges by $R$'s edges, with $R$'s new vertices freshly named and its labels "
    r"written in the bindings; a vertex only $L$ has is removed only if no remaining edge uses it.",
    maturity="stable",
    depends_on=("common.hypergraph",),
    sources=(
        SourceRef("hypergraph-paclet", "symbol:HypergraphRule", "implementation"),
        SourceRef("hypergraph-paclet", "symbol:HypergraphRuleApply", "implementation"),
    ),
    tags=("hypergraph", "rewriting"),
)
def apply_rule(
    rule: HypergraphRule,
    hg: Hypergraph,
    *,
    labeled: bool = True,
    label_patterns: bool = True,
    distinct_vertices: bool = True,
    distinct_edges: bool = True,
    restrict: Sequence[Any] | None = None,
    limit: int | None = None,
) -> list[Event]:
    """One event per match of the rule's input in ``hg``."""
    matches = rule_matches(
        rule,
        hg,
        labeled=labeled,
        label_patterns=label_patterns,
        distinct_vertices=distinct_vertices,
        distinct_edges=distinct_edges,
        restrict=restrict,
        limit=limit,
    )
    return [apply_match(rule, hg, m) for m in matches]
