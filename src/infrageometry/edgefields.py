r"""Fields as distributions of typed edges, and their algebra as rewriting.

A field is not a function on a whole manifold: it is a distribution of edges of its own type, partial,
finitary and with multiplicities (the user, 2026-10-29: "at the end of the day, it is distribution of
edges what defines fields"). Two kinds are primitive:

* a **vector field** of type $X$ is a multiset of typed arcs of the motion graph: **departures**
  $X(v \rightharpoonup e)$ from a vertex into an edge containing it, and **arrivals**
  $X(e \rightharpoonup v)$ from an edge to one of its vertices (the user, 2026-10-29). An arrival is a
  negative direction: $X(e \rightharpoonup v)$ is the opposite of $X(v \rightharpoonup e)$, and the two
  cancel. A step $a \to b$ of $X$ is a departure $a \rightharpoonup e$ followed by an arrival
  $e \rightharpoonup b$; read through the cancellation, it is $+1$ on the incidence $(a, e)$ and $-1$ on
  $(b, e)$;
* a **scalar field** of type $f$ is a multiset of unary edges $f(v)$ laid on the vertices; its value at
  $v$ is not given but *inferred*: the multiplicity of $f$ at $v$ (the user, 2026-10-29).

What would be algebra is a rewrite rule (the user's direction, 2026-10-29):

* **cancellation** $X(v \rightharpoonup e),\ X(e \rightharpoonup v) \to \varnothing$: opposite
  directions annihilate; the normal form is the *net* field, an integer on each incidence;
* **chains**: a rule with input types $(T_1, \dots, T_k)$ adds $G(p \to q)$ for every chain of edges
  $T_1(p \to \cdot), \dots, T_k(\cdot \to q)$ head to tail starting at a vertex. The steps of $X$ are the
  chains $(X, X)$; **flow composition** $X;Y$ is the chain $(X, X, Y, Y)$, first a step of $X$, then one of
  $Y$; **integration** is the closure $(X, X) \Rightarrow X^{+}$, $(X^{+}, X, X) \Rightarrow X^{+}$, whose
  round $k$ adds the displacements of time $k$;
* **bracket**: from $(X;Y)(a \to c)$ and $(Y;X)(a \to c')$ with $c \ne c'$ add the step $c' \to c$ of
  type $[X, Y]$ as a departure and an arrival through the smallest edges holding $c'$ and $c$ (or as one
  edge $c' \to c$ of higher order if they are not adjacent). The orientation is the classical one:
  flowing $s$ along $X$ then $t$ along $Y$ ends $st\,[X, Y]$ past flowing $Y$ first;
* **comparison**: two scalar values are compared by pairing off unary edges, one at each end, until one
  side runs out; what is left over is the difference, at the higher end.

Saturation applies additive rules everywhere until nothing new appears, round by round, recording which
edges produced which (the trace); what it adds is kept as a set (a derived edge is added once).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from frontier.hypergraphs.core import Edge, Hypergraph
from frontier.registry import Evidence, EvidenceKind, claim, define, symbol

from infrageometry.paths import Q02, adjacent
from infrageometry.scalar import DOWN, STAY, UP, ScalarFunction
from infrageometry.space import KEY, SpatialHypergraph, is_edge, point_key, point_text, sort_points

Point = Hashable

symbol(f"{KEY}.arrival", "e ⇀ v", r"e \rightharpoonup v", "an arrival: from the edge e to its vertex v, minus v ⇀ e")


@dataclass(frozen=True)
class FieldEdge:
    """A typed edge of a field: ``kind(tail → head)``, or a unary edge ``kind(tail)`` when ``head`` is None."""

    kind: str
    tail: Point
    head: Point | None = None

    @property
    def key(self) -> tuple:
        return (self.kind, point_key(self.tail), (-1,) if self.head is None else point_key(self.head))

    def __lt__(self, other: FieldEdge) -> bool:
        return self.key < other.key

    @property
    def unary(self) -> bool:
        return self.head is None

    @property
    def departure(self) -> bool:
        return self.head is not None and not is_edge(self.tail) and is_edge(self.head) and self.tail in self.head

    @property
    def arrival(self) -> bool:
        return self.head is not None and is_edge(self.tail) and not is_edge(self.head) and self.head in self.tail

    @property
    def half_step(self) -> bool:
        return self.departure or self.arrival

    @property
    def incidence(self) -> tuple[Point, frozenset]:
        """The incidence $(v, e)$ of a half-step."""
        if not self.half_step:
            raise ValueError(f"{self!r} is not a departure or an arrival")
        return (self.tail, self.head) if self.departure else (self.head, self.tail)

    @property
    def sign(self) -> int:
        """+1 for a departure, -1 for an arrival (a negative direction)."""
        return 1 if self.departure else -1 if self.arrival else 0

    def opposite(self) -> FieldEdge:
        if self.head is None:
            raise ValueError("a unary edge has no opposite")
        return FieldEdge(self.kind, self.head, self.tail)

    def __repr__(self) -> str:
        if self.head is None:
            return f"{self.kind}({point_text(self.tail)})"
        arrow = "⇀" if self.half_step else "→"
        return f"{self.kind}({point_text(self.tail)} {arrow} {point_text(self.head)})"


def _normal(counts) -> tuple:
    c: Counter = Counter()
    if isinstance(counts, Mapping):
        for e, n in counts.items():
            c[e] += int(n)
    else:
        for item in counts:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], FieldEdge):
                c[item[0]] += int(item[1])
            else:
                c[item] += 1
    if any(n < 0 for n in c.values()):
        raise ValueError("multiplicities are counts of edges: none is negative")
    return tuple(sorted(((e, n) for e, n in c.items() if n > 0), key=lambda en: en[0].key))


@define(
    f"{KEY}.edge_field",
    "Field of typed edges",
    r"A field of type $X$ on $H$ is a finite multiset of edges of that type laid on the space. A **vector "
    r"field** is made of departures $X(v \rightharpoonup e)$ and arrivals $X(e \rightharpoonup v)$, arcs of "
    r"the motion graph; an arrival is a negative direction, the opposite of the departure on the same "
    r"incidence. A **scalar field** is made of unary edges $f(v)$; its value at $v$ is inferred as their "
    r"multiplicity. Fields are partial (a vector field says nothing where it has no arcs) and several live "
    r"together as one multiset of typed edges, the *state* that rewrite rules act on. The user, 2026-10-29: "
    r"field objects are distributions of edges of different types; vector fields point from vertices to edges "
    r"and from edges to vertices; scalar fields are inferred from multiplicities of unary edges.",
    maturity="sketch",
    notation=(f"{KEY}.arrival",),
    depends_on=(f"{KEY}.spatial_hypergraph", f"{KEY}.tangent_vector"),
    tags=("infrageometry", "fields", "rewriting"),
)
@dataclass(frozen=True)
class FieldState:
    space: SpatialHypergraph
    counts: tuple = ()  # ((edge, multiplicity), ...), or any iterable of edges / mapping, normalised
    kinds: tuple = ()  # the order the types were introduced in: it fixes their colours

    def __post_init__(self) -> None:
        cs = _normal(self.counts)
        object.__setattr__(self, "counts", cs)
        seen = list(dict.fromkeys(self.kinds))
        for k in sorted({e.kind for e, _ in cs} - set(seen)):
            seen.append(k)
        object.__setattr__(self, "kinds", tuple(seen))

    @property
    def edges(self) -> frozenset:
        return frozenset(e for e, _ in self.counts)

    def count(self, e: FieldEdge) -> int:
        return dict(self.counts).get(e, 0)

    def of(self, kind: str) -> list[FieldEdge]:
        """The distinct edges of a type, in order."""
        return [e for e, _ in self.counts if e.kind == kind]

    def copies(self, kind: str) -> list[FieldEdge]:
        """The edges of a type with their repeats."""
        return [e for e, n in self.counts if e.kind == kind for _ in range(n)]

    def size(self, kind: str) -> int:
        return sum(n for e, n in self.counts if e.kind == kind)

    def map_of(self, kind: str) -> dict[Point, Point]:
        """The edges of a type with at most one leaving each point, as {tail: head}."""
        return {e.tail: e.head for e in self.of(kind)}

    def add(self, edges: Iterable[FieldEdge]) -> FieldState:
        """Add the edges not yet present, once each (how saturation adds)."""
        new = [e for e in dict.fromkeys(edges) if e not in self.edges]
        return self.plus(new)

    def plus(self, edges: Iterable[FieldEdge]) -> FieldState:
        """Add edges with their repeats (multiset sum)."""
        edges = list(edges)
        c = Counter(dict(self.counts))
        c.update(edges)
        kinds = list(self.kinds) + [k for k in dict.fromkeys(e.kind for e in edges) if k not in self.kinds]
        return FieldState(self.space, c, tuple(kinds))

    def minus(self, edges: Iterable[FieldEdge]) -> FieldState:
        c = Counter(dict(self.counts))
        c.subtract(edges)
        return FieldState(self.space, c, self.kinds)

    def join(self, other: FieldState) -> FieldState:
        if other.space != self.space:
            raise ValueError("the fields live on different spaces")
        out = self
        for k in other.kinds:
            out = out.plus(other.copies(k))
        return FieldState(self.space, out.counts, tuple(dict.fromkeys(self.kinds + other.kinds)))

    def only(self, kinds: Iterable[str]) -> FieldState:
        ks = list(kinds)
        return FieldState(self.space, tuple((e, n) for e, n in self.counts if e.kind in ks), tuple(ks))

    def is_vector_field(self, kind: str) -> bool:
        """Every edge of the type is a departure or an arrival."""
        return all(e.half_step for e in self.of(kind))

    def is_deterministic(self, kind: str) -> bool:
        """A vector field with one copy of each arc and at most one leaving each point: a flow can follow it."""
        es = self.of(kind)
        return (
            self.is_vector_field(kind) and all(self.count(e) == 1 for e in es) and len({e.tail for e in es}) == len(es)
        )

    def net(self, kind: str) -> dict[tuple, int]:
        """The vector field read through cancellation: on each incidence (v, e), departures minus arrivals."""
        out: Counter = Counter()
        for e, n in self.counts:
            if e.kind == kind and e.half_step:
                out[e.incidence] += e.sign * n
        return {k: v for k, v in sorted(out.items(), key=lambda kv: (point_key(kv[0][0]), point_key(kv[0][1]))) if v}

    def values(self, kind: str) -> dict[Point, int]:
        """A scalar field's inferred values: the multiplicity of its unary edges at each vertex (0 if none)."""
        out = dict.fromkeys(self.space.vertex_list(), 0)
        for e, n in self.counts:
            if e.kind == kind and e.unary:
                out[e.tail] += n
        return out

    def to_hypergraph(self) -> Hypergraph:
        """Space and fields as one hypergraph: the space edges unordered; each infinitesimal position a field
        edge touches as one more vertex (named by the edge, e.g. ``{1, 2}``); field edges directed, labelled by
        type and repeated by multiplicity; scalar quanta as directed unary edges."""

        def name(x):
            return point_text(x) if is_edge(x) else x

        extra = sorted({x for e in self.edges for x in (e.tail, e.head) if x is not None and is_edge(x)}, key=point_key)
        space = [Edge(sort_points(e), "Unordered") for e in self.space.edge_list()]
        fields = []
        for e, n in self.counts:
            pts = [name(e.tail)] if e.unary else [name(e.tail), name(e.head)]
            fields += [Edge(pts, "Directed", e.kind) for _ in range(n)]
        return Hypergraph(self.space.vertex_list() + [point_text(x) for x in extra], space + fields)

    def __repr__(self) -> str:
        counts = ", ".join(f"{k}: {self.size(k)}" for k in self.kinds)
        return f"FieldState({counts})"


# -- building fields ---------------------------------------------------------------------------


def through(space: SpatialHypergraph, a: Point, b: Point) -> list[frozenset]:
    """The smallest edges holding both ``a`` and ``b`` (for ``a == b``, the unary edge if there is one)."""
    if a == b:
        e = frozenset({a})
        return [e] if e in space.edges else []
    found = [e for e in space.star(a) if b in e]
    if not found:
        return []
    m = min(len(e) for e in found)
    return sort_points(e for e in found if len(e) == m)


def step_arcs(space: SpatialHypergraph, kind: str, a: Point, b: Point, via: frozenset | None = None) -> list[FieldEdge]:
    """The step ``a → b`` as a departure and an arrival through ``via`` (default: the one smallest edge)."""
    if via is None:
        options = through(space, a, b)
        if not options:
            raise ValueError(f"{point_text(a)} → {point_text(b)} is not a step of the space")
        if len(options) > 1:
            raise ValueError(
                f"the step {point_text(a)} → {point_text(b)} can pass through "
                + " or ".join(point_text(e) for e in options)
                + ": say which"
            )
        via = options[0]
    via = frozenset(via)
    if via not in space.edges or a not in via or b not in via:
        raise ValueError(f"{point_text(via)} is not an edge of the space holding {point_text(a)} and {point_text(b)}")
    return [FieldEdge(kind, a, via), FieldEdge(kind, via, b)]


def arc_field(space: SpatialHypergraph, kind: str, arcs: Iterable[tuple[Point, Point]]) -> FieldState:
    """The vector field of type ``kind`` with the given departures ``(v, e)`` and arrivals ``(e, v)``
    (repeats are multiplicities)."""
    edges = []
    for p, q in arcs:
        e = FieldEdge(kind, p, q)
        if not e.half_step or (e.departure and q not in space.edges) or (e.arrival and p not in space.edges):
            raise ValueError(f"{point_text(p)} ⇀ {point_text(q)} is not a departure or an arrival of the space")
        edges.append(e)
    return FieldState(space, edges, (kind,))


def vector_field(
    space: SpatialHypergraph, kind: str, steps: Mapping[Point, Point], via: Mapping | None = None
) -> FieldState:
    """The vector field of type ``kind`` stepping ``a → steps[a]``: for each, the departure into and the arrival
    out of the edge it passes (the smallest edge holding both, or ``via[a]``); ``a → a`` stays on ``{a}``."""
    edges = []
    for a, b in steps.items():
        edges += step_arcs(space, kind, a, b, (via or {}).get(a))
    return FieldState(space, edges, (kind,))


def direction_field(space: SpatialHypergraph, kind: str, delta, where=None, stay_elsewhere: bool = False):
    """On a space with coordinates: the step to the neighbour at ``position + delta``, where ``where``
    (a test on the position) holds; elsewhere a stationary step if ``stay_elsewhere`` and possible."""
    if not space.coordinates:
        raise ValueError("a direction field needs a space with coordinates (a grid)")
    at = {tuple(round(c, 6) for c in p): v for v, p in space.coordinates.items()}
    steps = {}
    for v in space.vertex_list():
        p = space.coordinates[v]
        if where is None or where(p):
            w = at.get(tuple(round(p[i] + delta[i], 6) for i in range(2)))
            if w is not None and adjacent(space, v, w):
                steps[v] = w
        elif stay_elsewhere and adjacent(space, v, v):
            steps[v] = v
    return vector_field(space, kind, steps)


@define(
    f"{KEY}.scalar_field",
    "Scalar field",
    r"A scalar field of type $f$ is a multiset of unary edges $f(v)$ on the vertices of $H$. Its value at $v$ "
    r"is inferred: the multiplicity $|f|_v$ of $f$ at $v$, a count, not a number given from outside. The map "
    r"version (a scalar function $H \to \mathcal{L}$) gives one by laying $f(v) - \min f$ copies at each $v$.",
    maturity="sketch",
    depends_on=(f"{KEY}.edge_field",),
    tags=("infrageometry", "fields", "functions"),
)
def scalar_field(space: SpatialHypergraph, kind: str, values: Mapping[Point, int]) -> FieldState:
    edges = []
    for v, n in values.items():
        if v not in space.vertices:
            raise ValueError(f"{point_text(v)} is not a position of the space")
        if int(n) < 0:
            raise ValueError("a multiplicity is a count: none is negative")
        edges += [FieldEdge(kind, v)] * int(n)
    return FieldState(space, edges, (kind,))


def scalar_field_of(fn: ScalarFunction, kind: str = "f") -> FieldState:
    """The scalar field of a scalar function (map version): ``fn(v) - min fn`` unary edges at each ``v``."""
    lo = min(fn.values.values(), default=0)
    return scalar_field(fn.space, kind, {v: n - lo for v, n in fn.values.items()})


# -- cancellation ----------------------------------------------------------------------------


@dataclass
class Cancellation:
    """Cancelling opposite arcs: ``pairs`` removed in order, from ``start`` to ``final``."""

    start: FieldState
    pairs: list = field(default_factory=list)

    @property
    def states(self) -> list[FieldState]:
        out = [self.start]
        for d, a in self.pairs:
            out.append(out[-1].minus([d, a]))
        return out

    @property
    def final(self) -> FieldState:
        return self.start.minus([e for pair in self.pairs for e in pair])


@define(
    f"{KEY}.cancellation",
    "Cancellation of opposite directions",
    r"The rule $X(v \rightharpoonup e),\ X(e \rightharpoonup v) \to \varnothing$ removes a departure with the "
    r"arrival on the same incidence and of the same type: an arrival is a negative direction. It terminates "
    r"and is confluent (it only lowers both counts on one incidence), and its normal form is the **net** "
    r"field, departures minus arrivals on each incidence. A stationary step $v \rightharpoonup \{v\} "
    r"\rightharpoonup v$ and a step followed by its reverse cancel to nothing: cancellation does not tell a "
    r"field that is zero somewhere from one not defined there.",
    maturity="sketch",
    depends_on=(f"{KEY}.edge_field",),
    tags=("infrageometry", "fields", "rewriting"),
)
def cancel(state: FieldState, kinds: Iterable[str] | None = None) -> Cancellation:
    ks = list(kinds) if kinds is not None else list(state.kinds)
    out = Cancellation(state)
    for e, n in state.counts:
        if e.kind in ks and e.departure:
            m = min(n, state.count(e.opposite()))
            out.pairs += [(e, e.opposite())] * m
    return out


# -- comparison of scalar values ---------------------------------------------------------------


@dataclass
class Comparison:
    """Comparing a scalar field at ``a`` and ``b`` by pairing off its unary edges: ``pairs`` removed, then
    ``left`` copies remain at ``higher`` (None when they balance)."""

    state: FieldState
    kind: str
    a: Point
    b: Point
    pairs: list
    left: int
    higher: Point | None

    @property
    def sign(self) -> str:
        """↑ when ``b`` is higher, ↓ when ``a`` is, ○ when they balance."""
        return STAY if self.higher is None else UP if self.higher == self.b else DOWN

    def states(self) -> list[FieldState]:
        """The two ends' quanta as they are paired off (only the type compared, only at ``a`` and ``b``)."""
        cur = FieldState(
            self.state.space,
            [FieldEdge(self.kind, v) for v in (self.a, self.b) for _ in range(self.state.values(self.kind)[v])],
            (self.kind,),
        )
        out = [cur]
        for pair in self.pairs:
            cur = cur.minus(pair)
            out.append(cur)
        return out

    def __repr__(self) -> str:
        if self.higher is None:
            return f"Comparison({self.kind}: {point_text(self.a)} and {point_text(self.b)} balance, ○)"
        return (
            f"Comparison({self.kind}: {point_text(self.higher)} is higher by {self.left}, "
            f"{self.sign} from {point_text(self.a)} to {point_text(self.b)})"
        )


@define(
    f"{KEY}.scalar_comparison",
    "Comparison of a scalar field",
    r"The rule $f(a),\ f(b) \to \varnothing$, applied to working copies of the unary edges at $a$ and $b$ "
    r"until one side runs out, compares the values: what is left is $\bigl||f|_b - |f|_a\bigr|$ copies at the "
    r"higher end. Along a step $a \to b$ it gives the **derivative** $\uparrow$, $\circ$ or $\downarrow$ and "
    r"its size: a difference found by cancelling, not by subtracting numbers.",
    maturity="sketch",
    depends_on=(f"{KEY}.scalar_field",),
    tags=("infrageometry", "fields", "functions", "rewriting"),
)
def compare(state: FieldState, kind: str, a: Point, b: Point) -> Comparison:
    vals = state.values(kind)
    na, nb = vals[a], vals[b]
    pairs = [(FieldEdge(kind, a), FieldEdge(kind, b))] * min(na, nb)
    higher = None if na == nb else (b if nb > na else a)
    return Comparison(state, kind, a, b, pairs, abs(nb - na), higher)


# -- rules -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldRule:
    """A rule that adds edges: ``name``; ``needs`` (the input types) and ``gives`` (the output type).

    ``kind`` is ``"chain"`` (edges of the types in ``needs`` head to tail, from a vertex, give one edge from
    the first tail to the last head) or ``"bracket"``.
    """

    name: str
    kind: str
    needs: tuple
    gives: str
    symbol: str = ""

    def matches(self, state: FieldState) -> list[tuple[tuple[FieldEdge, ...], FieldEdge]]:
        """Every application: the input edges and an edge it adds (already present ones included)."""
        out: list = []
        if self.kind == "chain":
            by_tail: dict[tuple, list[FieldEdge]] = {}
            for t in set(self.needs):
                for e in state.of(t):
                    if not e.unary:
                        by_tail.setdefault((t, e.tail), []).append(e)

            def extend(chain: tuple, i: int) -> None:
                if i == len(self.needs):
                    out.append((chain, FieldEdge(self.gives, chain[0].tail, chain[-1].head)))
                    return
                for e in by_tail.get((self.needs[i], chain[-1].head), []):
                    extend(chain + (e,), i + 1)

            for e in state.of(self.needs[0]):
                if not e.unary and not is_edge(e.tail):
                    extend((e,), 1)
        elif self.kind == "bracket":
            xy, yx = self.needs
            other: dict[Point, list[FieldEdge]] = {}
            for e in state.of(yx):
                other.setdefault(e.tail, []).append(e)
            for e in state.of(xy):
                for f in other.get(e.tail, []):
                    if f.head == e.head:
                        continue
                    vias = through(state.space, f.head, e.head)
                    if vias:
                        for via in vias:
                            out.append(((e, f), FieldEdge(self.gives, f.head, via)))
                            out.append(((e, f), FieldEdge(self.gives, via, e.head)))
                    else:
                        out.append(((e, f), FieldEdge(self.gives, f.head, e.head)))
        return out

    def __repr__(self) -> str:
        return f"FieldRule({self.name})"


def chain_rule(needs: Sequence[str], gives: str, name: str | None = None) -> FieldRule:
    return FieldRule(name or " ".join(needs) + f" ⇒ {gives}", "chain", tuple(needs), gives, ";")


def step_rule(x: str, gives: str | None = None) -> FieldRule:
    """$X(a \\rightharpoonup e),\\ X(e \\rightharpoonup b) \\Rightarrow X^{\\circ}(a \\to b)$: the integrated steps."""
    return chain_rule((x, x), gives or f"{x}°", f"steps of {x}")


@define(
    f"{KEY}.composition_rule",
    "Composition rule",
    r"For vector fields $X$ and $Y$ the composition rule adds $(X;Y)(a \to c)$ for every chain "
    r"$X(a \rightharpoonup e_1), X(e_1 \rightharpoonup b), Y(b \rightharpoonup e_2), Y(e_2 \rightharpoonup c)$: "
    r"a step of $X$, then one of $Y$. With $X = Y$ and the output fed back (the **closure** rules "
    r"$X, X \Rightarrow X^{+}$ and $X^{+}, X, X \Rightarrow X^{+}$) saturation gives the integrated flow. "
    r"Composition is defined where the edges meet, nowhere else.",
    maturity="sketch",
    depends_on=(f"{KEY}.edge_field",),
    tags=("infrageometry", "fields", "rewriting"),
)
def composition_rule(x: str, y: str, gives: str | None = None) -> FieldRule:
    return FieldRule(f"{x};{y}", "chain", (x, x, y, y), gives or f"{x};{y}", ";")


def closure_rules(x: str, gives: str | None = None) -> list[FieldRule]:
    """$X, X \\Rightarrow X^{+}$ and $X^{+}, X, X \\Rightarrow X^{+}$: their saturation is the integrated flow."""
    g = gives or f"{x}+"
    return [chain_rule((x, x), g, f"{x} {x} ⇒ {g}"), chain_rule((g, x, x), g, f"{g} {x} {x} ⇒ {g}")]


@define(
    f"{KEY}.bracket_rule",
    "Bracket rule",
    r"From $(X;Y)(a \to c)$ and $(Y;X)(a \to c')$ with $c \ne c'$ the bracket rule adds the step $c' \to c$ "
    r"of type $[X, Y]$, as the departure $c' \rightharpoonup e$ and the arrival $e \rightharpoonup c$ through "
    r"each smallest edge $e$ holding both; if $c'$ and $c$ are not adjacent it adds one edge $c' \to c$ of "
    r"higher order instead. Where $c = c'$ the square closes and the fields commute at $a$; where a composite "
    r"is missing the bracket is undefined. The orientation, from the end of $Y;X$ to the end of $X;Y$, is the "
    r"classical one: $\varphi^Y_t \varphi^X_s(a) - \varphi^X_s \varphi^Y_t(a) = st\,[X, Y](a) + \dots$ "
    r"(chosen by Claude as the most natural, at the user's word, 2026-10-29).",
    maturity="sketch",
    depends_on=(f"{KEY}.composition_rule",),
    tags=("infrageometry", "fields", "rewriting"),
)
def bracket_rules(x: str, y: str) -> list[FieldRule]:
    """Compose both ways, then close the squares."""
    xy, yx = f"{x};{y}", f"{y};{x}"
    return [composition_rule(x, y), composition_rule(y, x), FieldRule(f"[{x},{y}]", "bracket", (xy, yx), f"[{x},{y}]")]


@dataclass
class Saturation:
    """The rounds of a saturation: ``states[k]`` after round ``k``; ``events[k]`` the edges round ``k`` added,
    each with the rule and the input edges that produced it."""

    states: list = field(default_factory=list)
    events: list = field(default_factory=list)

    @property
    def final(self) -> FieldState:
        return self.states[-1]


@define(
    f"{KEY}.saturation",
    "Saturation",
    r"Saturating a state under rules that add edges applies every rule to every match, round after round, "
    r"adding each derived edge once, until a round adds nothing. It terminates on a finite space with finitely "
    r"many types, and its result does not depend on the order of application (the rules only add).",
    maturity="sketch",
    depends_on=(f"{KEY}.edge_field",),
    tags=("infrageometry", "rewriting"),
)
def saturate(state: FieldState, rules: Sequence[FieldRule], max_rounds: int = 1000) -> Saturation:
    trace = Saturation([state], [])
    for _ in range(max_rounds):
        cur = trace.states[-1]
        present = cur.edges
        new: dict[FieldEdge, tuple[FieldRule, tuple]] = {}
        for r in rules:
            for inputs, out in r.matches(cur):
                if out not in present and out not in new:
                    new[out] = (r, inputs)
        if not new:
            return trace
        added = sorted(new)
        trace.events.append([(new[e][0], new[e][1], e) for e in added])
        trace.states.append(cur.add(added))
    raise ValueError(f"no fixed point within {max_rounds} rounds")


def bracket(state: FieldState, x: str, y: str) -> Saturation:
    """Saturate under composition both ways and the bracket rule."""
    return saturate(state, bracket_rules(x, y))


def order(state: FieldState, e: FieldEdge) -> int:
    """How far apart the ends of an edge lie, in half-steps (arcs of the motion graph): 1 for a departure or an
    arrival, 2 for a step, 4 for a composite of two steps, 0 for a unary edge or a loop."""
    if e.unary or e.tail == e.head:
        return 0
    if e.half_step:
        return 1
    d = state.space.distance(e.tail, e.head) if not (is_edge(e.tail) or is_edge(e.head)) else None
    return 2 * d if d is not None else 0


claim(
    f"{KEY}.closure_is_integration",
    r"Saturating a vector field $X$ (a departure and an arrival for each step) under the closure rules gives "
    r"$X^{+}(a \to b)$ exactly when $b$ is reached from $a$ by one or more steps of $X$; round $k$ adds exactly "
    r"the points first reached at time $k$.",
    about=(f"{KEY}.composition_rule", f"{KEY}.saturation"),
    status="supported",
    evidence=[
        Evidence(EvidenceKind.ARGUMENT, "induction on the round: X+ edges of round k are X-chains of k steps"),
        Evidence(
            EvidenceKind.NUMERICAL,
            "random partial vector fields on random spaces: closure = reachability of the flow map",
            source=f"{Q02}/2026-10-29-half-arc-fields-checks.py",
        ),
    ],
)

claim(
    f"{KEY}.coordinate_fields_commute",
    r"On a grid, the east and north fields have no bracket edges: wherever both composites $X;Y$ and $Y;X$ are "
    r"defined they end at the same point. If north is replaced by north on even columns and stationary on odd "
    r"ones, the bracket is a partial vector field of single vertical steps (each a departure and an arrival), "
    r"one per unit square: south where the square starts on an even column, north where it starts on an odd "
    r"one.",
    about=(f"{KEY}.bracket_rule",),
    status="supported",
    evidence=[
        Evidence(EvidenceKind.ARGUMENT, "east then north and north then east both add (1, 1) where defined"),
        Evidence(
            EvidenceKind.NUMERICAL,
            "grids up to 7 x 7",
            source=f"{Q02}/2026-10-29-half-arc-fields-checks.py",
        ),
    ],
)

claim(
    f"{KEY}.cancellation_net_field",
    r"Cancellation of opposite arcs is terminating and confluent; two vector fields have the same normal form "
    r"exactly when they have the same net value (departures minus arrivals) on every incidence. A field and "
    r"its reverse (every arc reversed) cancel to nothing; a field of stationary steps cancels to nothing.",
    about=(f"{KEY}.cancellation",),
    status="supported",
    evidence=[
        Evidence(
            EvidenceKind.ARGUMENT,
            "each application lowers both counts on one incidence by one and touches no other; the normal form "
            "keeps max(d - a, 0) departures and max(a - d, 0) arrivals on each incidence",
        ),
        Evidence(
            EvidenceKind.NUMERICAL,
            "random multisets of arcs: every order of cancellation gives the same normal form",
            source=f"{Q02}/2026-10-29-half-arc-fields-checks.py",
        ),
    ],
)

claim(
    f"{KEY}.scalar_field_map_version",
    r"A scalar field (a multiset of unary edges) is the edge version of a scalar function $H \to \mathcal{L}$ "
    r"with tight edge values, up to a shift, exactly when on every edge its multiplicities lie within one "
    r"step of each other. Then for every departure $a \rightharpoonup e$ the derivative of the map version is "
    r"$\uparrow$ ($\downarrow$) exactly when comparing $a$ with some other vertex of $e$ leaves edges at that "
    r"vertex (at $a$), and $\circ$ when every comparison balances. Scalar fields in general are not so "
    r"constrained: they are the functions whose differences along a step are any counts.",
    about=(f"{KEY}.scalar_field", f"{KEY}.scalar_comparison", f"{KEY}.scalar_function"),
    status="supported",
    evidence=[
        Evidence(
            EvidenceKind.ARGUMENT,
            "a map to the line sends each edge into an edge {n} or {n, n+1}, so its values on an edge lie within "
            "one step; conversely such values define the tight map; the comparison leaves |f(b) - f(a)| copies "
            "at the higher end",
        ),
        Evidence(
            EvidenceKind.NUMERICAL,
            "every scalar function on 300 random spaces, and random multiplicities",
            source=f"{Q02}/2026-10-29-half-arc-fields-checks.py",
        ),
    ],
)
