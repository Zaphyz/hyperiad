r"""Vector fields and their flows.

A vector field chooses a tangent vector at each point: where to go *into*. An integrated motion
also has to come *out* of the infinitesimal position it entered, so a flow needs a second
piece of data, a **transit** at each edge: entering at $v$, leave at $w$. One step of the flow
is then the motion $v \to X(v) \to \tau_{X(v)}(v)$, and an integral curve repeats it. Both
pieces are choices inside $H$; nothing else is used.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping

from frontier.registry import Evidence, EvidenceKind, claim, define

from infrageometry.motion import Motion, TangentVector
from infrageometry.scalar import DOWN, UP, ScalarFunction
from infrageometry.space import EXPERIMENTS, KEY, SpatialHypergraph, edge_key, edge_text, point_text, sort_points

Point = Hashable


@define(
    f"{KEY}.vector_field",
    "Vector field",
    r"A vector field on $H$ is a section of the tangent bundle over some of the points: for each such "
    r"$v$, a tangent vector $X(v) = v \rightharpoonup e$. Where $X(v)$ is the stationary vector "
    r"$v \rightharpoonup \{v\}$ the field is at rest; a field can be everywhere defined only where every "
    r"point lies on an edge, and can rest at $v$ only if $\{v\} \in E$.",
    maturity="sketch",
    depends_on=(f"{KEY}.tangent_vector",),
    tags=("infrageometry", "fields"),
)
class VectorField:
    """``VectorField(space, {v: edge, ...})``: a departure chosen at some points."""

    __slots__ = ("space", "choice")

    def __init__(self, space: SpatialHypergraph, choice: Mapping[Point, object]) -> None:
        picked = {}
        for v, e in choice.items():
            t = TangentVector(v, frozenset(e))
            if t.edge not in space.edges:
                raise ValueError(f"{edge_text(t.edge)} is not an edge of the space")
            picked[v] = t.edge
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "choice", picked)

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError("VectorField is immutable")

    def at(self, v: Point) -> TangentVector | None:
        return TangentVector(v, self.choice[v]) if v in self.choice else None

    def __repr__(self) -> str:
        return "VectorField(" + ", ".join(repr(self.at(v)) for v in sort_points(self.choice)) + ")"


@define(
    f"{KEY}.transit",
    "Transit",
    r"A transit on $H$ says, for each edge $e$ and each vertex $v \in e$ entering it, a vertex "
    r"$\tau_e(v) \in e$ to leave by: the arrival that completes the departure $v \rightharpoonup e$. An "
    r"**exit field** is a transit that ignores the entry ($\tau_e$ constant), a memoryless rule: points "
    r"entering the same edge leave together. Where $\tau_e(v) = v$ the step returns (a stall).",
    maturity="sketch",
    depends_on=(f"{KEY}.motion_graph",),
    tags=("infrageometry", "fields"),
)
class Transit:
    """``Transit(space, {edge: {entry: exit}})``, or ``Transit.exit_field(space, {edge: exit})``."""

    __slots__ = ("space", "rule")

    def __init__(self, space: SpatialHypergraph, rule: Mapping[object, Mapping[Point, Point]]) -> None:
        r = {}
        for e, m in rule.items():
            e = frozenset(e)
            if e not in space.edges:
                raise ValueError(f"{edge_text(e)} is not an edge of the space")
            for a, b in m.items():
                if a not in e or b not in e:
                    raise ValueError(f"the transit of {edge_text(e)} must enter and leave by its vertices")
            r[e] = dict(m)
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "rule", r)

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError("Transit is immutable")

    @classmethod
    def exit_field(cls, space: SpatialHypergraph, exits: Mapping[object, Point]) -> Transit:
        return cls(space, {frozenset(e): dict.fromkeys(frozenset(e), w) for e, w in exits.items()})

    def leave(self, e: frozenset, entry: Point) -> Point | None:
        return self.rule.get(e, {}).get(entry)


@define(
    f"{KEY}.flow",
    "Flow and integral curve",
    r"A vector field $X$ with a transit $\tau$ has the **flow** $\varphi(v) = \tau_{X(v)}(v)$: one step "
    r"$v \to X(v) \to \varphi(v)$. The **integral curve** from $v$ is the motion "
    r"$v \to X(v) \to \varphi(v) \to X(\varphi(v)) \to \cdots$, followed until it reaches a point where "
    r"$X$ or $\tau$ is undefined (an end), a point where it rests ($\varphi(v) = v$, where it stays), or a "
    r"point it has visited (a cycle). Every map $\varphi$ with $\varphi(v) \in N[v]$, on points that lie on some edge, "
    r"is the flow of some $(X, \tau)$; with an exit field only the memoryless ones are.",
    maturity="sketch",
    depends_on=(f"{KEY}.vector_field", f"{KEY}.transit", f"{KEY}.motion"),
    tags=("infrageometry", "fields"),
)
class Flow:
    __slots__ = ("field", "transit")

    def __init__(self, field: VectorField, transit: Transit) -> None:
        if field.space != transit.space:
            raise ValueError("the field and the transit must live on the same space")
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "transit", transit)

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError("Flow is immutable")

    @property
    def space(self) -> SpatialHypergraph:
        return self.field.space

    def step(self, v: Point) -> tuple[frozenset, Point] | None:
        t = self.field.at(v)
        if t is None:
            return None
        w = self.transit.leave(t.edge, v)
        return None if w is None else (t.edge, w)

    def map(self) -> dict[Point, Point]:
        """$\\varphi$ where defined."""
        return {v: s[1] for v in self.space.vertex_list() if (s := self.step(v)) is not None}

    def integral_curve(self, v: Point, max_steps: int | None = None) -> Motion:
        pts: list = [v]
        seen = {v}
        while max_steps is None or len(pts) // 2 < max_steps:
            s = self.step(pts[-1])
            if s is None:
                break
            e, w = s
            if w == pts[-1]:  # a resting point: the curve stays there
                break
            pts += [e, w]
            if w in seen:
                break
            seen.add(w)
        return Motion(self.space, pts)

    def fate(self, v: Point) -> str:
        """``"end"``, ``"rest"`` or ``"cycle"``: how the integral curve from ``v`` finishes."""
        c = self.integral_curve(v)
        last = c.end
        if self.step(last) is None:
            return "end"
        return "rest" if self.step(last)[1] == last else "cycle"


def gradient_flow(f: ScalarFunction, *, ascending: bool = True) -> Flow:
    r"""The gradient flow of $f$: depart into an edge going up (down), leave at its highest (lowest) vertex.

    Ties are broken by the fixed order of edges and vertices. At a point with no such edge the
    field rests on $\{v\}$ if that is an edge, and is undefined otherwise.
    """
    space = f.space
    want = UP if ascending else DOWN
    choice, exits = {}, {}
    for v in space.vertex_list():
        moves = [t.edge for t, s in f.differential(v).items() if s == want]
        if moves:  # prefer an edge that really reaches the next level over an infinitesimal rise
            reach = (lambda e: -max(f.values[w] for w in e)) if ascending else (lambda e: min(f.values[w] for w in e))
            choice[v] = sorted(moves, key=lambda e: (reach(e), edge_key(e)))[0]
        elif frozenset({v}) in space.edges:
            choice[v] = frozenset({v})
    for e in space.edge_list():
        best = sort_points(e)
        exits[e] = (max if ascending else min)(best, key=lambda w: f.values[w])
    return Flow(VectorField(space, choice), Transit.exit_field(space, exits))


claim(
    f"{KEY}.distance_gradient_geodesics",
    r"For the distance function $d_v = d(v, \cdot)$ on a connected $H$, every integral curve of the "
    r"descending gradient flow is a geodesic ending at $v$: geodesics are the gradient lines of distance.",
    about=(f"{KEY}.flow", "infrageometry.scalar_function"),
    status="supported",
    evidence=[
        Evidence(
            EvidenceKind.ARGUMENT,
            "from w ≠ v some edge holds w and a vertex one step closer, so it goes down; leaving at its lowest "
            "vertex lowers the distance by exactly one per step, and a motion of d(v, w) steps is a geodesic",
        ),
        Evidence(
            EvidenceKind.NUMERICAL,
            "every point of 200 random connected H",
            source=f"{EXPERIMENTS}/2026-10-28-foundations-checks.py",
        ),
    ],
)


def describe_flow(flow: Flow) -> str:
    parts = []
    for v in flow.space.vertex_list():
        s = flow.step(v)
        parts.append(f"{point_text(v)} → {edge_text(s[0])} → {point_text(s[1])}" if s else f"{point_text(v)}: —")
    return "; ".join(parts)
