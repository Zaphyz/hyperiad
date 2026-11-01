r"""Scalar functions: maps into the scalar line.

A function is a map, not an assignment of numbers (the user, 2026-10-28). As in the continuum,
where a function is a map from a manifold to a one-dimensional non-compact manifold, a scalar
function on $H$ is a map $f: H \to \mathcal{L}$ into the **scalar line** $\mathcal{L}$: the
path graph on $\mathbb{Z}$ with an edge $\{n\}$ at every vertex as well as the edges
$\{n, n+1\}$. The unary edges are what make scalar functions possible: an edge of $H$ on which
$f$ is constant must go to an edge, and $\{n\}$ is the one that stays at $n$.

The integers here only name the vertices of the path; the structure used is the path itself.
The derivative of $f$ along a tangent vector is its pushforward, a tangent vector of
$\mathcal{L}$, and $\mathcal{L}$ has three at each point: down, stationary, up.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping

from frontier.registry import Evidence, EvidenceKind, claim, define, symbol

from infrageometry.maps import SpatialMap
from infrageometry.motion import Motion, TangentVector, tangent_space
from infrageometry.space import EXPERIMENTS, KEY, Region, SpatialHypergraph, edge_text, point_text

symbol(f"{KEY}.scalar_line", "𝓛", r"\mathcal{L}", "the scalar line: the path on Z with unary edges")
symbol(f"{KEY}.up", "↑", r"\uparrow", "the tangent vector n ⇀ {n, n+1} of the scalar line")
symbol(f"{KEY}.down", "↓", r"\downarrow", "the tangent vector n ⇀ {n-1, n} of the scalar line")
symbol(f"{KEY}.stationary", "○", r"\circ", "the stationary tangent vector n ⇀ {n} of the scalar line")

Point = Hashable
UP, STAY, DOWN = "↑", "○", "↓"


@define(
    f"{KEY}.scalar_line",
    "Scalar line",
    r"The scalar line $\mathcal{L}$ is the spatial hypergraph on $\mathbb{Z}$ with edges $\{n\}$ and "
    r"$\{n, n+1\}$: the path with a unary edge at each point, the hypergraphic one-dimensional "
    r"non-compact manifold. At each point $n$ it has three tangent vectors: $\downarrow = n "
    r"\rightharpoonup \{n-1, n\}$, $\circ = n \rightharpoonup \{n\}$, $\uparrow = n \rightharpoonup "
    r"\{n, n+1\}$. Computations use a finite window $\mathcal{L}_{[a,b]}$.",
    maturity="sketch",
    notation=(f"{KEY}.scalar_line", f"{KEY}.up", f"{KEY}.down", f"{KEY}.stationary"),
    depends_on=(f"{KEY}.spatial_hypergraph",),
    tags=("infrageometry", "functions"),
)
def scalar_line(lo: int, hi: int) -> SpatialHypergraph:
    if hi < lo:
        raise ValueError("the window [lo, hi] is empty")
    return SpatialHypergraph([[n] for n in range(lo, hi + 1)] + [[n, n + 1] for n in range(lo, hi)])


def _line_edge(x) -> frozenset:
    e = frozenset(x if isinstance(x, (list, tuple, set, frozenset)) else [x])
    if not (len(e) == 1 or (len(e) == 2 and max(e) - min(e) == 1)):
        raise ValueError(f"{edge_text(e)} is not an edge of the scalar line")
    return e


@define(
    f"{KEY}.scalar_function",
    "Scalar function",
    r"A scalar function on $H$ is a map $f: H \to \mathcal{L}$. It gives each position $v$ a point "
    r"$f(v) \in \mathbb{Z}$ and each infinitesimal position $e$ an edge $f(e)$ of $\mathcal{L}$ "
    r"containing $f[e]$: so the values on an edge lie within one step, and where $f$ is constant on $e$ "
    r"the edge value is $\{n\}$ (*stationary*) or one of $\{n-1, n\}$, $\{n, n+1\}$ (an *infinitesimal* "
    r"rise or dip, invisible at the positions). The **derivative** of $f$ along $v \rightharpoonup e$ is "
    r"$f_*(v \rightharpoonup e) \in \{\downarrow, \circ, \uparrow\}$. A point is **regular** when both "
    r"$\uparrow$ and $\downarrow$ occur at it, a **local maximum** when only $\downarrow$ and $\circ$ "
    r"occur and $\downarrow$ does, a **local minimum** dually, and **flat** when only $\circ$ occurs.",
    maturity="sketch",
    depends_on=(f"{KEY}.spatial_map", f"{KEY}.scalar_line"),
    tags=("infrageometry", "functions"),
)
class ScalarFunction:
    """``ScalarFunction(space, values, edge_values=None)``: ``values`` at the vertices; edge values
    (as ``[n]`` or ``[n, n+1]``) default to the tight value: ``{n}`` where $f$ is constant on the
    edge, else the one edge ``{n, n+1}`` holding the values."""

    __slots__ = ("space", "values", "edge_values", "map")

    def __init__(
        self,
        space: SpatialHypergraph,
        values: Mapping[Point, int],
        edge_values: Mapping[Iterable[Point], object] | None = None,
    ) -> None:
        vals = {v: int(values[v]) for v in space.vertices}
        for e in space.edge_list():
            got = sorted({vals[v] for v in e})
            if got[-1] - got[0] > 1:
                raise ValueError(
                    f"the values on the edge {edge_text(e)} are {', '.join(map(str, got))}: on one edge they must lie "
                    f"within one step, since the edge must go to an edge of the scalar line"
                )
        given = {frozenset(k): _line_edge(x) for k, x in (edge_values or {}).items()}
        lo = min(vals.values(), default=0) - 1
        hi = max(vals.values(), default=0) + 1
        line = scalar_line(lo, hi)
        emap = {e: given[e] for e in space.edges if e in given}
        f = SpatialMap(space, line, vals, emap)
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "values", vals)
        object.__setattr__(self, "edge_values", dict(f.edge_map))
        object.__setattr__(self, "map", f)

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError("ScalarFunction is immutable")

    def __call__(self, x: Point):
        return self.map(x)

    def derivative(self, t: TangentVector) -> str:
        n, fe = self.values[t.base], self.edge_values[t.edge]
        return STAY if fe == {n} else UP if fe == {n, n + 1} else DOWN

    def differential(self, v: Point) -> dict[TangentVector, str]:
        return {t: self.derivative(t) for t in tangent_space(self.space, v)}

    def kind(self, v: Point) -> str:
        signs = set(self.differential(v).values())
        if UP in signs and DOWN in signs:
            return "regular"
        if DOWN in signs:
            return "maximum"
        if UP in signs:
            return "minimum"
        return "flat"

    def critical_points(self) -> dict[Point, str]:
        return {v: k for v in self.space.vertex_list() if (k := self.kind(v)) != "regular"}

    def level_set(self, n: int) -> Region:
        """$f^{-1}(n)$: the positions at $n$ and the edges sent to $\\{n\\}$ (a closed subspace)."""
        return Region(
            frozenset(v for v, x in self.values.items() if x == n),
            frozenset(e for e, x in self.edge_values.items() if x == {n}),
        )

    def crossing(self, n: int) -> Region:
        """$f^{-1}(\\{n, n+1\\})$: the edges sent to the infinitesimal position between $n$ and $n+1$ (open)."""
        return Region(frozenset(), frozenset(e for e, x in self.edge_values.items() if x == {n, n + 1}))

    def sublevel(self, n: int) -> Region:
        """The points whose value lies at or below $n$ (an edge counts when its whole value does)."""
        return Region(
            frozenset(v for v, x in self.values.items() if x <= n),
            frozenset(e for e, x in self.edge_values.items() if max(x) <= n),
        )

    def push_motion(self, m: Motion) -> Motion:
        return self.map.push_motion(m)

    def __repr__(self) -> str:
        vs = ", ".join(f"{point_text(v)}: {self.values[v]}" for v in self.space.vertex_list())
        return f"ScalarFunction({vs})"


def distance_function(space: SpatialHypergraph, v: Point) -> ScalarFunction:
    """$w \\mapsto d(v, w)$, with each edge sent to the tight edge of its values."""
    dist = space.distances(v)
    if len(dist) != len(space.vertices):
        raise ValueError("the distance function needs a connected space")
    return ScalarFunction(space, dist)


claim(
    f"{KEY}.distance_is_scalar",
    r"For a connected $H$ with at least two points and a point $v$, $w \mapsto d(v, w)$ with each edge "
    r"sent to the edge of $\mathcal{L}$ spanned by its values is a scalar function; its only local "
    r"minimum is $v$, and its level sets are the spheres $S_r(v)$.",
    about=(f"{KEY}.scalar_function", f"{KEY}.neighbourhood"),
    status="supported",
    evidence=[
        Evidence(
            EvidenceKind.ARGUMENT,
            "the vertices of an edge are pairwise adjacent, so their distances differ by at most 1; every w ≠ v "
            "has a neighbour one step closer, so a departure towards it goes down",
        ),
        Evidence(
            EvidenceKind.NUMERICAL, "200 random connected H", source=f"{EXPERIMENTS}/2026-10-28-foundations-checks.py"
        ),
    ],
)

claim(
    f"{KEY}.crossings_separate",
    r"For a scalar function $f$ and $n \in \mathbb{Z}$, removing the infinitesimal level set "
    r"$f^{-1}(\{n, n+1\})$ from $X(H)$ separates the points with value $\le n$ from those with value "
    r"$\ge n+1$: every motion from one side to the other passes through an edge sent to $\{n, n+1\}$.",
    about=(f"{KEY}.scalar_function", f"{KEY}.incidence_space"),
    status="supported",
    evidence=[
        Evidence(
            EvidenceKind.ARGUMENT,
            "push the motion forward: a motion in the scalar line from at most n to at least n+1 passes the "
            "point {n, n+1}, the only point joining the two sides of X(L)",
        ),
        Evidence(
            EvidenceKind.NUMERICAL,
            "every n for 300 random scalar functions (random edge values) on random H",
            source=f"{EXPERIMENTS}/2026-10-28-foundations-checks.py",
        ),
    ],
)

claim(
    f"{KEY}.fundamental_theorem",
    r"For a scalar function $f$ and a motion $\gamma$ from $a$ to $b$, the pushforward $f_*\gamma$ is a "
    r"motion in $\mathcal{L}$ from $f(a)$ to $f(b)$: the derivatives met along $\gamma$ integrate, by "
    r"walking the scalar line, to the change from $f(a)$ to $f(b)$.",
    about=(f"{KEY}.scalar_function", f"{KEY}.motion"),
    status="supported",
    evidence=[Evidence(EvidenceKind.ARGUMENT, "maps send motions to motions and keep the ends")],
)
