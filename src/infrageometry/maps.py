r"""Maps of spatial hypergraphs and their pushforwards.

A map sends vertices and edges together (the user, 2026-10-28): positions to positions, and
each infinitesimal position to an infinitesimal position or, collapsing it, to a position
(collapsing maps allowed, the user, 2026-10-29). These are exactly the continuous maps of
incidence spaces that send closed points to closed points. A tangent vector whose edge is kept
is carried to a tangent vector; one whose edge collapses is carried to the point itself: the
motion pauses there.
"""

from __future__ import annotations

import itertools
from collections.abc import Hashable, Iterable, Iterator, Mapping

from frontier.registry import Evidence, EvidenceKind, claim, define, symbol

from infrageometry.motion import Motion, TangentVector, tangent_space
from infrageometry.space import (
    EXPERIMENTS,
    KEY,
    Region,
    SpatialHypergraph,
    edge_key,
    edge_text,
    is_edge,
    point_text,
    sort_points,
)

symbol(f"{KEY}.pushforward", "f_*", r"f_*", "the pushforward of a map of spatial hypergraphs on tangent vectors")

Point = Hashable


@define(
    f"{KEY}.spatial_map",
    "Map of spatial hypergraphs",
    r"A map $f: H \to H'$ sends vertices to vertices and each edge to an edge or a vertex, keeping "
    r"incidence: $v \in e \Rightarrow f(v) \le f(e)$, that is $f(v) \in f(e)$ for an edge image and "
    r"$f(v) = f(e)$ for a vertex image (the edge **collapses**). Equivalently, a continuous map "
    r"$X(H) \to X(H')$ sending positions to positions. The edge part is data: when all of $e$ goes to one "
    r"vertex $u$, $f(e)$ may be $u$ itself, $\{u\}$ (if it is an edge) or any edge through $u$. Maps "
    r"compose, and the identity is a map. Collapsing maps allowed by the user, 2026-10-29.",
    maturity="sketch",
    depends_on=(f"{KEY}.incidence_space",),
    tags=("infrageometry", "maps"),
)
class SpatialMap:
    """``SpatialMap(source, target, vertex_map, edge_map=None)``.

    ``edge_map`` sends edges (iterables of vertices) to target edges, or to a target vertex for
    a collapse. An edge it leaves out gets the *tight* image $f[e]$ when that is an edge of the
    target, else the only target edge containing $f[e]$, else (when $f[e]$ is one vertex on no
    edge) the collapse to it; if there are several choices, the edge must be given.
    """

    __slots__ = ("source", "target", "vertex_map", "edge_map")

    def __init__(
        self,
        source: SpatialHypergraph,
        target: SpatialHypergraph,
        vertex_map: Mapping[Point, Point],
        edge_map: Mapping[Iterable[Point], Iterable[Point] | Point] | None = None,
    ) -> None:
        vmap = {v: vertex_map[v] for v in source.vertices if v in vertex_map}
        missing = [v for v in source.vertex_list() if v not in vmap]
        if missing:
            raise ValueError(f"the vertex map does not say where {', '.join(map(point_text, missing))} go")
        for v, w in vmap.items():
            if w not in target.vertices:
                raise ValueError(f"vertex {point_text(v)} goes to {point_text(w)}, not a vertex of the target")
        given = {frozenset(k): _image_point(x) for k, x in (edge_map or {}).items()}
        emap: dict[frozenset, Point] = {}
        for e in source.edge_list():
            image = frozenset(vmap[v] for v in e)
            if e in given and not is_edge(given[e]):  # a collapse onto one vertex
                fe = given[e]
                if image != {fe}:
                    raise ValueError(
                        f"edge {edge_text(e)} collapses to {point_text(fe)}, but its vertices go to {edge_text(image)}"
                    )
            elif e in given:
                fe = given[e]
                if fe not in target.edges:
                    raise ValueError(f"edge {edge_text(e)} goes to {edge_text(fe)}, not an edge of the target")
                if not image <= fe:
                    raise ValueError(
                        f"edge {edge_text(e)} goes to {edge_text(fe)}, which does not contain the images "
                        f"{edge_text(image)} of its vertices"
                    )
            elif image in target.edges:
                fe = image
            else:
                choices = [x for x in target.edges if image <= x]
                if not choices and len(image) == 1:
                    choices = [next(iter(image))]  # a point on no edge: the edge can only collapse
                if len(choices) != 1:
                    where = "no target edge contains" if not choices else f"{len(choices)} target edges contain"
                    raise ValueError(f"edge {edge_text(e)}: {where} {edge_text(image)}; say where it goes")
                fe = choices[0]
            emap[e] = fe
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "vertex_map", vmap)
        object.__setattr__(self, "edge_map", emap)

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError("SpatialMap is immutable")

    def __call__(self, x: Point) -> Point:
        """The image of a point of $X(H)$: a vertex or an edge."""
        return self.edge_map[frozenset(x)] if isinstance(x, (frozenset, set)) else self.vertex_map[x]

    def pushforward(self, t: TangentVector | Point) -> TangentVector | Point:
        """$f_*(v \rightharpoonup e) = f(v) \rightharpoonup f(e)$; the point $f(v)$ itself when $e$ collapses.

        A point (a vector already collapsed: staying still) goes to its image point, so that
        $(g \circ f)_* = g_* \circ f_*$ holds with collapses too.
        """
        if not isinstance(t, TangentVector):
            return self.vertex_map[t]
        fe = self.edge_map[t.edge]
        return TangentVector(self.vertex_map[t.base], fe) if is_edge(fe) else fe

    def collapses(self) -> list[frozenset]:
        """The edges sent to vertices."""
        return [e for e, fe in self.edge_map.items() if not is_edge(fe)]

    def differential(self, v: Point) -> dict[TangentVector, TangentVector]:
        """$f_*$ at ``v``: each tangent vector at ``v`` with its image at $f(v)$."""
        return {t: self.pushforward(t) for t in tangent_space(self.source, v)}

    def is_immersion_at(self, v: Point) -> bool:
        images = list(self.differential(v).values())
        return all(isinstance(t, TangentVector) for t in images) and len(set(images)) == len(images)

    def is_submersion_at(self, v: Point) -> bool:
        return set(tangent_space(self.target, self.vertex_map[v])) <= set(self.differential(v).values())

    def is_immersion(self) -> bool:
        return all(self.is_immersion_at(v) for v in self.source.vertices)

    def is_submersion(self) -> bool:
        return all(self.is_submersion_at(v) for v in self.source.vertices)

    def is_local_isomorphism(self) -> bool:
        return self.is_immersion() and self.is_submersion()

    def is_injective(self) -> bool:
        """Injective on the points of $X(H)$: distinct vertices, distinct edges, no collapse."""
        vs, es = set(self.vertex_map.values()), set(self.edge_map.values())
        return len(vs) == len(self.vertex_map) and len(es) == len(self.edge_map) and not self.collapses()

    def is_embedding(self) -> bool:
        r"""Injective, and reflecting incidence: $f(v) \in f(e) \Rightarrow v \in e$."""
        return self.is_injective() and all(
            (self.vertex_map[v] in fe) == (v in e) for e, fe in self.edge_map.items() for v in self.source.vertices
        )

    def is_isomorphism(self) -> bool:
        return (
            self.is_embedding()
            and set(self.vertex_map.values()) == set(self.target.vertices)
            and set(self.edge_map.values()) == set(self.target.edges)
        )

    def image(self) -> Region:
        return Region(frozenset(self.vertex_map.values()), frozenset(x for x in self.edge_map.values() if is_edge(x)))

    def push_motion(self, m: Motion) -> Motion:
        """The image motion; where an edge collapses the motion pauses (repeated points are merged)."""
        pts: list = []
        for x in m.points:
            y = self(x)
            if not pts or pts[-1] != y:
                pts.append(y)
        return Motion(self.target, pts)

    def then(self, g: SpatialMap) -> SpatialMap:
        """The composite $g \\circ f$."""
        if g.source != self.target:
            raise ValueError("the second map must start where the first ends")
        vmap = {v: g.vertex_map[w] for v, w in self.vertex_map.items()}
        emap = {e: g(fe) for e, fe in self.edge_map.items()}
        return SpatialMap(self.source, g.target, vmap, emap)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SpatialMap) and (self.source, self.target, self.vertex_map, self.edge_map) == (
            other.source,
            other.target,
            other.vertex_map,
            other.edge_map,
        )

    def __hash__(self) -> int:
        return hash(("SpatialMap", frozenset(self.vertex_map.items()), frozenset(self.edge_map.items())))

    def __repr__(self) -> str:
        vs = ", ".join(f"{point_text(v)} ↦ {point_text(w)}" for v, w in sorted_items(self.vertex_map))
        es = ", ".join(
            f"{edge_text(e)} ↦ {point_text(x)}" for e, x in sorted(self.edge_map.items(), key=lambda p: edge_key(p[0]))
        )
        return f"SpatialMap({vs}; {es})"


def _image_point(x) -> Point:
    """An edge image as given: a collection of vertices is an edge, anything else a vertex (a collapse)."""
    return frozenset(x) if isinstance(x, (list, tuple, set, frozenset)) else x


def sorted_items(m: Mapping) -> list:
    keys = sort_points(m)
    return [(k, m[k]) for k in keys]


def identity_map(space: SpatialHypergraph) -> SpatialMap:
    return SpatialMap(space, space, {v: v for v in space.vertices}, {e: e for e in space.edges})


def all_maps(source: SpatialHypergraph, target: SpatialHypergraph) -> Iterator[SpatialMap]:
    """Every map ``source → target`` (small spaces only: |V'|^|V| vertex maps, then edge choices)."""
    vs = source.vertex_list()
    es = source.edge_list()
    for images in itertools.product(target.vertex_list(), repeat=len(vs)):
        vmap = dict(zip(vs, images, strict=True))
        choices = []
        for e in es:
            image = frozenset(vmap[v] for v in e)
            choices.append([x for x in target.edge_list() if image <= x] + (list(image) if len(image) == 1 else []))
        for pick in itertools.product(*choices):
            yield SpatialMap(source, target, vmap, dict(zip(es, pick, strict=True)))


def curve(m: Motion) -> SpatialMap:
    """The map $I_n \\to H$ of a motion from a vertex to a vertex with $n$ steps."""
    from infrageometry.motion import interval

    if is_edge(m.start) or is_edge(m.end):
        raise ValueError("a curve is a motion from a vertex to a vertex")
    pts = m.points
    vmap = {i: pts[2 * i] for i in range(len(pts) // 2 + 1)}
    emap = {frozenset({i, i + 1}): pts[2 * i + 1] for i in range(len(pts) // 2)}
    return SpatialMap(interval(len(pts) // 2), m.space, vmap, emap)


def motion_of_curve(c: SpatialMap) -> Motion:
    """The motion of a map from an interval (pausing where an edge of the interval collapses)."""
    n = len(c.source.vertices) - 1
    seq: list = [c(0)]
    for i in range(n):
        for y in (c(frozenset({i, i + 1})), c(i + 1)):
            if seq[-1] != y:
                seq.append(y)
    return Motion(c.target, seq)


claim(
    f"{KEY}.maps_are_typed_continuous_maps",
    r"The maps $H \to H'$ (collapses allowed) are exactly the continuous maps $X(H) \to X(H')$ sending "
    r"vertices to vertices; those without collapses are the ones that also send edges to edges.",
    about=(f"{KEY}.spatial_map", f"{KEY}.incidence_space"),
    status="supported",
    evidence=[
        Evidence(
            EvidenceKind.ARGUMENT,
            "a map of Alexandrov spaces is continuous iff it preserves the order; with vertices sent to "
            "vertices, preserving v <= e says f(v) lies in f(e), or equals it when f(e) is a vertex",
        ),
        Evidence(
            EvidenceKind.NUMERICAL,
            "all functions sending vertices to vertices, between 150 pairs of small random spaces: continuity "
            "(preimages of opens open) agrees with being a map",
            source=f"{EXPERIMENTS}/2026-10-28-foundations-checks.py",
        ),
    ],
)

claim(
    f"{KEY}.embedding_is_topological",
    r"A map is an embedding (injective, and $f(v) \in f(e) \Rightarrow v \in e$) exactly when it is a "
    r"homeomorphism of $X(H)$ onto its image with the subspace topology.",
    about=(f"{KEY}.spatial_map", f"{KEY}.subspace"),
    status="supported",
    evidence=[
        Evidence(
            EvidenceKind.ARGUMENT, "an injective monotone map of posets is an order embedding iff it reflects order"
        ),
        Evidence(
            EvidenceKind.NUMERICAL,
            "every map between 150 pairs of small random spaces",
            source=f"{EXPERIMENTS}/2026-10-28-foundations-checks.py",
        ),
    ],
)

claim(
    f"{KEY}.pushforward_chain_rule",
    r"$(g \circ f)_* = g_* \circ f_*$, the identity pushes forward to the identity, and maps send motions "
    r"to motions. A motion from a vertex to a vertex with $n$ steps is the same thing as a map "
    r"$I_n \to H$, and its velocity at $i$ is the pushforward of $i \rightharpoonup \{i, i+1\}$.",
    about=(f"{KEY}.spatial_map", f"{KEY}.interval", f"{KEY}.motion"),
    status="supported",
    evidence=[
        Evidence(EvidenceKind.ARGUMENT, "the pushforward of v ⇀ e is the pair of images; composition is pointwise"),
        Evidence(
            EvidenceKind.NUMERICAL,
            "composites of all maps between small spaces; round trip motion → curve → motion",
            source="infrageometry/tests/test_foundations.py",
        ),
    ],
)
