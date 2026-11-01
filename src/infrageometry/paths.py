r"""Paths as chains of edges, and homotopy as their rewriting across areas.

A **step path** is a word $v_0 v_1 \dots v_n$ of positions in which each two consecutive positions
lie on a common edge of space: a chain of binary *path edges* $v_{i-1} \to v_i$, each one step.
Paths are deformed by local rewrite rules (the user, 2026-10-29: "a rewrite rule that deforms the
binary edges of paths along vertices that are connected by higher-arity edges"):

* **cancel** $a\,b\,a \to a$: a step straight back is undone;
* **stall** $a\,a \to a$: staying put is no step;
* **contract** $a\,c\,b \to a\,b$ and its inverse **expand** $a\,b \to a\,c\,b$, when $a, b, c$ are
  distinct and lie in one edge of size at least $3$: the path is pushed across that area.

Two paths with the same ends are **area-homotopic** when a chain of rules (in either direction)
turns one into the other; the chain is a proof, a *derivation*. A proof that two paths are *not*
area-homotopic is a **barrier**: a set of adjacent pairs of positions that every area crosses an even
number of times (each triple inside an edge of size $\ge 3$ holds an even number of barrier pairs).
Every rule keeps the parity of the number of barrier crossings, so paths of different parity are not
homotopic.

Area homotopy is the primary homotopy (the user, 2026-10-29). Hyperedges are areas, filled in;
the homotopy of **motions** is derived from it: two motions are homotopic when their step paths are.
The incidence homotopy of motions (only backtracks cancel, hyperedges unfilled,
:mod:`infrageometry.motion`) refines it. The loops at a point up to area homotopy form the
fundamental group, presented here by generators (steps outside a spanning tree) and relators (one
per triangle inside an area).
"""

from __future__ import annotations

import itertools
from collections import deque
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

from frontier.registry import Evidence, EvidenceKind, claim, define

from infrageometry.space import KEY, SpatialHypergraph, edge_text, is_edge, point_text, sort_points

Point = Hashable
Q02 = "themes/hypergraph-geometry/questions/q02-rewriting-calculus/experiments"


@define(
    f"{KEY}.step_path",
    "Step path",
    r"A step path is a word $v_0 v_1 \dots v_n$ of positions with each two consecutive ones in a common "
    r"edge of space (a stall $v\,v$ needs the unary edge $\{v\}$): a chain of directed binary path edges "
    r"$v_{i-1} \to v_i$. It is the motion $v_0 \to e_1 \to v_1 \to \cdots$ with the infinitesimal "
    r"positions forgotten.",
    maturity="sketch",
    depends_on=(f"{KEY}.motion",),
    tags=("infrageometry", "path", "rewriting"),
)
@dataclass(frozen=True)
class StepPath:
    space: SpatialHypergraph
    vertices: tuple

    def __post_init__(self) -> None:
        vs = tuple(self.vertices)
        object.__setattr__(self, "vertices", vs)
        if not vs:
            raise ValueError("a path has at least one position")
        for v in vs:
            if is_edge(v) or v not in self.space.vertices:
                raise ValueError(f"{point_text(v)} is not a position of the space")
        for a, b in itertools.pairwise(vs):
            if not adjacent(self.space, a, b):
                what = "no unary edge to stay on" if a == b else "no edge holds both"
                raise ValueError(f"{point_text(a)} → {point_text(b)} is not a step: {what}")

    @property
    def start(self) -> Point:
        return self.vertices[0]

    @property
    def end(self) -> Point:
        return self.vertices[-1]

    def steps(self) -> list[tuple[Point, Point]]:
        return list(itertools.pairwise(self.vertices))

    def __len__(self) -> int:
        return len(self.vertices) - 1

    def __repr__(self) -> str:
        return "StepPath(" + " → ".join(point_text(v) for v in self.vertices) + ")"

    def __hash__(self) -> int:
        return hash(("StepPath", self.vertices))


def adjacent(space: SpatialHypergraph, a: Point, b: Point) -> bool:
    if a == b:
        return frozenset({a}) in space.edges
    return any(b in e for e in space.star(a))


def area_of(space: SpatialHypergraph, a: Point, b: Point, c: Point) -> frozenset | None:
    """An edge of size at least 3 holding the three distinct positions, if any (the first in order)."""
    if len({a, b, c}) < 3:
        return None
    found = [e for e in space.star(a) if len(e) >= 3 and b in e and c in e]
    return sorted(found, key=lambda e: (len(e), sort_points(e)))[0] if found else None


def path_of_motion(motion) -> StepPath:
    """The step path of a motion: its positions, the infinitesimal ones forgotten; a stall that has no
    unary edge to stay on is dropped (it is no step)."""
    out: list = []
    for v in (x for x in motion.points if not is_edge(x)):
        if out and out[-1] == v and frozenset({v}) not in motion.space.edges:
            continue
        out.append(v)
    return StepPath(motion.space, tuple(out))


# -- the rules -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PathRewrite:
    """One application of a rule: ``rule`` at ``position`` (index of the window's first letter)."""

    rule: str
    position: int
    before: tuple
    after: tuple
    area: frozenset | None = None

    def describe(self) -> str:
        w = " ".join(point_text(v) for v in self.before)
        r = " ".join(point_text(v) for v in self.after) or "·"
        where = f" across {edge_text(self.area)}" if self.area else ""
        return f"{self.rule}: {w} → {r}{where}"


SHORTENING = ("cancel", "stall", "contract")
ALL_RULES = ("cancel", "stall", "contract", "expand", "uncancel", "unstall")


def rewrites(p: StepPath, rules: Sequence[str] = SHORTENING) -> list[tuple[PathRewrite, StepPath]]:
    """Every single application of the given rules to ``p``, with its result."""
    s, vs, out = p.space, p.vertices, []

    def emit(rule, i, length, repl, area=None):
        new = vs[:i] + tuple(repl) + vs[i + length :]
        out.append((PathRewrite(rule, i, vs[i : i + length], tuple(repl), area), StepPath(s, new)))

    for i in range(len(vs)):
        if "cancel" in rules and i + 2 < len(vs) and vs[i] == vs[i + 2] and vs[i] != vs[i + 1]:
            emit("cancel", i, 3, [vs[i]])
        if "stall" in rules and i + 1 < len(vs) and vs[i] == vs[i + 1]:
            emit("stall", i, 2, [vs[i]])
        if "contract" in rules and i + 2 < len(vs):
            area = area_of(s, vs[i], vs[i + 2], vs[i + 1])
            if area is not None:
                emit("contract", i, 3, [vs[i], vs[i + 2]], area)
        if "expand" in rules and i + 1 < len(vs):
            a, b = vs[i], vs[i + 1]
            for c in sort_points(set().union(*(e for e in s.star(a) if len(e) >= 3 and b in e)) - {a, b}):
                emit("expand", i, 2, [a, c, b], area_of(s, a, b, c))
        if "uncancel" in rules:
            for b in sort_points(s.neighbourhood(vs[i], closed=False)):
                emit("uncancel", i, 1, [vs[i], b, vs[i]])
        if "unstall" in rules and frozenset({vs[i]}) in s.edges:
            emit("unstall", i, 1, [vs[i], vs[i]])
    return out


@define(
    f"{KEY}.area_homotopy",
    "Area homotopy",
    r"Step paths are rewritten by four local rules and their inverses: **cancel** $a\,b\,a \to a$, "
    r"**stall** $a\,a \to a$, and **contract** $a\,c\,b \to a\,b$ / **expand** $a\,b \to a\,c\,b$ when "
    r"$a, b, c$ are distinct and lie in one edge of size $\ge 3$: a path is deformed across an area of "
    r"space. Two paths with the same ends are **area-homotopic** when a chain of rule applications, a "
    r"**derivation**, turns one into the other; the derivation is the proof. Hyperedges are filled areas "
    r"here, unlike in the incidence space. Proposed by the user, 2026-10-29, and the primary homotopy: the "
    r"homotopy of motions is derived from it.",
    maturity="sketch",
    depends_on=(f"{KEY}.step_path",),
    tags=("infrageometry", "path", "rewriting"),
)
def find_derivation(
    p: StepPath, q: StepPath, *, max_length: int | None = None, max_paths: int = 200_000
) -> list[tuple[PathRewrite, StepPath]] | None:
    """A shortest derivation from ``p`` to ``q`` (all rules, both directions), or None if none is
    found among paths of at most ``max_length`` steps (default: the longer path plus 2)."""
    if p.space != q.space:
        raise ValueError("the paths live on different spaces")
    if (p.start, p.end) != (q.start, q.end):
        return None
    if p.vertices == q.vertices:
        return []
    limit = max_length if max_length is not None else max(len(p), len(q)) + 2
    parent: dict[tuple, tuple | None] = {p.vertices: None}
    queue = deque([p])
    while queue and len(parent) < max_paths:
        cur = queue.popleft()
        for rw, nxt in rewrites(cur, ALL_RULES):
            if len(nxt) > limit or nxt.vertices in parent:
                continue
            parent[nxt.vertices] = (cur.vertices, rw)
            if nxt.vertices == q.vertices:
                steps = []
                key = nxt.vertices
                while parent[key] is not None:
                    prev, rw_ = parent[key]
                    steps.append((rw_, StepPath(p.space, key)))
                    key = prev
                return steps[::-1]
            queue.append(nxt)
    return None


def reduce_path(p: StepPath) -> list[tuple[PathRewrite, StepPath]]:
    """Shortening rules only (cancel, stall, contract), leftmost first, until none applies."""
    out, cur = [], p
    while True:
        options = rewrites(cur, SHORTENING)
        if not options:
            return out
        rw, cur = min(options, key=lambda o: o[0].position)
        out.append((rw, cur))


# -- barriers: certificates of inequivalence -------------------------------------------------


@define(
    f"{KEY}.barrier",
    "Barrier",
    r"A barrier is a set $B$ of pairs $\{u, v\}$ of adjacent positions such that every triple of "
    r"distinct positions inside an edge of size $\ge 3$ contains an even number of pairs of $B$ (areas "
    r"cannot be crossed an odd number of times). The **crossing parity** of a step path is the parity of "
    r"the number of its steps across $B$. Every area-homotopy rule keeps it, so two paths with the same "
    r"ends and different crossing parities are not area-homotopic: the barrier is a proof of "
    r"inequivalence.",
    maturity="sketch",
    depends_on=(f"{KEY}.area_homotopy",),
    tags=("infrageometry", "path", "rewriting"),
)
@dataclass(frozen=True)
class Barrier:
    space: SpatialHypergraph
    pairs: frozenset

    def __post_init__(self) -> None:
        ps = frozenset(frozenset(x) for x in self.pairs)
        object.__setattr__(self, "pairs", ps)
        for pair in ps:
            if len(pair) != 2 or not adjacent(self.space, *pair):
                raise ValueError(f"{edge_text(pair)} is not a pair of adjacent positions")

    def is_barrier(self) -> bool:
        return not self.violations()

    def violations(self) -> list[frozenset]:
        """The triples inside an area that hold an odd number of barrier pairs."""
        bad = set()
        for e in self.space.edges:
            if len(e) >= 3:
                for t in itertools.combinations(sort_points(e), 3):
                    if sum(frozenset(p) in self.pairs for p in itertools.combinations(t, 2)) % 2:
                        bad.add(frozenset(t))
        return sorted(bad, key=sort_points)

    def crossings(self, p: StepPath) -> int:
        return sum(1 for a, b in p.steps() if a != b and frozenset({a, b}) in self.pairs)

    def __repr__(self) -> str:
        return "Barrier(" + ", ".join(edge_text(x) for x in sorted(self.pairs, key=sort_points)) + ")"


def find_barrier(p: StepPath, q: StepPath, *, max_size: int = 4) -> Barrier | None:
    """A barrier of at most ``max_size`` pairs on which ``p`` and ``q`` cross with different parity."""
    s = p.space
    pairs = sorted({frozenset(c) for e in s.edges for c in itertools.combinations(sort_points(e), 2)}, key=sort_points)
    for k in range(1, max_size + 1):
        for chosen in itertools.combinations(pairs, k):
            b = Barrier(s, frozenset(chosen))
            if (b.crossings(p) - b.crossings(q)) % 2 and b.is_barrier():
                return b
    return None


# -- example spaces --------------------------------------------------------------------------


def triangulated_disk(n: int = 6) -> SpatialHypergraph:
    """A disk: centre 0 and a rim 1..n, the triangles {0, i, i+1}."""
    import math

    edges = [[0, i, i % n + 1] for i in range(1, n + 1)]
    coords = {0: (0.0, 0.0)} | {
        i: (math.cos(2 * math.pi * (i - 1) / n), math.sin(2 * math.pi * (i - 1) / n)) for i in range(1, n + 1)
    }
    return SpatialHypergraph(edges, coordinates=coords)


def triangulated_annulus(n: int = 4) -> SpatialHypergraph:
    """An annulus: an inner ring a1..an round a hole, an outer ring b1..bn, triangles between them."""
    import math

    a = [f"a{i}" for i in range(1, n + 1)]
    b = [f"b{i}" for i in range(1, n + 1)]
    edges = []
    for i in range(n):
        j = (i + 1) % n
        edges += [[a[i], b[i], b[j]], [a[i], a[j], b[j]]]
    coords = {}
    for i in range(n):
        t = 2 * math.pi * i / n
        coords[a[i]] = (math.cos(t), math.sin(t))
        u = t - math.pi / n
        coords[b[i]] = (2.2 * math.cos(u), 2.2 * math.sin(u))
    return SpatialHypergraph(edges, coordinates=coords)


claim(
    f"{KEY}.barrier_parity_invariant",
    r"For a barrier $B$, every area-homotopy rule (cancel, stall, contract, expand and inverses) keeps the "
    r"parity of the number of crossings of $B$; so paths with different crossing parities are not "
    r"area-homotopic.",
    about=(f"{KEY}.barrier", f"{KEY}.area_homotopy"),
    status="supported",
    evidence=[
        Evidence(
            EvidenceKind.ARGUMENT,
            "cancel removes two crossings of one pair or none; a stall crosses nothing; contract and expand "
            "swap a c b for a b, changing the count by the number of barrier pairs among ab, ac, cb, which is "
            "even inside an area",
        ),
        Evidence(
            EvidenceKind.NUMERICAL,
            "every rewrite of 3000 random paths, against every barrier of up to 3 pairs, on random spaces",
            source=f"{Q02}/2026-10-29-rewriting-calculus-checks.py",
        ),
    ],
)

claim(
    f"{KEY}.motion_homotopy_implies_area_homotopy",
    r"Incidence-homotopic motions (same reduced form under $x\,y\,x \to x$) are homotopic (their step paths "
    r"are area-homotopic); the converse fails: two parallel connections between the same positions are "
    r"different motions but the same step.",
    about=(f"{KEY}.area_homotopy", f"{KEY}.incidence_homotopy", f"{KEY}.homotopy"),
    status="supported",
    evidence=[
        Evidence(
            EvidenceKind.ARGUMENT,
            "cancelling v e v leaves a stall v v; cancelling e w e in v e w e u turns v w u into v u, which is "
            "a contract across e when v, w, u are distinct in e, else a cancel or a stall",
        ),
        Evidence(
            EvidenceKind.NUMERICAL,
            "2000 random motions: the step path of the motion and of its reduced form are joined by a derivation",
            source=f"{Q02}/2026-10-29-rewriting-calculus-checks.py",
        ),
    ],
)

claim(
    f"{KEY}.shortening_not_confluent",
    r"The shortening rules alone (cancel, stall, contract) are terminating but not confluent: some paths "
    r"reduce to different irreducible paths that are nevertheless area-homotopic, so deciding homotopy needs "
    r"the expanding rules too (or another argument).",
    about=(f"{KEY}.area_homotopy",),
    status="supported",
    evidence=[
        Evidence(
            EvidenceKind.NUMERICAL,
            "counterexamples found among random paths on triangulated disks and annuli",
            source=f"{Q02}/2026-10-29-rewriting-calculus-checks.py",
        )
    ],
)


# -- homotopy of motions, derived; the fundamental group ---------------------------------------


@define(
    f"{KEY}.homotopy",
    "Homotopy of motions",
    r"Two motions with the same ends are **homotopic** when their step paths are area-homotopic: the homotopy "
    r"of motions is derived from area homotopy, the primary notion (the user, 2026-10-29). A derivation of "
    r"the step paths proves it, a barrier refutes it. Backtrack cancellation (incidence homotopy) implies it; "
    r"so does exchanging the infinitesimal position a step passes ($a \to e \to b$ for $a \to e' \to b$). "
    r"The closed motions at $v$ up to homotopy form the **fundamental group** $\pi_1(H, v)$.",
    maturity="sketch",
    depends_on=(f"{KEY}.area_homotopy", f"{KEY}.motion", f"{KEY}.incidence_homotopy"),
    tags=("infrageometry", "path", "rewriting"),
)
def homotopic(a, b, *, max_length: int | None = None) -> bool | None:
    """True with a derivation found, False with different ends or a barrier found, None when neither search
    decides (motions or step paths)."""
    p = a if isinstance(a, StepPath) else path_of_motion(a)
    q = b if isinstance(b, StepPath) else path_of_motion(b)
    if (p.start, p.end) != (q.start, q.end):
        return False
    if find_derivation(p, q, max_length=max_length) is not None:
        return True
    if find_barrier(p, q) is not None:
        return False
    return None


Word = tuple  # of (generator index, ±1)


def _free_reduce(w) -> tuple:
    out: list = []
    for g, s in w:
        if out and out[-1] == (g, -s):
            out.pop()
        else:
            out.append((g, s))
    return tuple(out)


def _cyclic_reduce(w) -> tuple:
    w = list(_free_reduce(w))
    while len(w) >= 2 and w[0] == (w[-1][0], -w[-1][1]):
        w = w[1:-1]
    return tuple(w)


def _inverse(w) -> tuple:
    return tuple((g, -s) for g, s in reversed(w))


def _substitute(w, g, value) -> tuple:
    """The word ``w`` with the generator ``g`` replaced by the word ``value``, cyclically reduced."""
    out: list = []
    for h, t in w:
        out += list(value if t > 0 else _inverse(value)) if h == g else [(h, t)]
    return _cyclic_reduce(out)


@dataclass
class GroupPresentation:
    """$\\langle$ generators $|$ relators $\\rangle$: ``loops[i]`` is the closed step path the generator
    ``names[i]`` stands for; relators are words of ``(i, ±1)``."""

    base: Point
    loops: list
    names: list
    relators: list

    def word_text(self, w) -> str:
        if not w:
            return "1"
        return " ".join(self.names[g] + ("⁻¹" if s < 0 else "") for g, s in w)

    def simplified(self) -> GroupPresentation:
        """Tietze moves: drop a generator that occurs once in some relator (it equals the rest of that relator),
        substituting it everywhere; drop trivial and repeated relators. The group is unchanged."""
        gens = list(range(len(self.names)))
        rels = [_cyclic_reduce(r) for r in self.relators]
        while True:
            rels = sorted({r for r in rels if r}, key=lambda r: (len(r), r))
            pick = None
            for r in rels:
                counts: dict = {}
                for g, _s in r:
                    counts[g] = counts.get(g, 0) + 1
                once = [g for g, n in counts.items() if n == 1]
                if once:
                    pick = (r, min(once))
                    break
            if pick is None:
                break
            r, g = pick
            i = next(k for k, (h, _s) in enumerate(r) if h == g)
            s = r[i][1]
            rest = r[i + 1 :] + r[:i]  # g^s · rest = 1, so g = rest^{-s}
            value = _inverse(rest) if s > 0 else rest

            rels = [_substitute(x, g, value) for x in rels if x != r]
            gens.remove(g)
        index = {g: k for k, g in enumerate(gens)}
        return GroupPresentation(
            self.base,
            [self.loops[g] for g in gens],
            [self.names[g] for g in gens],
            [tuple((index[h], t) for h, t in r) for r in rels],
        )

    def __repr__(self) -> str:
        gens = ", ".join(self.names)
        rels = ", ".join(self.word_text(r) for r in self.relators)
        return f"⟨{gens} | {rels}⟩" if rels else f"⟨{gens}⟩" if gens else "1"


@define(
    f"{KEY}.fundamental_group",
    "Fundamental group",
    r"$\pi_1(H, v)$ is the group of closed step paths at $v$ up to area homotopy, under concatenation. Choose a "
    r"spanning tree of the adjacency graph: each step $\{a, b\}$ outside it is a generator (the loop out along "
    r"the tree to $a$, across to $b$, back along the tree), and each triangle $\{a, b, c\}$ inside an area is a "
    r"relator $g_{ab}\,g_{bc}\,g_{ca}$ (tree steps are trivial). It is the edge-path group of the simplicial "
    r"complex $K(H)$ whose simplices are the subsets of edges.",
    maturity="sketch",
    depends_on=(f"{KEY}.homotopy",),
    tags=("infrageometry", "path", "rewriting"),
)
def fundamental_group(space: SpatialHypergraph, base: Point) -> GroupPresentation:
    if base not in space.vertices:
        raise ValueError(f"{point_text(base)} is not a position of the space")
    parent, order_, queue = {base: None}, [base], deque([base])
    while queue:
        v = queue.popleft()
        for w in sort_points(space.neighbourhood(v, closed=False)):
            if w != v and w not in parent:
                parent[w] = v
                order_.append(w)
                queue.append(w)
    comp = set(parent)

    def to_base(x):
        out = [x]
        while parent[out[-1]] is not None:
            out.append(parent[out[-1]])
        return out

    tree = {frozenset((v, parent[v])) for v in comp if parent[v] is not None}
    pairs = sorted(
        {frozenset((a, b)) for e in space.edges if e <= comp for a, b in itertools.combinations(sort_points(e), 2)}
        - tree,
        key=sort_points,
    )
    index = {pair: k for k, pair in enumerate(pairs)}
    loops, names = [], []
    for k, pair in enumerate(pairs):
        a, b = sort_points(pair)
        loops.append(StepPath(space, tuple(reversed(to_base(a))) + tuple(to_base(b))))
        names.append(f"g{k + 1}")

    def letter(a, b):
        pair = frozenset((a, b))
        if pair not in index:
            return ()
        return ((index[pair], 1 if sort_points(pair)[0] == a else -1),)

    triangles = {
        frozenset(t)
        for e in space.edges
        if len(e) >= 3 and e <= comp
        for t in itertools.combinations(sort_points(e), 3)
    }
    rels = []
    for t in sorted(triangles, key=sort_points):
        a, b, c = sort_points(t)
        w = _cyclic_reduce(letter(a, b) + letter(b, c) + letter(c, a))
        if w:
            rels.append(w)
    return GroupPresentation(base, loops, names, sorted(set(rels), key=lambda r: (len(r), r)))


claim(
    f"{KEY}.fundamental_group_edge_path",
    r"Closed step paths at $v$ up to area homotopy form a group isomorphic to the edge-path group of $K(H)$, the "
    r"simplicial complex of subsets of edges, hence to $\pi_1(|K(H)|, v)$; the presentation by a spanning "
    r"tree and the triangles inside areas presents it. Its abelianisation has rank $b_1(K(H))$, which is at "
    r"most $b_1(X(H))$.",
    about=(f"{KEY}.fundamental_group", f"{KEY}.homotopy"),
    status="supported",
    evidence=[
        Evidence(
            EvidenceKind.ARGUMENT,
            "the rules are the elementary equivalences of edge paths in the 2-skeleton of K(H): a b c ~ a c for a "
            "simplex {a, b, c} (with repeats allowed, giving cancel and stall); the edge-path group theorem "
            "(Spanier, Algebraic Topology 3.6) and the standard tree presentation",
        ),
        Evidence(
            EvidenceKind.NUMERICAL,
            "random spaces: rank of the abelianised presentation = b1 of K(H) by linear algebra; generator loops "
            "killed by the simplification are contractible by derivation",
            source=f"{Q02}/2026-10-29-half-arc-fields-checks.py",
        ),
    ],
)
