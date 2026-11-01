"""Hypergraphs: multisets of hyperedges, each a vertex sequence with a symmetry type.

Ported from the reference paclet (``src/Hypergraph.wl``):

* a :class:`Vertex` is an identifier, optionally with a label;
* an :class:`Edge` is a list of vertices with a symmetry type -- ``"Directed"``
  (the default), ``"Unordered"`` or ``"Cyclic"``, or an **input–output** type whose
  input block and output block each carry one of those three -- and optionally a label.
  Its vertex list is stored in the canonical order of its symmetry type (the least
  element of the orbit), so two edges are equal exactly when they are the same
  hyperedge;
* a :class:`Hypergraph` is a vertex set and a multiset of edges: edge order does
  not matter, multiplicity does. A label given to a vertex anywhere applies to
  every occurrence of that vertex.

Identifiers and labels may be any hashable values (lists are turned into tuples).
"""

from __future__ import annotations

import itertools
import math
import re
import warnings
from collections import Counter
from collections.abc import Hashable, Iterable, Sequence
from fractions import Fraction
from typing import Any

from frontier.registry import SourceRef, define

SYMMETRY_TYPES = ("Directed", "Unordered", "Cyclic")
_SYMMETRY_NAMES = {
    "directed": "Directed",
    "ordered": "Directed",
    "unordered": "Unordered",
    "undirected": "Unordered",
    "cyclic": "Cyclic",
}


class HypergraphError(ValueError):
    """An invalid vertex, edge or hypergraph. ``code`` names the reference paclet's message."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _hashable(x: Any) -> Hashable:
    if isinstance(x, list):
        return tuple(_hashable(v) for v in x)
    try:
        hash(x)
    except TypeError:
        raise HypergraphError("Vertex::spec", f"{x!r} cannot identify a vertex: it is not hashable") from None
    return x


def order_key(x: Any) -> tuple:
    """A total order on identifiers and labels of mixed types: numbers, strings, tuples, the rest."""
    if x is None:
        return (-1,)
    if isinstance(x, bool):
        return (0, int(x), 1)
    if isinstance(x, (int, float, Fraction)):
        return (0, x, 0)
    if isinstance(x, str):
        return (1, x)
    if isinstance(x, tuple):
        return (2, len(x), tuple(order_key(v) for v in x))
    own = getattr(x, "order_key", None)  # objects that know their order (e.g. FormalV, LabelToken)
    if callable(own):
        return (3, type(x).__name__, own())
    return (3, type(x).__name__, str(x))


# -- the input–output type ------------------------------------------------------------------
#
# An input–output edge has an input block and an output block of vertices (each nonempty), each
# block with one of the three basic symmetry types (the user, 2026-10-31). Its symmetry is written
# with the block sizes, ``"InputOutput(Unordered:2,Cyclic:3)"``, so that the symmetry string and the
# flat vertex list together determine the edge, as for the other types, and every piece of code
# that rebuilds an edge as ``Edge(e.ids, e.symmetry, e.label)`` keeps its blocks. One input and one
# output is the same edge as a directed binary edge: that type is written ``"Directed"``.

IO_TYPE = "InputOutput"
_IO_PATTERN = re.compile(r"^\s*inputoutput\s*\(\s*(\w+)\s*:\s*(\d+)\s*,\s*(\w+)\s*:\s*(\d+)\s*\)\s*$", re.IGNORECASE)
_IO_BARE = {"inputoutput", "input-output", "io"}


def _base_symmetry(sym: Any) -> str:
    s = _SYMMETRY_NAMES.get(sym.lower()) if isinstance(sym, str) else None
    if s is None:
        raise HypergraphError(
            "Edge::sym",
            f"{sym!r} is not a valid edge symmetry type; expected 'Directed', 'Unordered', 'Cyclic' or "
            f"'InputOutput(<type>:<k>,<type>:<m>)'",
        )
    return s


def io_symmetry(inputs: str, k: int, outputs: str, m: int) -> str:
    """The symmetry type of an input–output edge with ``k`` inputs and ``m`` outputs of the given types.

    Both blocks are nonempty; one input and one output is the directed binary type ``"Directed"``."""
    if k < 1 or m < 1:
        raise HypergraphError(
            "Edge::io", f"an input–output edge needs at least one input and one output, not {k} and {m}"
        )
    s_in, s_out = _base_symmetry(inputs), _base_symmetry(outputs)
    if k == 1 and m == 1:
        return "Directed"
    return f"{IO_TYPE}({s_in}:{k},{s_out}:{m})"


def parse_io(sym: str) -> tuple[str, int, str, int] | None:
    """``(input type, k, output type, m)`` of an input–output symmetry string, None for a basic type."""
    if not isinstance(sym, str) or not sym.startswith(IO_TYPE + "("):
        return None
    found = _IO_PATTERN.match(sym)
    return found.group(1), int(found.group(2)), found.group(3), int(found.group(4))


def normalize_symmetry(sym: str) -> str:
    """The canonical spelling of a symmetry type: a basic name, or a full input–output type."""
    found = _IO_PATTERN.match(sym) if isinstance(sym, str) else None
    if found:
        return io_symmetry(found.group(1), int(found.group(2)), found.group(3), int(found.group(4)))
    return _base_symmetry(sym)


def is_io(sym: str) -> bool:
    """Whether a (normalized) symmetry type is an input–output type with a block of two or more."""
    return parse_io(sym) is not None


def symmetry_blocks(sym: str, k: int) -> list[tuple[str, int, int]]:
    """The blocks of an edge of arity ``k``: ``(basic type, start, length)``, one block for a basic type."""
    io = parse_io(sym)
    if io is None:
        return [(sym, 0, k)]
    s_in, a, s_out, b = io
    if a + b != k:
        raise HypergraphError("Edge::io", f"{sym} has {a + b} vertices, not {k}")
    return [(s_in, 0, a), (s_out, a, b)]


def _block_orderings(sym: str, n: int) -> list[tuple[int, ...]]:
    identity = tuple(range(n))
    if sym == "Directed" or n <= 1:
        return [identity]
    if sym == "Cyclic":
        return [identity[r:] + identity[:r] for r in range(n)]
    return list(itertools.permutations(identity))


def symmetry_orderings(sym: str, k: int) -> list[tuple[int, ...]]:
    """The orderings of an edge's ``k`` positions that its symmetry type says are the same edge.

    For an input–output type, the products of the orderings of each block within that block."""
    blocks = symmetry_blocks(sym, k)
    if len(blocks) == 1:
        return _block_orderings(sym, k)
    (s_in, _, a), (s_out, _, b) = blocks
    return [p + tuple(a + j for j in q) for p in _block_orderings(s_in, a) for q in _block_orderings(s_out, b)]


def symmetry_size(sym: str, k: int) -> int:
    """How many orderings :func:`symmetry_orderings` lists, counted without listing them."""
    total = 1
    for s, _, n in symmetry_blocks(sym, k):
        total *= 1 if s == "Directed" or n <= 1 else n if s == "Cyclic" else math.factorial(n)
    return total


def symmetry_rank(sym: str) -> tuple:
    """A sort key for symmetry types: the three basic types first, in ``SYMMETRY_TYPES`` order."""
    io = parse_io(sym)
    if io is None:
        return (SYMMETRY_TYPES.index(sym),)
    s_in, a, s_out, b = io
    return (len(SYMMETRY_TYPES), SYMMETRY_TYPES.index(s_in), a, SYMMETRY_TYPES.index(s_out), b)


@define(
    "common.vertex",
    "Vertex",
    r"A vertex is an identifier, any value, optionally carrying a label. Two vertices are the same "
    r"when their identifiers are; a label of `None` means unlabelled.",
    maturity="stable",
    sources=(SourceRef("hypergraph-paclet", "symbol:Vertex", "implementation"),),
    tags=("hypergraph",),
)
class Vertex:
    __slots__ = ("id", "label")

    def __init__(self, id: Any, label: Any = None) -> None:  # noqa: A002 - the paclet's name
        if isinstance(id, Vertex):
            id, label = id.id, label if label is not None else id.label
        object.__setattr__(self, "id", _hashable(id))
        object.__setattr__(self, "label", None if label is None else _hashable(label))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Vertex is immutable")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Vertex) and (self.id, self.label) == (other.id, other.label)

    def __hash__(self) -> int:
        return hash(("Vertex", self.id, self.label))

    def sort_key(self) -> tuple:
        return (order_key(self.id), order_key(self.label))

    def __repr__(self) -> str:
        return f"Vertex({self.id!r})" if self.label is None else f"Vertex({self.id!r}, {self.label!r})"


def _canonical_order(vs: tuple[Vertex, ...], sym: str) -> tuple[Vertex, ...]:
    if is_io(sym):
        return tuple(v for s, i, n in symmetry_blocks(sym, len(vs)) for v in _canonical_order(vs[i : i + n], s))
    if sym == "Directed" or not vs:
        return vs
    if sym == "Unordered":
        return tuple(sorted(vs, key=Vertex.sort_key))
    rotations = [vs[k:] + vs[:k] for k in range(len(vs))]
    return min(rotations, key=lambda r: tuple(v.sort_key() for v in r))


@define(
    "common.hyperedge",
    "Hyperedge",
    r"A hyperedge is a list of vertices $(v_1, \dots, v_k)$ with a symmetry type: **directed** "
    r"(the order matters), **cyclic** (the order matters up to rotation) or **unordered** (only the "
    r"multiset of vertices matters), and optionally a label. $k$ is its arity; a vertex may repeat. "
    r"An **input–output** hyperedge $(u_1, \dots, u_a \mid w_1, \dots, w_b)$ has an input block and an output "
    r"block, $a, b \ge 1$, each with one of the three types, written `InputOutput(S:a,T:b)`; its orderings are "
    r"those of each block within that block, and one input and one output is the directed binary edge. "
    r"Its vertices are stored in the canonical order of its symmetry type, the least of its orbit, "
    r"so two hyperedges are equal exactly when they are the same hyperedge.",
    maturity="stable",
    depends_on=("common.vertex",),
    sources=(SourceRef("hypergraph-paclet", "symbol:Edge", "implementation"),),
    tags=("hypergraph",),
)
class Edge:
    __slots__ = ("vertices", "symmetry", "label")

    def __init__(self, vertices: Iterable[Any], symmetry: str | Sequence[str] = "Directed", label: Any = None) -> None:
        """``Edge(vertices, symmetry, label)``. An input–output edge is given either flat, with its full type
        (``Edge([a, b, c], "InputOutput(Unordered:2,Directed:1)")``), or as its two blocks with the two
        block types or the bare word ``"InputOutput"`` (both blocks directed):
        ``Edge([[a, b], [c]], ("Unordered", "Directed"))``."""
        if isinstance(vertices, (str, bytes)) or not isinstance(vertices, Iterable):
            raise HypergraphError("Edge::spec", f"{vertices!r} is not a list of vertices")
        blocks = _io_block_types(symmetry)
        if blocks is not None:
            parts = list(vertices)
            if len(parts) != 2 or any(isinstance(p, (str, bytes)) or not isinstance(p, Iterable) for p in parts):
                raise HypergraphError("Edge::io", f"{vertices!r} is not a pair of blocks (inputs, outputs)")
            ins, outs = list(parts[0]), list(parts[1])
            symmetry, vertices = io_symmetry(blocks[0], len(ins), blocks[1], len(outs)), ins + outs
        sym = normalize_symmetry(symmetry)
        vs = tuple(v if isinstance(v, Vertex) else Vertex(v) for v in vertices)
        if is_io(sym):
            symmetry_blocks(sym, len(vs))  # the sizes must add up
        object.__setattr__(self, "vertices", _canonical_order(vs, sym))
        object.__setattr__(self, "symmetry", sym)
        object.__setattr__(self, "label", None if label is None else _hashable(label))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Edge is immutable")

    @property
    def ids(self) -> tuple[Hashable, ...]:
        return tuple(v.id for v in self.vertices)

    @property
    def arity(self) -> int:
        return len(self.vertices)

    @property
    def blocks(self) -> tuple[tuple[Vertex, ...], ...]:
        """The vertices by block: one block for a basic type, (inputs, outputs) for an input–output type."""
        return tuple(self.vertices[i : i + n] for _, i, n in symmetry_blocks(self.symmetry, self.arity))

    @property
    def block_ids(self) -> tuple[tuple[Hashable, ...], ...]:
        return tuple(tuple(v.id for v in b) for b in self.blocks)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Edge) and (self.vertices, self.symmetry, self.label) == (
            other.vertices,
            other.symmetry,
            other.label,
        )

    def __hash__(self) -> int:
        return hash(("Edge", self.vertices, self.symmetry, self.label))

    def sort_key(self) -> tuple:
        return (tuple(v.sort_key() for v in self.vertices), symmetry_rank(self.symmetry), order_key(self.label))

    def skeleton(self) -> Edge:
        """The same edge without its label, over unlabelled vertices."""
        return Edge([Vertex(v.id) for v in self.vertices], self.symmetry)

    def __repr__(self) -> str:
        ids = list(self.ids)
        if self.label is not None:
            return f"Edge({ids!r}, {self.symmetry!r}, {self.label!r})"
        return f"Edge({ids!r})" if self.symmetry == "Directed" else f"Edge({ids!r}, {self.symmetry!r})"


def _io_block_types(symmetry: Any) -> tuple[str, str] | None:
    """The two block types when ``symmetry`` asks for an edge given as two blocks, else None."""
    if isinstance(symmetry, str):
        return ("Directed", "Directed") if symmetry.strip().lower() in _IO_BARE else None
    if isinstance(symmetry, (tuple, list)) and len(symmetry) == 2 and all(isinstance(s, str) for s in symmetry):
        return _base_symmetry(symmetry[0]), _base_symmetry(symmetry[1])
    raise HypergraphError("Edge::sym", f"{symmetry!r} is not a valid edge symmetry type")


def _to_edge(spec: Any) -> Edge:
    if isinstance(spec, Edge):
        return spec
    if isinstance(spec, (list, tuple)):
        return Edge(spec)
    raise HypergraphError("Edge::spec", f"{spec!r} is not a valid hyperedge specification")


@define(
    "common.hypergraph",
    "Hypergraph",
    r"A hypergraph is a set of vertices together with a finite **multiset** of hyperedges on them: "
    r"the order of the edges does not matter, their multiplicity does. Every vertex of an edge "
    r"belongs to the vertex set, and a label given to a vertex anywhere is the label of that vertex "
    r"everywhere. The *skeleton* forgets every vertex and edge label and keeps the symmetry types.",
    maturity="stable",
    depends_on=("common.vertex", "common.hyperedge"),
    sources=(
        SourceRef("hypergraph-paclet", "symbol:Hypergraph", "implementation"),
        SourceRef("heaps-of-fish", "section:The Plex Formalism", "background"),
    ),
    tags=("hypergraph",),
)
class Hypergraph:
    """``Hypergraph(edges)``, ``Hypergraph(vertices, edges)`` or ``Hypergraph(vertices)`` (edgeless).

    Edges are :class:`Edge` objects or lists of vertex identifiers (directed). A single
    list is read as edges when any element is an edge or a list, otherwise as vertices.
    """

    PROPERTIES = (
        "VertexList",
        "EdgeList",
        "VertexCount",
        "EdgeCount",
        "VertexIDs",
        "VertexLabels",
        "EdgeLabels",
        "EdgeSymmetries",
        "Arity",
        "Skeleton",
    )
    __slots__ = ("vertices", "edges")

    def __init__(self, *args: Any) -> None:
        if len(args) == 0:
            vs, es = [], []
        elif len(args) == 1:
            if isinstance(args[0], (str, bytes)) or not isinstance(args[0], Iterable):
                raise HypergraphError("Hypergraph::spec", f"{args[0]!r} is not a list of edges or vertices")
            x = list(args[0])
            if x and any(isinstance(item, (Edge, list, tuple)) for item in x):
                vs, es = [], x
            else:
                vs, es = x, []
        elif len(args) == 2:
            vs, es = list(args[0]), list(args[1])
        else:
            raise HypergraphError("Hypergraph::spec", "expected Hypergraph(edges) or Hypergraph(vertices, edges)")
        edges = [_to_edge(e) for e in es]
        declared = [v if isinstance(v, Vertex) else Vertex(v) for v in vs]
        incident = [v for e in edges for v in e.vertices]
        # Declared vertices come first, so an explicit vertex list wins label conflicts.
        labels: dict[Hashable, list[Any]] = {}
        for v in declared + incident:
            group = labels.setdefault(v.id, [])
            if v.label is not None and v.label not in group:
                group.append(v.label)
        canonical: dict[Hashable, Vertex] = {}
        for vid, labs in labels.items():
            if len(labs) > 1:
                warnings.warn(f"vertex {vid!r} was given more than one label {labs!r}; using {labs[0]!r}", stacklevel=2)
            canonical[vid] = Vertex(vid, labs[0] if labs else None)
        remapped = [Edge([canonical[v.id] for v in e.vertices], e.symmetry, e.label) for e in edges]
        object.__setattr__(self, "vertices", tuple(sorted(canonical.values(), key=Vertex.sort_key)))
        object.__setattr__(self, "edges", tuple(sorted(remapped, key=Edge.sort_key)))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Hypergraph is immutable")

    # -- accessors ---------------------------------------------------------

    @property
    def vertex_ids(self) -> list[Hashable]:
        return [v.id for v in self.vertices]

    def __len__(self) -> int:
        return len(self.edges)

    def prop(self, name: str) -> Any:
        values = {
            "VertexList": lambda: list(self.vertices),
            "EdgeList": lambda: list(self.edges),
            "VertexCount": lambda: len(self.vertices),
            "EdgeCount": lambda: len(self.edges),
            "VertexIDs": lambda: self.vertex_ids,
            "VertexLabels": lambda: [v.label for v in self.vertices],
            "EdgeLabels": lambda: [e.label for e in self.edges],
            "EdgeSymmetries": lambda: [e.symmetry for e in self.edges],
            "Arity": lambda: [e.arity for e in self.edges],
            "Skeleton": self.skeleton,
            "Properties": lambda: list(self.PROPERTIES),
        }
        if name not in values:
            raise KeyError(f"unknown hypergraph property {name!r}; known: {', '.join(self.PROPERTIES)}")
        return values[name]()

    def __getitem__(self, key: str) -> Any:
        if isinstance(key, str):
            return self.prop(key)
        raise TypeError("hypergraph properties are accessed by name, e.g. hg['Arity']")

    def skeleton(self) -> Hypergraph:
        return Hypergraph([Vertex(v.id) for v in self.vertices], [e.skeleton() for e in self.edges])

    def is_simple(self) -> bool:
        """No edge repeated, labels set aside; a vertex repeated within one edge is allowed."""
        counts = Counter(e.skeleton() for e in self.edges)
        return all(c == 1 for c in counts.values())

    def is_labelled(self) -> bool:
        return any(v.label is not None for v in self.vertices) or any(e.label is not None for e in self.edges)

    # -- value semantics ------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Hypergraph) and (self.vertices, self.edges) == (other.vertices, other.edges)

    def __hash__(self) -> int:
        return hash(("Hypergraph", self.vertices, self.edges))

    def __repr__(self) -> str:
        isolated = [v for v in self.vertices if all(v.id not in e.ids for e in self.edges)]
        labelled = [v for v in self.vertices if v.label is not None]
        if not isolated and not labelled:
            return f"Hypergraph({list(self.edges)!r})"
        return f"Hypergraph({list(self.vertices)!r}, {list(self.edges)!r})"


def simple_hypergraph_q(x: Any) -> bool:
    return isinstance(x, Hypergraph) and x.is_simple()


def as_edges(edges: Sequence[Any]) -> list[Edge]:
    return [_to_edge(e) for e in edges]
