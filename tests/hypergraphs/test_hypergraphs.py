"""The hypergraph object model and its SVG drawing.

Cases marked @ported reproduce the reference paclet's tests/BasicExamples.wls.
"""

import warnings

import numpy as np
import pytest
from frontier.conformance import ported
from frontier.hypergraphs import Edge, Hypergraph, HypergraphError, Vertex, simple_hypergraph_q
from frontier.hypergraphs.geometry import Arrowhead, Dots, hypergraph_scene
from frontier.hypergraphs.svg import hypergraph_drawing_args, hypergraph_svg


@ported("hypergraph-paclet", "tests/BasicExamples.wls", 25)
def test_vertices():
    assert Vertex(1, None) == Vertex(1)
    v = Vertex("x", "a")
    assert (v.id, v.label) == ("x", "a")
    assert Vertex([1, 2]).id == (1, 2)  # a list identifier becomes a tuple


@ported("hypergraph-paclet", "tests/BasicExamples.wls", 39, 41, 43, 45, 47, 49, 51)
def test_edge_identity_follows_symmetry():
    assert Edge([1, 2, 3], "Unordered") == Edge([3, 2, 1], "Unordered")
    assert Edge([1, 2, 3], "Cyclic") == Edge([3, 1, 2], "Cyclic")
    assert Edge([1, 2, 3], "Cyclic") != Edge([1, 3, 2], "Cyclic")
    assert Edge([1, 2, 3]) != Edge([3, 2, 1])
    assert Edge([1, 2]) != Edge([1, 2], "Unordered")
    assert Edge([1, 2], "Directed", "f") != Edge([1, 2])
    with pytest.raises(HypergraphError):
        Edge([1, 2], "Sideways")


def test_edge_canonical_order_and_defaults():
    assert Edge([1, 2, 3]).symmetry == "Directed"
    assert Edge([3, 1, 2], "cyclic").ids == (1, 2, 3)
    assert Edge([3, 2, 1], "undirected").ids == (1, 2, 3)
    assert Edge(["b", 2, "a", 1], "Unordered").ids == (1, 2, "a", "b")  # numbers before strings
    e = Edge([1, 2], "Cyclic", "f")
    assert (e.ids, e.symmetry, e.label, e.arity) == ((1, 2), "Cyclic", "f", 2)


@ported("hypergraph-paclet", "tests/BasicExamples.wls", 71, 83, 85, 87)
def test_hypergraph_is_a_multiset_of_edges():
    h1 = Hypergraph([Edge([1, 2]), Edge([2, 3]), Edge([1, 2, 3], "Unordered")])
    assert isinstance(h1, Hypergraph)
    assert Hypergraph([Edge([1, 2]), Edge([2, 3])]) == Hypergraph([Edge([2, 3]), Edge([1, 2])])
    assert Hypergraph([Edge([1, 2]), Edge([1, 2])])["EdgeCount"] == 2
    assert Hypergraph([Edge([1, 2]), Edge([1, 2])]) != Hypergraph([Edge([1, 2])])


def test_constructors_and_label_propagation():
    assert Hypergraph([1, 2, 3], [[1], [1, 2], [1, 2, 3]])["Arity"] == [1, 2, 3]
    assert Hypergraph([1, 2, 3, 4])["EdgeCount"] == 0  # a list of plain identifiers is a vertex list
    assert Hypergraph([])["VertexCount"] == 0
    assert Hypergraph([Edge([1, 2])])["VertexIDs"] == [1, 2]
    h = Hypergraph([Edge([Vertex(1, "a"), 2]), Edge([1, 3])])
    assert h["VertexLabels"] == ["a", None, None]
    assert all(v.label == "a" for e in h.edges for v in e.vertices if v.id == 1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        h = Hypergraph([Vertex(1, "a")], [[Vertex(1, "b"), 2]])
    assert h["VertexLabels"][0] == "a" and "more than one label" in str(w[0].message)
    with pytest.raises(HypergraphError):
        Edge("abc")


@ported("hypergraph-paclet", "tests/BasicExamples.wls", 98, 100, 102, 105)
def test_skeleton():
    h3 = Hypergraph([Vertex(1, "a"), Vertex(2, "b"), 3], [[1, 2], [2, 3]])
    sk = h3["Skeleton"]
    assert {v.label for v in sk.vertices} == {None}
    assert {v.label for e in sk.edges for v in e.vertices} == {None}
    assert Hypergraph([Edge([Vertex(1, "a"), 2], "Directed", "f")]).skeleton()["EdgeLabels"] == [None]
    h1 = Hypergraph([Edge([1, 2]), Edge([2, 3]), Edge([1, 2, 3], "Unordered")])
    assert h1.skeleton()["EdgeSymmetries"] == h1["EdgeSymmetries"]


@ported("hypergraph-paclet", "tests/BasicExamples.wls", 388, 389, 390, 393, 395, 397, 399, 401, 402)
def test_simplicity():
    assert simple_hypergraph_q(Hypergraph([Edge([1, 2]), Edge([2, 3])]))
    assert not simple_hypergraph_q(Hypergraph([Edge([1, 2]), Edge([1, 2])]))
    assert not simple_hypergraph_q(Hypergraph([Edge([1, 2], "Directed", "a"), Edge([1, 2], "Directed", "b")]))
    assert simple_hypergraph_q(Hypergraph([Edge([1, 1, 2], "Unordered")]))
    assert simple_hypergraph_q(Hypergraph([Edge([1, 2], "Directed"), Edge([1, 2], "Unordered")]))
    assert not simple_hypergraph_q(Hypergraph([Edge([1, 2], "Unordered"), Edge([2, 1], "Unordered")]))
    assert simple_hypergraph_q(Hypergraph([Edge([1, 2], "Directed"), Edge([2, 1], "Directed")]))
    assert simple_hypergraph_q(Hypergraph([]))
    assert not simple_hypergraph_q([1, 2])


def test_properties():
    h = Hypergraph([Vertex(1, "a"), 2], [Edge([1, 2, 3], "Cyclic", "f"), [3]])
    assert h["Properties"][:4] == ["VertexList", "EdgeList", "VertexCount", "EdgeCount"]
    assert (h["VertexCount"], h["EdgeCount"]) == (3, 2)
    assert h["EdgeLabels"] == ["f", None]  # edges in canonical order: (1, 2, 3) before (3)
    with pytest.raises(KeyError):
        h["Nonsense"]


# -- drawing --------------------------------------------------------------


def test_drawing_args_follow_the_hypergraph():
    h = Hypergraph([Vertex(1, "a"), 4], [Edge([1, 2, 3], "Cyclic"), Edge([3, 2], "Unordered", "g")])
    args = hypergraph_drawing_args(h)
    assert set(args["symmetry"]) == {"cyclic", "unordered"}
    assert 4 in args["vertices"]  # isolated vertices are drawn
    assert args["vertex_text"] == {1: "a"}  # labels are shown; identifiers only on request
    assert "g" in args["edge_text"]
    assert hypergraph_drawing_args(h, labels=True)["vertex_text"][1] == "1: a"


def test_scene_marks():
    scene = hypergraph_scene([(1, 2, 3), (3, 4), (4,)], symmetry="directed")
    kinds = [type(s).__name__ for s in scene.shapes]
    assert kinds.count("Arrowhead") == 1  # the binary edge; the hyperedge has half-arrowheads
    assert kinds.count("Area") == 1 + 3  # the region and its three barbs
    vertices = [s for s in scene.shapes if isinstance(s, Dots) and s.z == 3.0][0]
    assert len(vertices.centers) == 4
    assert all(isinstance(s, Arrowhead) for s in scene.shapes if type(s).__name__ == "Arrowhead")


def test_svg():
    h = Hypergraph([Edge([1, 2, 3], "Unordered"), Edge([3, 4]), Edge([4], "Directed"), Edge(["<x>", 1])])
    svg = hypergraph_svg(h, labels=True)
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert svg.count("<circle") >= 5  # five vertices and a unary disk
    assert "&lt;x&gt;" in svg and "<x>" not in svg  # identifiers are escaped
    assert 'fill="white"' in svg  # its own ground, as the session figures
    raw = hypergraph_svg([(1, 2), (2, 3, 4)], symmetry="directed", highlight=[1], title="t")
    assert "rgb(219,161,83)" in raw and ">t</text>" in raw
    assert hypergraph_svg(Hypergraph([1, 2, 3])).count("<circle") == 3


def test_layout_is_deterministic():
    a = hypergraph_scene([(1, 2, 3), (3, 4)])
    b = hypergraph_scene([(1, 2, 3), (3, 4)])
    assert np.allclose(a.points, b.points)
