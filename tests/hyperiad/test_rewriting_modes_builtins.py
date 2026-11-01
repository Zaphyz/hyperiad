from frontier.hypergraphs import Hypergraph, Saturation
from hypermatrix import ArrayObject

from hyperiad import Session


def run(code: str, **kw):
    s = Session(**kw)
    r = s.run(code)
    return r, s.globals.vars.get("ans")


COMP = (
    'comp = HypergraphRule([Edge([1, 2], "Directed", a), Edge([2, 3], "Directed", b)], [[1, 3]])\n'
    "g = Hypergraph([1, 2, 3], [[1, 2], [2, 3], [3, 3]])\n"
    "h = Hypergraph([1, 2, 3], [[2, 1], [3, 1]])\n"
)
SQUARE = (
    'sq = Hypergraph([Vertex("A", "A"), Vertex("B", "B"), Vertex("C", "C"), Vertex("D", "D")], '
    '[Edge(["A", "B"], "Directed", "f"), Edge(["B", "D"], "Directed", "g"), '
    'Edge(["A", "C"], "Directed", "h"), Edge(["C", "D"], "Directed", "k")])\n'
    "compose = HypergraphRule(Hypergraph([Vertex(1, X), Vertex(2, Y), Vertex(3, Z)], "
    '[Edge([1, 2], "Directed", x), Edge([2, 3], "Directed", y)]), '
    'Hypergraph([Vertex(1, X), Vertex(3, Z)], [Edge([1, 3], "Directed", o(y, x))]))\n'
    "identity = HypergraphRule(Hypergraph([Vertex(1, X)], []), "
    'Hypergraph([Vertex(1, X)], [Edge([1, 1], "Directed", id(X))]))\n'
    'units = [HypergraphRule([Edge([1, 2], "Directed", o(id(Y), x))], [Edge([1, 2], "Directed", x)]), '
    'HypergraphRule([Edge([1, 2], "Directed", o(x, id(X)))], [Edge([1, 2], "Directed", x)]), '
    'HypergraphRule([Edge([1, 2], "Directed", o(o(z, y), x))], [Edge([1, 2], "Directed", o(z, o(y, x)))])]\n'
)


def test_composition_and_its_hypermatrix():
    r, c = run(COMP + 'HypergraphCompose(comp, {"a": g, "b": h})')
    assert isinstance(c, Hypergraph) and sorted(e.ids for e in c.edges) == [(1, 1), (2, 1), (3, 1)]
    assert r["outputs"][0]["type"] == "svg"
    same = (
        'AdjacencyHypermatrix(HypergraphCompose(comp, {"a": g, "b": h}))[1] == '
        'CompositionAdjacencyProduct(comp, {"a": g, "b": h})'
    )
    assert run(COMP + same)[1] is True
    _, arr = run(COMP + 'CompositionAdjacencyProduct(comp, {"a": g, "b": g}, Multiplicity = "Boolean")')
    assert isinstance(arr, ArrayObject)
    _, d = run(COMP + "CompositionPlexDiagram(comp)")
    assert sorted(e.label for e in d.edges) == ["a", "b"] and [v.id for v in d.vertices if v.label] == [2]
    r, _ = run(COMP + 'HypergraphCompose(comp, {"a": g, "c": h})')
    assert r["status"] == "error" and "names none of the factors" in r["outputs"][-1]["data"]
    r, _ = run(COMP + 'HypergraphCompose(comp, {"a": g, "b": h}, Multiplicity = "Real")')
    assert r["status"] == "error" and "Multiplicity" in r["outputs"][-1]["data"]


def test_saturation_of_a_square():
    r, s = run(SQUARE + "HypergraphSaturate(sq, [compose, identity], Equations = units)")
    assert isinstance(s, Saturation) and s.saturated and len(s.final.edges) == 10
    assert r["outputs"][0]["type"] == "frames" and r["outputs"][0]["caption"].startswith("saturated after 1 rounds")
    _, groups = run(SQUARE + 'ParallelEdges(HypergraphSaturate(sq, [compose, identity], Equations = units)["Final"])')
    assert len(groups) == 1 and groups[0]["Vertices"] == ["A", "D"]
    commute = (
        'HypergraphRule([Edge([1, 2], "Directed", o("k", "h")), Edge([1, 2], "Directed", o("g", "f"))], '
        '[Edge([1, 2], "Directed", o("g", "f"))])'
    )
    _, groups = run(
        SQUARE
        + f'ParallelEdges(HypergraphSaturate(sq, [compose, identity], Equations = Join(units, [{commute}]))["Final"])'
    )
    assert groups == []
    _, rounds = run(SQUARE + 'HypergraphSaturate(sq, [compose, identity], MaxRounds = 2)["Saturated"]')
    assert rounds is False


def test_options_and_one_pass():
    _, evs = run(
        "HypergraphRuleApply(HypergraphRule([[1, 2], [2, 3]], [[1, 3]]), Hypergraph([[1, 1]]), "
        "DistinctVertices = false, DistinctEdges = false)"
    )
    assert len(evs) == 1
    _, op = run(
        COMP + 'HypergraphRuleOnePass(comp, Hypergraph([Edge([1, 2], "Directed", "a"), Edge([2, 3], "Directed", "b")]))'
    )
    assert [e.ids for e in op.edges] == [(1, 3)]


def test_list_utilities():
    code = (
        "[Join([1], [2, 3]), Flatten([[1, [2]], [3]]), Flatten([[1, [2]]], 1), Union([1, 2], [2, 3]), "
        "Tuples([1, 2], 2), Subsets([1, 2], 1), Fold((u, v) => u + v, 0, [1, 2, 3])]"
    )
    assert run(code)[1] == [[1, 2, 3], [1, 2, 3], [1, [2]], [1, 2, 3], [[1, 1], [1, 2], [2, 1], [2, 2]], [[1], [2]], 6]
    _, j = run("Join(Hypergraph([[1, 2]]), Hypergraph([[2, 3]]))")
    assert isinstance(j, Hypergraph) and len(j.edges) == 2 and j.vertex_ids == [1, 2, 3]
