import pytest
from frontier.hypergraphs import Edge, Hypergraph
from hyperiad.values import format_value

from hyperiad import Session


def run(code: str, **kw):
    s = Session(**kw)
    r = s.run(code)
    return r, s.globals.vars.get("ans")


def test_build_and_display():
    r, h = run('Hypergraph([[1, 2], [2, 3], Edge([1, 2, 3], "Unordered"), [3]])')
    assert isinstance(h, Hypergraph) and h["EdgeCount"] == 4
    out = r["outputs"][0]
    assert out["type"] == "svg" and out["data"].startswith("<svg")
    assert out["caption"] == "3 vertices · 4 edges · arity 1, 2, 3 · directed, unordered"


def test_edges_compare_by_symmetry():
    assert run('Edge([3, 1, 2], "Cyclic") == Edge([1, 2, 3], "Cyclic")')[1] is True
    assert run('Edge([1, 3, 2], "Cyclic") == Edge([1, 2, 3], "Cyclic")')[1] is False
    assert run('EdgeSymmetry(Edge([1, 2], "undirected"))')[1] == "Unordered"


def test_accessors_and_properties():
    code = 'h = Hypergraph([Vertex(a, "x"), b], [Edge([a, b, c], "Cyclic", "f"), [c]])\n'
    assert run(code + 'h["Arity"]')[1] == [3, 1]  # canonical edge order: (a, b, c) before (c)
    assert run(code + "VertexCount(h)")[1] == 3
    assert run(code + "VertexLabel(VertexList(h)[1])")[1] == "x"
    assert run(code + "EdgeLabel(EdgeList(h)[1])")[1] == "f"
    assert run(code + "SimpleHypergraphQ(h)")[1] is True
    assert run(code + "HypergraphSkeleton(h)")[1].is_labelled() is False


def test_values_print_back_as_hyperiad():
    r, _ = run('Edge([3, 1, 2], "Cyclic", "f")')
    assert r["outputs"][0]["data"] == 'Edge([1, 2, 3], "Cyclic", "f")'
    r, _ = run("EdgeList(Hypergraph([[2, 1]]))")
    assert r["outputs"][0]["data"] == "[Edge([2, 1])]"


def test_plot_options_and_budget():
    r, g = run('HypergraphPlot(Hypergraph([[1, 2], [2, 3, 4]]), VertexLabels = true, Highlight = [2], Title = "t")')
    assert "rgb(219,161,83)" in g.svg and ">t</text>" in g.svg
    r, _ = run("HypergraphPlot(Hypergraph([[1, 2]]), Highlight = [5])")
    assert r["status"] == "error" and "Highlight positions" in r["outputs"][0]["data"]
    r, _ = run("HypergraphPlot(Hypergraph(Table([i, i + 1, i + 2], [i, 400])))", budget=2e7)
    assert r["status"] == "too_expensive" and r["builtin"] == "HypergraphPlot"


def test_large_hypergraphs_display_as_summary():
    r, _ = run("Hypergraph(Table([i, i + 1], [i, 40]))")
    assert r["outputs"][0]["type"] == "text" and "41 vertices" in r["outputs"][0]["data"]


@pytest.mark.parametrize(
    "code, fragment",
    [
        ('Edge([1, 2], "Sideways")', "not a valid edge symmetry"),
        ('Edge("abc")', "not a list of vertices"),
        ("VertexCount(3)", "takes a hypergraph"),
        ("Hypergraph([[1, 2]])[1]", "taken by name"),
    ],
)
def test_errors(code, fragment):
    r, _ = run(code)
    assert r["status"] == "error" and fragment in r["outputs"][-1]["data"]


def test_hypergraph_objects_are_the_shared_ones():
    _, e = run("Edge([1, 2])")
    assert isinstance(e, Edge)


def test_isomorphism_builtins():
    code = 'l1 = Hypergraph([Vertex(1, "A"), Vertex(2, "B"), Vertex(3, "C")], [[1, 2], [2, 3]])\n'
    code += 'l3 = Hypergraph([Vertex(1, "C"), Vertex(2, "B"), Vertex(3, "A")], [[1, 2], [2, 3]])\n'
    assert run(code + "IsomorphicHypergraphQ(l1, l3)")[1] is True
    assert run(code + 'IsomorphicHypergraphQ(l1, l3, Labeled = false, Method = "BruteForce")')[1] is True
    assert run(code + "FindHypergraphIsomorphism(l1, l3)")[1] == {1: 1, 2: 2, 3: 3}
    found = run(code + "FindHypergraphIsomorphism(l1, l3, Labels = true)")[1]
    assert found["VertexLabels"] == {"A": "C", "B": "B", "C": "A"}
    assert run(code + "CanonicalHypergraph(l1) == CanonicalHypergraph(l3)")[1] is True
    assert run("IsomorphicHypergraphQ(Hypergraph([[1, 2], [2, 3]]), Hypergraph([[1, 2], [1, 3]]))")[1] is False


def test_canonical_forms_print_back_and_draw():
    r, h = run('CanonicalHypergraph(Hypergraph([Edge([7, 8], "Directed", "f")]))')
    assert format_value(h) == 'Hypergraph([Edge([1, 2], "Directed", LabelToken(1))])'
    assert run(format_value(h))[1] == h  # the printed form evaluates back to the same hypergraph
    assert r["outputs"][0]["type"] == "svg"


def test_isomorphism_options_and_budget():
    r, _ = run('CanonicalHypergraph(Hypergraph([[1, 2]]), Method = "Fast")')
    assert r["status"] == "error" and "Method is Automatic" in r["outputs"][-1]["data"]
    r, _ = run('CanonicalHypergraph(Hypergraph(Table([i, i + 1], [i, 12])), Method = "BruteForce")', budget=1e8)
    assert r["status"] == "too_expensive" and r["builtin"] == "CanonicalHypergraph"
    r, h = run("CanonicalHypergraph(Hypergraph(Table([i, i + 1], [i, 12])))", budget=1e8)
    assert r["status"] == "ok" and h["VertexCount"] == 13


def test_input_output_edges():
    code = 'io = Edge([[2, 1], [3, 4, 5]], ["Unordered", "Cyclic"], "f")\n'
    assert run(code + "EdgeSymmetry(io)")[1] == "InputOutput(Unordered:2,Cyclic:3)"
    assert run(code + "EdgeInputs(io)")[1] == [1, 2]
    assert run(code + "EdgeOutputs(io)")[1] == [3, 4, 5]
    assert run(code + "EdgeBlocks(io)")[1] == [[1, 2], [3, 4, 5]]
    assert run('Edge([[1], [2]], "InputOutput") == Edge([1, 2])')[1] is True
    r, _ = run(code + "io")
    assert r["outputs"][0]["data"] == 'Edge([[1, 2], [3, 4, 5]], ["Unordered", "Cyclic"], "f")'
    r, h = run(code + "Hypergraph([io])")
    assert r["outputs"][0]["type"] == "svg" and isinstance(h, Hypergraph)
    assert run('Edge([[], [1]], "InputOutput")')[0]["status"] == "error"
