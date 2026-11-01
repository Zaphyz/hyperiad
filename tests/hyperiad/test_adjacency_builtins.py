from hyperiad.values import format_value
from hypermatrix import ArrayObject, Hypermatrix

from hyperiad import Session


def run(code: str, **kw):
    s = Session(**kw)
    r = s.run(code)
    return r, s.globals.vars.get("ans")


H = 'h = Hypergraph([Edge([1, 2, 3], "Unordered"), Edge([3, 4]), Edge([1, 2])])\n'


def test_adjacency_hypermatrix_displays_one_picture_per_array():
    r, hm = run(H + "AdjacencyHypermatrix(h)")
    assert isinstance(hm, Hypermatrix) and len(hm) == 2
    assert [o["caption"] for o in r["outputs"]] == [
        "hypermatrix of 2 arrays · [1] 4×4 Integer array of order 2",
        "[2] 4×4×4 Integer array of order 3, symmetric",
    ]
    assert run(H + 'AdjacencyHypermatrix(h)["Symmetries"]')[1] == ["Rigid", "Symmetric"]
    assert run(H + "AdjacencyHypermatrix(h)[1]")[1] == ArrayObject(
        [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
    )


def test_labels_as_entries_and_their_condition():
    labelled = 'Hypergraph([Edge([1, 2], "Directed", "a"), Edge([2, 1])])'
    assert run(f"Normal(AdjacencyHypermatrix({labelled}, Labeled = true))")[1] == [[[0, "a"], [None, 0]]]
    twice = 'Hypergraph([Edge([1, 2], "Directed", "a"), Edge([1, 2], "Directed", "b")])'
    r, _ = run(f"AdjacencyHypermatrix({twice}, Labeled = true)")
    assert r["status"] == "error" and "repeated" in r["outputs"][-1]["data"]


def test_hypermatrix_values():
    _, hm = run("Hypermatrix([[1, 2], [[1, 2], [3, 4]]])")
    assert hm["Orders"] == [1, 2]
    assert run(format_value(hm))[1] == hm  # prints back to itself
    assert run("HypermatrixQ(Hypermatrix([1]))")[1] is True
    assert run("Normal(Hypermatrix([[[1, 2], [3, 4]], [5, 6]]))")[1] == [[5, 6], [[1, 2], [3, 4]]]
    r, _ = run("Hypermatrix([])")
    assert r["outputs"][0]["type"] == "text"
    r, _ = run("Hypermatrix([[1, 2]])[3]")
    assert r["status"] == "error" and "parts 1..1" in r["outputs"][-1]["data"]


def test_budget():
    r, _ = run('AdjacencyHypermatrix(Hypergraph([Edge(Range(8), "Unordered"), Edge(Range(9, 20))]))', budget=1e6)
    assert r["status"] == "too_expensive" and r["builtin"] == "AdjacencyHypermatrix"
