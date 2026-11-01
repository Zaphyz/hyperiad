from frontier.hypergraphs import Hypergraph, Term
from hypermatrix import ArrayObject

from hyperiad import Session


def run(code: str, **kw):
    s = Session(**kw)
    r = s.run(code)
    return r, s.globals.vars.get("ans")


MM = 'mm = PlexDiagram([[1, 2], [2, 3]], [2], Names = ["a", "b"])\n'
CHAIN = 'chain = PlexDiagram([[1, 2], [2, 3], [3, 4]], [2, 3], Names = ["a", "b", "c"])\nM = PlexRule(mm, "M")\n'
ARR = 'arr = {"a": [[1, 2], [3, 4]], "b": [[0, 1], [1, 0]], "c": [[2, 0], [0, 3]]}\n'


def test_plex_diagrams_are_drawn_hypergraphs():
    r, d = run(MM + "mm")
    assert isinstance(d, Hypergraph) and r["outputs"][0]["type"] == "svg"
    assert ">•</text>" not in r["outputs"][0]["data"] and ">a</text>" in r["outputs"][0]["data"]
    assert 'fill="rgb(25,38,84)"' in r["outputs"][0]["data"]  # the contracted index, solid dark blue
    assert run(MM + "[PlexFreeVertices(mm), PlexContractedVertices(mm), PlexNames(mm)]")[1] == [[1, 3], [2], ["a", "b"]]
    assert run("PlexMark")[1] == "•"


def test_products():
    assert run(MM + "PlexProduct(mm, [[[1, 2], [3, 4]], [[0, 1], [1, 0]]])")[1] == ArrayObject([[2, 1], [4, 3]])
    named = 'PlexProduct(mm, {"b": [[0, 1], [1, 0]], "a": [[1, 2], [3, 4]]}, Output = [3, 1])'
    assert run(MM + named)[1] == ArrayObject([[2, 4], [1, 3]])
    unordered = 'PlexDiagram([[1, 2, 3]], [3], Symmetry = "Unordered")'
    r, _ = run(f"PlexProduct({unordered}, [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])")
    assert r["status"] == "error" and "must be symmetric" in r["outputs"][-1]["data"]


FINAL = 'PlexEvents(M, e["Hypergraph"])[1]["Hypergraph"]'


def test_rewriting_is_symbolic_plex_algebra():
    _, labels = run(MM + CHAIN + f"Map(e => EdgeLabel(EdgeList({FINAL})[1]), PlexEvents(M, chain))")
    assert labels == [Term("M", [Term("M", ["a", "b"]), "c"]), Term("M", ["a", Term("M", ["b", "c"])])]
    value = f'PlexProduct({FINAL}, arr, Diagrams = {{"M": mm}}) == PlexProduct(chain, arr)'
    assert run(MM + CHAIN + ARR + f"Map(e => {value}, PlexEvents(M, chain))")[1] == [True, True]
    shared = 'PlexEvents(M, PlexDiagram([[1, 2], [2, 3], [2, 5]], [2], Names = ["a", "b", "d"]))'
    assert run(MM + CHAIN + shared)[1] == []


def test_budget():
    big = "PlexProduct(PlexDiagram([[1, 2, 3], [3, 4, 5]], [3]), [OneArray(3, 40), OneArray(3, 40)])"
    r, _ = run(big, budget=1e6)
    assert r["status"] == "too_expensive" and r["builtin"] == "PlexProduct"


def test_enumerate_plex_products():
    r, ds = run('EnumeratePlexProducts(2, 2, Equivalence = "isomorphism")')
    assert len(ds) == 16 and all(isinstance(d, Hypergraph) for d in ds)
    assert run("Length(EnumeratePlexProducts(2, 2))")[1] == 28
    assert run("Length(EnumeratePlexProducts(2, 2, Connected = false, Diagonals = true))")[1] == 62
    r, _ = run('EnumeratePlexProducts(2, 2, Equivalence = "homotopy")')
    assert r["status"] == "error" and "unknown equivalence" in r["outputs"][-1]["data"]
    r, _ = run("EnumeratePlexProducts(4, 4)", budget=1e6)
    assert r["status"] == "too_expensive"
