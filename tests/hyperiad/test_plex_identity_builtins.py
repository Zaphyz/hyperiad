import sympy
from frontier.hypergraphs import Hypergraph, Term
from hyperiad.plex_identity_builtins import PlexIdentity

from hyperiad import Session

OPS = (
    'mm = PlexDiagram([[1, 2], [2, 3]], [2], Names = ["x", "y"])\n'
    'tr = PlexDiagram([[2, 1]], Names = ["x"])\n'
    'ops = {"M": mm, "T": tr}\n'
)
FISH = (
    'fish = PlexDiagram([["i", "j", "p"], ["q", "r", "p"], ["q", "r", "k"]], ["p", "q", "r"], '
    'Names = ["x", "y", "z"])\n'
    "long = PlexDiagram([[1, 2, 3], [4, 5, 3], [4, 5, 6], [7, 8, 6], [7, 8, 9]], [3, 4, 5, 6, 7, 8], "
    'Names = ["a", "b", "c", "d", "e"])\n'
)


def run(code: str, **kw):
    s = Session(**kw)
    r = s.run(code)
    return r, s.globals.vars.get("ans")


def test_equal_q():
    assert run(OPS + "PlexEqualQ(T(T(a)), a, ops)")[1] is True
    assert run(OPS + "PlexEqualQ(T(M(a, b)), M(T(b), T(a)), ops)")[1] is True
    assert run(OPS + "PlexEqualQ(M(M(a, b), c), M(a, M(b, c)), ops)")[1] is True
    assert run(OPS + "PlexEqualQ(M(a, b), M(b, a), ops)")[1] is False


def test_identities_are_drawn_with_their_equation():
    r, ids = run(OPS + "PlexIdentities(PlexTerms(ops, [a, b], MaxOperations = 3), ops)")
    assert len(ids) == 8 and all(isinstance(x, PlexIdentity) for x in ids)
    assert [o["type"] for o in r["outputs"]] == ["svg"] * 8
    assert "[8] M(T(b), T(a)) = T(M(a, b))" in [o["caption"] for o in r["outputs"]]
    assert 'fill="rgb(25,38,84)"' in r["outputs"][1]["data"]  # the contracted index
    _, d = run(OPS + 'PlexIdentities([M(M(a, b), c), M(a, M(b, c))], ops)[1]["Diagram"]')
    assert isinstance(d, Hypergraph) and sorted(v.id for v in d.vertices) == ["i", "j", "p", "q"]
    assert len(run(OPS + "PlexIdentities([M(a, b), M(b, a)], ops, All = true)")[1]) == 2


def test_term_diagram():
    _, d = run(OPS + "PlexTermDiagram(M(a, T(b)), ops)")
    assert sorted((e.label.name, e.ids) for e in d.edges) == [("a", ("i", "p")), ("b", ("j", "p"))]
    r, _ = run(OPS + "PlexTermDiagram(a, ops)")
    assert r["status"] == "error" and "needs the order" in r["outputs"][-1]["data"]
    assert len(run(OPS + "PlexTermDiagram(a, ops, Order = 2)")[1].edges) == 1


def test_fish_semiheap_by_confluence_and_search():
    a, b, c, d, e = sympy.symbols("a b c d e")
    _, terms = run(FISH + 'PlexNormalTerms(long, PlexRule(fish, "F"))')
    assert set(terms) == {
        Term("F", [Term("F", [a, b, c]), d, e]),
        Term("F", [a, Term("F", [d, c, b]), e]),
        Term("F", [a, b, Term("F", [c, d, e])]),
    }
    assert len(run(FISH + 'PlexIdentities(PlexNormalTerms(long, PlexRule(fish, "F")), {"F": fish})')[1]) == 1
    _, forms = run(FISH + 'PlexNormalForms(long, PlexRule(fish, "F"))')
    assert len(forms) == 3 and all(len(f.edges) == 1 for f in forms)
    _, ids = run(FISH + 'PlexIdentities(PlexTerms({"F": fish}, [a, b, c, d, e], Order = 3), {"F": fish})')
    assert len(ids) == 120 and {len(x["Terms"]) for x in ids} == {3}


def test_errors_and_budget():
    r, _ = run(OPS + "PlexEqualQ(M(a), a, ops)")
    assert r["status"] == "error" and "takes 2 arguments" in r["outputs"][-1]["data"]
    r, _ = run(OPS + "PlexIdentities([a], [mm])")
    assert r["status"] == "error" and "operations as" in r["outputs"][-1]["data"]
    r, _ = run(OPS + 'PlexNormalForms(mm, PlexRule(tr, "T"))')
    assert r["status"] == "error" and "must have more plexes" in r["outputs"][-1]["data"]
    r, _ = run(OPS + "PlexTerms(ops, [a, b, c, d, e, f], MaxOperations = 5)", budget=1e6)
    assert r["status"] == "too_expensive" and r["builtin"] == "PlexTerms"
