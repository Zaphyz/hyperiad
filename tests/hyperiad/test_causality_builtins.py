from frontier.conformance import ported
from frontier.hypergraphs.causality import RewritingSystem, SystemGraph
from hyperiad.values import RuleValue, format_value

from hyperiad import Session


def run(code: str, **kw):
    s = Session(**kw)
    r = s.run(code)
    return r, s.globals.vars.get("ans")


MEET = (
    "sub = HypergraphRule([[1, 2]], [[1, 3], [3, 2]])\n"
    "meet = HypergraphRewritingSystem(sub, Hypergraph([[1, 2], [2, 3]]))\n"
)


@ported("hypergraph-paclet", "tests/CausalityExamples.wls", 37, 43)
def test_systems_from_arrows_and_q():
    _, arrows = run("HypergraphRewritingSystem([[1, 2]] -> [[1, 3], [3, 2]], [[1, 2]])")
    _, rules = run("HypergraphRewritingSystem(HypergraphRule([[1, 2]], [[1, 3], [3, 2]]), Hypergraph([[1, 2]]))")
    assert isinstance(arrows, RewritingSystem) and arrows == rules
    assert run(
        MEET + "[HypergraphRewritingSystemQ(meet), HypergraphRewritingSystemQ(sub), HypergraphRewritingSystemQ(7)]"
    )[1] == [True, False, False]


def test_properties_by_call_and_by_part():
    assert run(MEET + 'meet["EventCount", 2]')[1] == 8
    assert run(MEET + 'meet("StateCount", 2)')[1] == 9
    _, edges = run(MEET + 'meet("CausalEdges", 2, CausalDependence = "Edges")')
    assert [(c["From"], c["To"]) for c in edges] == [(1, 3), (1, 5), (2, 7), (2, 8)]
    assert run(MEET + 'meet["EventList", 2][3]["InputState"]')[1] == 2
    assert run(MEET + 'meet["Options"]["CausalDependence"]')[1] == "Elements"
    assert run("CausalDependenceTypes")[1] == ["Elements", "Edges", "Vertices"]


def test_graphs_display_drawn():
    r, g = run(MEET + 'meet("CausalGraph", 2)')
    assert isinstance(g, SystemGraph)
    out = r["outputs"][0]
    assert out["type"] == "svg" and out["caption"] == "causal graph · 8 vertices · 6 edges · dependence by elements"
    assert run(MEET + 'meet("CausalGraph", 2)["EdgeList"]')[1][0] == RuleValue(1, 3)
    assert run(MEET + 'meet("BranchialGraph", 1)["EdgeList"]')[1] == [[2, 3]]
    assert run(MEET + 'meet("EvolutionCausalGraph", 1)["VertexList"]')[1][0] == ["State", 1]
    r, _ = run(MEET + 'meet("EvolutionGraph", 4)')
    assert r["outputs"][0]["kind"] == "warning" and "MaxDrawnVertices" in r["outputs"][0]["data"]
    assert r["outputs"][1]["type"] == "svg"  # drawn with plain vertices


def test_systems_and_event_records_display():
    r, _ = run(MEET + "meet")
    assert r["outputs"][0]["caption"].startswith("rewriting system · 1 rule · initial state 3 vertices, 2 edges")
    assert [o["caption"] for o in r["outputs"]][-1] == "initial state"
    r, _ = run(MEET + 'meet["EventList", 1]')
    assert [o["caption"] for o in r["outputs"]] == [
        "[1] event 1 · generation 1 · rule 1 · state 1 → 2",
        "[2] event 2 · generation 1 · rule 1 · state 1 → 3",
    ]
    _, held = run('HypergraphRewritingSystem(HypergraphRule([[1, 2]], [[2, 1]]), [[1, 2]], CausalDependence = "Edges")')
    assert run(format_value(held))[1] == held


def test_standalone_causal_graph_and_budget():
    _, g = run("HypergraphCausalGraph([[1, 2]] -> [[1, 3], [3, 2]], Hypergraph([[1, 2]]), 3)")
    assert (g.vertex_count, g.edge_count) == (9, 8)  # 1 + 2 + 6 events, each depending on the one before
    _, g = run(MEET + 'HypergraphCausalGraph(meet, 2, CausalDependence = "Edges")')
    assert g.edge_count == 4
    r, _ = run(MEET + 'meet("CausalGraph", 6)', budget=1e5)
    assert r["status"] == "too_expensive" and r["builtin"] == "RewritingSystemProperty"
    r, _ = run(MEET + 'meet("Nope", 1)')
    assert r["status"] == "error" and "unknown property" in r["outputs"][-1]["data"]
    r, _ = run(MEET + "HypergraphRewritingSystem(sub, [[1, 2]], Nonsense = 1)")
    assert r["status"] == "error" and "unknown option" in r["outputs"][-1]["data"]
