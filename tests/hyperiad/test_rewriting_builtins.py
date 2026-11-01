import sympy
from frontier.conformance import ported
from frontier.hypergraphs import Event, FormalV, HypergraphRule, Term
from hyperiad.values import format_value

from hyperiad import Session


def run(code: str, **kw):
    s = Session(**kw)
    r = s.run(code)
    return r, s.globals.vars.get("ans")


SUB = "sub = HypergraphRule([[1, 2]], [[1, 3], [3, 2]])\npath2 = Hypergraph([[1, 2], [2, 3]])\n"


def test_rules_display_drawn_and_print_back():
    r, rule = run(SUB + "sub")
    assert isinstance(rule, HypergraphRule)
    out = r["outputs"][0]
    assert out["type"] == "svg" and out["caption"] == "rule: 1 edge, 2 vertices → 2 edges, 3 vertices · creates [3]"
    text = format_value(rule)
    assert text == "HypergraphRule(Hypergraph([Edge([1, 2])]), Hypergraph([Edge([1, 3]), Edge([3, 2])]))"
    assert run(text)[1] == rule
    assert run(SUB + 'sub["NewVertices"]')[1] == [3]


@ported("hypergraph-paclet", "tests/RewritingExamples.wls", 40)
def test_rule_q():
    assert run(SUB + "HypergraphRuleQ(sub)")[1] is True
    assert run("HypergraphRuleQ([[1, 2]] -> [[2, 1]])")[1] is False


def test_applying_a_rule_and_reading_events():
    r, events = run(SUB + "sub(path2)")
    assert len(events) == 2 and all(isinstance(e, Event) for e in events)
    assert [o["type"] for o in r["outputs"]] == ["svg", "svg"]
    assert r["outputs"][0]["caption"] == "[1] event: edges [1] rewritten · creates [FormalV(1)]"
    assert run(SUB + 'sub(path2)[1]["MatchEdgePositions"]')[1] == [1]  # counted from 1, as the paclet does
    assert run(SUB + 'sub(path2)[2]["EdgeOrigins"]')[1] == [1, None, None]
    assert run(SUB + 'VertexCount(sub(path2)[1]["Hypergraph"])')[1] == 4
    assert run(SUB + "HypergraphRuleApply(sub, path2) == sub(path2)")[1] is True
    assert run(SUB + 'sub(path2)[1]["NewVertices"]')[1] == [FormalV(1)]


def test_matches_and_options():
    _, ms = run(SUB + "HypergraphRuleMatches(sub, path2)")
    assert ms[1] == {"MatchEdgePositions": [2], "RuleVertexMap": {1: 2, 2: 3}, "LabelBindings": {}}
    loop = "HypergraphRuleMatches(HypergraphRule([[1, 2]], []), Hypergraph([[1, 1]])"
    assert run(loop + ")")[1] == []
    assert len(run(loop + ", DistinctVertices = false)")[1]) == 1


def test_label_variables_and_composite_labels():
    code = 'hl = Hypergraph([Edge([1, 2], "Directed", f), Edge([2, 3], "Directed", g)])\n'
    code += 'comp = HypergraphRule([Edge([1, 2], "Directed", x), Edge([2, 3], "Directed", y)], '
    code += '[Edge([1, 3], "Directed", Row(["(", x, y, ")"]))])\n'
    r, ev = run(code + "comp(hl)[1]")
    f, g = sympy.symbols("f g")
    assert ev.hypergraph.edges[0].label == Term("Row", [["(", f, g, ")"]])
    assert ev.label_bindings == {sympy.Symbol("x"): f, sympy.Symbol("y"): g}
    assert "binds {x: f, y: g}" in r["outputs"][0]["caption"]
    assert ">(fg)</text>" in r["outputs"][0]["data"]  # the composite label, drawn
    label = run(code + 'EdgeLabel(EdgeList(comp(hl)[1]["Hypergraph"])[1])')[1]
    assert format_value(label) == 'Row(["(", f, g, ")"])'

    def count(pattern, opts=""):
        return len(
            run(code + f'HypergraphRuleMatches(HypergraphRule([Edge([1, 2], "Directed", {pattern})], []), hl{opts})')[1]
        )

    assert (count("Verbatim(f)"), count("x"), count("x", ", LabelPatterns = false")) == (1, 2, 0)


def test_undefined_functions_stay_symbolic():
    assert run("h(x) + 1")[1] == sympy.Function("h")(sympy.Symbol("x")) + 1
    assert run('Subscript("1", p)')[1] == Term("Subscript", ["1", sympy.Symbol("p")])


@ported("hypergraph-paclet", "tests/RewritingExamples.wls", 528)
def test_plot_highlights_coordinates_and_embedding():
    _, g = run(
        "HypergraphPlot(Hypergraph([[1, 2], [2, 3], [1, 2]]), Highlight = [Edge([2, 3])], HighlightVertices = [3])"
    )
    assert g.svg.count("rgb(219,161,83)") >= 2
    _, emb = run("HypergraphEmbedding(Hypergraph([[1, 2], [2, 3]]), VertexCoordinates = [[0, 0], [1, 0], [2, 0]])")
    assert emb == [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    r, _ = run("HypergraphPlot(Hypergraph([[1, 2]]), VertexCoordinates = {1: [0, 0], 2: [1, 1]})")
    assert r["status"] == "ok"


def test_rule_and_event_plots():
    r, g = run(SUB + "HypergraphRulePlot(sub, VertexLabels = true)")
    assert g.svg.startswith("<svg") and r["outputs"][0]["type"] == "svg"
    r, gs = run(SUB + "HypergraphEventPlot(sub, path2)")
    assert len(gs) == 2 and [o["type"] for o in r["outputs"]] == ["svg", "svg"]
    r, _ = run("HypergraphEventPlot(42)")
    assert r["status"] == "error" and "takes an event" in r["outputs"][-1]["data"]


def test_errors_and_budget():
    r, _ = run('HypergraphRule([[1, 2]], "nope")')
    assert r["status"] == "error" and "not a hypergraph" in r["outputs"][-1]["data"]
    r, _ = run(SUB + 'sub["Nonsense"]')
    assert r["status"] == "error" and "unknown property" in r["outputs"][-1]["data"]
    big = "HypergraphRule([[1, 2], [2, 3], [3, 4], [4, 5]], [])(Hypergraph(Table([i, i + 1], [i, 60])))"
    r, _ = run(big, budget=1e6)
    assert r["status"] == "too_expensive" and r["builtin"] == "HypergraphRuleApply"
