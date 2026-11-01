"""One pass, composition products and saturation (frontier.hypergraphs.composition)."""

from collections import Counter

import pytest
import sympy
from frontier.hypergraphs import (
    Edge,
    FormalV,
    Hypergraph,
    HypergraphError,
    HypergraphRule,
    Term,
    Vertex,
    hypergraph_compose,
    hypergraph_saturate,
    one_pass,
    parallel_edges,
)

x, y, z, X, Y, Z = sympy.symbols("x y z X Y Z")
COMPOSE = HypergraphRule(
    Hypergraph([Edge([1, 2], "Directed", "a"), Edge([2, 3], "Directed", "b")]), Hypergraph([Edge([1, 3])])
)
G = Hypergraph([1, 2, 3], [Edge([1, 2]), Edge([2, 3]), Edge([3, 3])])
H = Hypergraph([1, 2, 3], [Edge([2, 1]), Edge([3, 1])])


def comp(g, f):
    return Term("∘", [g, f])


def test_graph_composition():
    assert Counter(e.ids for e in hypergraph_compose(COMPOSE, {"a": G, "b": H}).edges) == {
        (1, 1): 1,
        (2, 1): 1,
        (3, 1): 1,
    }
    square = hypergraph_compose(COMPOSE, {"a": G, "b": G})
    assert sorted(e.ids for e in square.edges) == [(1, 3), (2, 3), (3, 3)]
    loops = Hypergraph([1, 2], [Edge([1, 1]), Edge([1, 1]), Edge([1, 2])])
    counted = hypergraph_compose(COMPOSE, {"a": loops, "b": loops})
    boolean = hypergraph_compose(COMPOSE, {"a": loops, "b": loops}, multiplicity="Boolean")
    assert Counter(e.ids for e in counted.edges) == {(1, 1): 4, (1, 2): 2}  # 2x2 loop pairs, 2 paths to 2
    assert sorted(e.ids for e in boolean.edges) == [(1, 1), (1, 2)]
    same = HypergraphRule(
        Hypergraph([Edge([1, 2], "Directed", "a"), Edge([2, 3], "Directed", "a")]), Hypergraph([Edge([1, 3])])
    )
    assert len(hypergraph_compose(same, {"a": loops}).edges) == 6  # A·A
    assert len(hypergraph_compose(same, {"a": loops}, distinct_edges=True).edges) == 4  # A·A − diag(loops)


def test_composition_with_labels():
    rule = HypergraphRule(
        Hypergraph([Edge([1, 2], "Directed", Term("a", [x])), Edge([2, 3], "Directed", Term("b", [y]))]),
        Hypergraph([Edge([1, 3], "Directed", comp(y, x))]),
    )
    gl = Hypergraph(
        [1, 2, 3], [Edge([1, 2], "Directed", "f"), Edge([2, 3], "Directed", "g"), Edge([2, 3], "Directed", "h")]
    )
    labelled = hypergraph_compose(rule, {"a": gl, "b": gl}, multiplicity="Labels")
    assert sorted(str(e.label) for e in labelled.edges) == ["∘(g, f)", "∘(h, f)"]  # a formal sum of two words
    assert all(e.label is None for e in hypergraph_compose(rule, {"a": gl, "b": gl}).edges)


def test_composition_errors():
    with pytest.raises(HypergraphError, match="names none of the factors"):
        hypergraph_compose(COMPOSE, {"a": G, "c": H})
    grow = HypergraphRule(Hypergraph([Edge([1, 2], "Directed", "a")]), Hypergraph([Edge([1, 5])]))
    with pytest.raises(HypergraphError, match="creates vertices"):
        hypergraph_compose(grow, {"a": G})
    with pytest.raises(HypergraphError, match="multiplicity"):
        hypergraph_compose(COMPOSE, {"a": G, "b": H}, multiplicity="Real")


def test_one_pass_names_created_vertices_apart():
    grow = HypergraphRule(Hypergraph([Edge([1, 2])]), Hypergraph([Edge([1, 5]), Edge([5, 2])]))
    out = one_pass(grow, Hypergraph([Edge([1, 2]), Edge([2, 3])]))
    assert sorted(out.vertex_ids, key=str) == sorted([1, 2, 3, FormalV(1), FormalV(2)], key=str)
    assert len(out.edges) == 4


SQUARE = Hypergraph(
    [Vertex(v, v) for v in "ABCD"],
    [Edge(["A", "B"], "Directed", "f"), Edge(["B", "D"], "Directed", "g"),
     Edge(["A", "C"], "Directed", "h"), Edge(["C", "D"], "Directed", "k")],
)  # fmt: skip
COMPOSITION = HypergraphRule(
    Hypergraph([Vertex(1, X), Vertex(2, Y), Vertex(3, Z)], [Edge([1, 2], "Directed", x), Edge([2, 3], "Directed", y)]),
    Hypergraph([Vertex(1, X), Vertex(2, Y), Vertex(3, Z)], [Edge([1, 3], "Directed", comp(y, x))]),
)
IDENTITY = HypergraphRule(
    Hypergraph([Vertex(1, X)], []), Hypergraph([Vertex(1, X)], [Edge([1, 1], "Directed", Term("id", [X]))])
)
UNITS = [
    HypergraphRule([Edge([1, 2], "Directed", comp(Term("id", [Y]), x))], [Edge([1, 2], "Directed", x)]),
    HypergraphRule([Edge([1, 2], "Directed", comp(x, Term("id", [X])))], [Edge([1, 2], "Directed", x)]),
    HypergraphRule(Hypergraph([Edge([1, 2], "Directed", comp(comp(z, y), x))]),
                   Hypergraph([Edge([1, 2], "Directed", comp(z, comp(y, x)))])),
]  # fmt: skip


def test_saturating_a_square_into_its_category():
    free = hypergraph_saturate(SQUARE, [COMPOSITION])
    assert free.saturated and [len(s.edges) for s in free.states] == [4, 6]
    assert parallel_edges(free.final) == {(("A", "D"), "Directed"): [comp("g", "f"), comp("k", "h")]}
    syntactic = hypergraph_saturate(SQUARE, [COMPOSITION, IDENTITY], max_rounds=3)
    assert not syntactic.saturated and [len(s.edges) for s in syntactic.states] == [4, 10, 26, 114]
    category = hypergraph_saturate(SQUARE, [COMPOSITION, IDENTITY], equations=UNITS)
    assert category.saturated and len(category.final.edges) == 10
    commute = HypergraphRule(
        Hypergraph([Edge([1, 2], "Directed", comp("k", "h")), Edge([1, 2], "Directed", comp("g", "f"))]),
        Hypergraph([Edge([1, 2], "Directed", comp("g", "f"))]),
    )
    commuting = hypergraph_saturate(SQUARE, [COMPOSITION, IDENTITY], equations=[*UNITS, commute])
    assert commuting.saturated and parallel_edges(commuting.final) == {}


def test_identify_by_label_makes_object_creation_stop():
    terminal = HypergraphRule(  # every object has an arrow to a terminal object 1
        Hypergraph([Vertex(1, X)], []),
        Hypergraph([Vertex(1, X), Vertex(2, "1")], [Edge([1, 2], "Directed", Term("!", [X]))]),
    )
    two = Hypergraph([Vertex("a", "A"), Vertex("b", "B")], [])
    merged = hypergraph_saturate(two, [terminal], identify_by_label=True)
    assert merged.saturated and len(merged.final.vertices) == 3
    assert sorted(str(e.label) for e in merged.final.edges) == ["!(1)", "!(A)", "!(B)"]
    apart = hypergraph_saturate(two, [terminal], max_rounds=3)
    assert not apart.saturated and [len(s.vertices) for s in apart.states] == [2, 4, 8, 16]


def test_equations_that_undo_each_other_are_reported():
    there = HypergraphRule(Hypergraph([Edge([1, 2], "Directed", "p")]), Hypergraph([Edge([1, 2], "Directed", "q")]))
    back = HypergraphRule(Hypergraph([Edge([1, 2], "Directed", "q")]), Hypergraph([Edge([1, 2], "Directed", "p")]))
    with pytest.raises(HypergraphError, match="did not reach a normal form"):
        hypergraph_saturate(
            Hypergraph([Edge([1, 2], "Directed", "p")]), [], equations=[there, back], max_equation_steps=50
        )


def test_congruence_applies_label_equations_inside_labels():
    chain = Hypergraph(
        [Vertex(i, f"O{i}") for i in range(6)], [Edge([i, i + 1], "Directed", f"m{i}") for i in range(5)]
    )
    inside = hypergraph_saturate(chain, [COMPOSITION, IDENTITY], equations=UNITS, congruence=True)
    outside = hypergraph_saturate(chain, [COMPOSITION, IDENTITY], equations=UNITS)

    def morphisms(s):
        return sum(1 for e in s.final.edges if e.ids[0] != e.ids[1])

    assert inside.saturated and morphisms(inside) == 15  # one per pair i < j
    assert outside.saturated and morphisms(outside) == 21  # bracketings of one morphism stay apart
    cycle = Hypergraph([Vertex(1, "A"), Vertex(2, "B")], [Edge([1, 2], "Directed", "f"), Edge([2, 1], "Directed", "g")])
    capped = hypergraph_saturate(cycle, [COMPOSITION, IDENTITY], equations=UNITS, congruence=True, max_rounds=3)
    assert not capped.saturated and capped.stopped_by == "MaxRounds"
    big = hypergraph_saturate(
        cycle, [COMPOSITION, IDENTITY], equations=UNITS, congruence=True, max_rounds=50, max_edges=30
    )
    assert not big.saturated and big.stopped_by == "MaxEdges"
