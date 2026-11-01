"""The input–output symmetry type (user's specification, 2026-10-31; themes/algebroid-foundations BRIEF)."""

import itertools
import random

import pytest
import sympy
from frontier.hypergraphs import (
    Edge,
    Hypergraph,
    HypergraphError,
    HypergraphRule,
    Term,
    apply_rule,
    canonical_hypergraph,
    hypergraph_compose,
    hypergraph_saturate,
    isomorphic,
    rule_matches,
)
from frontier.hypergraphs.canon import canonical_data
from frontier.hypergraphs.core import io_symmetry, normalize_symmetry, parse_io, symmetry_orderings

TYPES = ("Directed", "Unordered", "Cyclic")


def test_spelling_blocks_and_canonical_order():
    e = Edge([["b", "a"], ["d", "c", "e"]], ("Unordered", "Cyclic"), "f")
    assert e.symmetry == "InputOutput(Unordered:2,Cyclic:3)"
    assert e.block_ids == (("a", "b"), ("c", "e", "d"))  # each block in its own canonical order
    assert Edge(["b", "a", "e", "d", "c"], "inputoutput(unordered:2, cyclic:3)", "f") == e
    assert parse_io(e.symmetry) == ("Unordered", 2, "Cyclic", 3)
    assert normalize_symmetry("InputOutput(directed:1,directed:1)") == "Directed"
    assert Edge([["a", "b"], ["c"]], "InputOutput").symmetry == "InputOutput(Directed:2,Directed:1)"


def test_one_input_one_output_is_the_directed_binary_edge():
    for s, t in itertools.product(TYPES, TYPES):
        assert Edge([["x"], ["y"]], (s, t), "f") == Edge(["x", "y"], "Directed", "f")
    assert Edge([["x"], ["x"]], "InputOutput") == Edge(["x", "x"])  # an endomorphism loop


def test_blocks_are_nonempty_and_sizes_add_up():
    for blocks in ([[], ["a"]], [["a"], []], [[], []]):
        with pytest.raises(HypergraphError):
            Edge(blocks, ("Directed", "Directed"))
    with pytest.raises(HypergraphError):
        Edge(["a", "b"], "InputOutput(Directed:2,Directed:1)")
    with pytest.raises(HypergraphError):
        Edge([["a"], ["b"], ["c"]], ("Directed", "Directed"))


def test_repeated_vertices_follow_the_other_types():
    e = Edge([["x", "x"], ["x", "y"]], ("Unordered", "Directed"))
    assert e.ids == ("x", "x", "x", "y")
    assert Edge([["x", "y"], ["x"]], ("Unordered", "Directed")) == Edge([["y", "x"], ["x"]], ("Unordered", "Directed"))


def test_orderings_are_the_products_of_the_block_orderings():
    for s, t in itertools.product(TYPES, TYPES):
        for a, b in [(2, 1), (1, 3), (2, 3), (3, 2)]:
            sym = io_symmetry(s, a, t, b)
            perms = symmetry_orderings(sym, a + b)
            size = {
                "Directed": lambda n: 1,
                "Cyclic": lambda n: n,
                "Unordered": lambda n: len(list(itertools.permutations(range(n)))),
            }  # noqa: E501
            assert len(set(perms)) == size[s](a) * size[t](b)
            assert all(set(p[:a]) == set(range(a)) and set(p[a:]) == set(range(a, a + b)) for p in perms)


def _random_io_hypergraph(rng, n, m):
    edges = []
    for _ in range(m):
        a, b = rng.randint(1, 3), rng.randint(1, 3)
        s, t = rng.choice(TYPES), rng.choice(TYPES)
        edges.append(
            Edge([[rng.randrange(n) for _ in range(a)], [rng.randrange(n) for _ in range(b)]], (s, t), rng.choice("fg"))
        )
    return Hypergraph(list(range(n)), edges)


def _shuffled(hg, rng):
    """The same hypergraph under a random vertex renaming, each edge's blocks written in a random equivalent order."""
    perm = list(range(len(hg.vertices)))
    rng.shuffle(perm)
    edges = []
    for e in hg.edges:
        ids = [perm[i] for i in e.ids]
        o = rng.choice(symmetry_orderings(e.symmetry, e.arity))
        edges.append(Edge([ids[p] for p in o], e.symmetry, e.label))
    return Hypergraph(list(range(len(hg.vertices))), edges)


def test_canonical_form_is_an_isomorphism_invariant():
    rng = random.Random(11)
    for _ in range(60):
        g = _random_io_hypergraph(rng, 5, 4)
        h = _shuffled(g, rng)
        assert canonical_hypergraph(g) == canonical_hypergraph(h)
        assert isomorphic(g, h)
        brute = canonical_data(g, method="bruteforce").form
        assert brute == canonical_data(h, method="bruteforce").form


def test_block_types_and_block_roles_are_told_apart():
    a = Hypergraph([Edge([["x", "y"], ["z"]], ("Unordered", "Directed"))])
    b = Hypergraph([Edge([["x"], ["y", "z"]], ("Directed", "Unordered"))])
    c = Hypergraph([Edge([["x", "y"], ["z"]], ("Directed", "Directed"))])
    d = Hypergraph([Edge(["x", "y", "z"], "Directed")])
    assert not isomorphic(a, b) and not isomorphic(a, c) and not isomorphic(c, d)
    # a directed block (x, y) is not the same as (y, x), up to isomorphism it is
    assert Edge([["x", "y"], ["z"]], "InputOutput") != Edge([["y", "x"], ["z"]], "InputOutput")
    assert isomorphic(Hypergraph([Edge([["x", "y"], ["z"]], "InputOutput")]),
                      Hypergraph([Edge([["y", "x"], ["z"]], "InputOutput")]))  # fmt: skip


def test_matching_within_blocks():
    f, g = sympy.symbols("f g")
    pattern = Hypergraph([Edge([[1, 2], [3]], ("Unordered", "Directed"), f)])
    host = Hypergraph([Edge([["p", "q"], ["r"]], ("Unordered", "Directed"), "F")])
    assert len(rule_matches(HypergraphRule(pattern, pattern), host)) == 2  # the input block in either order
    wrong = Hypergraph([Edge([["p"], ["q", "r"]], ("Directed", "Unordered"), "F")])
    assert rule_matches(HypergraphRule(pattern, pattern), wrong) == []
    # a directed binary pattern matches the one-input, one-output edge, which is the same edge
    arrow = HypergraphRule(Hypergraph([Edge([1, 2], "Directed", f)]), Hypergraph([Edge([2, 1], "Directed", g)]))
    assert len(rule_matches(arrow, Hypergraph([Edge([["a"], ["b"]], "InputOutput", "h")]))) == 1


def test_polycategory_composition_along_one_object():
    f, g = sympy.symbols("f g")
    uu, du = ("Unordered", "Unordered"), ("Directed", "Unordered")
    rule = HypergraphRule(
        Hypergraph([Edge([[1, 2], [3, 4]], uu, f), Edge([[3], [5, 6]], du, g)]),
        Hypergraph([1, 2, 4, 5, 6], [Edge([[1, 2], [4, 5, 6]], uu, Term("∘", [g, f]))]),
    )
    host = Hypergraph([Edge([["p", "q"], ["y", "x"]], uu, "F"), Edge([["x"], ["u", "v"]], du, "G")])
    results = {ev.hypergraph for ev in apply_rule(rule, host)}
    assert results == {
        Hypergraph([Edge([["p", "q"], ["u", "v", "y"]], ("Unordered", "Unordered"), Term("∘", ("G", "F")))])
    }


def test_saturation_and_composition_accept_input_output_edges():
    f, g = sympy.symbols("f g")
    plug = HypergraphRule(  # multicategory: plug the output of g into the first input of f
        Hypergraph([Edge([[1, 2], [3]], "InputOutput", g), Edge([[3, 4], [5]], "InputOutput", f)]),
        Hypergraph([1, 2, 4, 5], [Edge([[1, 2, 4], [5]], "InputOutput", Term("∘", [f, g]))]),
    )
    diagram = Hypergraph([Edge([["a", "b"], ["c"]], "InputOutput", "G"), Edge([["c", "d"], ["e"]], "InputOutput", "F")])
    sat = hypergraph_saturate(diagram, [plug])
    assert Edge([["a", "b", "d"], ["e"]], "InputOutput", Term("∘", ("F", "G"))) in sat.final.edges
    composed = hypergraph_compose(
        HypergraphRule(
            Hypergraph([Edge([[1, 2], [3]], "InputOutput", "a"), Edge([[3, 4], [5]], "InputOutput", "b")]),
            Hypergraph([Edge([[1, 2, 4], [5]], "InputOutput")]),
        ),
        {"a": Hypergraph([Edge([[0, 1], [2]], "InputOutput")]), "b": Hypergraph([Edge([[2, 0], [1]], "InputOutput")])},
    )
    assert composed.edges == (Edge([[0, 1, 0], [1]], "InputOutput"),)


def test_drawing_two_lobes_and_a_waist():
    from frontier.hypergraphs.geometry import Area, Arrowhead, io_blocks
    from frontier.hypergraphs.pictures import hypergraph_picture
    from frontier.hypergraphs.svg import hypergraph_svg

    assert io_blocks("inputoutput(unordered:2,directed:3)") == ("unordered", 2, "directed", 3)
    for blocks, types in [
        ([["a"], ["d", "e"]], ("Directed", "Unordered")),
        ([["a", "b", "c"], ["d", "e"]], ("Unordered", "Directed")),
        ([["a", "b"], ["d", "e", "g"]], ("Cyclic", "Cyclic")),
        ([["a", "a"], ["a", "b"]], ("Unordered", "Directed")),  # repeated vertices
        ([["a", "b"], ["b", "a"]], ("Unordered", "Unordered")),  # both blocks on the same place
    ]:
        hg = Hypergraph([Edge(blocks, types, "f")])
        scene = hypergraph_picture(hg)
        assert sum(isinstance(s, Arrowhead) for s in scene.shapes) >= 1  # the waist's arrowhead
        assert any(isinstance(s, Area) and s.fill is not None for s in scene.shapes)
        assert hypergraph_svg(hg).startswith("<svg")
    # one input and one output is drawn exactly as the directed binary edge
    one = hypergraph_picture(Hypergraph([Edge([["a"], ["b"]], "InputOutput")]))
    two = hypergraph_picture(Hypergraph([Edge(["a", "b"], "Directed")]))
    assert [type(s) for s in one.shapes] == [type(s) for s in two.shapes]
