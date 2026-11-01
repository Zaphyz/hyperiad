"""Canonical forms and isomorphism, ported from the paclet's BasicExamples (sections 5-10)."""

import random

import pytest
from frontier.conformance import ported
from frontier.hypergraphs import (
    Edge,
    Hypergraph,
    HypergraphError,
    LabelToken,
    Vertex,
    apply_vertex_map,
    canonical_data,
    canonical_hypergraph,
    find_isomorphism,
    isomorphic,
)

H, E, V = Hypergraph, Edge, Vertex
METHODS = ("automatic", "bruteforce")

a = H([[1, 2], [2, 3]])
b = H([[10, 20], [20, 30]])
c = H([[1, 2], [1, 3]])
au = H([E([1, 2], "Unordered"), E([2, 3], "Unordered")])
cu = H([E([1, 2], "Unordered"), E([1, 3], "Unordered")])


@ported("hypergraph-paclet", "tests/BasicExamples.wls", 116, 117, 125, 126)
def test_plain_isomorphism():
    assert isomorphic(a, b) is True
    assert isomorphic(a, c) is False  # directed: a path is not an out-star
    assert isomorphic(au, cu) is True  # undirected, they are
    assert isomorphic(a, au) is False
    assert find_isomorphism(a, b) == {1: 10, 2: 20, 3: 30}
    assert find_isomorphism(a, c) == {}


l1 = H([V(1, "A"), V(2, "B"), V(3, "C")], [[1, 2], [2, 3]])
l2 = H([V(10, "A"), V(20, "B"), V(30, "C")], [[10, 20], [20, 30]])
l3 = H([V(1, "C"), V(2, "B"), V(3, "A")], [[1, 2], [2, 3]])


@ported("hypergraph-paclet", "tests/BasicExamples.wls", 134, 137, 138, 145, 146, 153, 154)
def test_labels_count_up_to_renaming():
    assert isomorphic(l1, l2) is True
    assert isomorphic(l1, l3) is True  # by the renaming A <-> C
    assert isomorphic(l1, l3, labeled=False) is True
    m1 = H([V(1, "A"), V(2, "A"), V(3, "B")], [[1, 2], [2, 3]])
    m2 = H([V(1, "A"), V(2, "B"), V(3, "B")], [[1, 2], [2, 3]])
    assert isomorphic(m1, m2) is False  # the case upstream got wrong
    assert isomorphic(m1, m2, labeled=False) is True
    t1 = H([E([1, 2], "Directed", "p"), E([2, 3], "Directed", "q")])
    t2 = H([E([1, 2], "Directed", "q"), E([2, 3], "Directed", "p")])
    assert isomorphic(t1, t2) is True
    assert isomorphic(t1, t2, labeled=False) is True


g1 = H([E([1, 2], "Directed", "a"), E([2, 3], "Directed", "b")])
g2 = H([E([4, 5], "Directed", "x"), E([5, 6], "Directed", "y")])
g3 = H([E([4, 5], "Directed", "x"), E([5, 6], "Directed", "x")])


@ported("hypergraph-paclet", "tests/BasicExamples.wls", 161, 162, 166, 167, 168)
def test_label_correspondences():
    assert isomorphic(g1, g2) is True
    assert find_isomorphism(g1, g2, labels=True) == {
        "Vertices": {1: 4, 2: 5, 3: 6},
        "VertexLabels": {},
        "EdgeLabels": {"a": "x", "b": "y"},
    }
    assert isomorphic(g1, g3) is False
    assert isomorphic(g1, g3, labeled=False) is True
    assert find_isomorphism(g1, g3, labels=True) == {"Vertices": {}, "VertexLabels": {}, "EdgeLabels": {}}


@ported("hypergraph-paclet", "tests/BasicExamples.wls", 180, 181, 184)
def test_vertex_and_edge_labels_correspond_independently():
    i1 = H([E([V(1, "A"), V(2, "A")], "Directed", "f"), E([V(2, "A"), V(3, "B")], "Directed", "g")])
    i2 = H([E([V(1, "P"), V(2, "P")], "Directed", "g"), E([V(2, "P"), V(3, "Q")], "Directed", "f")])
    i3 = H([E([V(1, "P"), V(2, "Q")], "Directed", "f"), E([V(2, "Q"), V(3, "Q")], "Directed", "g")])
    assert isomorphic(i1, i2) is True
    found = find_isomorphism(i1, i2, labels=True)
    assert (found["VertexLabels"], found["EdgeLabels"]) == ({"A": "P", "B": "Q"}, {"f": "g", "g": "f"})
    assert isomorphic(i1, i3) is False


@ported("hypergraph-paclet", "tests/BasicExamples.wls", 192, 193, 200, 201, 205, 208, 209)
def test_forced_and_free_vertex_maps():
    k1 = H([E([1, 2], "Directed", "f"), E([1, 3], "Directed", "f"), E([1, 4], "Directed", "g")])
    k2 = H([E([1, 2], "Directed", "p"), E([1, 3], "Directed", "q"), E([1, 4], "Directed", "q")])
    assert isomorphic(k1, k2) is True
    assert find_isomorphism(k1, k2)[4] == 2  # the lone g must go to the lone p
    e1 = H([E([1, 2], "Directed", "f"), E([2, 3], "Directed", "g")])
    e2 = H([E([7, 8], "Directed", "f"), E([8, 9], "Directed", "g")])
    assert isomorphic(e1, e2) is True
    assert find_isomorphism(e1, e2) == {1: 7, 2: 8, 3: 9}
    s1 = H([E([1, 2], "Directed", "f"), E([1, 3], "Directed", "g")])
    s2 = H([E([1, 2], "Directed", "g"), E([1, 3], "Directed", "f")])
    assert isomorphic(s1, s2) is True
    # The paclet asserts the identity here; exchanging 2 and 3 is an isomorphism too. We
    # happen to find the identity, but what the test pins down is that the map is valid.
    found = find_isomorphism(s1, s2, labels=True)
    assert found["Vertices"] == {1: 1, 2: 2, 3: 3}
    assert found["EdgeLabels"] == {"f": "g", "g": "f"}
    assert apply_vertex_map(s1, found["Vertices"], found["VertexLabels"], found["EdgeLabels"]) == s2


@ported("hypergraph-paclet", "tests/BasicExamples.wls", 215, 217, 220, 223)
def test_labelled_edges_negative_cases():
    f12, u12 = H([E([1, 2], "Directed", "f")]), H([E([1, 2])])
    assert isomorphic(f12, u12) is False
    assert isomorphic(f12, u12, labeled=False) is True
    assert isomorphic(H([E([1, 2, 3], "Cyclic", "f")]), H([E([1, 2, 3], "Directed", "f")])) is False
    ff = H([E([1, 2], "Directed", "f"), E([1, 2], "Directed", "f")])
    fg = H([E([1, 2], "Directed", "f"), E([1, 2], "Directed", "g")])
    assert isomorphic(ff, fg) is False


@ported("hypergraph-paclet", "tests/BasicExamples.wls", 235, 237, 241, 242)
def test_vertex_and_edge_labels_together():
    def bb(f, g, base=5):
        return H(
            [
                E([V(base, "A"), V(base + 1, "B")], "Directed", f),
                E([V(base + 1, "B"), V(base + 2, "C")], "Cyclic", g),
            ]
        )

    b1 = bb("f", "g", base=1)
    assert isomorphic(b1, bb("f", "g")) is True
    assert isomorphic(b1, bb("g", "f")) is True  # the renaming f <-> g absorbs the exchange
    assert isomorphic(b1, bb("f", "f")) is False  # a renaming cannot merge two labels
    assert isomorphic(b1, bb("f", "f"), labeled=False) is True


@ported("hypergraph-paclet", "tests/BasicExamples.wls", 248)
def test_canonical_form():
    assert canonical_hypergraph(a) == canonical_hypergraph(b)
    assert canonical_hypergraph(l1) == H(
        [V(1, LabelToken(1)), V(2, LabelToken(2)), V(3, LabelToken(3))], [[1, 2], [2, 3]]
    )
    assert canonical_hypergraph(l1, labeled=False) == H([[1, 2], [2, 3]])
    assert str(LabelToken(2)) == "ℓ2"


@ported("hypergraph-paclet", "tests/BasicExamples.wls", 258, 262)
def test_mixed_symmetries():
    x1 = H([E([1, 2, 3], "Unordered"), E([3, 4]), E([1, 4, 5], "Cyclic")])
    x2 = H([E([30, 20, 10], "Unordered"), E([30, 40]), E([40, 50, 10], "Cyclic")])
    x3 = H([E([1, 2, 3], "Unordered"), E([3, 4]), E([1, 5, 4], "Cyclic")])
    assert isomorphic(x1, x2) is True
    assert apply_vertex_map(x1, find_isomorphism(x1, x2)) == x2
    assert isomorphic(x1, x3) is False  # the cyclic edge's orientation


@ported("hypergraph-paclet", "tests/BasicExamples.wls", 271)
def test_identifiers_can_be_anything():
    w1 = H([E(["a", "b"]), E(["b", [1, 2]])])
    w2 = H([E(["x", "y"]), E(["y", "z"])])
    assert isomorphic(w1, w2) is True
    assert find_isomorphism(w1, w2) == {"a": "x", "b": "y", (1, 2): "z"}


@ported("hypergraph-paclet", "tests/BasicExamples.wls", 480, 481, 483, 485, 486)
def test_identifier_inside_another_identifier():
    # The paclet's \[FormalV][k], which a rewrite gives to new vertices beside vertices named 1, 2, ...
    def fv(k):
        return ("v", k)

    path_a = H([[1, fv(1)], [fv(1), 2], [2, fv(2)], [fv(2), 3]])
    path_b = H([[1, fv(3)], [fv(3), 2], [2, fv(4)], [fv(4), 3]])
    plain = H([[1, 2], [2, 3], [3, 4], [4, 5]])
    assert isomorphic(path_a, path_b) is True
    assert canonical_hypergraph(path_a) == canonical_hypergraph(path_b) == canonical_hypergraph(plain)
    assert isinstance(find_isomorphism(path_a, path_b), dict) and find_isomorphism(path_a, path_b)
    assert canonical_hypergraph(path_a, labeled=False) == canonical_hypergraph(path_b, labeled=False)


# -- section 10: randomized checks, with the paclet's generators ------------------------------


def random_hg(rng):
    n = rng.randint(2, 5)
    return H(
        [
            E(
                [rng.randint(1, n) for _ in range(rng.randint(1, 3))],
                rng.choice(["Directed", "Unordered", "Cyclic"]),
                rng.choice([None, "f", "g"]),
            )
            for _ in range(rng.randint(1, 4))
        ]
    )


def random_edge_labelled(rng):
    n = rng.randint(2, 5)
    return H(
        [
            E(
                [rng.randint(1, n) for _ in range(rng.randint(2, 3))],
                rng.choice(["Directed", "Unordered", "Cyclic"]),
                rng.choice(["f", "g", "h"]),
            )
            for _ in range(rng.randint(1, 4))
        ]
    )


def relabel_randomly(rng, hg):
    ids = hg.vertex_ids
    return apply_vertex_map(hg, dict(zip(ids, rng.sample(range(100, 100 + len(ids)), len(ids)), strict=True)))


def randomly_labelled(rng):
    g = random_hg(rng)
    vertices = [V(v.id, rng.choice([None, "A", "B"])) for v in g.vertices]
    return H(vertices, [E(e.ids, e.symmetry, e.label) for e in g.edges])


def rename_labels(hg, vr, er):
    return apply_vertex_map(hg, {i: i for i in hg.vertex_ids}, vr, er)


@ported("hypergraph-paclet", "tests/BasicExamples.wls", 303)
def test_random_relabellings():
    rng = random.Random(20260917)
    for _ in range(300):
        g = random_hg(rng)
        r = relabel_randomly(rng, g)
        for lab in (True, False):
            for m in METHODS:
                assert isomorphic(g, r, labeled=lab, method=m)
                assert canonical_hypergraph(g, labeled=lab, method=m) == canonical_hypergraph(r, labeled=lab, method=m)


@ported("hypergraph-paclet", "tests/BasicExamples.wls", 321)
def test_random_edge_labelled_relabellings():
    rng = random.Random(1)
    for _ in range(200):
        g = random_edge_labelled(rng)
        r = relabel_randomly(rng, g)
        for m in METHODS:
            assert isomorphic(g, r, method=m)
            assert canonical_hypergraph(g, method=m) == canonical_hypergraph(r, method=m)


@ported("hypergraph-paclet", "tests/BasicExamples.wls", 332)
def test_methods_agree():
    rng = random.Random(2)
    for _ in range(300):
        g, h = random_hg(rng), random_hg(rng)
        for lab in (True, False):
            assert isomorphic(g, h, labeled=lab) == isomorphic(g, h, labeled=lab, method="bruteforce")


@ported("hypergraph-paclet", "tests/BasicExamples.wls", 340)
def test_labelled_implies_skeleton_isomorphism():
    rng = random.Random(3)
    for _ in range(300):
        g, h = random_hg(rng), random_hg(rng)
        assert not isomorphic(g, h) or isomorphic(g, h, labeled=False)


@ported("hypergraph-paclet", "tests/BasicExamples.wls", 359)
def test_injective_renamings_keep_the_class():
    rng = random.Random(4)
    for _ in range(200):
        g = randomly_labelled(rng)
        h = rename_labels(g, {"A": "A1", "B": "B1"}, {"f": "f1", "g": "g1"})
        assert isomorphic(g, h) and isomorphic(g, h, method="bruteforce")
        assert canonical_hypergraph(g) == canonical_hypergraph(h)


@ported("hypergraph-paclet", "tests/BasicExamples.wls", 368)
def test_collapsing_labels_changes_the_class():
    rng = random.Random(5)
    for _ in range(300):
        g = randomly_labelled(rng)
        if len({v.label for v in g.vertices} - {None}) >= 2:
            assert not isomorphic(g, rename_labels(g, {"A": "Z", "B": "Z"}, {}))


@ported("hypergraph-paclet", "tests/BasicExamples.wls", 383)
def test_found_maps_are_isomorphisms():
    rng = random.Random(6)
    for _ in range(300):
        g = random_hg(rng)
        h = relabel_randomly(rng, g)
        iso = find_isomorphism(g, h)
        assert sorted(iso) == sorted(g.vertex_ids) and sorted(iso.values()) == sorted(h.vertex_ids)
        assert apply_vertex_map(g, iso) == h


# -- beyond the paclet --------------------------------------------------------------------------


def test_found_label_maps_carry_g_onto_h():
    rng = random.Random(7)
    for _ in range(200):
        g = randomly_labelled(rng)
        h = relabel_randomly(rng, rename_labels(g, {"A": "B", "B": "A"}, {"f": "x", "g": "y"}))
        found = find_isomorphism(g, h, labels=True)
        assert apply_vertex_map(g, found["Vertices"], found["VertexLabels"], found["EdgeLabels"]) == h


def test_symmetric_hypergraphs_are_pruned():
    # K_12 has 12! orderings; automorphism pruning keeps the search to a few dozen leaves
    k12 = H([E([i, j], "Unordered") for i in range(12) for j in range(i + 1, 12)])
    assert canonical_data(k12).leaves < 200
    units = H([[i] for i in range(12)])
    assert canonical_hypergraph(units) == H([[i] for i in range(1, 13)])


def test_errors():
    with pytest.raises(HypergraphError, match="unknown method"):
        canonical_hypergraph(a, method="Fast")
    with pytest.raises(HypergraphError, match="exceed"):
        canonical_hypergraph(H([[i] for i in range(9)]), method="bruteforce")
    assert canonical_hypergraph(H([])) == H([])
