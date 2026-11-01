"""The adjacency hypermatrix, ported from the paclet's HypermatrixExamples.wls section 10."""

import random
import string

import numpy as np
import pytest
from frontier.conformance import ported
from frontier.hypergraphs import Edge, Hypergraph, canonical_hypergraph, isomorphic
from hypermatrix import ArrayError, adjacency_hypermatrix, hypermatrix_edges
from hypermatrix.adjacency import EDGE_TO_ARRAY_SYMMETRY
from hypermatrix.arrays import canonical_index

H, E = Hypergraph, Edge


def first(hg, **kw):
    return adjacency_hypermatrix(hg, **kw).arrays[0]


def positions(entries, test):
    arr = np.array(entries, dtype=object)
    return [tuple(int(i) for i in idx) for idx in np.ndindex(arr.shape) if test(arr[idx])]


@ported("hypergraph-paclet", "tests/HypermatrixExamples.wls", 331, 332, 333, 334, 336)
def test_one_array_per_arity_and_symmetry():
    hm = adjacency_hypermatrix(H([E([1, 2, 3], "Unordered"), E([3, 4]), E([1, 2])]))
    assert len(hm) == 2
    assert hm["Orders"] == [2, 3]
    assert hm["Dimensions"] == [[4, 4], [4, 4, 4]]
    assert hm["Symmetries"] == ["Rigid", "Symmetric"]
    assert hm.arrays[0].entries == [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]]


@ported("hypergraph-paclet", "tests/HypermatrixExamples.wls", 341, 345, 349, 353)
def test_the_symmetry_types_correspond():
    def ones(sym):
        return positions(first(H([E([1, 2, 3], sym)])).entries, lambda v: v == 1)

    assert ones("Cyclic") == [(0, 1, 2), (1, 2, 0), (2, 0, 1)]  # the rotations only
    assert len(ones("Unordered")) == 6
    assert len(ones("Directed")) == 1
    three = H([E([1, 2], "Directed"), E([1, 2], "Unordered"), E([1, 2], "Cyclic")])
    assert adjacency_hypermatrix(three)["Symmetries"] == ["Symmetric", "Cyclic", "Rigid"]


@ported("hypergraph-paclet", "tests/HypermatrixExamples.wls", 359, 363, 367, 371, 375, 379, 381)
def test_multiplicity_repeated_vertices_and_degenerate_arities():
    assert first(H([E([1, 2]), E([1, 2]), E([2, 1])])).entries == [[0, 2], [1, 0]]
    loop = first(H([E([1, 1, 2], "Unordered")]))
    assert loop.symmetry == "Symmetric"
    assert len(positions(loop.entries, lambda v: v > 0)) == 3
    assert first(H([E([]), E([]), E([1])])).entries == 2
    assert adjacency_hypermatrix(H([E([]), E([1]), E([2]), E([1, 2])])).arrays[1].entries == [1, 1]
    assert len(adjacency_hypermatrix(H([]))) == 0
    with pytest.raises(ArrayError, match="not a hypergraph"):
        adjacency_hypermatrix([1, 2])


@ported("hypergraph-paclet", "tests/HypermatrixExamples.wls", 385, 389)
def test_labels_play_no_part_when_counting():
    hl = H([E([1, 2], "Directed", "x"), E([2, 3], "Directed", "y")])
    assert adjacency_hypermatrix(hl) == adjacency_hypermatrix(H([E([1, 2]), E([2, 3])]))
    assert adjacency_hypermatrix(hl) == adjacency_hypermatrix(hl.skeleton())


def _edges_of_hypergraph(hg):
    index = {v: k for k, v in enumerate(hg.vertex_ids)}
    return sorted(
        (canonical_index(tuple(index[i] for i in e.ids), EDGE_TO_ARRAY_SYMMETRY[e.symmetry]), e.symmetry)
        for e in hg.edges
    )


@ported("hypergraph-paclet", "tests/HypermatrixExamples.wls", 431)
def test_the_hypermatrix_determines_the_edge_multiset():
    rng = random.Random(20260924)
    for _ in range(300):
        hg = H(
            [
                E(
                    [rng.randint(1, rng.randint(1, 4)) for _ in range(rng.randint(0, 3))],
                    rng.choice(["Directed", "Unordered", "Cyclic"]),
                )
                for _ in range(rng.randint(0, 5))
            ]
        )
        assert hypermatrix_edges(adjacency_hypermatrix(hg)) == _edges_of_hypergraph(hg)


h_num = H([E([1, 2]), E([2, 3])])
h_str = H([E(["z", "a"]), E(["a", "m"])])
g_num = H([E([1, 2, 3], "Unordered"), E([1, 1, 2], "Unordered")])
g_str = H([E(["z", "a", "m"], "Unordered"), E(["z", "z", "a"], "Unordered")])


@ported("hypergraph-paclet", "tests/HypermatrixExamples.wls", 438, 442, 448, 450, 454, 463, 470, 474)
def test_which_order_the_indices_run_in():
    assert adjacency_hypermatrix(H([E([3, 1]), E([2, 3])])) == adjacency_hypermatrix(H([E([2, 3]), E([3, 1])]))
    assert H([E([3, 1]), E([2, 3])]).vertex_ids == [1, 2, 3]
    assert h_str.vertex_ids == ["a", "m", "z"]
    assert isomorphic(h_num, h_str) and adjacency_hypermatrix(h_num) != adjacency_hypermatrix(h_str)
    q = [1, 2, 0]  # InversePermutation[{3, 1, 2}], counted from 0
    m_num, m_str = (np.array(first(h).entries) for h in (h_num, h_str))
    assert (m_num[np.ix_(q, q)] == m_str).all()
    a_num, a_str = (np.array(first(h).entries) for h in (g_num, g_str))
    assert (a_num[np.ix_(q, q, q)] == a_str).all()  # the permutation acts in every slot at once
    canon = [adjacency_hypermatrix(canonical_hypergraph(h)) for h in (h_num, h_str, g_num, g_str)]
    assert canon[0] == canon[1] and canon[2] == canon[3]


@ported("hypergraph-paclet", "tests/HypermatrixExamples.wls", 491)
def test_canonicalizing_makes_it_an_invariant():
    rng = random.Random(20260925)
    for _ in range(200):
        n = rng.randint(1, 4)
        hg = H(
            [
                E(
                    [rng.randint(1, n) for _ in range(rng.randint(1, 3))],
                    rng.choice(["Directed", "Unordered", "Cyclic"]),
                )
                for _ in range(rng.randint(1, 4))
            ]
        )
        names = dict(zip(hg.vertex_ids, rng.sample(string.ascii_lowercase[:20], len(hg.vertices)), strict=True))
        hg2 = H([E([names[i] for i in e.ids], e.symmetry) for e in hg.edges])
        assert adjacency_hypermatrix(canonical_hypergraph(hg)) == adjacency_hypermatrix(canonical_hypergraph(hg2))


@ported("hypergraph-paclet", "tests/HypermatrixExamples.wls", 495, 499, 505, 511, 514)
def test_multiplicities_are_integer_multiples():
    assert first(H([E([1, 2]), E([1, 2]), E([2, 1])])).entries == [[0, 2], [1, 0]]
    triple = first(H([E([1, 2, 3], "Unordered")] * 3)).entries
    assert set(np.array(triple).flatten()) - {0} == {3} and len(positions(triple, lambda v: v == 3)) == 6
    double = first(H([E([1, 2, 3], "Cyclic")] * 2)).entries
    assert set(np.array(double).flatten()) - {0} == {2} and len(positions(double, lambda v: v == 2)) == 3
    assert first(H([E([])] * 5)).entries == 5
    assert first(H([E([1, 2], "Unordered"), E([2, 1], "Unordered")])).entries == [[0, 2], [2, 0]]


h_lab = H([E([1, 2], "Directed", "a"), E([2, 3], "Directed", "b"), E([1, 2, 3], "Unordered", "c")])


@ported("hypergraph-paclet", "tests/HypermatrixExamples.wls", 523, 525, 527, 529, 532, 537, 542, 546)
def test_labels_as_entries():
    counted, labelled = adjacency_hypermatrix(h_lab), adjacency_hypermatrix(h_lab, labeled=True)
    assert counted == adjacency_hypermatrix(h_lab, labeled=False)
    assert counted["Domains"] == ["Integer", "Integer"]
    assert labelled["Domains"] == ["Expression", "Expression"]
    assert labelled.arrays[0].entries == [[0, "a", 0], [0, 0, "b"], [0, 0, 0]]
    last = labelled.arrays[-1]
    assert (len(positions(last.entries, lambda v: v == "c")), last.symmetry) == (6, "Symmetric")
    assert (counted["Orders"], counted["Symmetries"]) == (labelled["Orders"], labelled["Symmetries"])
    assert first(H([E([], "Directed", "empty")]), labeled=True).entries == "empty"
    mixed = H([E([1, 2], "Directed", "a"), E([2, 1], "Directed")])
    assert first(mixed, labeled=True).entries == [[0, "a"], [None, 0]]  # an unlabelled edge is None, absence 0


@ported("hypergraph-paclet", "tests/HypermatrixExamples.wls", 552, 557, 561, 566)
def test_the_simplicity_labels_need():
    twice = H([E([1, 2], "Directed", "a"), E([1, 2], "Directed", "b")])
    with pytest.raises(ArrayError, match="repeated edge"):
        adjacency_hypermatrix(twice, labeled=True)
    assert first(twice).entries == [[0, 2], [0, 0]]
    split = H([E([1, 2], "Directed", "d"), E([1, 2], "Unordered", "u")])
    assert adjacency_hypermatrix(split, labeled=True)["Symmetries"] == ["Symmetric", "Rigid"]
    loop = first(H([E([1, 1, 2], "Unordered", "r")]), labeled=True)
    assert set(np.array(loop.entries, dtype=object).flatten()) == {0, "r"}
