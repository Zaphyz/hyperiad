"""Enumeration of closed plex products and the equivalences acting on them."""

import itertools

import numpy as np
import pytest
from frontier.hypergraphs import canonical_hypergraph
from hypermatrix import ArrayObject, array_multiply, plex_free, plex_names, plex_product
from hypermatrix.plex_enumeration import (
    EQUIVALENCES,
    PlexShape,
    all_plex_shapes,
    enumerate_plex_products,
    operation_hypergraph,
    operation_table,
    set_partitions,
)

MATMUL = PlexShape(2, 2, (0, 1, 1, 2), (0, 2))  # a_ij b_jk -> c_ik
RNG = np.random.default_rng(3)


def test_set_partitions_are_counted_by_bell_numbers():
    assert [sum(1 for _ in set_partitions(m)) for m in range(7)] == [1, 1, 2, 5, 15, 52, 203]


def test_census_of_binary_products_on_matrices():
    connected = {e: len(enumerate_plex_products(2, 2, equivalence=e)) for e in EQUIVALENCES}
    assert connected == {"strict": 28, "isomorphism": 16, "parastrophy": 10, "hypergraph": 9}
    every = {
        e: len(enumerate_plex_products(2, 2, equivalence=e, connected=False, diagonals=True)) for e in EQUIVALENCES
    }
    assert every == {"strict": 62, "isomorphism": 37, "parastrophy": 22, "hypergraph": 19}
    assert MATMUL in enumerate_plex_products(2, 2)


def test_the_diagram_computes_the_shape():
    diagram = MATMUL.diagram()
    assert plex_names(diagram) == ["x1", "x2"] and plex_free(diagram) == [1, 2]
    a, b = RNG.integers(-2, 3, (3, 3)), RNG.integers(-2, 3, (3, 3))
    assert plex_product(diagram, [a, b]) == ArrayObject(a @ b)
    for shape in all_plex_shapes(2, 3):
        arrays = [RNG.integers(-2, 3, (2, 2)) for _ in range(3)]
        assert plex_product(shape.diagram(), arrays) == array_multiply([ArrayObject(x) for x in arrays], shape.spec())


def test_a_common_transposition_is_an_algebra_isomorphism():
    for shape in all_plex_shapes(3, 2):
        for tau in itertools.permutations(range(3)):
            moved = shape.act((0, 1), (0, 1, 2), tau)
            arrays = [RNG.integers(-2, 3, (2, 2, 2)) for _ in range(2)]
            lhs = np.transpose(array_multiply([ArrayObject(x) for x in arrays], shape.spec()).data, tau)
            rhs = array_multiply([ArrayObject(np.transpose(x, tau)) for x in arrays], moved.spec()).data
            assert (np.asarray(lhs, dtype=float) == np.asarray(rhs, dtype=float)).all()


def test_hypergraph_equivalence_is_isomorphism_of_the_diagrams():
    shapes = list(all_plex_shapes(2, 2, connected=False, diagonals=True))
    pairs = {(s.canonical("hypergraph"), canonical_hypergraph(s.diagram())) for s in shapes}
    assert len(pairs) == len({p[0] for p in pairs}) == len({p[1] for p in pairs})


def test_distinct_shapes_are_distinct_operations_from_size_two():
    shapes = list(all_plex_shapes(2, 2, connected=False, diagonals=True))
    assert len({frozenset(operation_table(s, 2).items()) for s in shapes}) == len(shapes)
    assert len({frozenset(operation_table(s, 1).items()) for s in shapes}) == 1  # size 1: all collapse


def test_operations_are_partial_maps_on_indices():
    table = operation_table(MATMUL, 2)
    assert table[((0, 1), (1, 1))] == (0, 1) and ((0, 1), (0, 1)) not in table
    assert len(table) == 2**3  # one product per value of the three indices
    hg = operation_hypergraph(MATMUL, 2)
    assert (len(hg.vertices), len(hg.edges)) == (4, 8)


def test_unknown_equivalence():
    with pytest.raises(ValueError, match="unknown equivalence"):
        MATMUL.canonical("homotopy")
