"""Hypermatrices (canonically ordered lists of arrays), ported from the paclet's HypermatrixExamples.wls
sections 6-7, and the drawing of a hypermatrix as one picture per array."""

import itertools
import random

import numpy as np
import pytest
import sympy
from frontier.conformance import ported
from hypermatrix import ArrayError, ArrayObject, Hypermatrix, generate_symbolic_array, hypermatrix_svgs
from hypermatrix.hypermatrix import SYMMETRY_ORDER

ARRAYS = [
    ArrayObject(np.ones((2, 2, 2), dtype=int), "Symmetric"),
    ArrayObject([[1, 2], [3, 4]]),
    ArrayObject([1, 2]),
    ArrayObject([1, 2, 3]),
    ArrayObject([[1, 2], [2, 1]], "Symmetric"),
    ArrayObject([[1, 2], [2, 1]], "Cyclic"),
]


@ported("hypergraph-paclet", "tests/HypermatrixExamples.wls", 152, 153, 154, 156, 157, 158)
def test_canonical_order():
    hm = Hypermatrix(ARRAYS)
    assert hm["Orders"] == [1, 1, 2, 2, 2, 3]
    assert hm["Dimensions"][:2] == [[2], [3]]
    assert hm["Symmetries"][2:5] == ["Symmetric", "Cyclic", "Rigid"]
    assert isinstance(hm, Hypermatrix)
    assert len(hm) == 6 == hm["Length"]
    assert hm["Properties"] == ["Arrays", "Length", "Orders", "Dimensions", "Symmetries", "Domains", "Entries"]


@ported("hypergraph-paclet", "tests/HypermatrixExamples.wls", 160, 163, 166, 168, 171, 173)
def test_construction():
    assert Hypermatrix(ArrayObject([1, 2]), ArrayObject([1, 2, 3])) == Hypermatrix(
        [ArrayObject([1, 2]), ArrayObject([1, 2, 3])]
    )
    assert Hypermatrix([[1, 2], [[1, 2], [3, 4]]]).arrays[0] == ArrayObject([1, 2], "Rigid")
    assert Hypermatrix([generate_symbolic_array("a", [2, 2]), ArrayObject([1, 2])])["Orders"] == [1, 2]
    nope = Hypermatrix(["nope"])
    assert isinstance(nope, Hypermatrix) and nope.normal() == ["nope"]  # a bare value is a 0-array
    with pytest.raises(ArrayError, match="not an array"):
        Hypermatrix([[[1, 2], [3]]])
    assert Hypermatrix([ArrayObject([[1, 2], [3, 4]]), ArrayObject([1, 2])]).normal() == [[1, 2], [[1, 2], [3, 4]]]


def _random_data(rng):
    n = rng.randint(1, 3)
    if rng.choice([True, False]):
        dims = [rng.randint(1, 3)] * n
    else:
        dims = [rng.randint(1, 3) for _ in range(n)]
    return np.array([rng.randint(0, 5) for _ in range(int(np.prod(dims)))], dtype=object).reshape(dims).tolist()


@ported("hypergraph-paclet", "tests/HypermatrixExamples.wls", 198)
def test_order_and_dimensions_agree_with_the_entries():
    rng = random.Random(20260919)
    for _ in range(300):
        data = _random_data(rng)
        arr, shape = ArrayObject(data), np.array(data, dtype=object).shape
        assert arr.order == len(shape) and arr.dims == list(shape)
        assert arr.regular == (len(set(shape)) <= 1)
        assert arr.entries == data


@ported("hypergraph-paclet", "tests/HypermatrixExamples.wls", 208, 217)
def test_symmetrized_arrays_are_accepted():
    rng = random.Random(1)
    for _ in range(200):
        n, d = rng.randint(2, 3), rng.randint(2, 3)
        data = np.array([rng.randint(0, 4) for _ in range(d**n)]).reshape((d,) * n)
        sym = sum(np.transpose(data, p) for p in itertools.permutations(range(n)))
        assert ArrayObject(sym, "Symmetric").symmetry == "Symmetric"
        assert ArrayObject(sym, "Cyclic").symmetry == "Cyclic"
        cyc = sum(np.transpose(data, [(i + r) % n for i in range(n)]) for r in range(n))
        assert ArrayObject(cyc, "Cyclic").symmetry == "Cyclic"


@ported("hypergraph-paclet", "tests/HypermatrixExamples.wls", 224, 233)
def test_canonical_order_is_a_function_of_the_set():
    rng = random.Random(2)
    for _ in range(300):
        arrays = [ArrayObject(_random_data(rng)) for _ in range(rng.randint(2, 6))]
        hm = Hypermatrix(arrays)
        assert hm == Hypermatrix(rng.sample(arrays, len(arrays)))
        keys = [(a.order, a.dims, SYMMETRY_ORDER.index(a.symmetry)) for a in hm.arrays]
        assert keys == sorted(keys)


@ported("hypergraph-paclet", "tests/HypermatrixExamples.wls", 316, 322)
def test_one_picture_per_array():
    pictures = hypermatrix_svgs(Hypermatrix([[1, 2], [[1, 2], [3, 4]]]))
    assert len(pictures) == 2 and all(p.startswith("<svg") for p in pictures)
    with pytest.raises(ArrayError, match="not a hypermatrix"):
        hypermatrix_svgs([1, 2])


def test_symbolic_and_numeric_entries_sort_without_error():
    x = sympy.Symbol("x")
    assert len(Hypermatrix([ArrayObject([x, 1]), ArrayObject([1, 2]), ArrayObject(["s", 2])])) == 3
