"""Test cases ported from the reference paclet's tests/HypermatrixExamples.wls, sections 1-5."""

from fractions import Fraction

import numpy as np
import pytest
import sympy
from frontier.conformance import ported
from hypermatrix import ArrayError, ArrayObject, FiniteField, generate_symbolic_array


@ported("hypergraph-paclet", "tests/HypermatrixExamples.wls", 22, 23, 24, 25, 26, 27, 29, 31, 33, 34, 35, 38, 40)
def test_value_domain_is_read_off_the_entries():
    x, y = sympy.symbols("x y")
    f2, f3 = FiniteField(2), FiniteField(3)
    assert ArrayObject([True, False]).domain == "Boolean"
    assert ArrayObject([1, 2, 3]).domain == "Integer"
    assert ArrayObject([Fraction(1, 2), 3]).domain == "Rational"
    assert ArrayObject([1.5, 2]).domain == "Real"
    assert ArrayObject([1 + sympy.I, 2]).domain == "Complex"
    assert ArrayObject([f2(0), f2(1)]).domain is f2
    assert ArrayObject([1, 2]).domain == "Integer"
    assert ArrayObject(["a", "b"]).domain == "Expression"
    assert ArrayObject([x, y]).domain == "Expression"
    assert ArrayObject([1, "a"]).domain == "Expression"
    assert isinstance(ArrayObject([1.5]), ArrayObject) and isinstance(ArrayObject(["anything", x + 1]), ArrayObject)
    assert ArrayObject([f3(1)]).domain is f3
    assert ArrayObject([f2(1), f3(1)]).domain == "Expression"


@ported("hypergraph-paclet", "tests/HypermatrixExamples.wls", 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62)
def test_order_dimensions_regularity():
    v = ArrayObject([1, 2, 3])
    m = ArrayObject([[1, 2, 3], [4, 5, 6]])
    t = ArrayObject(np.fromfunction(lambda i, j, k: i + j + k + 3, (2, 2, 2), dtype=int))
    assert (v.order, v.dims) == (1, [3])
    assert (m.order, m.dims) == (2, [2, 3])
    assert (t.order, t.dims) == (3, [2, 2, 2])
    assert m.regular is False and t.regular is True
    assert m["Dimensions"] == [2, 3]
    assert v.entries == [1, 2, 3] and v["Entries"] == v.entries
    s = ArrayObject(5)
    assert (s.order, s.dims, s.regular, s.entries) == (0, [], True, 5)
    assert generate_symbolic_array("q", []).entries == sympy.Symbol("q")
    with pytest.raises(ArrayError):
        ArrayObject([[1, 2], [3]])


@ported("hypergraph-paclet", "tests/HypermatrixExamples.wls", 69, 71, 73, 74, 75, 77, 79, 84, 86, 87)
def test_declared_symmetry_is_verified():
    sym2, asym2 = [[1, 2], [2, 1]], [[1, 2], [3, 4]]
    assert ArrayObject(sym2, "Symmetric").symmetry == "Symmetric"
    with pytest.raises(ArrayError):
        ArrayObject(asym2, "Symmetric")
    assert ArrayObject(asym2).symmetry == "Rigid"
    assert ArrayObject(sym2).symmetry == "Rigid"
    with pytest.raises(ArrayError):
        ArrayObject([[1, 2, 3], [4, 5, 6]], "Symmetric")
    with pytest.raises(ArrayError):
        ArrayObject(sym2, "Wobbly")
    assert ArrayObject(sym2, "symmetric").symmetry == "Symmetric"
    orbit = {(1, 2, 3), (2, 3, 1), (3, 1, 2)}
    cyc3 = [[[int((i, j, k) in orbit) for k in (1, 2, 3)] for j in (1, 2, 3)] for i in (1, 2, 3)]
    assert ArrayObject(cyc3, "Cyclic").symmetry == "Cyclic"
    with pytest.raises(ArrayError):
        ArrayObject(cyc3, "Symmetric")
    assert ArrayObject(sym2, "Cyclic").symmetry == "Cyclic"


@ported(
    "hypergraph-paclet", "tests/HypermatrixExamples.wls", 93, 96, 99, 101, 103, 105, 108, 112, 115, 118, 120, 121, 122
)
def test_generate_symbolic_array():
    a = sympy.IndexedBase("a")
    g = generate_symbolic_array("a", [2, 2])
    assert g.entries == [[a[1, 1], a[1, 2]], [a[2, 1], a[2, 2]]]
    g23 = generate_symbolic_array("a", [2, 3])
    assert (g23.order, g23.dims, g23.domain) == (2, [2, 3], "Expression")
    assert isinstance(g23, ArrayObject)
    e = np.array(generate_symbolic_array("a", [3, 3], "Symmetric").entries, dtype=object)
    assert bool(np.all(e == e.T))
    assert generate_symbolic_array("a", [2, 2], "Symmetric").entries == [[a[1, 1], a[1, 2]], [a[1, 2], a[2, 2]]]

    def distinct(sym):
        return len(set(np.array(generate_symbolic_array("a", [3, 3, 3], sym).entries, dtype=object).ravel()))

    assert [distinct(s) for s in ("Rigid", "Cyclic", "Symmetric")] == [27, 11, 10]
    assert (
        generate_symbolic_array("a", [3, 3, 3], "Cyclic").entries
        != generate_symbolic_array("a", [3, 3, 3], "Symmetric").entries
    )
    assert (
        generate_symbolic_array("a", [3, 3], "Cyclic").entries
        == generate_symbolic_array("a", [3, 3], "Symmetric").entries
    )
    A = sympy.IndexedBase("A")
    assert generate_symbolic_array("A", [2]).entries == [A[1], A[2]]
    with pytest.raises(ArrayError):
        generate_symbolic_array([1, 2], [2])
    with pytest.raises(ArrayError):
        generate_symbolic_array("a", [2, 0])
    with pytest.raises(ArrayError):
        generate_symbolic_array("a", [2, 3], "Symmetric")


@ported("hypergraph-paclet", "tests/HypermatrixExamples.wls", 128, 131, 133, 135)
def test_predicates_and_properties():
    v = ArrayObject([1, 2, 3])
    assert [isinstance(x, ArrayObject) for x in (v, generate_symbolic_array("a", [2]), [1, 2], 5)] == [
        True,
        True,
        False,
        False,
    ]
    assert v["Properties"] == ["Order", "Dimensions", "Symmetry", "Domain", "Entries", "RegularQ"]
    assert [v["Order"], v["Dimensions"], v["Domain"], v["RegularQ"]] == [1, [3], "Integer", True]
    with pytest.raises(KeyError):
        v["Nonsense"]
