"""Test cases ported from the reference paclet's tests/ArrayAlgebraExamples.wls.

Each test cites the paclet lines it reproduces (`frontier wl coverage`). The paclet's
random arrays are replaced by our own seeded ones; the properties checked are the same.
"""

import random
from fractions import Fraction

import numpy as np
import pytest
import sympy
from frontier.conformance import ported
from hypermatrix import (
    ArrayError,
    ArrayObject,
    FiniteField,
    array_add,
    array_multiply,
    array_times,
    generate_symbolic_array,
    identity_array,
    partial_identity_array,
    zero_array,
)


def rand(rng: random.Random, dims, lo=-3, hi=3):
    if not dims:
        return rng.randint(lo, hi)
    return [rand(rng, dims[1:], lo, hi) for _ in range(dims[0])]


@pytest.fixture
def mats():
    rng = random.Random(11)
    return rand(rng, [3, 3]), rand(rng, [3, 3]), rand(rng, [3]), rand(rng, [3])


def E(x):
    return x.entries if isinstance(x, ArrayObject) else x


# -- A. ArrayMultiply against known contractions ---------------------------


@ported("hypergraph-paclet", "tests/ArrayAlgebraExamples.wls", 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34)
def test_known_contractions(mats):
    a, b, pv, qv = mats
    A, B, P, Q = (np.array(x) for x in mats)
    assert E(array_multiply([a, b], "ij,jk->ik")) == (A @ B).tolist()
    assert array_multiply([a, b], ([[1, 2], [2, 3]], [1, 3])) == array_multiply([a, b], "ij,jk->ik")
    assert E(array_multiply([a, b], "ij,ij->ij")) == (A * B).tolist()
    assert E(array_multiply([a], "ij->ji")) == A.T.tolist()
    assert E(array_multiply([a], "ii->")) == int(np.trace(A))
    assert array_multiply([a], "ii->").order == 0
    assert E(array_multiply([a], "ii->i")) == np.diag(A).tolist()
    assert E(array_multiply([pv, qv], "i,i->")) == int(P @ Q)
    assert E(array_multiply([pv, qv], "i,j->ij")) == np.outer(P, Q).tolist()
    assert E(array_multiply([pv, qv, pv], "i,i,i")) == int(np.sum(P * Q * P))
    assert E(array_multiply([a, pv], "ij,j->i")) == (A @ P).tolist()
    assert array_multiply([ArrayObject(a), ArrayObject(b)], "ij,jk->ik") == array_multiply([a, b], "ij,jk->ik")


@ported("hypergraph-paclet", "tests/ArrayAlgebraExamples.wls", 43, 45, 47)
def test_ternary_products_from_the_paper():
    rng = random.Random(12)
    x, y, z = (np.array(rand(rng, [3, 3, 3])) for _ in range(3))
    R = range(3)
    bm = [[[sum(x[i, j, q] * y[i, q, k] * z[q, j, k] for q in R) for k in R] for j in R] for i in R]
    assert E(array_multiply([x, y, z], "ijp,ipk,pjk->ijk")) == bm
    triforce = [
        [[sum(x[i, r, q] * y[r, j, s] * z[q, s, k] for r in R for q in R for s in R) for k in R] for j in R] for i in R
    ]
    assert E(array_multiply([x, y, z], "ipq,pjr,qrk->ijk")) == triforce
    fish = [
        [[sum(x[i, j, q] * y[r, s, q] * z[r, s, k] for q in R for r in R for s in R) for k in R] for j in R] for i in R
    ]
    assert E(array_multiply([x, y, z], "ijp,qrp,qrk->ijk")) == fish


@ported("hypergraph-paclet", "tests/ArrayAlgebraExamples.wls", 51, 52, 53, 55)
def test_errors(mats):
    a, b, *_ = mats
    with pytest.raises(ArrayError, match="names 2 arrays"):
        array_multiply([a], "ij,jk->ik")
    with pytest.raises(ArrayError) as e:
        array_multiply([a, b], "i,jk->ik")
    assert e.value.code == "ArrayMultiply::rank"
    with pytest.raises(ArrayError) as e:
        array_multiply([[[1, 2], [1, 2]], [[1, 2, 3]] * 3], "ij,jk->ik")
    assert e.value.code == "ArrayMultiply::dims"
    with pytest.raises(ArrayError) as e:
        array_multiply([a], "ij->ijk")
    assert e.value.code == "ArrayMultiply::out"


# -- B. Special arrays ----------------------------------------------------


@ported("hypergraph-paclet", "tests/ArrayAlgebraExamples.wls", 64, 65, 67, 68, 69, 70, 71, 74, 75, 76)
def test_special_arrays():
    assert identity_array(2, 3).entries == np.eye(3, dtype=int).tolist()
    assert partial_identity_array(3, 2, [1, 2, 3]) == identity_array(3, 2)
    assert zero_array(3, 2).symmetry == "Symmetric"
    assert identity_array(3, 2).symmetry == "Symmetric"
    assert partial_identity_array(3, 2, [1, 2]).symmetry == "Rigid"
    assert zero_array(2, 2, "Real").domain == "Real"
    f = FiniteField(2)
    assert identity_array(2, 2, f).entries == [[f(1), f(0)], [f(0), f(1)]]
    assert zero_array([2, 3]).dims == [2, 3]
    assert zero_array([2, 3]).symmetry == "Rigid"
    with pytest.raises(ArrayError):
        partial_identity_array(3, 2, [1, 5])


@ported("hypergraph-paclet", "tests/ArrayAlgebraExamples.wls", 80)
def test_identity_is_an_identity(mats):
    a = mats[0]
    assert array_multiply([identity_array(2, 3).entries, a], "ij,jk->ik") == ArrayObject(a)


@ported("hypergraph-paclet", "tests/ArrayAlgebraExamples.wls", 85, 86, 87, 88, 90, 92)
def test_arrays_of_order_zero(mats):
    a = mats[0]
    A = np.array(a)
    assert ArrayObject(7).order == 0
    assert zero_array([]).entries == 0
    assert zero_array(0, 3) == zero_array([])
    assert E(array_multiply([ArrayObject(3), a], ",ij->ij")) == (3 * A).tolist()
    assert E(array_multiply([a, a], "ij,ji->")) == int(np.trace(A @ A))
    assert generate_symbolic_array("s0", []).entries == sympy.Symbol("s0")


# -- B2. Element-wise operations ------------------------------------------

U2, V2 = [[1, 2], [3, 4]], [[5, 6], [7, 8]]
U, V = np.array(U2), np.array(V2)


@ported("hypergraph-paclet", "tests/ArrayAlgebraExamples.wls", 98, 99, 100, 101, 102, 103, 104, 106, 107, 108, 109, 110)
def test_elementwise_numeric():
    assert E(array_add(U2, V2)) == (U + V).tolist()
    assert E(array_times(U2, V2)) == (U * V).tolist()
    assert E(array_add(U2, V2, U2)) == (2 * U + V).tolist()
    assert E(array_add(U2)) == U2
    assert E(array_times(2, U2)) == (2 * U).tolist()
    assert E(array_add(1, U2)) == (1 + U).tolist()
    assert array_times(2, U2) == array_times(ArrayObject(2), U2)
    assert E(ArrayObject(U2) + ArrayObject(V2)) == (U + V).tolist()
    assert E(ArrayObject(U2) * ArrayObject(V2)) == (U * V).tolist()
    assert E(ArrayObject(U2) - ArrayObject(V2)) == (U - V).tolist()
    half = ArrayObject(U2) / 2
    assert E(half) == [[Fraction(1, 2), 1], [Fraction(3, 2), 2]] and half.domain == "Rational"
    assert isinstance(array_add(U2, V2), ArrayObject)


@ported("hypergraph-paclet", "tests/ArrayAlgebraExamples.wls", 111, 113)
def test_elementwise_symbolic():
    sc = sympy.IndexedBase("sc")
    assert E(2 * generate_symbolic_array("sc", [2])) == [2 * sc[1], 2 * sc[2]]
    assert E(ArrayObject([1, 2]) + generate_symbolic_array("sc", [2])) == [1 + sc[1], 2 + sc[2]]


@ported("hypergraph-paclet", "tests/ArrayAlgebraExamples.wls", 115, 116, 120, 124)
def test_elementwise_rules():
    with pytest.raises(ArrayError) as e:
        array_add([1, 2], [1, 2, 3])
    assert e.value.code == "ArrayAdd::dims"
    s1 = ArrayObject([[1, 2], [2, 3]], "Symmetric")
    assert array_add(s1, ArrayObject([[0, 1], [1, 0]], "Symmetric")).symmetry == "Symmetric"
    assert array_add(s1, ArrayObject(U2)).symmetry == "Rigid"
    with pytest.raises(ArrayError):
        array_add()


# -- D. Randomized checks -------------------------------------------------


@ported("hypergraph-paclet", "tests/ArrayAlgebraExamples.wls", 231)
def test_random_matrix_products():
    rng = random.Random(20260920)
    for _ in range(200):
        u, w = rand(rng, [3, 4], -4, 4), rand(rng, [4, 2], -4, 4)
        assert E(array_multiply([u, w], "ij,jk->ik")) == (np.array(u) @ np.array(w)).tolist()


@ported("hypergraph-paclet", "tests/ArrayAlgebraExamples.wls", 240)
def test_relabelling_does_not_matter():
    rng = random.Random(1)
    for _ in range(200):
        u, w, t = (rand(rng, [2, 2, 2]) for _ in range(3))
        assert array_multiply([u, w, t], "ijp,ipk,pjk->ijk") == array_multiply([u, w, t], "xyz,xzw,zyw->xyw")


@ported("hypergraph-paclet", "tests/ArrayAlgebraExamples.wls", 247)
def test_string_and_index_list_forms_agree():
    rng = random.Random(2)
    for _ in range(200):
        u, w = rand(rng, [3, 3]), rand(rng, [3, 3])
        assert array_multiply([u, w], "ij,jk->ik") == array_multiply([u, w], ([[1, 2], [2, 3]], [1, 3]))


@ported("hypergraph-paclet", "tests/ArrayAlgebraExamples.wls", 255)
def test_identity_for_random_dimensions():
    rng = random.Random(3)
    for _ in range(200):
        d = rng.randint(1, 4)
        u = rand(rng, [d, d], -4, 4)
        assert E(array_multiply([identity_array(2, d).entries, u], "ij,jk->ik")) == u


def test_large_integers_stay_exact():
    # not in the paclet (WL integers are unbounded); guards our int64 fast path
    big = [[2**40] * 2] * 2
    assert E(array_multiply([big, big], "ij,jk->ik"))[0][0] == 2 * 2**80
    assert E(array_times(big, big))[0][0] == 2**80
