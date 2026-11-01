from fractions import Fraction

import pytest
import sympy
from hypermatrix import ArrayObject

from hyperiad import Session


def value(code: str, session: Session | None = None):
    s = session or Session()
    r = s.run(code)
    assert r["status"] == "ok", r["outputs"]
    return s.globals.vars.get("ans")


def outputs(code: str, **kw):
    return Session(**kw).run(code)


def test_exact_arithmetic():
    assert value("1/2 + 1/3") == Fraction(5, 6)
    assert value("6/3") == 2 and isinstance(value("6/3"), int)
    assert value("2^-2") == Fraction(1, 4)
    assert value("2^100") == 2**100
    assert value("1.5 * 2") == 3.0


def test_symbolic():
    x, y = sympy.symbols("x y")
    assert value("Expand((x + y)^2)") == x**2 + 2 * x * y + y**2
    assert value("x == x") is True
    r = outputs("x^2 + 1")
    assert r["outputs"][0]["type"] == "latex"


def test_variables_functions_and_closures():
    s = Session()
    assert value("a = 3", s) == 3
    assert value("sq(x) = x^2\nsq(a)", s) == 9
    assert value("add(k) = x => x + k\nadd(10)(5)", s) == 15
    assert value("fact(n) = If(n <= 1, 1, n * fact(n - 1))\nfact(10)", s) == 3628800
    assert value("Map(x => x * a, [1, 2])", s) == [3, 6]


def test_lists_parts_tables():
    assert value("[1, 2] + [3, 4]") == [4, 6]
    assert value("[10, 20, 30][-1]") == 30
    assert value("[[1, 2], [3, 4]][2, 1]") == 3
    assert value('{"a": 1}["a"]') == 1
    assert value("Table(i * j, [i, 2], [j, [5, 7]])") == [[5, 7], [10, 14]]
    assert value("Total(Range(100))") == 5050
    assert value("Select(Range(10), n => n > 7)") == [8, 9, 10]


def test_arrays_through_the_language():
    a = value('m = [[1, 2], [3, 4]]\nArrayMultiply([m, m], "ij,jk->ik")')
    assert a == ArrayObject([[7, 10], [15, 22]])
    assert value("ArrayMultiply([[[1, 2], [3, 4]], [[1, 2], [3, 4]]], [[1, 2], [2, 3]] -> [1, 3])") == a
    assert value('IdentityArray(3, 2)["Symmetry"]') == "Symmetric"
    assert value("ArrayObject([[1, 2], [3, 4]]) / 2")["Domain"] == "Rational"
    assert value("ArrayObject([[1, 2], [3, 4]])[2]") == ArrayObject([3, 4])
    assert value("ArrayObject([[1, 2], [3, 4]])[2][1]") == 3
    assert value('GenerateSymbolicArray("s", [2, 2])[2, 1]') == sympy.IndexedBase("s")[2, 1]
    assert value("ArrayObject([[1, 2], [3, 4]])[-1, -1]") == 4
    g = value('GenerateSymbolicArray(a, [2], "Rigid")')
    assert g.entries == [sympy.IndexedBase("a")[1], sympy.IndexedBase("a")[2]]
    assert value("FiniteField(2)(1) + FiniteField(2)(1)") == value("FiniteField(2)(0)")


def test_arrays_display_as_pictures():
    r = outputs("IdentityArray(2, 3)")
    out = r["outputs"][0]
    assert out["type"] == "svg" and out["data"].startswith("<svg")
    assert out["caption"] == "3×3 Integer array of order 2, symmetric"


def test_budget_refuses_before_computing():
    r = outputs('ArrayMultiply([OneArray(3, 60), OneArray(3, 60)], "ijk,jkl->il")', budget=1e6)
    assert r["status"] == "too_expensive"
    out = r["outputs"][-1]
    assert out["kind"] == "too_expensive" and out["budget"] == 1e6 and out["estimate"] > 1e6
    assert r["estimate"] == 2 * 60**4 and r["builtin"] == "ArrayMultiply"
    assert outputs('ArrayMultiply([OneArray(2, 3), OneArray(2, 3)], "ij,jk->ik")', budget=1e6)["status"] == "ok"
    # no budget: no refusal
    assert outputs('ArrayMultiply([OneArray(2, 30), OneArray(2, 30)], "ij,jk->ik")')["status"] == "ok"


@pytest.mark.parametrize(
    "code, fragment",
    [
        ("1/0", "division by zero"),
        ('ArrayMultiply([[1, 2]], "ij->ji")', "has 1 indices"),
        ("Range(1, 2, 0)", "step cannot be 0"),
        ("ArrayMultiply = 3", "built in"),
        ("List = 3", "built in"),
        ("f(x) = x\nf(1, 2)", "takes 1 argument"),
        ("loop(n) = loop(n + 1)\nloop(0)", "Recursion too deep"),
        ("[1, 2][5]", "outside 1..2"),
        ("If(x, 1, 2)", "neither true nor false"),
        ("Range()", "wrong arguments"),
        ("[1, 2] ^ 2", "powers of arrays"),
        ('"a" + 1', "is not a number"),
        ("3(4)", "is not a function"),
    ],
)
def test_errors_are_reported_not_raised(code, fragment):
    r = outputs(code)
    assert r["status"] == "error"
    assert fragment in r["outputs"][-1]["data"]


def test_print_and_suppression():
    r = outputs('Print("hello", [1, 2]); 5;')
    assert [o["data"] for o in r["outputs"]] == ['"hello"', "[1, 2]"]
    assert outputs("x = 5;")["outputs"] == []


def test_help_names_and_concepts():
    assert "Einstein" in value('Help("ArrayMultiply")')
    assert "ArrayMultiply" in value('Names("Array*")')
    r = outputs('Concept("hypermatrix.array_multiply")')
    assert r["outputs"][0]["type"] == "markdown" and "hypergraph-paclet" in r["outputs"][0]["data"]
    assert "hypermatrix.array" in value('Concepts("hypermatrix")')
