import re

import pytest
from hyperiad.expr import E, S
from hyperiad.syntax import ParseError, parse, tokenize, unparse

CORPUS = [
    "1 + 2 * 3",
    "(1 + 2) * 3",
    "a - b - c",
    "a - (b - c)",
    "2 ^ 3 ^ 2",
    "(2 ^ 3) ^ 2",
    "-x ^ 2",
    "(-x) ^ 2",
    "a ^ (-b)",
    "2 - -3",
    "-a * b",
    "a / b / c",
    "not a == b",
    "a and b or c",
    "a and (b or c)",
    "x <= 3 and y > 2",
    "a -> b -> c",
    "(a -> b) -> c",
    "f(x)",
    "f(x, y, Labeled = true, Mode = 2)",
    "f(x)(y)",
    "(x => x ^ 2)(3)",
    "x => x + 1",
    "(x, y) => x * y",
    "() => 1",
    "x => y => x + y",
    '[1, 2.5, "s", null, true, false]',
    "[[1, 2], [3, 4]]",
    '{"a": 1, 2: [3]}',
    "a[1]",
    "a[1, 2]",
    'a["Order"]',
    "f(x)[2]",
    "x = 3",
    "f(x, y) = x + y",
    "a = b = 2",
    'ArrayMultiply([a, b], "ij,jk->ik")',
    "ArrayMultiply([a, b], [[1, 2], [2, 3]] -> [1, 3])",
    "Table(i * j, [i, 3], [j, 1, 4, 2])",
    "x = 1\ny = 2\nx + y",
    "x = 1; y = 2;",
    "1e-05 + .5 + 3.",
    '"quote \\" and \\n newline"',
]


@pytest.mark.parametrize("src", CORPUS)
def test_round_trip(src):
    tree = parse(src)
    assert parse(unparse(tree)) == tree


def test_trees():
    assert parse("1 + 2 * 3") == E("Plus", 1, E("Times", 2, 3))
    assert parse("2 ^ 3 ^ 2") == E("Power", 2, E("Power", 3, 2))
    assert parse("-x ^ 2") == E("Minus", E("Power", S("x"), 2))
    assert parse("f(x, k = 1)") == E("f", S("x"), E("Option", S("k"), 1))
    assert parse("x => x") == E("Function", E("List", S("x")), S("x"))
    assert parse("a[1, 2]") == E("Part", S("a"), 1, 2)
    assert parse("{1: 2}") == E("Association", E("Rule", 1, 2))
    assert parse("f(x) = x") == E("Set", E("f", S("x")), S("x"))
    assert parse("a; b;") == E("CompoundExpression", S("a"), S("b"), None)
    assert parse("Plus(1, 2)") == parse("1 + 2")  # operators are ordinary heads


def test_printer_output():
    assert unparse(parse("(1+2)*3")) == "(1 + 2) * 3"
    assert unparse(parse("f(x,k=1)")) == "f(x, k = 1)"
    assert unparse(parse("a;b;")) == "a\nb;"
    assert unparse(E("Plus", 1, 2, 3)) == "Plus(1, 2, 3)"  # not an operator at that arity


def test_newlines_inside_brackets_continue():
    assert parse("f(1,\n 2)") == E("f", 1, 2)
    assert parse("[1,\n2]") == E("List", 1, 2)
    assert parse("1 + \\\n 2") == E("Plus", 1, 2)


def test_comments_and_blank_lines():
    assert parse("# a comment\n\nx  # trailing\n\n") == S("x")


@pytest.mark.parametrize(
    "src, fragment",
    [
        ("1 +", "end of input"),
        ("f(1", "expected ',' or ')'"),
        ("a < b < c", "do not chain"),
        ("(a, b)", "exactly one expression"),
        ("(1, 2) => 3", "parameters must be names"),
        ("1 2", "separated by newlines"),
        ("3 = x", "only a name or a call"),
        ("x @ y", "unexpected character"),
    ],
)
def test_errors(src, fragment):
    with pytest.raises(ParseError, match=re.escape(fragment)):
        parse(src)


def test_error_position():
    with pytest.raises(ParseError) as e:
        parse("x = 1\ny = (2 +")
    assert (e.value.line, e.value.col) == (2, 9)


def test_tokens_carry_positions():
    toks = tokenize("ab\n  cd")
    assert [(t.kind, t.text, t.line, t.col) for t in toks] == [
        ("name", "ab", 1, 1),
        ("newline", "\n", 1, 3),
        ("name", "cd", 2, 3),
        ("end", "", 2, 5),
    ]
