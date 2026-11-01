"""Arithmetic on Hyperiad values: exact on integers and fractions, entry by entry on arrays.

Lists take part in arithmetic as arrays (``[1, 2] + [3, 4]`` is ``[4, 6]``); a result
stays a list unless an :class:`ArrayObject` took part, in which case it is an array.
"""

from __future__ import annotations

from fractions import Fraction
from numbers import Number
from typing import Any

import sympy
from hypermatrix import ArrayError, ArrayObject, FieldElement, array_add, array_times
from hypermatrix.arrays import reciprocal

from hyperiad.errors import HyperiadError


def norm(v: Any) -> Any:
    """Canonical exact numbers: integral fractions become ints, SymPy numbers become Python ones."""
    if isinstance(v, Fraction) and v.denominator == 1:
        return int(v.numerator)
    if isinstance(v, sympy.Basic):
        if v.is_Integer:
            return int(v)
        if v.is_Rational:
            return Fraction(int(v.p), int(v.q))
        if v is sympy.true or v is sympy.false:
            return bool(v)
    return v


def _arrayish(x: Any) -> bool:
    return isinstance(x, (ArrayObject, list))


def _scalar(x: Any) -> bool:
    return isinstance(x, (Number, sympy.Basic, FieldElement)) and not isinstance(x, bool)


def _check_scalar(op: str, *xs: Any) -> None:
    for x in xs:
        if not _scalar(x):
            from hyperiad.values import format_value

            raise HyperiadError(f"{op}: {format_value(x, 80)} is not a number, symbol or array")


def _array_result(result: ArrayObject, inputs: tuple) -> Any:
    return result if any(isinstance(x, ArrayObject) for x in inputs) else result.entries


def _array_op(fn, op: str, a: Any, b: Any) -> Any:
    try:
        return _array_result(fn(a, b), (a, b))
    except ArrayError as e:
        raise HyperiadError(f"{op}: {e}") from None


def plus(a: Any, b: Any) -> Any:
    if _arrayish(a) or _arrayish(b):
        return _array_op(array_add, "Plus", a, b)
    _check_scalar("Plus", a, b)
    return norm(a + b)


def times(a: Any, b: Any) -> Any:
    if _arrayish(a) or _arrayish(b):
        return _array_op(array_times, "Times", a, b)
    _check_scalar("Times", a, b)
    return norm(a * b)


def minus(a: Any) -> Any:
    return times(-1, a)


def subtract(a: Any, b: Any) -> Any:
    return plus(a, minus(b))


def divide(a: Any, b: Any) -> Any:
    if _arrayish(b):
        raise HyperiadError("Divide: an array can be divided by a scalar only")
    _check_scalar("Divide", b)
    if b == 0 and not isinstance(b, sympy.Basic):
        raise HyperiadError("Divide: division by zero")
    try:
        r = reciprocal(b)
    except (TypeError, ZeroDivisionError) as e:
        raise HyperiadError(f"Divide: {e}") from None
    return times(a, r)


def power(a: Any, b: Any) -> Any:
    if _arrayish(a) or _arrayish(b):
        raise HyperiadError("Power: powers of arrays are not defined; use ArrayMultiply or ArrayTimes")
    _check_scalar("Power", a, b)
    if isinstance(a, (int, Fraction)) and isinstance(b, int) and b < 0:
        if a == 0:
            raise HyperiadError("Power: 0 to a negative power")
        return norm(Fraction(1) / Fraction(a) ** -b)
    if isinstance(a, (int, Fraction)) and isinstance(b, Fraction):
        return norm(sympy.Rational(a) ** sympy.Rational(b.numerator, b.denominator))
    return norm(a**b)


def _symbolic(*xs: Any) -> bool:
    return any(isinstance(x, sympy.Basic) and not x.is_number for x in xs)


def equal(a: Any, b: Any) -> Any:
    if _symbolic(a, b):
        return norm(sympy.Eq(sympy.sympify(a), sympy.sympify(b)))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(equal(x, y) is True for x, y in zip(a, b, strict=True))
    try:
        result = a == b
    except Exception:  # noqa: BLE001 - incomparable values are simply unequal
        return False
    return bool(result) if isinstance(result, (bool, sympy.logic.boolalg.BooleanAtom)) else result is True


def unequal(a: Any, b: Any) -> Any:
    r = equal(a, b)
    return (not r) if isinstance(r, bool) else norm(sympy.Not(r))


def compare(op: str, a: Any, b: Any) -> Any:
    _check_scalar(op, a, b)
    fn = {"Less": "__lt__", "LessEqual": "__le__", "Greater": "__gt__", "GreaterEqual": "__ge__"}[op]
    if _symbolic(a, b) or isinstance(a, sympy.Basic) or isinstance(b, sympy.Basic):
        rel = {"Less": sympy.Lt, "LessEqual": sympy.Le, "Greater": sympy.Gt, "GreaterEqual": sympy.Ge}[op]
        return norm(rel(sympy.sympify(a), sympy.sympify(b)))
    try:
        return bool(getattr(a, fn)(b))
    except TypeError as e:
        raise HyperiadError(f"{op}: {e}") from None


def truth(v: Any, what: str) -> bool:
    v = norm(v)
    if isinstance(v, bool):
        return v
    raise HyperiadError(f"{what}: the condition is neither true nor false")
