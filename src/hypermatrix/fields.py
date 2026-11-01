"""Prime fields GF(p), as array entries.

Only prime fields are implemented; the reference paclet's ``FiniteField[p, n]``
with ``n > 1`` is not needed yet.
"""

from __future__ import annotations

from functools import total_ordering


def _is_prime(p: int) -> bool:
    if p < 2:
        return False
    k = 2
    while k * k <= p:
        if p % k == 0:
            return False
        k += 1
    return True


class FiniteField:
    """The field with ``p`` elements. ``FiniteField(2)(1)`` is the element 1 of GF(2)."""

    _cache: dict[int, FiniteField] = {}

    def __new__(cls, p: int, n: int = 1) -> FiniteField:
        if n != 1:
            raise NotImplementedError("only prime fields GF(p) are implemented")
        if not isinstance(p, int) or not _is_prime(p):
            raise ValueError(f"{p!r} is not a prime")
        if p not in cls._cache:
            field = super().__new__(cls)
            field.p = p
            cls._cache[p] = field
        return cls._cache[p]

    def __call__(self, value: int) -> FieldElement:
        return FieldElement(value, self)

    @property
    def zero(self) -> FieldElement:
        return FieldElement(0, self)

    @property
    def one(self) -> FieldElement:
        return FieldElement(1, self)

    def __repr__(self) -> str:
        return f"FiniteField({self.p})"

    def __reduce__(self):
        return (FiniteField, (self.p,))


@total_ordering
class FieldElement:
    __slots__ = ("value", "field")

    def __init__(self, value: int, field: FiniteField) -> None:
        self.value = int(value) % field.p
        self.field = field

    def _coerce(self, other: object) -> FieldElement | None:
        if isinstance(other, FieldElement):
            if other.field is not self.field:
                raise ValueError(f"cannot combine elements of {self.field} and {other.field}")
            return other
        if isinstance(other, int) and not isinstance(other, bool):
            return FieldElement(other, self.field)
        return None

    def __add__(self, other):
        o = self._coerce(other)
        return NotImplemented if o is None else FieldElement(self.value + o.value, self.field)

    __radd__ = __add__

    def __mul__(self, other):
        o = self._coerce(other)
        return NotImplemented if o is None else FieldElement(self.value * o.value, self.field)

    __rmul__ = __mul__

    def __sub__(self, other):
        o = self._coerce(other)
        return NotImplemented if o is None else FieldElement(self.value - o.value, self.field)

    def __rsub__(self, other):
        o = self._coerce(other)
        return NotImplemented if o is None else FieldElement(o.value - self.value, self.field)

    def __neg__(self):
        return FieldElement(-self.value, self.field)

    def inverse(self) -> FieldElement:
        if self.value == 0:
            raise ZeroDivisionError("0 has no inverse")
        return FieldElement(pow(self.value, -1, self.field.p), self.field)

    def __truediv__(self, other):
        o = self._coerce(other)
        return NotImplemented if o is None else self * o.inverse()

    def __rtruediv__(self, other):
        o = self._coerce(other)
        return NotImplemented if o is None else o * self.inverse()

    def __pow__(self, k: int):
        if k < 0:
            return self.inverse() ** -k
        return FieldElement(pow(self.value, k, self.field.p), self.field)

    def __eq__(self, other) -> bool:
        if isinstance(other, FieldElement):
            return self.field is other.field and self.value == other.value
        return NotImplemented

    def __lt__(self, other) -> bool:
        if isinstance(other, FieldElement) and other.field is self.field:
            return self.value < other.value
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.field.p, self.value))

    def __repr__(self) -> str:
        return f"FiniteField({self.field.p})({self.value})"

    def __str__(self) -> str:
        return f"{self.value}"
