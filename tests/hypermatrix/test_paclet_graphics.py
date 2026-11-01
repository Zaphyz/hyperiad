"""Test cases ported from the reference paclet's tests/HypermatrixExamples.wls, section 8 (flat drawing)."""

import pytest
from frontier.conformance import ported
from hypermatrix import (
    ArrayError,
    ArrayObject,
    array_nesting,
    array_svg,
    generate_symbolic_array,
    identity_array,
    one_array,
)


def is_svg(s: str) -> bool:
    return s.startswith("<svg") and s.endswith("</svg>")


@ported("hypergraph-paclet", "tests/HypermatrixExamples.wls", 238, 239, 240, 241, 242, 243, 244, 248, 249)
def test_every_order_draws_flat():
    assert is_svg(array_svg(ArrayObject(7)))
    assert is_svg(array_svg([1, 2, 3]))
    assert is_svg(array_svg([[1, 2], [3, 4]]))
    for n in range(3, 6):
        assert is_svg(array_svg(one_array(n, 2)))
    assert is_svg(array_svg(identity_array(2, 3)))
    assert is_svg(array_svg(generate_symbolic_array("a", [2, 2])))


@ported("hypergraph-paclet", "tests/HypermatrixExamples.wls", 252, 254, 255, 257, 259, 261, 263)
def test_nesting():
    assert [array_nesting(n) for n in range(8)] == [[], [1], [2], [1, 2], [2, 2], [1, 2, 2], [2, 2, 2], [1, 2, 2, 2]]
    assert array_nesting(3) == [1, 2]
    assert all(k <= 2 for n in range(13) for k in array_nesting(n))
    assert all(sum(array_nesting(n)) == n for n in range(13))
    with pytest.raises(ArrayError):
        array_svg(one_array(3, 2), nesting=[3])
    with pytest.raises(ArrayError):
        array_svg([[1, 2], [3, 4]], nesting=[1])
    assert is_svg(array_svg(one_array(3, 2), nesting=[2, 1]))


@ported("hypergraph-paclet", "tests/HypermatrixExamples.wls", 270, 274, 276, 278)
def test_options_and_limits():
    assert 'class="hd-index"' not in array_svg([[1, 2], [3, 4]], indices=False)
    with pytest.raises(ArrayError):
        array_svg(one_array(3, 12))
    assert is_svg(array_svg(one_array(3, 4), max_entries=1000))
    with pytest.raises(ArrayError):
        array_svg([[1, 2], [3]])


def test_entries_are_escaped_and_scalars_unnumbered():
    svg = array_svg(ArrayObject(["<a&b>"]))
    assert "&lt;a&amp;b&gt;" in svg and "<a&b>" not in svg
    assert 'class="hd-index"' not in array_svg(ArrayObject(7))
