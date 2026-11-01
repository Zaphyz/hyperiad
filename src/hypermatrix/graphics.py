"""Drawing arrays in the plane, as SVG.

The flat family of the reference paclet (``src/ArrayGraphics.wl``): a row of cells at
order one, a grid at order two, and above that a nesting of those -- a grid whose cells
hold grids. :func:`array_nesting` says how the indices are split into blocks; the
innermost block holds the entries and every outer block holds pictures.

The SVG is self-contained (its own styles, light and dark) and uses CSS classes, so a
host page may restyle it.
"""

from __future__ import annotations

import html
from fractions import Fraction
from typing import Any

import numpy as np
from frontier import SourceRef, define

from hypermatrix.arrays import ArrayError, as_array, entries_of
from hypermatrix.fields import FieldElement

CELL_H = 26.0
CHAR_W = 7.2
PAD = 6.0
INDEX_W = 16.0
INDEX_H = 14.0
OUTER_GAP = 8.0

STYLE = """
.hd-cell{fill:#dfe5f7;stroke:#4a5a9a;stroke-opacity:.9;stroke-width:1}
.hd-frame{fill:none;stroke:#4a5a9a;stroke-opacity:.45;stroke-width:1}
.hd-entry{fill:#1d1d1f;font:12px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
text-anchor:middle;dominant-baseline:central}
.hd-index{fill:#5a6bb0;font:italic 10px system-ui,sans-serif;text-anchor:middle;dominant-baseline:central}
@media (prefers-color-scheme:dark){
.hd-cell{fill:#2e3760;stroke:#a9b6ec}.hd-frame{stroke:#a9b6ec}.hd-entry{fill:#f2f2f4}.hd-index{fill:#a9b6ec}}
"""


@define(
    "hypermatrix.array_nesting",
    "Array nesting",
    r"How the indices of an $n$-array are split into blocks for drawing: blocks of the largest size "
    r"allowed ($2$ in the plane, $3$ in space) with any remainder taken off the front. A $3$-array is "
    r"drawn as a row of matrices, the first index choosing the matrix.",
    maturity="working",
    sources=(SourceRef("hypergraph-paclet", "symbol:ArrayNesting", "implementation"),),
    tags=("rendering",),
)
def array_nesting(n: int, b: int = 2) -> list[int]:
    if n < 0 or b < 1:
        raise ValueError("order must be >= 0 and block size >= 1")
    if n == 0:
        return []
    if n <= b:
        return [n]
    q, r = divmod(n, b)
    return ([r] if r else []) + [b] * q


def format_entry(x: Any) -> str:
    """Compact text for one entry."""
    if isinstance(x, bool):
        return "True" if x else "False"
    if isinstance(x, Fraction):
        return f"{x.numerator}/{x.denominator}"
    if isinstance(x, float):
        return f"{x:.6g}"
    if isinstance(x, complex):
        return f"{x.real:.4g}{x.imag:+.4g}i"
    if isinstance(x, FieldElement):
        return str(x.value)
    return str(x)


class _Pic:
    """A laid-out picture: size plus SVG body drawn with its top-left at the origin."""

    def __init__(self, w: float, h: float, body: str) -> None:
        self.w, self.h, self.body = w, h, body


def _grid(rows: int, cols: int, cell_w: float, cell_h: float, cell, indices: bool, outer: bool) -> _Pic:
    left = INDEX_W if indices and rows > 1 else 0.0
    top = INDEX_H if indices else 0.0
    parts = []
    for i in range(rows):
        for j in range(cols):
            x, y = left + j * cell_w, top + i * cell_h
            cls = "hd-frame" if outer else "hd-cell"
            parts.append(
                f'<rect class="{cls}" x="{x + 1:.1f}" y="{y + 1:.1f}" width="{cell_w - 2:.1f}" '
                f'height="{cell_h - 2:.1f}" rx="4"/>'
            )
            parts.append(cell(i, j, x, y))
    if indices:
        for j in range(cols):
            parts.append(f'<text class="hd-index" x="{left + (j + 0.5) * cell_w:.1f}" y="{top / 2:.1f}">{j + 1}</text>')
        if rows > 1:
            for i in range(rows):
                parts.append(
                    f'<text class="hd-index" x="{left / 2:.1f}" y="{top + (i + 0.5) * cell_h:.1f}">{i + 1}</text>'
                )
    return _Pic(left + cols * cell_w, top + rows * cell_h, "".join(parts))


def _block_shape(dims: list[int], k: int) -> tuple[int, int]:
    if k == 0:
        return 1, 1
    if k == 1:
        return 1, dims[0]
    return dims[0], dims[1]


def _index(k: int, i: int, j: int) -> tuple[int, ...]:
    return () if k == 0 else (j,) if k == 1 else (i, j)


def _layout(arr: np.ndarray, nesting: list[int], indices: bool) -> _Pic:
    k, rest = (nesting[0], nesting[1:]) if nesting else (0, [])
    rows, cols = _block_shape(list(arr.shape), k)
    if not rest:  # innermost block: entries
        texts = [format_entry(entries_of(np.asarray(arr[idx], dtype=object))) for idx in np.ndindex(*arr.shape)]
        widest = max((len(t) for t in texts), default=1)
        cell_w = max(28.0, widest * CHAR_W + 2 * PAD)

        def entry(i: int, j: int, x: float, y: float) -> str:
            text = format_entry(entries_of(np.asarray(arr[_index(k, i, j)], dtype=object)))
            return (
                f'<text class="hd-entry" x="{x + cell_w / 2:.1f}" y="{y + CELL_H / 2:.1f}">{html.escape(text)}</text>'
            )

        # a 0-array has no indices, so nothing to number (the reference numbers its one cell 1)
        return _grid(rows, cols, cell_w, CELL_H, entry, indices and k > 0, outer=False)
    inner = [[_layout(arr[_index(k, i, j)], rest, indices) for j in range(cols)] for i in range(rows)]
    cw = max(p.w for row in inner for p in row) + 2 * OUTER_GAP
    ch = max(p.h for row in inner for p in row) + 2 * OUTER_GAP

    def picture(i: int, j: int, x: float, y: float) -> str:
        p = inner[i][j]
        dx, dy = x + (cw - p.w) / 2, y + (ch - p.h) / 2
        return f'<g transform="translate({dx:.1f},{dy:.1f})">{p.body}</g>'

    return _grid(rows, cols, cw, ch, picture, indices, outer=True)


@define(
    "hypermatrix.array_graphics",
    "Plane picture of an array",
    r"`ArrayGraphics(A)` draws an array in the plane: a cell at order $0$, a row at order $1$, a grid at "
    r"order $2$, and a nesting of rows and grids above that, as split by the array nesting. Index "
    r"numbers run along the top and down the left.",
    maturity="working",
    depends_on=("hypermatrix.array", "hypermatrix.array_nesting"),
    sources=(SourceRef("hypergraph-paclet", "symbol:ArrayGraphics", "implementation"),),
    tags=("rendering",),
)
def array_svg(a: Any, nesting: list[int] | None = None, indices: bool = True, max_entries: int = 512) -> str:
    arr = as_array(a).data
    nest = array_nesting(arr.ndim) if nesting is None else list(nesting)
    if sum(nest) != arr.ndim or any(k < 1 for k in nest):
        raise ArrayError("ArrayGraphics::nesting", f"nesting {nest} is not positive integers summing to {arr.ndim}")
    if any(k > 2 for k in nest):
        raise ArrayError("ArrayGraphics::block", f"nesting {nest} asks for a block of more than two indices")
    if arr.size > max_entries:
        raise ArrayError(
            "ArrayGraphics::big", f"the array has {arr.size} entries, more than {max_entries} drawn legibly"
        )
    pic = _layout(arr, nest, indices)
    w, h = pic.w + 2, pic.h + 2
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" class="hd-array" width="{w:.0f}" height="{h:.0f}" '
        f'viewBox="-1 -1 {w:.0f} {h:.0f}" role="img" aria-label="array of dimensions {list(arr.shape)}">'
        f"<style>{STYLE}</style>{pic.body}</svg>"
    )


def hypermatrix_svgs(hm: Any, **options: Any) -> list[str]:
    """One flat picture per array of a hypermatrix, in its order (the paclet's ``HypermatrixGraphics``)."""
    from hypermatrix.hypermatrix import Hypermatrix

    if not isinstance(hm, Hypermatrix):
        raise ArrayError("HypermatrixGraphics::hm", f"{hm!r} is not a hypermatrix")
    return [array_svg(a, **options) for a in hm.arrays]
