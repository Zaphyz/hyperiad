r"""Devices: constructions outside the hypergraphic core, kept for side directions.

The user (2026-10-28): every construction of infrageometry rests on hypergraphs and discrete
rewriting; real numbers enter only in a side direction, as a device. What lives here is such a
device: useful to orient the research, never a definition the rest depends on.
"""

from __future__ import annotations

import math
from collections.abc import Hashable

from frontier.registry import define

from infrageometry.space import KEY, SpatialHypergraph

Point = Hashable


@define(
    f"{KEY}.ball_growth",
    "Ball growth",
    r"The ball growth of $H$ at $v$ is the sequence of counts $|B_0(v)|, |B_1(v)|, |B_2(v)|, \dots$: how "
    r"many points lie within $r$ steps. It is hypergraphic (counting only); comparing its growth with "
    r"powers $r^d$ is the device :func:`growth_dimension`.",
    maturity="sketch",
    depends_on=(f"{KEY}.neighbourhood",),
    tags=("infrageometry", "dimension"),
)
def ball_growth(space: SpatialHypergraph, v: Point, rmax: int) -> list[int]:
    dist = space.distances(v, rmax)
    return [sum(1 for d in dist.values() if d <= r) for r in range(rmax + 1)]


@define(
    f"{KEY}.growth_dimension",
    "Growth dimension (device)",
    r"A device, outside the hypergraphic core: the growth dimension at $v$ and scale $r \ge 1$ is "
    r"$\log\big(|B_{r+1}(v)| / |B_r(v)|\big) / \log\big((r+1)/r\big)$, the local slope of $\log |B_r|$ "
    r"against $\log r$; on the grid $\mathbb{Z}^n$ it tends to $n$. It measures large-scale dimension, "
    r"which the link dimension (a local count) does not see: the grid $\mathbb{Z}^2$ has link dimension 1.",
    maturity="sketch",
    depends_on=(f"{KEY}.ball_growth",),
    tags=("infrageometry", "dimension", "device"),
)
def growth_dimension(space: SpatialHypergraph, v: Point, r: int) -> float:
    if r < 1:
        raise ValueError("the growth dimension needs r >= 1")
    counts = ball_growth(space, v, r + 1)
    return math.log(counts[r + 1] / counts[r]) / math.log((r + 1) / r)


@define(
    f"{KEY}.linear_tangent_space",
    "Linear tangent space (device)",
    r"A device: the free abelian group $\mathbb{Z}[T_vH]$ on the tangent vectors at $v$, whose elements "
    r"are formal combinations of departures, of rank $\deg v$. Kept to compare with the continuum; the "
    r"core works with $T_vH$ as a set.",
    maturity="sketch",
    depends_on=(f"{KEY}.tangent_vector",),
    tags=("infrageometry", "tangent", "device"),
)
def linear_tangent_rank(space: SpatialHypergraph, v: Point) -> int:
    return space.degree(v)
