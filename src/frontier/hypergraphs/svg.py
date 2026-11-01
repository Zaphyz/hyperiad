"""Hypergraph drawings as SVG: the same scene the matplotlib renderer paints.

Used by Hyperiad (in the browser kernel, where matplotlib is not loaded) and by any
web page. Layout units map to points at :data:`POINTS_PER_SPAN` over the drawing's
span, the scale of a 3.6-inch matplotlib figure, and marks keep their sizes in points,
so the SVG and the session figures look the same. The picture carries its own white
ground, as the figures do.
"""

from __future__ import annotations

import html
from collections.abc import Hashable, Mapping, Sequence

import numpy as np

from frontier.hypergraphs.core import Hypergraph
from frontier.hypergraphs.geometry import (
    ArcStroke,
    Area,
    Arrowhead,
    Curve,
    Dots,
    Label,
    Scene,
    hypergraph_scene,
)
from frontier.hypergraphs.pictures import (
    Coordinates,
    event_scene,
    hypergraph_drawing_args,
    hypergraph_picture,
    rule_scene,
    system_graph_scene,
)
from frontier.hypergraphs.rewriting import Event, HypergraphRule
from frontier.registry import SourceRef, define

POINTS_PER_SPAN = 160.0  # a figure 3.6 in wide, axes 0.775 of it, data margins 0.12 each side
MARGIN = 6.0
PX_PER_PT = 1.75  # drawn larger than print size, for the screen; proportions are unchanged
ROW_ZOOM = 0.6  # a rule or an event shows two drawings side by side, each smaller
FONT = "DejaVu Sans, Verdana, Arial, sans-serif"


def _color(attr: str, rgba) -> str:
    if rgba is None:
        return f'{attr}="none"'
    r, g, b, a = rgba
    rgb = f"rgb({round(255 * r)},{round(255 * g)},{round(255 * b)})"
    return f'{attr}="{rgb}"' + (f' {attr}-opacity="{a:.3g}"' if a < 1 else "")


def _pts(points: np.ndarray) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def scene_svg(scene: Scene, *, width: float | None = None, zoom: float = 1.0) -> str:
    """Paint a scene as a standalone SVG. ``width`` (px) or ``zoom`` rescales the whole picture."""
    scale = POINTS_PER_SPAN / scene.span

    def to_pt(p: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(p, dtype=float))
        return np.stack([p[:, 0] * scale, -p[:, 1] * scale], axis=1)

    body: list[str] = []
    extent: list[np.ndarray] = []  # boxes [xmin, ymin, xmax, ymax] of everything painted

    def grow(points: np.ndarray, pad: float = 0.0) -> None:
        extent.append(np.r_[points.min(axis=0) - pad, points.max(axis=0) + pad])

    ordered = sorted(enumerate(scene.shapes), key=lambda t: (t[1].z, t[0]))  # matplotlib's painting order
    for _, s in ordered:
        if isinstance(s, Area):
            p = to_pt(s.points)
            grow(p)
            stroke = _color("stroke", s.stroke) + (f' stroke-width="{s.width:.2f}"' if s.stroke else "")
            body.append(f'<polygon points="{_pts(p)}" {_color("fill", s.fill)} {stroke} stroke-linejoin="round"/>')
        elif isinstance(s, Curve):
            p = to_pt(s.points)
            grow(p, s.width)
            cap = "round" if s.round_cap else "square"
            body.append(
                f'<polyline points="{_pts(p)}" fill="none" {_color("stroke", s.color)} stroke-width="{s.width:.2f}" '
                f'stroke-linecap="{cap}" stroke-linejoin="round"/>'
            )
        elif isinstance(s, Dots):
            p = to_pt(s.centers)
            grow(p, s.radius + s.width)
            stroke = _color("stroke", s.stroke) + (f' stroke-width="{s.width:.2f}"' if s.stroke else "")
            for x, y in p:
                body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{s.radius:.2f}" {_color("fill", s.fill)} {stroke}/>')
        elif isinstance(s, Arrowhead):
            tail, head = to_pt(s.tail)[0], to_pt(s.head)[0]
            u = head - tail
            norm = float(np.hypot(*u))
            if norm < 1e-12:
                continue
            u /= norm
            tip = head - s.shrink * u
            base = tip - 0.4 * s.size * u
            n = np.array([-u[1], u[0]]) * 0.2 * s.size
            tri = np.array([tip, base + n, base - n])
            grow(tri)
            body.append(f'<polygon points="{_pts(tri)}" {_color("fill", s.color)}/>')
        elif isinstance(s, ArcStroke):
            t = np.radians(np.linspace(s.theta1, s.theta2, 48))
            arc = s.center + s.radius * np.stack([np.cos(t), np.sin(t)], axis=1)
            p = to_pt(arc)
            grow(p, s.width)
            body.append(
                f'<polyline points="{_pts(p)}" fill="none" {_color("stroke", s.color)} '
                f'stroke-width="{s.width:.2f}" stroke-linecap="butt"/>'
            )
        elif isinstance(s, Label):
            at = to_pt(s.anchor)[0] + np.array([s.offset[0], -s.offset[1]])
            half = np.array([0.31 * s.size * max(len(s.text), 1), 0.6 * s.size])
            grow(np.array([at - half, at + half]))
            halo = (
                f'{_color("stroke", s.outline)} stroke-width="{0.16 * s.size:.2f}" '
                'stroke-linejoin="round" paint-order="stroke" '
                if s.outline
                else ""
            )
            body.append(
                f'<text x="{at[0]:.2f}" y="{at[1]:.2f}" {_color("fill", s.color)} {halo}font-size="{s.size:.1f}" '
                f'text-anchor="middle" dominant-baseline="central">{html.escape(s.text)}</text>'
            )
    if extent:
        box = np.array(extent)
        x0, y0 = box[:, 0].min() - MARGIN, box[:, 1].min() - MARGIN
        x1, y1 = box[:, 2].max() + MARGIN, box[:, 3].max() + MARGIN
    else:
        x0, y0, x1, y1 = -MARGIN, -MARGIN, MARGIN, MARGIN
    title = ""
    if scene.title:
        y0 -= 16
        title = (
            f'<text x="{(x0 + x1) / 2:.2f}" y="{y0 + 9:.2f}" font-size="10" text-anchor="middle" '
            f'dominant-baseline="central" fill="rgb(0,0,0)">{html.escape(scene.title)}</text>'
        )
    w, h = x1 - x0, y1 - y0
    px = width if width is not None else w * PX_PER_PT * zoom
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" class="hd-hypergraph" width="{px:.0f}" '
        f'height="{px * h / w:.0f}" viewBox="{x0:.2f} {y0:.2f} {w:.2f} {h:.2f}" role="img" '
        f'font-family="{FONT}" aria-label="hypergraph drawing">'
        f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{w:.2f}" height="{h:.2f}" fill="white" rx="4"/>'
        f"{title}{''.join(body)}</svg>"
    )


@define(
    "common.hypergraph_plot",
    "Hypergraph drawing",
    r"`HypergraphPlot(hg)` draws a hypergraph in the reference paclet's style: binary edges as lines, "
    r"hyperedges of arity $\ge 3$ as translucent concave regions, unary edges as disks round their "
    r"vertex; directed edges show their order with arrowheads (internal half-arrowheads growing along "
    r"the sequence on hyperedges), cyclic edges a circular badge. Labels are written beside what they "
    r"name.",
    maturity="working",
    depends_on=("common.hypergraph",),
    sources=(SourceRef("hypergraph-paclet", "symbol:HypergraphPlot", "implementation"),),
    tags=("hypergraph", "rendering"),
)
def hypergraph_svg(
    hg: Hypergraph | Sequence[Sequence[Hashable]],
    *,
    symmetry: str | Sequence[str] = "unordered",
    labels: bool = False,
    highlight: Sequence[int] = (),
    highlight_vertices: Sequence[Hashable] = (),
    title: str = "",
    seed: int = 0,
    width: float | None = None,
    coordinates: Coordinates | None = None,
    edge_tints: Mapping[Hashable, str] | Sequence[str | None] | None = None,
) -> str:
    """SVG drawing of a :class:`Hypergraph`, or of raw edges with ``symmetry`` as in ``draw_hypergraph``.

    ``edge_tints``: for a Hypergraph, a map from edge labels to tint names; for raw edges, one tint per edge."""
    if isinstance(hg, Hypergraph):
        scene = hypergraph_picture(
            hg,
            labels=labels,
            highlight=highlight,
            highlight_vertices=highlight_vertices,
            title=title,
            seed=seed,
            coordinates=coordinates,
            edge_tints=edge_tints if isinstance(edge_tints, Mapping) else None,
        )
    else:
        tints = list(edge_tints) if edge_tints is not None and not isinstance(edge_tints, Mapping) else []
        args = {"edges": [tuple(e) for e in hg], "symmetry": symmetry, "labels": labels, "edge_tints": tints}
        scene = hypergraph_scene(**args, highlight=highlight, title=title, seed=seed)
    return scene_svg(scene, width=width)


@define(
    "common.hypergraph_rule_plot",
    "Rule and event drawings",
    r"A rewrite rule is drawn as its input beside its output, with the vertices it destroys and "
    r"creates highlighted; an event as the hypergraph it acted on, the match highlighted, beside the "
    r"result, the created vertices and edges highlighted. Both sides share one layout.",
    maturity="working",
    depends_on=("common.hypergraph_plot", "common.hypergraph_rule"),
    sources=(
        SourceRef("hypergraph-paclet", "symbol:HypergraphRulePlot", "implementation"),
        SourceRef("hypergraph-paclet", "symbol:HypergraphEventPlot", "implementation"),
    ),
    tags=("hypergraph", "rewriting", "rendering"),
)
def rule_svg(
    rule: HypergraphRule, *, labels: bool = False, highlight: bool = True, shared: bool = True, width=None
) -> str:
    scene = rule_scene(rule, labels=labels, highlight=highlight, shared=shared)
    return scene_svg(scene, width=width, zoom=ROW_ZOOM)


def event_svg(event: Event, *, labels: bool = False, shared: bool = True, width=None) -> str:
    return scene_svg(event_scene(event, labels=labels, shared=shared), width=width, zoom=ROW_ZOOM)


def system_graph_svg(sg, *, title: str = "", width=None) -> str:
    """A graph of a rewriting system (evolution, states, causal, branchial, ...), drawn."""
    return scene_svg(system_graph_scene(sg, title=title), width=width)


__all__ = ["system_graph_svg", "event_svg", "hypergraph_drawing_args", "hypergraph_svg", "rule_svg", "scene_svg"]
