"""Scenes of hypergraphs, rules and events: what is drawn, before any renderer paints it.

A rule is drawn as its input beside its output, an event as what it acted on beside what it
made, following the paclet's ``src/RewritingGraphics.wl``: the match is highlighted on the
left and the creations on the right, and both sides share a layout, so that reading one
against the other shows what changed. Both renderers (:mod:`frontier.hypergraphs.svg` and
:mod:`frontier.viz.hypergraph`) paint these scenes.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

from frontier.hypergraphs.core import Hypergraph, HypergraphError
from frontier.hypergraphs.geometry import (
    BRANCHIAL_COLOR,
    CAUSAL_COLOR,
    EVENT_COLOR,
    PANEL_MARGIN,
    STATE_COLOR,
    STATE_EDGE_COLOR,
    TOKEN_COLOR,
    Label,
    Scene,
    graph_scene,
    hypergraph_layout,
    hypergraph_scene,
    layered_layout,
    scene_row,
)
from frontier.hypergraphs.labels import CONTRACTED_MARK, label_text
from frontier.hypergraphs.rewriting import Event, HypergraphRule

Coordinates = Mapping[Hashable, Sequence[float]] | Sequence[Sequence[float]]

_SYMMETRY = {"Directed": "directed", "Unordered": "unordered", "Cyclic": "cyclic"}


def _drawn_symmetry(sym: str) -> str:
    """An edge's symmetry type as the geometry names it (input–output types in lower case)."""
    return _SYMMETRY.get(sym) or sym.lower()


def hypergraph_drawing_args(
    hg: Hypergraph, *, labels: bool = False, edge_tints: Mapping[Hashable, str] | None = None
) -> dict[str, Any]:
    """The scene arguments for a :class:`Hypergraph`: its edges, symmetries, isolated vertices and labels.

    ``edge_tints`` maps edge labels to tint names (``geometry.EDGE_TINTS``): those edges take that colour."""
    vertex_text: dict[Hashable, str] = {}
    marked = [v.id for v in hg.vertices if v.label == CONTRACTED_MARK]
    for v in hg.vertices:
        if v.label == CONTRACTED_MARK:  # shown by the vertex's colour, not by text
            if labels:
                vertex_text[v.id] = str(v.id)
        elif v.label is not None:
            vertex_text[v.id] = f"{v.id}: {label_text(v.label)}" if labels else label_text(v.label)
        elif labels:
            vertex_text[v.id] = str(v.id)
    return {
        "edges": [e.ids for e in hg.edges],
        "symmetry": [_drawn_symmetry(e.symmetry) for e in hg.edges],
        "vertices": hg.vertex_ids,
        "vertex_text": vertex_text,
        "edge_text": [None if e.label is None else label_text(e.label) for e in hg.edges],
        "marked_vertices": marked,
        "edge_tints": [(edge_tints or {}).get(e.label) for e in hg.edges],
    }


def _fixed(hg: Hypergraph, coordinates: Coordinates | None) -> dict | None:
    """``coordinates`` as a map from vertex to position; a bare list is taken in vertex order."""
    if coordinates is None:
        return None
    if isinstance(coordinates, Mapping):
        return dict(coordinates)
    coords = list(coordinates)
    if len(coords) > len(hg.vertices):
        raise HypergraphError("HypergraphPlot::coords", f"{len(coords)} positions for {len(hg.vertices)} vertices")
    return dict(zip(hg.vertex_ids, coords, strict=False))


def hypergraph_positions(hg: Hypergraph, *, coordinates: Coordinates | None = None, seed: int = 0) -> dict:
    """The layout of ``hg``: a position per vertex. Given ``coordinates`` hold; the rest settle among them."""
    args = hypergraph_drawing_args(hg)
    return hypergraph_layout(
        args["edges"], symmetry=args["symmetry"], seed=seed, vertices=args["vertices"], fixed=_fixed(hg, coordinates)
    )


def hypergraph_embedding(hg: Hypergraph, *, coordinates: Coordinates | None = None, seed: int = 0) -> list:
    """The paclet's ``HypergraphEmbedding``: the positions of the vertices, in vertex order."""
    pos = hypergraph_positions(hg, coordinates=coordinates, seed=seed)
    return [pos[v] for v in hg.vertex_ids]


def hypergraph_picture(
    hg: Hypergraph,
    *,
    labels: bool = False,
    highlight: Sequence[int] = (),
    highlight_vertices: Sequence[Hashable] = (),
    title: str = "",
    seed: int = 0,
    coordinates: Coordinates | None = None,
    edge_tints: Mapping[Hashable, str] | None = None,
) -> Scene:
    args = hypergraph_drawing_args(hg, labels=labels, edge_tints=edge_tints)
    layout = hypergraph_positions(hg, coordinates=coordinates, seed=seed)
    return hypergraph_scene(
        **args, layout=layout, highlight=highlight, highlight_vertices=highlight_vertices, title=title, seed=seed
    )


def rule_scene(rule: HypergraphRule, *, labels: bool = False, highlight: bool = True, shared: bool = True) -> Scene:
    """The rule's input beside its output; destroyed vertices highlighted on the left, created on the right."""
    lhs, rhs = rule.input, rule.output
    coords = hypergraph_positions(lhs) if shared else None
    left = hypergraph_picture(
        lhs, labels=labels, highlight_vertices=rule.deleted_vertices if highlight else (), coordinates=coords
    )
    right = hypergraph_picture(
        rhs, labels=labels, highlight_vertices=rule.new_vertices if highlight else (), coordinates=coords
    )
    return scene_row([left, right])


def event_scene(event: Event, *, labels: bool = False, shared: bool = True) -> Scene:
    """What the event acted on, its match highlighted, beside what it made, its creations highlighted."""
    coords = hypergraph_positions(event.input) if shared else None
    left = hypergraph_picture(
        event.input,
        labels=labels,
        highlight=event.match_edge_positions,
        highlight_vertices=event.match_vertices,
        coordinates=coords,
    )
    right = hypergraph_picture(
        event.hypergraph,
        labels=labels,
        highlight=event.created_edge_positions,
        highlight_vertices=event.new_vertices,
        coordinates=coords,
    )
    return scene_row([left, right])


_EDGE_COLORS = {
    "evolution": STATE_EDGE_COLOR,
    "causal": CAUSAL_COLOR,
    "branchial": BRANCHIAL_COLOR,
    "token": STATE_EDGE_COLOR,
}
_NODE_STYLES = {"state": ("disk", STATE_COLOR), "event": ("square", EVENT_COLOR), "token": ("disk", TOKEN_COLOR)}
STATE_PANEL = (1.0, 1.0)  # width, at most this high (graph units)
EVENT_PANEL = (2.4, 1.0)


def _number(v) -> str:
    return str(v[1] if isinstance(v, tuple) else v)


def system_graph_scene(sg, *, title: str = "") -> Scene:
    """A graph of a rewriting system (:class:`frontier.hypergraphs.causality.SystemGraph`), drawn.

    States and events are drawn as small framed pictures when ``sg.drawn``, otherwise as plain
    marks numbered as the evolution numbers them; tokens are always plain.
    """
    nodes, edges = list(sg.graph.nodes), list(sg.graph.edges)
    zoom = float(sg.options.get("PanelSize", 90) or 90) / 90.0
    panels: dict = {}
    if sg.decide_drawn():
        for v in nodes:
            kind = sg.kinds[v]
            if kind == "state":
                w, h = STATE_PANEL
                panels[v] = (hypergraph_picture(sg.payload[v]), w * zoom, h * zoom)
            elif kind == "event":
                w, h = EVENT_PANEL
                panels[v] = (event_scene(sg.payload[v].data), w * zoom, h * zoom)
    if panels:
        dx = max(p[1] for p in panels.values()) * PANEL_MARGIN
        dy = max(p[2] for p in panels.values()) * PANEL_MARGIN * 1.3
    else:
        dx = dy = 0.6
    if sg.layered:
        pos = layered_layout(nodes, edges, dx=dx, dy=dy)
    else:
        pos = _ring(nodes, dx)
    scene = graph_scene(
        nodes,
        edges,
        pos,
        node_style={v: _NODE_STYLES[sg.kinds[v]] for v in nodes},
        edge_colors={e: _EDGE_COLORS[k] for e, k in sg.edge_kinds.items()},
        directed=sg.graph.is_directed(),
        panels=panels,
        title=title,
    )
    for v in nodes:
        if v not in panels and sg.kinds[v] != "token":
            scene.shapes.append(Label(scene.points[nodes.index(v)], (8.0, 8.0), _number(v), size=8.0))
    return scene


def _ring(nodes, dx: float) -> dict:
    import math

    n = len(nodes)
    if n == 1:
        return {nodes[0]: (0.0, 0.0)}
    radius = max(dx * n / (2 * math.pi), dx * 0.6)
    angles = [math.pi / 2 - 2 * math.pi * k / n for k in range(n)]  # clockwise from the top
    return {v: (radius * math.cos(t), radius * math.sin(t)) for v, t in zip(nodes, angles, strict=True)}


__all__ = [
    "system_graph_scene",
    "event_scene",
    "hypergraph_drawing_args",
    "hypergraph_embedding",
    "hypergraph_picture",
    "hypergraph_positions",
    "rule_scene",
]
