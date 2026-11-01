"""Turning values into notebook outputs.

An output is a dict with a ``type`` the front end knows how to show:

* ``svg``      -- a picture (``data``: SVG markup), with an optional ``caption``
* ``latex``    -- typeset mathematics (``data``: LaTeX), with ``text`` as plain fallback
* ``text``     -- plain text in Hyperiad syntax
* ``markdown`` -- prose with ``$...$`` math (used for definitions from the registry)
* ``error``    -- an error message; ``kind`` is ``error`` or ``too_expensive``
* ``frames``   -- pictures shown one at a time (``frames``: SVG markup, ``labels``), with
  play / step controls: an animation when ``autoplay``, a stepper otherwise

Toolkits register how their own values display with :func:`register_display`.

Following the reference paclet's principle, objects are shown drawn: an array value
displays as its picture, with a one-line description to check it against.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import sympy
from frontier.hypergraphs import Event, Hypergraph, HypergraphRule, parse_io
from frontier.hypergraphs.causality import EventRecord, RewritingSystem, SystemGraph
from frontier.hypergraphs.svg import event_svg, hypergraph_svg, rule_svg, system_graph_svg
from hypermatrix import ArrayError, ArrayObject, FiniteField, Hypermatrix, array_svg

from hyperiad.values import Frames, Graphic, event_property, format_value

_DISPLAYERS: list[tuple[type, Callable[[Any], list[dict]]]] = []


def register_display(cls: type, show: Callable[[Any], list[dict]]) -> None:
    """Values of ``cls`` display as ``show(value)``, a list of outputs; checked before the built-in kinds."""
    _DISPLAYERS.insert(0, (cls, show))


_VIEWBOX = re.compile(r'viewBox="([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+)"')


def steady_frames(svgs: list[str]) -> list[str]:
    """Give every frame the union of their boxes, so the picture does not move between frames."""
    boxes = [tuple(map(float, m.groups())) for s in svgs if (m := _VIEWBOX.search(s))]
    if len(boxes) != len(svgs) or not svgs:
        return svgs
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    out = []
    for s, (bx, by, bw, bh) in zip(svgs, boxes, strict=True):
        m = re.search(r'width="([\d.]+)" height="([\d.]+)"', s)
        px = float(m.group(1)) / bw if m else 1.75
        s = _VIEWBOX.sub(f'viewBox="{x0:.2f} {y0:.2f} {x1 - x0:.2f} {y1 - y0:.2f}"', s, count=1)
        if m:
            s = s.replace(m.group(0), f'width="{(x1 - x0) * px:.0f}" height="{(y1 - y0) * px:.0f}"', 1)
        s = s.replace(
            f'<rect x="{bx:.2f}" y="{by:.2f}" width="{bw:.2f}" height="{bh:.2f}"',
            f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{x1 - x0:.2f}" height="{y1 - y0:.2f}"',
            1,
        )
        out.append(s)
    return out


def _registered(value: Any) -> Callable[[Any], list[dict]] | None:
    for cls, show in _DISPLAYERS:
        if isinstance(value, cls):
            return show
    return None


def frames_output(f: Frames) -> dict:
    return {
        "type": "frames",
        "frames": steady_frames(list(f.svgs)),
        "labels": list(f.labels),
        "caption": f.title,
        "autoplay": f.play,
        "interval": f.interval,
    }


def value_picture(value: Any) -> tuple[str, str]:
    """The picture a value displays as (the first, for a value shown as several), with its caption."""
    for out in display(value):
        if out["type"] == "svg":
            return out["data"], out.get("caption", "")
        if out["type"] == "frames":
            return out["frames"][0], out.get("caption", "")
    raise ValueError(f"{format_value(value, 60)} has no picture")


def value_svg(value: Any) -> str:
    return value_picture(value)[0]


def error_output(message: str, kind: str = "error") -> dict:
    return {"type": "error", "kind": kind, "data": message}


def describe_array(a: ArrayObject) -> str:
    domain = f"GF({a.domain.p})" if isinstance(a.domain, FiniteField) else a.domain
    dims = "×".join(map(str, a.dims)) if a.dims else "scalar"
    sym = "" if a.symmetry == "Rigid" else f", {a.symmetry.lower()}"
    return f"{dims} {domain} array of order {a.order}{sym}"


# A hypergraph larger than this displays as a summary, as in the reference paclet
# ($HypergraphSummaryThresholds); HypergraphPlot draws it anyway.
MAX_VERTICES, MAX_EDGES, MAX_ELEMENTS = 32, 128, 196


def _kind(sym: str) -> str:
    """A symmetry type in a caption: an input–output type as its block types, e.g. ``unordered→cyclic``."""
    io = parse_io(sym)
    return sym if io is None else f"{io[0]}→{io[2]}"


def describe_hypergraph(h: Hypergraph) -> str:
    arities = sorted(set(h["Arity"]))
    kinds = sorted({_kind(s) for s in h["EdgeSymmetries"]})
    nv, ne = h["VertexCount"], h["EdgeCount"]
    parts = [f"{nv} vert{'ex' if nv == 1 else 'ices'}", f"{ne} edge{'' if ne == 1 else 's'}"]
    if arities:
        parts.append("arity " + ", ".join(map(str, arities)))
    if kinds:
        parts.append(", ".join(k.lower() for k in kinds))
    if not h.is_simple():
        parts.append("repeated edges")
    return " · ".join(parts)


def large_hypergraph(h: Hypergraph) -> bool:
    nv, ne = h["VertexCount"], h["EdgeCount"]
    return nv > MAX_VERTICES or ne > MAX_EDGES or nv + ne > MAX_ELEMENTS


def _count(n: int, word: str) -> str:
    plural = "vertices" if word == "vertex" else word + "s"
    return f"{n} {word if n == 1 else plural}"


def describe_rule(r: HypergraphRule) -> str:
    sides = [f"{_count(len(h.edges), 'edge')}, {_count(len(h.vertices), 'vertex')}" for h in (r.input, r.output)]
    parts = [f"rule: {sides[0]} → {sides[1]}"]
    if r.new_vertices:
        parts.append(f"creates {format_value(r.new_vertices)}")
    if r.deleted_vertices:
        parts.append(f"destroys {format_value(r.deleted_vertices)}")
    return " · ".join(parts)


def describe_event(ev: Event) -> str:
    parts = [f"event: edges {format_value(event_property(ev, 'MatchEdgePositions'))} rewritten"]
    if ev.new_vertices:
        parts.append(f"creates {format_value(ev.new_vertices)}")
    if ev.deleted_vertices:
        parts.append(f"destroys {format_value(ev.deleted_vertices)}")
    if ev.label_bindings:
        parts.append(f"binds {format_value(ev.label_bindings)}")
    return " · ".join(parts)


GRAPH_NAMES = {
    "CausalGraph": "causal graph",
    "EvolutionGraph": "evolution graph",
    "StatesGraph": "states graph",
    "BranchialGraph": "branchial graph",
    "EvolutionCausalGraph": "evolution-causal graph",
    "TokenEventGraph": "token-event graph",
}
MAX_GRAPH_VERTICES = 400  # a larger graph displays as a summary


def describe_graph(g: SystemGraph) -> str:
    parts = [GRAPH_NAMES.get(g.name, g.name), _count(g.vertex_count, "vertex"), _count(g.edge_count, "edge")]
    if g.options.get("CausalDependence") and g.name in ("CausalGraph", "EvolutionCausalGraph", "TokenEventGraph"):
        parts.append(f"dependence by {str(g.options['CausalDependence']).lower()}")
    return " · ".join(parts)


def describe_record(r: EventRecord) -> str:
    return (
        f"event {r.number} · generation {r.generation} · rule {r.rule_index} · state {r.input_state} → {r.output_state}"
    )


def describe_system(s: RewritingSystem) -> str:
    init = s.initial
    return (
        f"rewriting system · {_count(len(s.rules), 'rule')} · initial state {_count(len(init.vertices), 'vertex')}, "
        f"{_count(len(init.edges), 'edge')}"
    )


def _too_large(x: Any) -> bool:
    if isinstance(x, EventRecord):
        x = x.data
    graphs = (
        [x]
        if isinstance(x, Hypergraph)
        else [x.input, x.output]
        if isinstance(x, HypergraphRule)
        else [x.input, x.hypergraph]
    )
    return any(large_hypergraph(h) for h in graphs)


def _picture(x: Any) -> dict:
    if isinstance(x, EventRecord):
        return {"type": "svg", "data": event_svg(x.data), "caption": describe_record(x)}
    if isinstance(x, Hypergraph):
        return {"type": "svg", "data": hypergraph_svg(x), "caption": describe_hypergraph(x)}
    if isinstance(x, HypergraphRule):
        return {"type": "svg", "data": rule_svg(x), "caption": describe_rule(x)}
    return {"type": "svg", "data": event_svg(x), "caption": describe_event(x)}


MAX_PICTURES = 12  # a list of hypergraphs, rules or events shows this many drawn


def display(value: Any) -> list[dict]:
    if value is None:
        return []
    show = _registered(value)
    if show is not None:
        return show(value)
    if isinstance(value, list) and 0 < len(value) <= MAX_PICTURES and all(_registered(x) for x in value):
        outs = [_registered(x)(x) for x in value]  # a short list of toolkit values: one picture each
        if all(len(o) == 1 and o[0]["type"] == "svg" for o in outs):
            pictures = [o[0] for o in outs]
            for k, out in enumerate(pictures, 1):
                out["caption"] = f"[{k}] {out.get('caption', '')}"
            return pictures
    if isinstance(value, Frames):
        return [frames_output(value)]
    drawable = (Hypergraph, HypergraphRule, Event, EventRecord)
    if isinstance(value, SystemGraph):
        if value.vertex_count > MAX_GRAPH_VERTICES:
            return [{"type": "text", "data": describe_graph(value)}]
        return [{"type": "svg", "data": system_graph_svg(value), "caption": describe_graph(value)}]
    if isinstance(value, RewritingSystem):
        if len(value.rules) <= MAX_PICTURES and not any(_too_large(r) for r in value.rules):
            outs = [_picture(r) for r in value.rules]
            if not large_hypergraph(value.initial):
                outs.append({"type": "svg", "data": hypergraph_svg(value.initial), "caption": "initial state"})
            outs[0]["caption"] = describe_system(value) + " · " + outs[0]["caption"]
            return outs
        return [{"type": "text", "data": describe_system(value)}]
    if isinstance(value, (HypergraphRule, Event, EventRecord)):
        if _too_large(value):
            return [{"type": "text", "data": format_value(value, 600)}]
        return [_picture(value)]
    if (
        isinstance(value, list)
        and 0 < len(value) <= MAX_PICTURES
        and all(isinstance(x, drawable) and not _too_large(x) for x in value)
    ):
        outs = [_picture(x) for x in value]
        for k, out in enumerate(outs, 1):
            out["caption"] = f"[{k}] {out['caption']}"
        return outs
    if isinstance(value, list) and 0 < len(value) <= MAX_PICTURES and all(isinstance(x, Graphic) for x in value):
        return [{"type": "svg", "data": g.svg, "caption": f"[{k}] {g.title}".strip()} for k, g in enumerate(value, 1)]
    if isinstance(value, Hypergraph):
        if large_hypergraph(value):
            return [{"type": "text", "data": f"Hypergraph: {describe_hypergraph(value)} (draw it with HypergraphPlot)"}]
        return [{"type": "svg", "data": hypergraph_svg(value), "caption": describe_hypergraph(value)}]
    if isinstance(value, Hypermatrix):
        head = f"hypermatrix of {_count(len(value), 'array')}"
        if not value.arrays:
            return [{"type": "text", "data": f"Hypermatrix([]) · {head}"}]
        outs = []
        for k, a in enumerate(value.arrays, 1):
            try:
                outs.append({"type": "svg", "data": array_svg(a), "caption": f"[{k}] {describe_array(a)}"})
            except ArrayError:
                outs.append({"type": "text", "data": f"[{k}] {describe_array(a)}, too large to draw"})
        outs[0]["caption"] = f"{head} · {outs[0].get('caption', '')}"
        return outs
    if isinstance(value, ArrayObject):
        try:
            return [{"type": "svg", "data": array_svg(value), "caption": describe_array(value)}]
        except ArrayError:  # too large to draw legibly
            return [{"type": "text", "data": f"{describe_array(value)}, too large to draw\n{format_value(value, 600)}"}]
    if isinstance(value, Graphic):
        return [{"type": "svg", "data": value.svg, "caption": value.title}]
    if isinstance(value, dict) and value.get("__display__") == "concept":
        return [{"type": "markdown", "data": value["markdown"]}]
    if isinstance(value, sympy.Basic) and not (value.is_Number and value.is_real):
        return [{"type": "latex", "data": sympy.latex(value), "text": format_value(value)}]
    return [{"type": "text", "data": format_value(value)}]
