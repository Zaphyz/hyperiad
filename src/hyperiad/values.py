"""Runtime values that are not plain Python data, and how any value is written as text.

Evaluation turns trees into values: numbers (exact where possible), strings, lists,
dicts, arrays (:class:`hypermatrix.ArrayObject`), SymPy expressions for symbolic
results, and the small classes below.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import sympy
from frontier.hypergraphs import (
    Edge,
    Event,
    FormalV,
    Hypergraph,
    HypergraphRule,
    LabelToken,
    Term,
    Verbatim,
    Vertex,
    parse_io,
)
from frontier.hypergraphs.causality import CausalEdge, EventRecord, RewritingSystem, SystemGraph
from hypermatrix import ArrayObject, FieldElement, FiniteField, Hypermatrix

from hyperiad.errors import HyperiadError
from hyperiad.syntax import unparse


def event_property(ev: Event, key: str) -> Any:
    """An event's entry under the paclet's key, with positions counted from 1 as Hyperiad counts."""
    value = ev[key]
    if key == "MatchEdgePositions":
        return [p + 1 for p in value]
    if key == "EdgeOrigins":
        return [None if o is None else o + 1 for o in value]
    if key in ("MatchVertices", "MatchEdges", "NewVertices", "NewEdges", "DeletedVertices"):
        return list(value)
    return value


def node_value(v: Any) -> Any:
    """A graph vertex as Hyperiad shows it: a number, or [kind, number] such as ["Event", 3]."""
    if isinstance(v, tuple):
        return [node_value(x) for x in v]
    return v


GRAPH_PROPERTIES = ("VertexList", "EdgeList", "VertexCount", "EdgeCount", "AcyclicQ", "Name")


def graph_property(g: SystemGraph, key: Any) -> Any:
    if key == "VertexList":
        return [node_value(v) for v in g.vertices]
    if key == "EdgeList":
        if g.graph.is_directed():
            return [RuleValue(node_value(a), node_value(b)) for a, b in g.edges]
        return [[node_value(a), node_value(b)] for a, b in g.edges]
    if key == "VertexCount":
        return g.vertex_count
    if key == "EdgeCount":
        return g.edge_count
    if key == "AcyclicQ":
        return g.acyclic()
    if key == "Name":
        return g.name
    raise HyperiadError(f"a graph has the properties {', '.join(GRAPH_PROPERTIES)}")


def causal_edge_value(ce: CausalEdge) -> dict:
    return {k: ce[k] for k in CausalEdge.KEYS}


def match_value(m) -> dict:
    """A match as the paclet reports it, positions from 1."""
    return {
        "MatchEdgePositions": [p + 1 for p in m.edge_positions],
        "RuleVertexMap": dict(m.vertex_map),
        "LabelBindings": dict(m.label_bindings),
    }


@dataclass(frozen=True)
class RuleValue:
    lhs: Any
    rhs: Any


@dataclass(frozen=True)
class OptionValue:
    name: str
    value: Any


@dataclass(eq=False)
class FunctionValue:
    params: tuple[str, ...]
    body: Any  # an expression tree
    scope: Any  # the defining scope (closure)
    name: str | None = None


@dataclass(frozen=True)
class Graphic:
    """A picture produced by a drawing function."""

    svg: str
    title: str = ""


@dataclass(frozen=True)
class Frames:
    """Pictures shown one at a time with controls: an animation (``play``) or a stepper.

    ``labels`` name the frames (shown beside the controls); ``interval`` is in milliseconds.
    """

    svgs: tuple[str, ...]
    labels: tuple[str, ...] = ()
    title: str = ""
    play: bool = True
    interval: int = 900


def sym_str(x: sympy.Basic) -> str:
    return sympy.sstr(x).replace("**", "^")


def format_value(v: Any, limit: int = 4000) -> str:
    """Text for a value, in Hyperiad syntax where there is one."""
    text = _fmt(v)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _fmt(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, Fraction):
        return f"{v.numerator}/{v.denominator}"
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, complex):
        return f"{_fmt(v.real)} + {_fmt(v.imag)}*I"
    if isinstance(v, str):
        return unparse(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    if isinstance(v, frozenset):  # an edge as a point of an incidence space
        from frontier.hypergraphs.core import order_key

        return "{" + ", ".join(_fmt(x) for x in sorted(v, key=order_key)) + "}"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_fmt(k)}: {_fmt(x)}" for k, x in v.items()) + "}"
    if isinstance(v, Hypermatrix):
        return f"Hypermatrix({_fmt(list(v.arrays))})"
    if isinstance(v, ArrayObject):
        sym = "" if v.symmetry == "Rigid" else f', "{v.symmetry}"'
        return f"ArrayObject({_fmt(v.entries)}{sym})"
    if isinstance(v, Term):
        return f"{v.head}({', '.join(_fmt(a) for a in v.args)})"
    if isinstance(v, Verbatim):
        return f"Verbatim({_fmt(v.value)})"
    if isinstance(v, FormalV):
        return f"FormalV({v.index})"
    if isinstance(v, HypergraphRule):
        return f"HypergraphRule({_fmt(v.input)}, {_fmt(v.output)})"
    if isinstance(v, RewritingSystem):
        opts = "".join(f", {k} = {_fmt(x)}" for k, x in v.options.items())
        return f"HypergraphRewritingSystem({_fmt(list(v.rules))}, {_fmt(v.initial)}{opts})"
    if isinstance(v, EventRecord):
        where = f"state {v.input_state} → {v.output_state}"
        return f"<event {v.number}: generation {v.generation}, rule {v.rule_index}, {where}>"
    if isinstance(v, SystemGraph):
        return f"<{v.name}: {v.vertex_count} vertices, {v.edge_count} edges>"
    if isinstance(v, Event):
        made = ", ".join(_fmt(e) for e in v.new_edges)
        return f"<event: edges {_fmt(event_property(v, 'MatchEdgePositions'))} rewritten to [{made}]>"
    if isinstance(v, LabelToken):
        return f"LabelToken({v.index})"
    if isinstance(v, Vertex):
        return f"Vertex({_fmt(v.id)})" if v.label is None else f"Vertex({_fmt(v.id)}, {_fmt(v.label)})"
    if isinstance(v, Edge) and len(v.block_ids) == 2:  # input–output: written as its two blocks and their types
        s_in, _, s_out, _ = parse_io(v.symmetry)
        tail = "" if v.label is None else f", {_fmt(v.label)}"
        return f"Edge({_fmt([list(b) for b in v.block_ids])}, {_fmt([s_in, s_out])}{tail})"
    if isinstance(v, Edge):
        tail = "" if v.symmetry == "Directed" and v.label is None else f", {_fmt(v.symmetry)}"
        tail += "" if v.label is None else f", {_fmt(v.label)}"
        return f"Edge({_fmt(list(v.ids))}{tail})"
    if isinstance(v, Hypergraph):
        isolated = [x for x in v.vertices if all(x.id not in e.ids for e in v.edges)]
        if isolated or any(x.label is not None for x in v.vertices):
            return f"Hypergraph({_fmt(list(v.vertices))}, {_fmt(list(v.edges))})"
        return f"Hypergraph({_fmt(list(v.edges))})"
    if isinstance(v, FieldElement):
        return f"FiniteField({v.field.p})({v.value})"
    if isinstance(v, FiniteField):
        return f"FiniteField({v.p})"
    if isinstance(v, RuleValue):
        return f"{_fmt(v.lhs)} -> {_fmt(v.rhs)}"
    if isinstance(v, OptionValue):
        return f"{v.name} = {_fmt(v.value)}"
    if isinstance(v, FunctionValue):
        params = v.params[0] if len(v.params) == 1 else "(" + ", ".join(v.params) + ")"
        return f"{params} => {unparse(v.body)}"
    if isinstance(v, Graphic):
        return f"Graphic({v.title!r})" if v.title else "Graphic(...)"
    if isinstance(v, Frames):
        return f"<{len(v.svgs)} frames{': ' + v.title if v.title else ''}>"
    if isinstance(v, sympy.Basic):
        return sym_str(v)
    return repr(v)
