"""Hypergraphs, shared by every direction: the object model and its drawing.

* :mod:`frontier.hypergraphs.core` -- :class:`Vertex`, :class:`Edge`, :class:`Hypergraph`
  (ported from the reference paclet's ``src/Hypergraph.wl``), with the input–output symmetry type
* :mod:`frontier.hypergraphs.canon` -- canonical forms and isomorphism
* :mod:`frontier.hypergraphs.labels` -- labels as patterns, and labels built from labels
* :mod:`frontier.hypergraphs.rewriting` -- rewrite rules, matches and events
* :mod:`frontier.hypergraphs.composition` -- one pass, composition products, saturation
* :mod:`frontier.hypergraphs.geometry` -- layout and the shapes of the drawing
* :mod:`frontier.hypergraphs.svg` -- the drawing as SVG, for Hyperiad and web pages

The drawing has one geometry and two renderers: :func:`frontier.viz.draw_hypergraph`
(matplotlib, for session figures) and :func:`hypergraph_svg` here. Neither module
needs the other's library, so the browser kernel loads no matplotlib.
"""

from frontier.hypergraphs.canon import (
    LabelToken,
    apply_vertex_map,
    canonical_data,
    canonical_hypergraph,
    find_isomorphism,
    isomorphic,
)
from frontier.hypergraphs.composition import (
    MULTIPLICITIES,
    Saturation,
    hypergraph_compose,
    hypergraph_saturate,
    one_pass,
    parallel_edges,
)
from frontier.hypergraphs.core import (
    IO_TYPE,
    SYMMETRY_TYPES,
    Edge,
    Hypergraph,
    HypergraphError,
    Vertex,
    io_symmetry,
    parse_io,
    simple_hypergraph_q,
    symmetry_orderings,
    symmetry_size,
)
from frontier.hypergraphs.labels import CONTRACTED_MARK, Term, Verbatim, is_label_variable, label_text, substitute
from frontier.hypergraphs.rewriting import (
    Event,
    FormalV,
    HypergraphRule,
    Match,
    apply_match,
    apply_rule,
    rule_matches,
)

__all__ = [
    "MULTIPLICITIES",
    "Saturation",
    "hypergraph_compose",
    "hypergraph_saturate",
    "one_pass",
    "parallel_edges",
    "CONTRACTED_MARK",
    "Event",
    "FormalV",
    "HypergraphRule",
    "Match",
    "Term",
    "Verbatim",
    "apply_match",
    "apply_rule",
    "is_label_variable",
    "label_text",
    "rule_matches",
    "substitute",
    "LabelToken",
    "apply_vertex_map",
    "canonical_data",
    "canonical_hypergraph",
    "find_isomorphism",
    "isomorphic",
    "IO_TYPE",
    "SYMMETRY_TYPES",
    "io_symmetry",
    "parse_io",
    "symmetry_orderings",
    "symmetry_size",
    "Edge",
    "Hypergraph",
    "HypergraphError",
    "Vertex",
    "simple_hypergraph_q",
]
