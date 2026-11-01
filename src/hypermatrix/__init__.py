"""Hypermatrix algebra toolkit.

Definitions register under keys ``hypermatrix.*`` in :data:`frontier.REGISTRY` when
their module is imported. Import every definitions module from here so that
loading the toolkit registers all of them.
"""

from hypermatrix.adjacency import adjacency_hypermatrix, adjacency_size, hypermatrix_edges
from hypermatrix.algebra import (
    array_add,
    array_multiply,
    array_multiply_cost,
    array_times,
    identity_array,
    one_array,
    parse_index_spec,
    partial_identity_array,
    zero_array,
)
from hypermatrix.arrays import ArrayError, ArrayObject, as_array, generate_symbolic_array
from hypermatrix.composition import (
    composition_adjacency,
    composition_adjacency_product,
    composition_plex_diagram,
)
from hypermatrix.fields import FieldElement, FiniteField
from hypermatrix.graphics import array_nesting, array_svg, hypermatrix_svgs
from hypermatrix.hypermatrix import Hypermatrix
from hypermatrix.plex import (
    PLEX_MARK,
    plex_contracted,
    plex_diagram,
    plex_events,
    plex_free,
    plex_names,
    plex_product,
    plex_rule,
)
from hypermatrix.plex_enumeration import PlexShape, enumerate_plex_products
from hypermatrix.plex_identities import (
    diagram_key,
    plex_identity_classes,
    plex_normal_forms,
    plex_term_diagram,
    plex_term_key,
    plex_term_value,
    plex_terms,
)

DIRECTION = "hypermatrix"

__all__ = [
    "DIRECTION",
    "PLEX_MARK",
    "PlexShape",
    "ArrayError",
    "ArrayObject",
    "FieldElement",
    "FiniteField",
    "Hypermatrix",
    "adjacency_hypermatrix",
    "adjacency_size",
    "composition_adjacency",
    "composition_adjacency_product",
    "composition_plex_diagram",
    "array_add",
    "array_multiply",
    "array_multiply_cost",
    "array_nesting",
    "array_svg",
    "array_times",
    "as_array",
    "generate_symbolic_array",
    "hypermatrix_edges",
    "hypermatrix_svgs",
    "identity_array",
    "one_array",
    "enumerate_plex_products",
    "parse_index_spec",
    "diagram_key",
    "plex_contracted",
    "plex_diagram",
    "plex_events",
    "plex_free",
    "plex_identity_classes",
    "plex_names",
    "plex_normal_forms",
    "plex_product",
    "plex_rule",
    "plex_term_diagram",
    "plex_term_key",
    "plex_term_value",
    "plex_terms",
    "partial_identity_array",
    "zero_array",
]
