"""Infrageometry toolkit.

Definitions register under keys ``infrageometry.*`` in :data:`frontier.REGISTRY` when
their module is imported. Import every definitions module from here so that
loading the toolkit registers all of them.

The foundations (theme ``hypergraph-geometry``, question q01): a spatial hypergraph is space;
its incidence space gives the topology; departures vertex → edge are tangent vectors; motions,
maps, scalar functions, vector fields and local models are built from hypergraphs alone.

* :mod:`infrageometry.space` -- spatial hypergraphs, the incidence space, neighbourhoods, subspaces
* :mod:`infrageometry.motion` -- tangent vectors, the motion graph, motions, homotopy, motion as rewriting
* :mod:`infrageometry.maps` -- maps (vertices and edges together) and their pushforwards
* :mod:`infrageometry.scalar` -- the scalar line and scalar functions
* :mod:`infrageometry.fields` -- vector fields, transits, flows
* :mod:`infrageometry.models` -- pointed balls, charts, local models, example spaces
* :mod:`infrageometry.paths` -- step paths, area homotopy (primary), homotopy of motions, the fundamental group
* :mod:`infrageometry.edgefields` -- fields as typed edges (vector fields of arcs, scalar fields of unary
  edges) and their rewriting
* :mod:`infrageometry.devices` -- real-valued devices kept outside the core
"""

DIRECTION = "infrageometry"

from infrageometry import devices, edgefields, fields, maps, models, motion, paths, scalar, space  # noqa: E402,F401
from infrageometry.devices import ball_growth, growth_dimension  # noqa: E402
from infrageometry.edgefields import (  # noqa: E402
    FieldEdge,
    FieldRule,
    FieldState,
    arc_field,
    bracket,
    bracket_rules,
    cancel,
    closure_rules,
    compare,
    composition_rule,
    direction_field,
    saturate,
    scalar_field,
    scalar_field_of,
    vector_field,
)
from infrageometry.fields import Flow, Transit, VectorField, gradient_flow  # noqa: E402
from infrageometry.maps import SpatialMap, all_maps, curve, identity_map, motion_of_curve  # noqa: E402
from infrageometry.models import (  # noqa: E402
    atlas,
    ball_type,
    chart,
    cycle,
    grid,
    is_combinatorial_surface,
    klein_grid,
    locally_modelled,
    octahedron,
    path,
    pointed_ball,
    torus7,
    torus_grid,
    triangle_example,
    with_unary,
)
from infrageometry.motion import (  # noqa: E402
    Motion,
    TangentVector,
    geodesic,
    incidence_group_generators,
    incidence_homotopic,
    interval,
    motion_graph,
    reduce_motion,
    reduction_steps,
    tangent_bundle,
    tangent_space,
    token_rewriting_system,
    token_state,
)
from infrageometry.paths import (  # noqa: E402
    Barrier,
    StepPath,
    find_barrier,
    find_derivation,
    fundamental_group,
    homotopic,
    path_of_motion,
    reduce_path,
    rewrites,
    triangulated_annulus,
    triangulated_disk,
)
from infrageometry.scalar import DOWN, STAY, UP, ScalarFunction, distance_function, scalar_line  # noqa: E402
from infrageometry.space import IncidenceSpace, Region, SpatialHypergraph, link_dimension  # noqa: E402

__all__ = [
    "Barrier",
    "FieldEdge",
    "FieldRule",
    "FieldState",
    "StepPath",
    "arc_field",
    "bracket",
    "bracket_rules",
    "cancel",
    "closure_rules",
    "compare",
    "composition_rule",
    "fundamental_group",
    "incidence_group_generators",
    "incidence_homotopic",
    "scalar_field",
    "scalar_field_of",
    "direction_field",
    "find_barrier",
    "find_derivation",
    "path_of_motion",
    "reduce_path",
    "rewrites",
    "saturate",
    "triangulated_annulus",
    "triangulated_disk",
    "vector_field",
    "DOWN",
    "STAY",
    "UP",
    "Flow",
    "IncidenceSpace",
    "Motion",
    "Region",
    "ScalarFunction",
    "SpatialHypergraph",
    "SpatialMap",
    "TangentVector",
    "Transit",
    "VectorField",
    "all_maps",
    "atlas",
    "ball_growth",
    "ball_type",
    "chart",
    "curve",
    "cycle",
    "distance_function",
    "geodesic",
    "gradient_flow",
    "grid",
    "growth_dimension",
    "homotopic",
    "identity_map",
    "interval",
    "is_combinatorial_surface",
    "klein_grid",
    "link_dimension",
    "locally_modelled",
    "motion_graph",
    "motion_of_curve",
    "octahedron",
    "path",
    "pointed_ball",
    "reduce_motion",
    "reduction_steps",
    "scalar_line",
    "tangent_bundle",
    "tangent_space",
    "token_rewriting_system",
    "token_state",
    "torus7",
    "torus_grid",
    "triangle_example",
    "with_unary",
]
