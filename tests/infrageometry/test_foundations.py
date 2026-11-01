import itertools

import pytest
from frontier.hypergraphs import Hypergraph

import infrageometry as ig
from infrageometry import (
    Flow,
    IncidenceSpace,
    Motion,
    Region,
    ScalarFunction,
    SpatialHypergraph,
    SpatialMap,
    TangentVector,
    Transit,
    VectorField,
)

H = ig.triangle_example()


def test_space_is_simple_and_undirected():
    assert SpatialHypergraph([[2, 1], [1, 2]]) == SpatialHypergraph([[1, 2]])
    with pytest.raises(ValueError):
        SpatialHypergraph([[]])
    with pytest.raises(ValueError):
        SpatialHypergraph.from_hypergraph(Hypergraph([[1, 1, 2]]))
    with pytest.raises(ValueError):
        SpatialHypergraph([[frozenset({1})]])
    assert SpatialHypergraph.from_hypergraph(H.to_hypergraph()) == H
    assert SpatialHypergraph([[1], [1, 2]]).star(1) == {frozenset({1}), frozenset({1, 2})}


def test_neighbourhoods_links_and_dimension():
    assert H.star(2) == {frozenset({1, 2}), frozenset({1, 2, 3}), frozenset({2, 6})}
    assert H.neighbourhood(2) == {1, 2, 3, 6}
    assert H.link(2) == SpatialHypergraph([[1], [1, 3], [6]])
    assert H.ball(1, 2) == {1, 2, 3, 4, 6} and H.sphere(1, 3) == {5}
    assert H.distance(1, 5) == 3 and SpatialHypergraph([[1], [2]]).distance(1, 2) is None
    assert ig.link_dimension(H) == 2 and ig.link_dimension(ig.path(3), 1) == 1
    assert ig.link_dimension(ig.grid(3, 3)) == 1 and ig.link_dimension(ig.octahedron()) == 2
    assert ig.link_dimension(SpatialHypergraph([[1]])) == 0
    assert ig.ball_growth(ig.grid(9, 9), (4, 4), 3) == [1, 5, 13, 25]


def test_incidence_space():
    x = IncidenceSpace(H)
    u2 = x.minimal_open(2)
    assert x.is_open(u2) and not x.is_closed(u2) and x.closure(u2).vertices == H.neighbourhood(2)
    assert x.frontier(x.open_hull(Region(frozenset({1})))) == Region(frozenset({2, 3}))
    s = Region.of([1, 2, 3, frozenset({1, 2}), frozenset({1, 2, 3})])
    assert x.is_closed(s) and x.subspace(s) == H.induced({1, 2, 3}) and x.frontier(s) == Region(frozenset({2, 3}))
    inscribed = Region.of([1, 2, frozenset({1, 2, 3})])  # an edge seen through two of its vertices
    assert x.subspace(inscribed) == SpatialHypergraph([[1, 2]]) and not x.is_closed(inscribed)
    assert x.is_connected(inscribed) and not x.is_connected(Region.of([1, 5]))
    assert x.first_betti() == 2 and IncidenceSpace(ig.cycle(5)).first_betti() == 1


def test_tangent_vectors_and_motion_graph():
    assert [t.edge for t in ig.tangent_space(H, 2)] == [frozenset({1, 2}), frozenset({2, 6}), frozenset({1, 2, 3})]
    assert TangentVector(1, frozenset({1})).stationary
    with pytest.raises(ValueError):
        TangentVector(4, frozenset({1, 2}))
    assert len(ig.tangent_bundle(H)) == H.incidences() == 12
    g = ig.motion_graph(H)
    assert g.number_of_nodes() == 11 and g.number_of_edges() == 24


def test_motions_geodesics_and_curves():
    m = Motion(H, [1, {1, 2, 3}, 3, {3, 4}, 4])
    assert len(m) == 4 and m.steps() == [(1, frozenset({1, 2, 3}), 3), (3, frozenset({3, 4}), 4)]
    assert m.velocity(2) == TangentVector(3, frozenset({3, 4})) and m.incoming(2) == TangentVector(
        3, frozenset({1, 2, 3})
    )
    assert m.velocity(1) is None and m.is_proper()
    assert not Motion(H, [1, {1, 2}, 1]).is_proper()
    with pytest.raises(ValueError):
        Motion(H, [1, {3, 4}, 4])
    with pytest.raises(ValueError):
        Motion(H, [1, 2])
    for v, w in itertools.product(H.vertex_list(), repeat=2):
        assert len(ig.geodesic(H, v, w).steps()) == H.distance(v, w)
    assert ig.geodesic(H, 1, frozenset({4, 5, 6})).points[-2] in (4, 6)
    c = ig.curve(m)
    assert c.source == ig.interval(2) and ig.motion_of_curve(c) == m
    assert c.pushforward(TangentVector(1, frozenset({1, 2}))) == m.velocity(2)


def test_homotopy_by_rewriting():
    m = Motion(H, [1, {1, 2}, 2, {1, 2, 3}, 3, {1, 2, 3}, 2, {1, 2}, 1])
    steps = ig.reduction_steps(m)
    assert steps[-1] == ig.reduce_motion(m) == Motion(H, [1])
    assert all(len(a) - len(b) == 2 for a, b in itertools.pairwise(steps))
    loop1 = Motion(H, [1, {1, 2}, 2, {1, 2, 3}, 1])
    assert ig.reduce_motion(loop1) == loop1 and not ig.incidence_homotopic(loop1, Motion(H, [1]))
    assert ig.incidence_homotopic(loop1.then(loop1.reverse()), Motion(H, [1]))
    assert ig.homotopic(loop1, Motion(H, [1]))  # derived from area homotopy: the area {1, 2, 3} fills it
    gens = ig.incidence_group_generators(H, 1)
    assert len(gens) == IncidenceSpace(H).first_betti() and all(g.is_closed() for g in gens)


def test_maps_and_pushforward():
    c6, c3 = ig.cycle(6), ig.cycle(3)
    cover = SpatialMap(c6, c3, {i: i % 3 for i in range(6)})
    assert cover.is_local_isomorphism() and not cover.is_embedding()
    tri = SpatialHypergraph([[0, 1, 2]])
    fold = SpatialMap(c3, tri, {0: 0, 1: 1, 2: 2})
    assert fold.is_submersion() and not fold.is_immersion() and fold.is_injective() is False
    inscribe = SpatialMap(SpatialHypergraph([[0, 1]]), tri, {0: 0, 1: 1})
    assert inscribe.is_embedding() and inscribe.edge_map[frozenset({0, 1})] == frozenset({0, 1, 2})
    with pytest.raises(ValueError):
        SpatialMap(SpatialHypergraph([[0, 1]]), ig.cycle(4), {0: 0, 1: 2})  # 0 and 2 share no edge
    line = ig.scalar_line(0, 2)
    with pytest.raises(ValueError):  # {1} lies in three edges of the line: ambiguous only if not itself an edge
        SpatialMap(SpatialHypergraph([[0, 1]]), SpatialHypergraph([[1, 2], [0, 1]]), {0: 1, 1: 1})
    assert SpatialMap(SpatialHypergraph([[0, 1]]), line, {0: 1, 1: 1}).edge_map[frozenset({0, 1})] == {1}
    for f in ig.all_maps(ig.path(2), c3):
        for g in ig.all_maps(c3, tri):
            fg = f.then(g)
            assert all(fg.pushforward(t) == g.pushforward(f.pushforward(t)) for t in ig.tangent_bundle(ig.path(2)))
    assert ig.identity_map(H).is_isomorphism()


def test_scalar_functions():
    f = ig.distance_function(H, 1)
    assert f.critical_points() == {1: "minimum", 5: "maximum"}
    assert f.level_set(2) == Region(frozenset({4, 6})) and f.crossing(1).edges == {frozenset({2, 6}), frozenset({3, 4})}
    assert set(f.differential(2).values()) == {ig.UP, ig.DOWN}
    assert f(frozenset({4, 5, 6})) == {2, 3}
    flat = ScalarFunction(ig.path(1), {0: 5, 1: 5})
    assert flat.kind(0) == "flat" and flat(frozenset({0, 1})) == {5}
    bump = ScalarFunction(ig.path(1), {0: 5, 1: 5}, {(0, 1): [5, 6]})  # an infinitesimal rise between them
    assert bump.kind(0) == "minimum" and bump.kind(1) == "minimum"
    with pytest.raises(ValueError):
        ScalarFunction(ig.path(1), {0: 0, 1: 2})  # values on an edge must lie within one step
    pushed = f.push_motion(ig.geodesic(H, 1, 5))
    assert pushed.points == (0, frozenset({0, 1}), 1, frozenset({1, 2}), 2, frozenset({2, 3}), 3)


def test_flows():
    f = ig.distance_function(H, 1)
    down = ig.gradient_flow(f, ascending=False)
    assert down.integral_curve(5) == Motion(H, [5, {4, 5, 6}, 4, {3, 4}, 3, {1, 2, 3}, 1])
    assert down.fate(1) == "end" and down.integral_curve(1) == Motion(H, [1])
    c = ig.cycle(4)
    spin = Flow(
        VectorField(c, {i: {i, (i + 1) % 4} for i in range(4)}),
        Transit.exit_field(c, {(i, (i + 1) % 4): (i + 1) % 4 for i in range(4)}),
    )
    assert spin.map() == {0: 1, 1: 2, 2: 3, 3: 0} and spin.fate(0) == "cycle"
    rest = ig.with_unary(ig.path(2))
    still = Flow(
        VectorField(rest, {v: {v} for v in rest.vertices}), Transit.exit_field(rest, {(v,): v for v in rest.vertices})
    )
    assert still.fate(1) == "rest" and len(still.integral_curve(1)) == 0


def test_local_models():
    model = ig.grid(11, 11)
    assert ig.locally_modelled(ig.torus_grid(6, 6), model, [(5, 5)], 2)
    assert not ig.locally_modelled(ig.torus_grid(5, 6), model, [(5, 5)], 2)
    assert ig.locally_modelled(ig.klein_grid(6, 6), model, [(5, 5)], 2)
    charts = ig.atlas(ig.torus_grid(4, 4), model, [(5, 5)], 1)
    assert all(c is not None and c.is_isomorphism() for c in charts.values())
    assert ig.is_combinatorial_surface(ig.octahedron()) and ig.is_combinatorial_surface(ig.torus7())
    assert not ig.is_combinatorial_surface(SpatialHypergraph([[1, 2, 3], [1, 3, 4]]))


def test_token_rewriting_states_graph():
    sg = ig.token_rewriting_system(H, 1, MaxStates=4000)("StatesGraph", 6)  # every point is 6 half-steps from 1
    assert sg.vertex_count == len(H.points())
    sym = ig.cycle(4)  # the square: points up to symmetry are one vertex and one edge
    assert ig.token_rewriting_system(sym, 0)("StatesGraph", 3).vertex_count == 2


def test_collapsing_maps():
    squash = SpatialMap(ig.path(2), ig.path(1), {0: 0, 1: 1, 2: 1}, {(1, 2): 1})
    assert squash.collapses() == [frozenset({1, 2})] and squash(frozenset({1, 2})) == 1
    assert squash.pushforward(TangentVector(1, frozenset({1, 2}))) == 1  # the departure collapses to the point
    assert squash.pushforward(TangentVector(1, frozenset({0, 1}))) == TangentVector(1, frozenset({0, 1}))
    assert not squash.is_immersion() and not squash.is_submersion() and not squash.is_injective()  # T_2 collapses
    walk = Motion(ig.path(2), [0, {0, 1}, 1, {1, 2}, 2])
    assert squash.push_motion(walk).points == (0, frozenset({0, 1}), 1)  # the motion pauses at 1
    with pytest.raises(ValueError):
        SpatialMap(ig.path(1), ig.path(1), {0: 0, 1: 1}, {(0, 1): 0})  # its vertices do not all go to 0
    point = SpatialHypergraph([], [0])
    assert SpatialMap(ig.cycle(3), point, dict.fromkeys(range(3), 0)).collapses()  # forced: no edge to go to
    to_line = SpatialMap(ig.path(1), ig.scalar_line(0, 1), {0: 0, 1: 0}, {(0, 1): 0})
    assert to_line.then(ig.identity_map(ig.scalar_line(0, 1))) == to_line
    assert len(list(ig.all_maps(ig.path(1), ig.path(1)))) == 6  # 2 with the edge kept, 2 collapsing, 2 constant-kept
