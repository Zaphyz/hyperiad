import pytest
from infrageometry.edgefields import (
    FieldState,
    arc_field,
    bracket,
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
from infrageometry.paths import ALL_RULES, reduce_path, rewrites

import infrageometry as ig
from infrageometry import Barrier, StepPath, find_barrier, find_derivation

DISK = ig.triangulated_disk(6)
ANNULUS = ig.triangulated_annulus(4)


def test_step_paths():
    assert len(StepPath(DISK, (1, 2, 0))) == 2
    with pytest.raises(ValueError):
        StepPath(DISK, (1, 3))  # not in one edge
    with pytest.raises(ValueError):
        StepPath(DISK, (1, 1))  # a stall needs a unary edge
    m = ig.Motion(DISK, [1, {0, 1, 2}, 1, {0, 1, 2}, 2])
    assert ig.path_of_motion(m).vertices == (1, 2)


def test_area_rules_and_derivations():
    p, q = StepPath(DISK, (1, 2, 3, 4)), StepPath(DISK, (1, 0, 4))
    steps = find_derivation(p, q)
    assert [rw.rule for rw, _ in steps] == ["expand", "contract", "contract"] and steps[-1][1] == q
    assert {rw.rule for rw, _ in rewrites(StepPath(DISK, (1, 0, 1)), ALL_RULES)} >= {"cancel", "expand", "uncancel"}
    assert reduce_path(StepPath(DISK, (1, 2, 3, 4, 0, 1)))[-1][1].vertices == (1,)
    h = ig.triangle_example()
    assert find_derivation(StepPath(h, (1, 2)), StepPath(h, (1, 3, 2)))  # across the area {1, 2, 3}


def test_barriers_prove_inequivalence():
    p, q = StepPath(ANNULUS, ("a1", "a2", "a3")), StepPath(ANNULUS, ("a1", "a4", "a3"))
    assert find_derivation(p, q, max_length=6) is None
    b = find_barrier(p, q)
    assert b is not None and b.is_barrier() and (b.crossings(p) - b.crossings(q)) % 2 == 1
    assert not Barrier(ANNULUS, frozenset({frozenset({"a1", "a2"})})).is_barrier()  # an area crossed once
    loop = StepPath(ig.triangle_example(), (2, 6, 4, 3, 2))
    assert find_barrier(loop, StepPath(ig.triangle_example(), (2,))) is not None


def test_composition_closure_and_bracket():
    g = ig.with_unary(ig.grid(3, 3))
    x, y = direction_field(g, "X", (1, 0)), direction_field(g, "Y", (0, 1))
    assert x.is_vector_field("X") and x.is_deterministic("X") and x.size("X") == 12  # 6 steps, 2 arcs each
    assert all(e.departure for e in x.of("X") if not isinstance(e.tail, frozenset))
    s = saturate(x.join(y), [composition_rule("X", "Y")]).final
    assert len(s.of("X;Y")) == 4 and all(ig.edgefields.order(s, e) == 4 for e in s.of("X;Y"))
    assert not bracket(x.join(y), "X", "Y").final.of("[X,Y]")
    y2 = direction_field(g, "Y", (0, 1), where=lambda p: int(p[0]) % 2 == 0, stay_elsewhere=True)
    b = bracket(x.join(y2), "X", "Y").final
    assert len(b.of("[X,Y]")) == 8 and b.is_vector_field("[X,Y]")  # 4 steps, one per unit square
    c = saturate(x, closure_rules("X"))
    assert [len(r) for r in c.events] == [6, 3]  # round k adds the time-k displacements
    with pytest.raises(ValueError):
        vector_field(g, "Z", {(0, 0): (2, 2)})  # not a step
    hg = b.to_hypergraph()
    assert sum(1 for e in hg.edges if e.label == "[X,Y]") == 8


def test_cancellation_and_net():
    g = ig.with_unary(ig.grid(3, 3))
    x = direction_field(g, "X", (1, 0))
    back = FieldState(g, [e.opposite() for e in x.copies("X")])
    assert not cancel(x.join(back)).final.edges  # a field and its reverse cancel
    stay = vector_field(g, "S", {(1, 1): (1, 1)})
    assert stay.size("S") == 2 and not cancel(stay).final.edges  # zero and undefined become one
    assert set(x.net("X").values()) == {1, -1}
    with pytest.raises(ValueError):
        arc_field(g, "B", [((0, 0), frozenset({(1, 1), (1, 2)}))])  # not an incidence


def test_scalar_fields_by_multiplicity():
    g = ig.grid(2, 3)
    f = scalar_field(g, "f", {(0, 0): 3, (0, 1): 1})
    assert f.values("f")[(0, 0)] == 3 and f.values("f")[(1, 2)] == 0 and f.size("f") == 4
    c = compare(f, "f", (0, 0), (0, 1))
    assert (c.sign, c.left, c.higher, len(c.pairs)) == ("↓", 2, (0, 0), 1) and len(c.states()) == 2
    assert compare(f, "f", (1, 1), (1, 2)).sign == "○"
    fn = ig.distance_function(g, (0, 0))
    sf = scalar_field_of(fn)
    for t in ig.tangent_bundle(g):
        others = [compare(sf, "f", t.base, w).sign for w in t.edge if w != t.base]
        assert fn.derivative(t) == ("○" if not others or set(others) == {"○"} else next(s for s in others if s != "○"))


def test_derived_homotopy_and_fundamental_group():
    h = ig.triangle_example()
    loop = ig.Motion(h, [1, {1, 2}, 2, {1, 2, 3}, 1])  # two parallel connections: filled by the area
    assert not ig.incidence_homotopic(loop, ig.Motion(h, [1])) and ig.homotopic(loop, ig.Motion(h, [1]))
    hole = StepPath(h, (2, 6, 4, 3, 2))
    assert ig.homotopic(hole, StepPath(h, (2,))) is False
    assert repr(ig.fundamental_group(DISK, 1).simplified()) == "1"
    assert len(ig.fundamental_group(ANNULUS, "a1").simplified().names) == 1
    t = ig.fundamental_group(ig.torus7(), 0).simplified()
    assert len(t.names) == 2 and len(t.relators) == 1 and len(t.relators[0]) == 4  # ⟨a, b | a b a⁻¹ b⁻¹⟩
