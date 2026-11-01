"""Hyperiad builtins for the rewriting calculus of infrageometry (theme hypergraph-geometry, q02).

Step paths and their homotopy by deformation across areas (derivations prove equivalence, barriers
prove inequivalence; the homotopy of motions and the fundamental group derive from it), and fields
as typed edges: vector fields of departures and arrivals, scalar fields of unary edges counted by
multiplicity. Their algebra (cancellation, composition, integration, bracket, comparison) is
rewriting. Values display drawn: paths green, further paths and fields in the
desaturated palette, what a rewrite touched in amber.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from frontier.hypergraphs.svg import scene_svg
from infrageometry.edgefields import Cancellation, Comparison, FieldEdge, Saturation, direction_field, order
from infrageometry.paths import ALL_RULES, GroupPresentation, PathRewrite
from infrageometry.space import point_text

from hyperiad.display import register_display
from hyperiad.errors import HyperiadError
from hyperiad.geometry_builtins import ZOOM, _int, _motion, _pairs, _point, _space, _vertex
from hyperiad.kernel import Session, builtin
from hyperiad.values import Frames, FunctionValue, Graphic, format_value
from infrageometry import (
    Barrier,
    FieldRule,
    FieldState,
    StepPath,
    arc_field,
    bracket_rules,
    cancel,
    closure_rules,
    compare,
    composition_rule,
    find_barrier,
    find_derivation,
    fundamental_group,
    path_of_motion,
    reduce_path,
    rewrites,
    saturate,
    scalar_field,
    scalar_field_of,
    triangulated_annulus,
    triangulated_disk,
    vector_field,
)
from infrageometry import pictures as pic

RULE_ZOOM = 0.8  # a rule shows two tiny spaces side by side: drawn smaller than a full picture


def _svg(scene) -> str:
    return scene_svg(scene, zoom=ZOOM)


# -- values ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PathStep:
    """One rewrite of a path and its result."""

    rewrite: PathRewrite
    path: StepPath

    def __repr__(self) -> str:
        return f"<{self.rewrite.describe()} ⟹ {self.path!r}>"


@dataclass(frozen=True)
class Derivation:
    """A chain of rewrites from ``source`` to ``target`` (``steps`` is None when none was found)."""

    source: StepPath
    target: StepPath
    steps: tuple | None
    note: str = ""

    def __repr__(self) -> str:
        if self.steps is None:
            return f"<no derivation found {self.note}>"
        return f"<derivation of {len(self.steps)} rewrites>"


def _path(x: Any, name: str) -> StepPath:
    if isinstance(x, StepPath):
        return x
    raise HyperiadError(f"{name} takes a step path (from StepPath), got {format_value(x, 60)}")


def _state(x: Any, name: str) -> FieldState:
    if isinstance(x, FieldState):
        return x
    if isinstance(x, Saturation):
        return x.final
    raise HyperiadError(f"{name} takes fields (from VectorFieldEdges or DirectionField), got {format_value(x, 60)}")


def _rules(x: Any, name: str) -> list[FieldRule]:
    rules = x if isinstance(x, list) else [x]
    flat = []
    for r in rules:
        flat += r if isinstance(r, list) else [r]
    if not all(isinstance(r, FieldRule) for r in flat):
        raise HyperiadError(f"{name} takes field rules (from CompositionRule, ClosureRules, BracketRules)")
    return flat


def _describe_path(p: StepPath) -> str:
    return f"step path {point_text(p.start)} to {point_text(p.end)} · {len(p)} steps: " + " → ".join(
        point_text(v) for v in p.vertices
    )


def _show_path(p: StepPath) -> list[dict]:
    return [{"type": "svg", "data": _svg(pic.step_path_scene(p)), "caption": _describe_path(p)}]


def _show_path_step(s: PathStep) -> list[dict]:
    rw = s.rewrite
    window = s.path.vertices[rw.position : rw.position + len(rw.after)] or s.path.vertices[:1]
    scene = pic.step_path_scene(s.path, window=window, area=rw.area)
    return [
        {
            "type": "svg",
            "data": _svg(scene),
            "caption": f"{rw.describe()} ⟹ " + " ".join(map(point_text, s.path.vertices)),
        }
    ]


def derivation_frames(d: Derivation, interval: int = 1400) -> Frames:
    svgs, labels = [], []
    svgs.append(_svg(pic.step_path_scene(d.source, others=[(d.target, pic.TANGENT)])))
    labels.append("start: " + " ".join(map(point_text, d.source.vertices)) + "  (target in blue)")
    for k, (rw, p) in enumerate(d.steps or (), 1):
        window = p.vertices[rw.position : rw.position + len(rw.after)] or p.vertices[:1]
        svgs.append(_svg(pic.step_path_scene(p, others=[(d.target, pic.TANGENT)], window=window, area=rw.area)))
        labels.append(f"{k}. {rw.describe()}")
    return Frames(tuple(svgs), tuple(labels), f"a derivation in {len(d.steps or ())} rewrites", True, interval)


def _show_derivation(d: Derivation) -> list[dict]:
    if d.steps is None:
        scene = pic.step_path_scene(d.source, others=[(d.target, pic.TANGENT)])
        return [{"type": "svg", "data": _svg(scene), "caption": f"no derivation found {d.note}".strip()}]
    from hyperiad.display import frames_output

    return [frames_output(derivation_frames(d))]


def _show_barrier(b: Barrier) -> list[dict]:
    scene = pic.space_scene(b.space, edge_points="none")
    pic._barrier_marks(scene, pic.layout_of(b.space), b.pairs)
    ok = "a barrier" if b.is_barrier() else f"not a barrier: {len(b.violations())} areas crossed an odd number of times"
    return [{"type": "svg", "data": _svg(scene), "caption": f"{b!r} · {ok}"}]


def describe_state(s: FieldState) -> str:
    names = pic.kind_color_names(s.kinds)

    def what(k):
        es = s.of(k)
        if es and all(e.unary for e in es):
            return f"{s.size(k)} unary edges"
        if es and all(e.half_step for e in es):
            return f"{s.size(k)} arcs"
        return f"{s.size(k)} edges"

    return "fields · " + " · ".join(f"{k} ({names[k]}): {what(k)}" for k in s.kinds)


def _show_state(s: FieldState) -> list[dict]:
    return [{"type": "svg", "data": _svg(pic.field_state_scene(s)), "caption": describe_state(s)}]


def _show_saturation(sat: Saturation) -> list[dict]:
    added = sum(len(r) for r in sat.events)
    caption = f"saturated in {len(sat.events)} rounds, {added} edges added · " + describe_state(sat.final)
    return [{"type": "svg", "data": _svg(pic.field_state_scene(sat.final)), "caption": caption}]


def _show_rule(r: FieldRule) -> list[dict]:
    svg = scene_svg(pic.field_rule_scene(r), zoom=RULE_ZOOM)
    return [{"type": "svg", "data": svg, "caption": f"rule {r.name}: adds {r.gives} edges"}]


register_display(StepPath, _show_path)
register_display(PathStep, _show_path_step)
register_display(Derivation, _show_derivation)
register_display(Barrier, _show_barrier)
register_display(FieldState, _show_state)
register_display(Saturation, _show_saturation)
register_display(FieldRule, _show_rule)


def cancellation_frames(c: Cancellation, interval: int = 900) -> Frames:
    states = c.states
    svgs = [_svg(pic.field_state_scene(states[0]))]
    labels = [f"start: {sum(n for _e, n in c.start.counts)} edges"]
    for k, pair in enumerate(c.pairs, 1):
        svgs.append(_svg(pic.field_state_scene(states[k - 1], new=list(pair))))
        labels.append(f"{k}. cancel {pair[0]!r} with {pair[1]!r}")
    svgs.append(_svg(pic.field_state_scene(c.final)))
    labels.append("normal form: " + describe_state(c.final))
    return Frames(tuple(svgs), tuple(labels), f"{len(c.pairs)} pairs of opposite arcs cancelled", True, interval)


def _show_cancellation(c: Cancellation) -> list[dict]:
    from hyperiad.display import frames_output

    return [frames_output(cancellation_frames(c))]


def comparison_frames(c: Comparison, interval: int = 1000) -> Frames:
    states = c.states()
    svgs, labels = [], []
    for k, st in enumerate(states):
        nxt = list(c.pairs[k]) if k < len(c.pairs) else []
        svgs.append(_svg(pic.field_state_scene(st, kinds=[c.kind], new=nxt)))
        labels.append(f"{k} pairs cancelled" + (" · next pair in amber" if nxt else f" · {c!r}"))
    return Frames(tuple(svgs), tuple(labels), repr(c), False, interval)


def _show_comparison(c: Comparison) -> list[dict]:
    from hyperiad.display import frames_output

    return [frames_output(comparison_frames(c))]


def _show_group(g: GroupPresentation) -> list[dict]:
    out = [{"type": "text", "data": f"π1 at {point_text(g.base)}: {g!r}"}]
    for name, loop in zip(g.names, g.loops, strict=True):
        out.append(
            {"type": "svg", "data": _svg(pic.step_path_scene(loop)), "caption": f"{name}: {_describe_path(loop)}"}
        )
    return out


register_display(Cancellation, _show_cancellation)
register_display(Comparison, _show_comparison)
register_display(GroupPresentation, _show_group)
register_display(FieldEdge, lambda e: [{"type": "text", "data": repr(e)}])


# -- paths and their homotopy -------------------------------------------------------------------


@builtin(
    "StepPath",
    "StepPath(h, [v0, v1, ...]) is the path through these positions, each two consecutive ones in a common edge "
    "of h (a chain of binary path edges); drawn as green arcs.",
)
def _step_path(h, vs):
    s = _space(h, "StepPath")
    if not isinstance(vs, list):
        raise HyperiadError("StepPath takes a list of positions")
    return StepPath(s, tuple(_vertex(s, v, "StepPath") for v in vs))


@builtin("PathOfMotion", "PathOfMotion(m) is the step path of a motion: its positions, infinitesimal ones forgotten.")
def _path_of_motion(m):
    return path_of_motion(_motion(m, "PathOfMotion"))


@builtin("PathPositions", "PathPositions(p) lists the positions of a step path.")
def _path_positions(p):
    return list(_path(p, "PathPositions").vertices)


@builtin(
    "PathRewrites",
    "PathRewrites(p) lists every single rewrite of p by the shortening rules (cancel, stall, contract); "
    "PathRewrites(p, All = true) adds the expanding ones.",
)
def _path_rewrites(p, All=False):  # noqa: N803
    from infrageometry.paths import SHORTENING

    p = _path(p, "PathRewrites")
    return [PathStep(rw, q) for rw, q in rewrites(p, ALL_RULES if All else SHORTENING)]


@builtin("HomotopyRules", "HomotopyRules() draws the rules of area homotopy: contract, expand, cancel, stall.")
def _homotopy_rules():
    return [Graphic(scene_svg(scene, zoom=RULE_ZOOM), title) for title, scene in pic.homotopy_rule_scenes()]


@builtin(
    "HomotopyDerivation",
    "HomotopyDerivation(p, q) searches for a shortest chain of rewrites (all rules, both ways) turning p into q, "
    "among paths of at most MaxLength steps (default: the longer path plus 2); it plays step by step.",
    cost=lambda p, q, MaxLength=None: 1e3 * 3 ** (MaxLength or max(len(p), len(q)) + 2),
)
def _homotopy_derivation(p, q, MaxLength=None):  # noqa: N803
    p, q = _path(p, "HomotopyDerivation"), _path(q, "HomotopyDerivation")
    limit = None if MaxLength is None else _int(MaxLength, "HomotopyDerivation")
    steps = find_derivation(p, q, max_length=limit)
    bound = limit if limit is not None else max(len(p), len(q)) + 2
    return Derivation(p, q, None if steps is None else tuple(steps), f"among paths of at most {bound} steps")


@builtin("DerivationSteps", "DerivationSteps(d) lists the rewrites of a derivation, each with its result.")
def _derivation_steps(d):
    if not isinstance(d, Derivation) or d.steps is None:
        raise HyperiadError("DerivationSteps takes a derivation that was found")
    return [PathStep(rw, q) for rw, q in d.steps]


@builtin("ReducePath", "ReducePath(p) shortens p by cancel, stall and contract, leftmost first, until none applies.")
def _reduce_path(p):
    p = _path(p, "ReducePath")
    steps = reduce_path(p)
    return Derivation(p, steps[-1][1] if steps else p, tuple(steps))


@builtin(
    "Barrier", "Barrier(h, [[u, v], ...]) is a set of pairs of adjacent positions, drawn as amber bars across them."
)
def _barrier(h, pairs):
    s = _space(h, "Barrier")
    return Barrier(s, frozenset(frozenset(_point(x)) for x in pairs))


@builtin("BarrierQ", "BarrierQ(b) is true when every area holds an even number of its pairs.")
def _barrier_q(b):
    if not isinstance(b, Barrier):
        raise HyperiadError("BarrierQ takes a barrier")
    return b.is_barrier()


@builtin("Crossings", "Crossings(b, p) counts the steps of p across the barrier b.")
def _crossings(b, p):
    if not isinstance(b, Barrier):
        raise HyperiadError("Crossings takes a barrier and a path")
    return b.crossings(_path(p, "Crossings"))


@builtin(
    "FindBarrier",
    "FindBarrier(p, q) searches for a barrier (at most MaxSize pairs, default 4) that p and q cross with different "
    "parity: a proof that they are not area-homotopic. Drawn with both paths.",
)
def _find_barrier(p, q, MaxSize=4):  # noqa: N803
    p, q = _path(p, "FindBarrier"), _path(q, "FindBarrier")
    b = find_barrier(p, q, max_size=_int(MaxSize, "FindBarrier"))
    if b is None:
        return None
    caption = f"{b!r}: crossed {b.crossings(p)} and {b.crossings(q)} times"
    return Graphic(_svg(pic.step_path_scene(p, others=[(q, pic.TANGENT)], barrier=b)), caption)


@builtin(
    "FundamentalGroup",
    "FundamentalGroup(h, v) presents π1(h, v), loops at v up to area homotopy: one generator per step outside a "
    "spanning tree, one relator per triangle inside an area, then simplified (Simplify = false keeps it raw). Drawn "
    "with the generator loops.",
)
def _fundamental_group(h, v, Simplify=True):  # noqa: N803
    s = _space(h, "FundamentalGroup")
    g = fundamental_group(s, _vertex(s, v, "FundamentalGroup"))
    return g.simplified() if Simplify else g


@builtin(
    "AreaHomotopicQ",
    "AreaHomotopicQ(p, q) is true when a derivation is found, false when a barrier separates them, and "
    '"undecided" otherwise (within the default search bounds).',
)
def _area_homotopic_q(p, q):
    p, q = _path(p, "AreaHomotopicQ"), _path(q, "AreaHomotopicQ")
    if (p.start, p.end) != (q.start, q.end):
        return False
    if find_derivation(p, q) is not None:
        return True
    return False if find_barrier(p, q) is not None else "undecided"


@builtin("TriangulatedDisk", "TriangulatedDisk(n) is a disk: a centre 0 and a rim 1..n, the n triangles {0, i, i+1}.")
def _disk(n=6):
    return triangulated_disk(_int(n, "TriangulatedDisk"))


@builtin(
    "TriangulatedAnnulus",
    "TriangulatedAnnulus(n) is an annulus of 2n triangles round a hole: inner ring a1..an, outer b1..bn.",
)
def _annulus(n=4):
    return triangulated_annulus(_int(n, "TriangulatedAnnulus"))


# -- fields of typed edges ----------------------------------------------------------------------


@builtin(
    "VectorFieldEdges",
    'VectorFieldEdges(h, "X", {a: b, ...}) is the vector field of type X stepping a → b: the departure from a '
    "into the smallest edge holding both and the arrival from it at b (b = a stays on the unary edge {a}).",
)
def _vector_field_edges(h, kind, steps):
    s = _space(h, "VectorFieldEdges")
    return vector_field(s, str(kind), {_point(a): _point(b) for a, b in _pairs(steps, "VectorFieldEdges")})


def _coords(p):
    return [int(c) if float(c).is_integer() else c for c in p]


@builtin(
    "DirectionField",
    'DirectionField(h, "X", [dx, dy]) on a space drawn with coordinates (a grid) steps every position to the '
    "neighbour at +[dx, dy]. Where = (p => ...) keeps the positions whose coordinates p pass the test; "
    "Stay = true puts a stationary edge at the others (where there is a unary edge).",
    session=True,
)
def _direction_field(session: Session, h, kind, delta, Where=None, Stay=False):  # noqa: N803
    s = _space(h, "DirectionField")
    test = None
    if Where is not None:
        if not isinstance(Where, FunctionValue):
            raise HyperiadError("DirectionField: Where is a function of the coordinates, p => ...")

        def test(p):
            return bool(session.apply(Where, [_coords(p)]))

    return direction_field(s, str(kind), tuple(delta), where=test, stay_elsewhere=bool(Stay))


@builtin("JoinFields", "JoinFields(f, g, ...) puts fields on one space together, keeping their types and colours.")
def _join_fields(*states):
    out = _state(states[0], "JoinFields")
    for s in states[1:]:
        out = out.join(_state(s, "JoinFields"))
    return out


@builtin(
    "VectorFieldArcs",
    'VectorFieldArcs(h, "X", [[v, [edge]], [[edge], v], ...]) is the vector field with these departures v ⇀ e and '
    "arrivals e ⇀ v (an arrival is a negative direction); repeats count.",
)
def _vector_field_arcs(h, kind, arcs):
    s = _space(h, "VectorFieldArcs")
    pairs = []
    for a in arcs:
        if not (isinstance(a, list) and len(a) == 2):
            raise HyperiadError("VectorFieldArcs takes a list of [v, edge] departures and [edge, v] arrivals")
        pairs.append((_point(a[0]), _point(a[1])))
    try:
        return arc_field(s, str(kind), pairs)
    except ValueError as err:
        raise HyperiadError(f"VectorFieldArcs: {err}") from None


@builtin(
    "Cancel",
    "Cancel(fields) cancels each departure v ⇀ e with an arrival e ⇀ v of the same type, until none is left: "
    "it plays the pairs, and ends in the net field.",
)
def _cancel(s):
    return cancel(_state(s, "Cancel"))


@builtin("CancelledFields", "CancelledFields(c) is the state a cancellation ends in.")
def _cancelled(c):
    if not isinstance(c, Cancellation):
        raise HyperiadError("CancelledFields takes a cancellation (from Cancel)")
    return c.final


@builtin(
    "ScalarField",
    'ScalarField(h, "f", {v: n, ...}) lays n unary edges of type f at each v: a scalar field, its values inferred '
    "from the multiplicities (0 where there are none).",
)
def _scalar_field(h, kind, values):
    s = _space(h, "ScalarField")
    try:
        return scalar_field(
            s,
            str(kind),
            {_vertex(s, v, "ScalarField"): _int(n, "ScalarField") for v, n in _pairs(values, "ScalarField")},
        )
    except ValueError as err:
        raise HyperiadError(f"ScalarField: {err}") from None


@builtin(
    "ScalarFieldOf",
    'ScalarFieldOf(f, "f") is the scalar field of a scalar function (map version): f(v) - min f unary edges at v.',
)
def _scalar_field_of(fn, kind="f"):
    from infrageometry.scalar import ScalarFunction

    if not isinstance(fn, ScalarFunction):
        raise HyperiadError("ScalarFieldOf takes a scalar function (from ScalarFunction or DistanceFunction)")
    return scalar_field_of(fn, str(kind))


@builtin("FieldValues", 'FieldValues(s, "f") lists [v, n]: the multiplicity of the unary edges of type f at each v.')
def _field_values(s, kind):
    return [[v, n] for v, n in _state(s, "FieldValues").values(str(kind)).items()]


@builtin(
    "CompareValues",
    'CompareValues(s, "f", a, b) compares a scalar field at a and b by cancelling unary edges in pairs, one at '
    "each end, until one side runs out: what is left is the difference, at the higher end. It plays the pairs.",
)
def _compare_values(s, kind, a, b):
    st = _state(s, "CompareValues")
    return compare(st, str(kind), _vertex(st.space, a, "CompareValues"), _vertex(st.space, b, "CompareValues"))


@builtin("ComparisonSign", "ComparisonSign(c) is ↑ when the second end is higher, ↓ when the first is, ○ when equal.")
def _comparison_sign(c):
    if not isinstance(c, Comparison):
        raise HyperiadError("ComparisonSign takes a comparison (from CompareValues)")
    return c.sign


@builtin("ComparisonSize", "ComparisonSize(c) is how many unary edges are left after the comparison.")
def _comparison_size(c):
    if not isinstance(c, Comparison):
        raise HyperiadError("ComparisonSize takes a comparison (from CompareValues)")
    return c.left


@builtin("FieldEdges", 'FieldEdges(s, "X") lists the edges of type X.')
def _field_edges(s, kind):
    return _state(s, "FieldEdges").of(str(kind))


@builtin("FieldTypes", "FieldTypes(s) lists the types of the edges present, in order of appearance.")
def _field_types(s):
    return list(_state(s, "FieldTypes").kinds)


@builtin("ShowFields", 'ShowFields(s, ["X", ...]) draws only the edges of the given types (in their usual colours).')
def _show_fields(s, kinds):
    st = _state(s, "ShowFields")
    caption = " · ".join(f"{k} ({pic.kind_color_names(st.kinds)[k]}): {len(st.of(k))}" for k in kinds)
    return Graphic(_svg(pic.field_state_scene(st, kinds=[str(k) for k in kinds])), caption)


@builtin("VectorFieldQ", 'VectorFieldQ(s, "X") is true when the X edges are all departures and arrivals.')
def _vector_field_q(s, kind):
    return _state(s, "VectorFieldQ").is_vector_field(str(kind))


@builtin(
    "DeterministicQ",
    'DeterministicQ(s, "X") is true when the X arcs are single and at most one leaves each point: a flow can follow.',
)
def _deterministic_q(s, kind):
    return _state(s, "DeterministicQ").is_deterministic(str(kind))


@builtin(
    "EdgeOrder",
    "EdgeOrder(s, e) is how many half-steps apart the ends of a field edge lie: 1 for a departure or an arrival, "
    "2 for a step, 4 for a composite of two steps.",
)
def _edge_order(s, e):
    if not isinstance(e, FieldEdge):
        raise HyperiadError("EdgeOrder takes fields and one of their edges")
    return order(_state(s, "EdgeOrder"), e)


@builtin(
    "CompositionRule",
    'CompositionRule("X", "Y") adds (X;Y)(a → c) for every step of X from a to b followed by a step of Y from b to c '
    "(each step a departure and an arrival).",
)
def _composition_rule(x, y):
    return composition_rule(str(x), str(y))


@builtin(
    "ClosureRules",
    'ClosureRules("X") are X X ⇒ X+ (a step) and X+ X X ⇒ X+ (one step more): saturated, the integrated flow.',
)
def _closure_rules(x):
    return closure_rules(str(x))


@builtin(
    "BracketRules",
    'BracketRules("X", "Y") composes both ways and, for (X;Y)(a → c) and (Y;X)(a → c\'), c ≠ c\', adds the step '
    "c' → c of type [X,Y] as a departure and an arrival.",
)
def _bracket_rules(x, y):
    return bracket_rules(str(x), str(y))


def _saturate_cost(s, rules, **_):
    st = _state(s, "Saturate")
    return 50.0 * len(st.space.vertices) ** 2 * max(len(_rules(rules, "Saturate")), 1) * max(len(st.edges), 1)


@builtin(
    "Saturate",
    "Saturate(fields, rules) applies the rules everywhere, round after round, until nothing new appears; "
    "it shows the result (Final(s) is the state).",
    cost=_saturate_cost,
)
def _saturate(s, rules):
    return saturate(_state(s, "Saturate"), _rules(rules, "Saturate"))


@builtin("Final", "Final(saturation) is the state a saturation ends in.")
def _final(sat):
    if not isinstance(sat, Saturation):
        raise HyperiadError("Final takes a saturation")
    return sat.final


@builtin("Rounds", "Rounds(saturation) lists how many edges each round added.")
def _rounds(sat):
    if not isinstance(sat, Saturation):
        raise HyperiadError("Rounds takes a saturation")
    return [len(r) for r in sat.events]


@builtin(
    "AnimateSaturation",
    "AnimateSaturation(fields, rules) plays a saturation round by round, the edges each round adds in amber. "
    "Types = [...] draws only those types.",
    cost=_saturate_cost,
)
def _animate_saturation(s, rules, Types=None, Interval=1200):  # noqa: N803
    sat = saturate(_state(s, "AnimateSaturation"), _rules(rules, "AnimateSaturation"))
    kinds = [str(k) for k in Types] if Types is not None else None
    svgs, labels = [_svg(pic.field_state_scene(sat.states[0], kinds=kinds))], ["round 0: the fields given"]
    for k, (st, events) in enumerate(zip(sat.states[1:], sat.events, strict=True), 1):
        new = [out for _r, _i, out in events]
        svgs.append(_svg(pic.field_state_scene(st, kinds=kinds, new=new)))
        by = {}
        for _r, _i, out in events:
            by[out.kind] = by.get(out.kind, 0) + 1
        labels.append(f"round {k}: added " + ", ".join(f"{n} {t}" for t, n in by.items()))
    return Frames(tuple(svgs), tuple(labels), f"saturation in {len(sat.events)} rounds", True, int(Interval))


@builtin(
    "FieldHypergraph",
    "FieldHypergraph(s) is space and fields as one hypergraph: field edges directed and labelled by type.",
)
def _field_hypergraph(s):
    return _state(s, "FieldHypergraph").to_hypergraph()
