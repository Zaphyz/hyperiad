"""Infrageometry in Hyperiad: builtins, drawings, frames and the example notebook."""

import json

from hyperiad.server import build, example_notebook, precompute

from hyperiad import Session


def run(session: Session, code: str) -> dict:
    result = session.run(code)
    assert result["status"] == "ok", result["outputs"]
    return result


def text(session: Session, code: str) -> str:
    outs = run(session, code)["outputs"]
    assert outs[-1]["type"] == "text", outs
    return outs[-1]["data"]


def test_space_topology_and_tangent_vectors():
    s = Session(budget=2e7)
    run(s, "h = ExampleSpace()")
    assert text(s, "[OpenQ(MinimalOpen(h, 2)), ClosedQ(Closure(MinimalOpen(h, 2)))]") == "[true, true]"
    assert text(s, "FirstBetti(h)") == "2"
    assert text(s, "Star(h, 2)") == "[{1, 2}, {2, 6}, {1, 2, 3}]"
    out = run(s, "TangentSpace(h, 2)")["outputs"][0]
    assert out["type"] == "svg" and "3 tangent vectors" in out["caption"] and "<svg" in out["data"]
    assert text(s, "Length(TangentSpace(h, 2))") == "3"  # still a list
    assert "2 ⇀ {1, 2}" in text(s, "TangentBundle(h)")


def test_motions_homotopy_and_maps():
    s = Session(budget=2e7)
    run(s, "h = ExampleSpace(); g = Geodesic(h, 1, 5)")
    assert text(s, "HalfSteps(g)") == "6"
    assert text(s, "HomotopicQ(JoinMotions(g, ReverseMotion(g)), Motion(h, [1]))") == "true"
    pictures = run(s, "IncidenceGroupGenerators(h, 1)")["outputs"]
    assert len(pictures) == 2 and all(o["type"] == "svg" for o in pictures) and pictures[0]["caption"].startswith("[1]")
    run(s, "cover = SpatialMap(CycleSpace(6), CycleSpace(3), {0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 2})")
    assert text(s, "[LocalIsomorphismQ(cover), EmbeddingQ(cover)]") == "[true, false]"
    assert text(s, "Pushforward(CurveOfMotion(g), Departure(1, [1, 2])) == Velocity(g, 3)") == "true"
    err = s.run("SpatialMap(PathSpace(1), CycleSpace(4), {0: 0, 1: 2})")
    assert err["status"] == "error" and "no target edge contains" in err["outputs"][0]["data"]


def test_scalar_functions_and_flows():
    s = Session(budget=2e7)
    run(s, "h = ExampleSpace(); f = DistanceFunction(h, 1)")
    assert text(s, "CriticalPoints(f)") == '{1: "minimum", 5: "maximum"}'
    assert '"↑"' in text(s, "Differential(f, 2)")
    bump = "ScalarFunction(PathSpace(3), [0, 0, 0, 0], EdgeValues = [[1, 2] -> [0, 1]])"
    assert text(s, f"Map(v => PointKind({bump}, v), [0, 1, 2, 3])") == '["flat", "minimum", "minimum", "flat"]'
    err = s.run("ScalarFunction(PathSpace(1), {0: 0, 1: 2})")
    assert err["status"] == "error" and "within one step" in err["outputs"][0]["data"]
    assert text(s, "HalfSteps(IntegralCurve(GradientFlow(f, Ascending = false), 5))") == "6"


def test_frames_from_manipulate_animate_and_list_animate():
    s = Session(budget=2e7)
    run(s, "h = ExampleSpace()")
    out = run(s, "Manipulate(TangentSpace(h, v), [v, Positions(h)])")["outputs"][0]
    assert out["type"] == "frames" and len(out["frames"]) == 6 and not out["autoplay"]
    assert out["labels"][0].startswith("v = 1 · T_1H")
    boxes = {f.split('viewBox="')[1].split('"')[0] for f in out["frames"]}
    assert len(boxes) == 1  # steady: every frame has the same box
    assert run(s, "Animate(Ball(h, 1, r), [r, 0, 3])")["outputs"][0]["autoplay"]
    anim = run(s, "AnimateMotion(Geodesic(h, 1, 5))")["outputs"][0]
    assert len(anim["frames"]) == 7 and anim["labels"][1] == "1: at {1, 2} (infinitesimal position)"
    assert len(run(s, "ListAnimate([h, Link(h, 2)])")["outputs"][0]["frames"]) == 2
    bad = s.run("ListAnimate([1, 2])")
    assert bad["status"] == "error" and "drawn" in bad["outputs"][0]["data"]


def test_token_rewriting_system():
    s = Session(budget=2e7)
    run(s, "t = SpatialHypergraph([[1, 2, 3], [3, 4], [4, 1]]); sys = TokenRewritingSystem(t, 1)")
    out = run(s, 'sys("StatesGraph", 5)')["outputs"][0]
    assert "5 vertices" in out["caption"]  # 7 points of X(t), 1 and 3 exchanged by a symmetry


def test_infrageometry_notebook_runs(tmp_path):
    nb = example_notebook("infrageometry-foundations")
    assert nb["contents"] and nb["title"]
    cells = precompute(nb, 2e7)["cells"]
    failed = [c["source"] for c in cells if c.get("type") == "code" and c["status"] != "ok"]
    assert failed == ["ScalarFunction(PathSpace(1), {0: 0, 1: 2})"]  # the one deliberate error
    page = build(tmp_path, None, "pyodide/", 2e7, False, None, kernel=False, notebook="infrageometry-foundations")
    html = page.read_text()
    assert '"kernel": "none"' in html and '"toc": true' in html and not (tmp_path / "pysrc.json").exists()
    assert json.loads(html.split("window.HYPERIAD_NOTEBOOK = ")[1].split(";</script>")[0].replace("<\\/", "</"))
