"""The rewriting calculus in Hyperiad: homotopy derivations, barriers, fields and their saturation."""

from hyperiad import Session


def run(s: Session, code: str) -> list[dict]:
    r = s.run(code)
    assert r["status"] == "ok", r["outputs"]
    return r["outputs"]


def test_homotopy_derivation_and_barrier():
    s = Session(budget=2e7)
    run(s, "d = TriangulatedDisk(6); p = StepPath(d, [1, 2, 3, 4]); q = StepPath(d, [1, 0, 4])")
    frames = run(s, "HomotopyDerivation(p, q)")[0]
    assert frames["type"] == "frames" and len(frames["frames"]) == 4 and "across {0, 3, 4}" in frames["labels"][-1]
    assert run(s, "AreaHomotopicQ(p, q)")[0]["data"] == "true"
    run(s, 'A = TriangulatedAnnulus(4); u = StepPath(A, ["a1", "a2", "a3"]); w = StepPath(A, ["a1", "a4", "a3"])')
    assert run(s, "AreaHomotopicQ(u, w)")[0]["data"] == "false"
    assert "crossed 1 and 0 times" in run(s, "FindBarrier(u, w)")[0]["caption"]
    assert run(s, 'BarrierQ(Barrier(A, [["a1", "a2"]]))')[0]["data"] == "false"
    assert len(run(s, "HomotopyRules()")) == 4
    err = s.run("StepPath(d, [1, 3])")
    assert err["status"] == "error" and "not a step" in err["outputs"][0]["data"]


def test_fields_saturation_and_bracket():
    s = Session(budget=2e7)
    run(s, 'g = WithUnaryEdges(GridSpace(3, 3)); X = DirectionField(g, "X", [1, 0])')
    run(s, 'Up = DirectionField(g, "N", [0, 1])')
    run(s, 'A = DirectionField(g, "A", [0, 1], Where = (c => EvenQ(c[1])), Stay = true)')
    assert run(s, 'Length(FieldEdges(Saturate(JoinFields(X, Up), BracketRules("X", "N")), "[X,N]"))')[0]["data"] == "0"
    run(s, 'b = Saturate(JoinFields(X, A), BracketRules("X", "A"))')
    assert run(s, 'Length(FieldEdges(b, "[X,A]"))')[0]["data"] == "8"  # 4 steps: a departure and an arrival each
    assert run(s, 'VectorFieldQ(b, "[X,A]")')[0]["data"] == "true"
    assert run(s, "Rounds(b)")[0]["data"] == "[10, 8]"  # both composites, then the squares
    anim = run(s, 'AnimateSaturation(X, ClosureRules("X"), Types = ["X+"])')[0]
    assert anim["type"] == "frames" and anim["labels"][1] == "round 1: added 6 X+"
    rule = run(s, 'CompositionRule("X", "A")')[0]
    assert rule["type"] == "svg" and "X;A" in rule["caption"]
    assert run(s, "[Mod(7, 3), EvenQ(4), OddQ(4)]")[0]["data"] == "[1, true, false]"


def test_cancellation_scalar_fields_and_homotopy():
    s = Session(budget=2e7)
    run(s, 'g = WithUnaryEdges(GridSpace(2, 3)); X = VectorFieldEdges(g, "X", {1: 2, 2: 1, 5: 5})')
    c = run(s, "c = Cancel(X)")[0]
    assert c["type"] == "frames" and "3 pairs" in c["caption"]
    assert run(s, 'Length(FieldEdges(CancelledFields(c), "X"))')[0]["data"] == "0"
    run(s, 'V = VectorFieldArcs(g, "V", [[1, [1, 2]], [[1, 2], 2]])')
    assert run(s, 'DeterministicQ(V, "V")')[0]["data"] == "true"
    run(s, 'f = ScalarField(g, "f", {1: 3, 2: 1})')
    assert run(s, 'FieldValues(f, "f")')[0]["data"].startswith("[[1, 3], [2, 1]")
    run(s, 'k = CompareValues(f, "f", 2, 1)')
    assert run(s, "[ComparisonSign(k), ComparisonSize(k)]")[0]["data"] == '["↑", 2]'
    run(s, "h = ExampleSpace(); loop = Motion(h, [1, [1, 2], 2, [1, 2, 3], 1])")
    assert run(s, "[HomotopicQ(loop, Motion(h, [1])), IncidenceHomotopicQ(loop, Motion(h, [1]))]")[0]["data"] == (
        "[true, false]"
    )
    g = run(s, 'FundamentalGroup(TriangulatedAnnulus(4), "a1")')
    assert g[0]["data"].startswith("π1 at a1: ⟨g") and len(g) == 2
