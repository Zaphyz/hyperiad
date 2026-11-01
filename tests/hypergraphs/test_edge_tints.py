"""Special hyperedges keep the geometry and only change colour."""

from frontier.hypergraphs.geometry import EDGE_LINE, EDGE_TINTS, Area, Curve, hypergraph_scene
from frontier.hypergraphs.svg import hypergraph_svg

EDGES = [(1, 2, 3), (3, 4, 5)]


def _shapes(scene, kind):
    return [s for s in scene.shapes if isinstance(s, kind)]


def test_tint_changes_colours_not_geometry():
    plain = hypergraph_scene(EDGES, symmetry="directed", seed=1)
    tinted = hypergraph_scene(EDGES, symmetry="directed", seed=1, edge_tints=["red", None])
    a, b = _shapes(plain, Curve), _shapes(tinted, Curve)
    assert len(a) == len(b) and all((x.points == y.points).all() for x, y in zip(a, b, strict=True))
    line, _ = EDGE_TINTS["red"]
    assert any(c.color == line for c in b) and any(c.color == EDGE_LINE for c in b)
    assert any(
        isinstance(s, Area) and s.fill == EDGE_TINTS["white"][1]
        for s in hypergraph_scene(EDGES, symmetry="directed", seed=1, edge_tints=["white"]).shapes
    )


def test_svg_accepts_tints_by_label():
    from frontier.hypergraphs import Edge, Hypergraph

    hg = Hypergraph([1, 2, 3, 4, 5], [Edge([1, 2, 3], "Directed", "x"), Edge([3, 4, 5], "Directed", "W")])
    assert "<svg" in hypergraph_svg(hg, edge_tints={"W": "white"})


def test_white_label_is_an_outlined_white_letter():
    from frontier.hypergraphs.geometry import EDGE_TINT_LABELS, Label

    scene = hypergraph_scene(EDGES, symmetry="directed", seed=1, edge_text=["W", "x"], edge_tints=["white", None])
    labels = {s.text: s for s in scene.shapes if isinstance(s, Label)}
    assert (labels["W"].color, labels["W"].outline) == EDGE_TINT_LABELS["white"]
    assert labels["x"].outline is None
