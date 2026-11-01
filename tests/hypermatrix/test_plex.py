"""Plex diagrams, plex products and plex rewriting.

Plex diagrams are not in the paclet's tested code: the checks here come from the paper's §4 and
from the identities checked in the paclet's prototype notebook reference/PlexAlgebra.wl.
"""

import itertools
import random

import numpy as np
import pytest
from frontier.hypergraphs import Edge, Hypergraph, Term, canonical_hypergraph, isomorphic
from frontier.hypergraphs.svg import hypergraph_svg
from hypermatrix import (
    PLEX_MARK,
    ArrayError,
    ArrayObject,
    identity_array,
    partial_identity_array,
    plex_contracted,
    plex_diagram,
    plex_events,
    plex_free,
    plex_names,
    plex_product,
    plex_rule,
)

RNG = np.random.default_rng(20261027)


def rand(*dims):
    return RNG.integers(-2, 3, dims)


def symmetric(d, n=3):
    a = rand(*([d] * n))
    return sum(np.transpose(a, p) for p in itertools.permutations(range(n)))


def same(x, y):
    return ArrayObject(x) == ArrayObject(y)


matmul = plex_diagram([[1, 2], [2, 3]], [2], names=["a", "b"])
bm = plex_diagram([[1, 2, 4], [1, 4, 3], [4, 2, 3]], [4], names=["x", "y", "z"])  # the prototype's plex3


def test_a_plex_diagram_is_a_labelled_hypergraph():
    assert isinstance(matmul, Hypergraph)
    assert [e.label for e in matmul.edges] == ["a", "b"]
    assert (plex_free(matmul), plex_contracted(matmul)) == ([1, 3], [2])
    assert [v.label for v in matmul.vertices] == [None, PLEX_MARK, None]
    assert plex_names(bm) == ["x", "y", "z"]
    assert plex_names(plex_diagram([[1, 2], [2, 3]])) == ["p1", "p2"]
    with pytest.raises(ArrayError, match="lie on no plex"):
        plex_diagram([[1, 2]], [5])
    # markings are labels, so isomorphism respects them
    other = plex_diagram([["u", "v"], ["v", "w"]], ["v"], names=["a", "b"])
    assert isomorphic(matmul, other)
    assert not isomorphic(matmul, plex_diagram([[1, 2], [2, 3]], [1], names=["a", "b"]))
    assert canonical_hypergraph(matmul) == canonical_hypergraph(other)


def test_plex_products_are_the_contractions_they_draw():
    a, b = rand(3, 3), rand(3, 3)
    assert same(plex_product(matmul, {"a": a, "b": b}).data, a @ b)
    assert same(plex_product(matmul, [a, b]).data, a @ b)  # positional, in plex-name order
    x, y = rand(3, 3, 3), rand(3, 3, 3)
    diamond = plex_diagram([[1, 2, 3], [2, 3, 4]], [2, 3], names=["x", "y"])  # the prototype's DiamProd
    assert same(plex_product(diamond, [x, y]).data, np.einsum("ipq,pqj->ij", x, y))
    for spec in ("ijp,ipk,pjk->ijk", "ipq,pjq,pqk->ijk", "ipq,pjr,qrk->ijk", "ijp,qrp,qrk->ijk", "ipq,pjq,pjk->ijk"):
        ins, out = spec.split("->")  # BM, Blades, TriForce, Fish, Boat of the prototype
        parts = [list(w) for w in ins.split(",")]
        diagram = plex_diagram(parts, sorted(set(ins.replace(",", "")) - set(out)), names=["x", "y", "z"])
        arrays = [rand(3, 3, 3) for _ in range(3)]
        assert plex_free(diagram) == list(out)
        assert same(plex_product(diagram, arrays).data, np.einsum(spec, *arrays))


def test_an_incidence_keeps_every_index():
    m, a = rand(2, 3), rand(3, 2, 2)
    incidence = plex_diagram([["i", "j"], ["j", "l", "k"]], [], names=["m", "a"])
    assert plex_free(incidence) == ["i", "j", "k", "l"]
    assert same(plex_product(incidence, [a, m]).data, np.einsum("ij,jlk->ijkl", m, a))


def test_output_order_can_be_given():
    a, b = rand(2, 3), rand(3, 4)
    assert same(plex_product(matmul, [a, b], output=[3, 1]).data, (a @ b).T)
    with pytest.raises(ArrayError, match="must list the free vertices"):
        plex_product(matmul, [a, b], output=[1, 2])


def test_edge_symmetry_is_array_symmetry():
    sym_diagram = plex_diagram([Edge([1, 2, 4], "Unordered", "x"), Edge([4, 3], "Directed", "m")], [4])
    s, m = symmetric(3), rand(3, 3)
    assert same(plex_product(sym_diagram, {"x": s, "m": m}).data, np.einsum("ijp,pk->ijk", s, m))
    with pytest.raises(ArrayError, match="must be symmetric"):
        plex_product(sym_diagram, {"x": rand(3, 3, 3), "m": m})
    # on an unordered edge the vertex order written is immaterial
    flipped = plex_diagram([Edge([4, 2, 1], "Unordered", "x"), Edge([4, 3], "Directed", "m")], [4])
    assert flipped == sym_diagram


def test_the_prototype_identities():
    # reference/PlexAlgebra.wl, section "Plex Diagram -> Array Product"
    al, be, ga, de = (symmetric(3) for _ in range(4))
    # The prototype checks BM(α, β, γ) == BM(α, γ, β) for symmetric arrays. As arrays that is false;
    # it holds after exchanging the first two indices, i.e. as plexes (up to reordering).
    assert plex_product(bm, [al, be, ga]) != plex_product(bm, [al, ga, be])
    assert plex_product(bm, [al, be, ga], output=[2, 1, 3]) == plex_product(bm, [al, ga, be])
    ir = partial_identity_array(3, 2, [1, 2])
    ig, ib = partial_identity_array(3, 2, [2, 3]), partial_identity_array(3, 2, [1, 3])
    ident = identity_array(3, 2)
    assert plex_product(bm, [ir, ir, ir]) == ir
    assert same(plex_product(bm, [ident, ident, ident]).data, ident.data)
    al2 = symmetric(2)
    assert same(plex_product(bm, [al2, ig, ib]).data, al2)
    i3 = identity_array(3, 3)

    def bmp(*xs):
        return plex_product(bm, list(xs))

    assert bmp(i3, be, bmp(i3, ga, de)) == bmp(i3, bmp(i3, be, ga), de)


def test_drawing_shows_the_markings():
    from frontier.hypergraphs.geometry import MARKED_VERTEX

    dark = "rgb({},{},{})".format(*(round(255 * c) for c in MARKED_VERTEX[:3]))
    svg = hypergraph_svg(matmul)
    assert svg.count(f'fill="{dark}"') == 1  # the one contracted index, drawn solid dark
    assert f">{PLEX_MARK}</text>" not in svg and ">a</text>" in svg
    labelled = hypergraph_svg(matmul, labels=True)
    assert ">2</text>" in labelled and ">2: •</text>" not in labelled  # its name, without the dot


# -- rewriting -------------------------------------------------------------------------------------

chain = plex_diagram([[1, 2], [2, 3], [3, 4]], [2, 3], names=["a", "b", "c"])


def test_a_plex_diagram_is_a_rewrite_rule():
    rule = plex_rule(matmul, "M")
    assert rule["DeletedVertices"] == [2]  # the contracted index
    assert [e.ids for e in rule.output.edges] == [(1, 3)]
    label = rule.output.edges[0].label
    assert isinstance(label, Term) and label.head == "M" and len(label.args) == 2


def test_associativity_appears_as_confluence():
    rule = plex_rule(matmul, "M")
    first = plex_events(rule, chain)
    assert [ev.match_edge_positions for ev in first] == [[0, 1], [1, 2]]
    finals = [ev2.hypergraph for ev in first for ev2 in plex_events(rule, ev.hypergraph)]
    labels = [h.edges[0].label for h in finals]
    assert labels == [Term("M", [Term("M", ["a", "b"]), "c"]), Term("M", ["a", Term("M", ["b", "c"])])]
    arrays = {"a": rand(2, 3), "b": rand(3, 4), "c": rand(4, 2)}
    values = [plex_product(h, arrays, diagrams={"M": matmul}) for h in finals]
    assert values[0] == values[1] == plex_product(chain, arrays)


def test_only_events_that_really_contract_count():
    # vertex 2 is also used by the plex d, so contracting a·b there is not legitimate
    shared = plex_diagram([[1, 2], [2, 3], [2, 5]], [2], names=["a", "b", "d"])
    rule = plex_rule(matmul, "M")
    assert len(rule(shared)) > 0 and plex_events(rule, shared) == []


def _random_case(rng):
    m = rng.randint(3, 6)
    edges, used = [], set()
    for k in range(rng.randint(2, 4)):
        arity = rng.choice([1, 2, 2, 3])
        vs = rng.sample(range(1, m + 1), arity)
        sym = "Unordered" if arity == 3 and rng.random() < 0.3 else "Directed"
        edges.append(Edge(vs, sym, f"e{k}"))
        used |= set(vs)
    marked = [v for v in used if rng.random() < 0.5]
    d = plex_diagram(edges, marked)
    env = {}
    for e in d.edges:
        env[e.label] = symmetric(2, e.arity) if e.symmetry == "Unordered" else rand(*([2] * e.arity))
    chosen = rng.sample(list(d.edges), rng.randint(1, len(d.edges)))
    inside = {v for e in chosen for v in e.ids}
    outside = {v for e in d.edges if e not in chosen for v in e.ids}
    internal = [v for v in plex_contracted(d) if v in inside and v not in outside]
    p = plex_diagram(chosen, internal)
    return d, p, env


def test_rewriting_then_evaluating_equals_evaluating():
    """The claim hypermatrix.plex_rewriting_evaluates, on random plex diagrams and arrays."""
    rng = random.Random(7)
    checked = 0
    for _ in range(200):
        d, p, env = _random_case(rng)
        target = plex_product(d, env)
        for ev in plex_events(plex_rule(p, "P"), d):
            assert plex_product(ev.hypergraph, env, diagrams={"P": p}) == target
            checked += 1
    assert checked >= 150
