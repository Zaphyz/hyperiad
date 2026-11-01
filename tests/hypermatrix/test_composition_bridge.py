"""Composition of hypergraphs = plex products of adjacency hypermatrices (hypermatrix.composition)."""

import random

import numpy as np
import pytest
import sympy
from frontier.hypergraphs import Edge, Hypergraph, HypergraphRule, Term, hypergraph_compose
from hypermatrix import composition_adjacency, composition_adjacency_product, composition_plex_diagram


def rule_of(spec: str, sym="Directed", out_sym="Directed", names="abc") -> HypergraphRule:
    ins, out = spec.split("->")
    lhs = [Edge(list(p), sym, names[j]) for j, p in enumerate(ins.split(","))]
    return HypergraphRule(Hypergraph(lhs), Hypergraph([Edge(list(out), out_sym)]))


def random_factor(rng, n, k, m, sym):
    return Hypergraph(
        list(range(n)), [Edge([rng.randrange(n) for _ in range(k)], sym) for _ in range(rng.randrange(m + 1))]
    )


CASES = [  # spec, input symmetry, output symmetry, vertices, max edges per factor
    ("ij,jk->ik", "Directed", "Directed", 4, 7),
    ("ijp,qrp,qrk->ijk", "Directed", "Directed", 3, 4),
    ("ijp,ipk,pjk->ijk", "Directed", "Directed", 3, 4),
    ("ipq,pjr,qrk->ijk", "Directed", "Directed", 3, 4),
    ("ij,jk->ik", "Unordered", "Directed", 4, 7),
    ("ij,jk->ik", "Unordered", "Unordered", 4, 7),
    ("ij,jk->ik", "Directed", "Unordered", 4, 7),
    ("ijp,qrp,qrk->ijk", "Unordered", "Unordered", 3, 4),
    ("ijp,pkq,qij->ijk", "Cyclic", "Cyclic", 3, 4),
]


@pytest.mark.parametrize("spec,sym,out_sym,n,m", CASES)
def test_composition_is_the_plex_product(spec, sym, out_sym, n, m):
    rng = random.Random(hash((spec, sym, out_sym)) % 1000)
    rule = rule_of(spec, sym, out_sym)
    k = len(spec.split(",")[0])
    for _ in range(25):
        factors = {name: random_factor(rng, n, k, m, sym) for name in "abc"[: len(spec.split(","))]}
        result = hypergraph_compose(rule, factors)
        expected = np.asarray(composition_adjacency_product(rule, factors).data)
        assert np.array_equal(composition_adjacency(result, rule, list(range(n))), expected)


def test_one_factor_twice_and_mixed_arities():
    rng = random.Random(3)
    square = rule_of("ij,jk->ik", names="aa")
    mixed = HypergraphRule(Hypergraph([Edge([1, 2], "Directed", "a"), Edge([2, 3, 4], "Directed", "a")]),
                           Hypergraph([Edge([1, 4])]))  # fmt: skip
    diagram, out = composition_plex_diagram(mixed)
    assert sorted(e.label for e in diagram.edges) == ["a[2,Directed]", "a[3,Directed]"] and out == [1, 4]
    for _ in range(25):
        g = random_factor(rng, 4, 2, 7, "Directed")
        r = hypergraph_compose(square, {"a": g})
        assert np.array_equal(composition_adjacency(r, square, list(range(4))),
                              np.asarray(composition_adjacency_product(square, {"a": g}).data))  # fmt: skip
        both = Hypergraph(list(range(3)), list(g.edges[:3]) + list(random_factor(rng, 3, 3, 4, "Directed").edges))
        both = Hypergraph(list(range(4)), [e for e in both.edges])
        r = hypergraph_compose(mixed, {"a": both})
        assert np.array_equal(composition_adjacency(r, mixed, list(range(4))),
                              np.asarray(composition_adjacency_product(mixed, {"a": both}).data))  # fmt: skip


def test_injective_and_boolean_variants():
    rng = random.Random(5)
    for spec, n in (("ij,jk->ik", 4), ("ijp,qrp,qrk->ijk", 4)):
        rule = rule_of(spec)
        k = len(spec.split(",")[0])
        for _ in range(15):
            factors = {name: random_factor(rng, n, k, 6, "Directed") for name in "abc"[: len(spec.split(","))]}
            injective = hypergraph_compose(rule, factors, distinct_vertices=True)
            assert np.array_equal(
                composition_adjacency(injective, rule, list(range(n))),
                np.asarray(composition_adjacency_product(rule, factors, distinct_vertices=True).data),
            )
            boolean = hypergraph_compose(rule, factors, multiplicity="Boolean")
            assert np.array_equal(
                composition_adjacency(boolean, rule, list(range(n))),
                np.asarray(composition_adjacency_product(rule, factors, multiplicity="Boolean").data),
            )
    # fmt: skip


def test_labels_are_the_path_semiring_product():
    x, y = sympy.symbols("x y", commutative=False)
    words = sympy.symbols("f0:6", commutative=False)
    rule = HypergraphRule(
        Hypergraph([Edge([1, 2], "Directed", Term("a", [x])), Edge([2, 3], "Directed", Term("b", [y]))]),
        Hypergraph([Edge([1, 3], "Directed", x * y)]),
    )
    rng = random.Random(7)
    n = 3
    for _ in range(20):
        factors = {}
        mats = {}
        for name in "ab":
            edges = [Edge([rng.randrange(n), rng.randrange(n)], "Directed", rng.choice(words)) for _ in range(5)]
            factors[name] = Hypergraph(list(range(n)), edges)
            m = [[sympy.Integer(0)] * n for _ in range(n)]
            for e in edges:
                m[e.ids[0]][e.ids[1]] += e.label
            mats[name] = m
        result = hypergraph_compose(rule, factors, multiplicity="Labels")
        got = [[sympy.Integer(0)] * n for _ in range(n)]
        for e in result.edges:
            got[e.ids[0]][e.ids[1]] += e.label
        for i in range(n):
            for k in range(n):
                want = sum((mats["a"][i][j] * mats["b"][j][k] for j in range(n)), sympy.Integer(0))
                assert sympy.expand(got[i][k] - want) == 0


IO_CASES = [  # input-output factors: (blocks per factor, block types per factor), output blocks and types
    # polycategory-like: f:(i j | p k), g:(p | l m) composed along p
    ([("ij", "pk"), ("p", "lm")], [("Unordered", "Unordered"), ("Directed", "Unordered")], ("ij", "lmk"),
     ("Unordered", "Unordered")),
    # multicategory: g:(i j | p), f:(p k | l), output (i j k | l)
    ([("ij", "p"), ("pk", "l")], [("Directed", "Directed"), ("Directed", "Directed")], ("ijk", "l"),
     ("Directed", "Directed")),
    # cyclic blocks, output reversed in type
    ([("ijk", "p"), ("p", "lm")], [("Cyclic", "Directed"), ("Directed", "Cyclic")], ("ijk", "lm"),
     ("Cyclic", "Cyclic")),
]  # fmt: skip


def random_io_factor(rng, n, sizes, types, m):
    return Hypergraph(
        list(range(n)),
        [
            Edge([[rng.randrange(n) for _ in range(sizes[0])], [rng.randrange(n) for _ in range(sizes[1])]], types)
            for _ in range(rng.randrange(m + 1))
        ],
    )


@pytest.mark.parametrize("blocks,types,out,out_types", IO_CASES)
def test_input_output_composition_is_the_plex_product(blocks, types, out, out_types):
    rng = random.Random(len(blocks[0][0]) + 7 * len(out[1]))
    lhs = [Edge([list(b[0]), list(b[1])], t, "ab"[j]) for j, (b, t) in enumerate(zip(blocks, types, strict=True))]
    rule = HypergraphRule(Hypergraph(lhs), Hypergraph([Edge([list(out[0]), list(out[1])], out_types)]))
    for _ in range(20):
        factors = {
            "ab"[j]: random_io_factor(rng, 3, (len(b[0]), len(b[1])), t, 4)
            for j, (b, t) in enumerate(zip(blocks, types, strict=True))
        }
        result = hypergraph_compose(rule, factors)
        expected = np.asarray(composition_adjacency_product(rule, factors).data)
        assert np.array_equal(composition_adjacency(result, rule, list(range(3))), expected)
