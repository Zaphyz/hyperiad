import numpy as np
import pytest
from frontier.hypergraphs import Edge, Hypergraph, Term, Vertex, label_text
from hypermatrix import (
    ArrayError,
    plex_diagram,
    plex_identity_classes,
    plex_normal_forms,
    plex_product,
    plex_rule,
    plex_term_diagram,
    plex_term_key,
    plex_term_value,
    plex_terms,
)

M = plex_diagram([["i", "j"], ["j", "k"]], ["j"], names=["x", "y"])
T = Hypergraph([Vertex("i"), Vertex("j")], [Edge(["j", "i"], "Directed", "x")])
F = plex_diagram([["i", "j", "p"], ["q", "r", "p"], ["q", "r", "k"]], ["p", "q", "r"], names=["x", "y", "z"])
OPS = {"M": M, "T": T, "F": F}


def t(head, *args):
    return Term(head, args)


def key(term, order=2):
    return plex_term_key(term, OPS, order=order)


def test_transposition_and_matrix_identities():
    a, b, c = "abc"
    assert key(t("T", t("T", a))) == key(a)
    assert key(t("T", t("M", a, b))) == key(t("M", t("T", b), t("T", a)))
    assert key(t("M", t("M", a, b), c)) == key(t("M", a, t("M", b, c)))
    assert key(t("M", a, b)) != key(t("M", b, a))  # names are fixed, not up to renaming
    assert key(t("T", a)) != key(a)  # the output order counts
    assert key(t("T", t("M", a, a))) == key(t("M", t("T", a), t("T", a)))  # a repeated variable


def test_expansion_agrees_with_evaluation():
    rng = np.random.default_rng(1)
    arrays = {v: rng.integers(-3, 4, (3, 3)) for v in "abc"}
    term = t("M", t("T", t("M", "a", t("T", "b"))), "c")
    hg, out = plex_term_diagram(term, OPS)
    direct = plex_product(hg, arrays, output=out)
    by_steps = plex_term_value(term, OPS, arrays)
    expected = (arrays["a"] @ arrays["b"].T).T @ arrays["c"]
    assert np.array_equal(np.asarray(direct.data, dtype=int), expected)
    assert np.array_equal(np.asarray(by_steps.data, dtype=int), expected)


def test_search_finds_the_antihomomorphism():
    terms = plex_terms({"M": M, "T": T}, ("a", "b"), order=2, max_operations=3)
    classes = plex_identity_classes(terms, {"M": M, "T": T})
    texts = [[label_text(x) for x in c] for c in classes]
    assert ["M(T(b), T(a))", "T(M(a, b))"] in texts
    assert len(terms) == 20 and len(classes) == 8


def test_fish_semiheap_by_confluence_and_by_search():
    long = plex_diagram(
        [[1, 2, 3], [4, 5, 3], [4, 5, 6], [7, 8, 6], [7, 8, 9]], [3, 4, 5, 6, 7, 8], names=list("abcde")
    )
    forms = plex_normal_forms(long, [plex_rule(F, "F")])
    found = sorted(label_text(f.edges[0].label) for f in forms)
    assert found == ["F(F(a, b, c), d, e)", "F(a, F(d, c, b), e)", "F(a, b, F(c, d, e))"]
    terms = plex_terms({"F": F}, tuple("abcde"), order=3, max_operations=2)
    two = [x for x in terms if isinstance(x, Term) and any(isinstance(y, Term) for y in x.args)]
    classes = plex_identity_classes(two, {"F": F})
    assert len(two) == 360 and sorted({len(c) for c in classes}) == [3] and len(classes) == 120


def test_errors():
    with pytest.raises(ArrayError, match="takes 2 arguments"):
        plex_term_diagram(t("M", "a"), OPS)
    with pytest.raises(ArrayError, match="needs the order"):
        plex_term_diagram("a", OPS)
    with pytest.raises(ArrayError, match="more plexes in its input"):
        plex_normal_forms(M, [plex_rule(T, "T")])
