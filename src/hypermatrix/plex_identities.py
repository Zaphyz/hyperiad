"""Equational identities of plex products: terms, their diagrams, and the search for identities.

A **term** is an expression in named plex products: a :class:`Term` ``P(t1, ..., tk)`` whose head
names a plex diagram and whose arguments are terms or variables (any other label). Plex algebra
turns terms and diagrams into each other in two directions:

* **expansion** (term to diagram, :func:`plex_term_diagram`): substitute each operation's diagram
  for it, gluing an argument's output vertices to the slot it fills and giving every contracted
  vertex a fresh name. The result is one plex diagram whose edges are the variables;
* **confluence** (diagram to terms, :func:`plex_normal_forms`): rewrite a diagram with plex rules
  (:func:`hypermatrix.plex.plex_rule`) in every possible order until no rule applies. Each normal
  form is a term, and by ``hypermatrix.plex_rewriting_evaluates`` every one of them has the
  value of the diagram, so the distinct normal forms are an identity (the paper's associativity
  as confluence).

Two terms are equal as operations when their diagrams are isomorphic by a map that fixes every
variable and the output order (:func:`plex_term_key` is a complete invariant for that). The
search (:func:`plex_identity_classes`) groups terms by it; :func:`plex_terms` enumerates the
multilinear terms to search among.
"""

from __future__ import annotations

import itertools
from collections.abc import Hashable, Mapping, Sequence
from functools import cache
from typing import Any

from frontier.hypergraphs import Edge, Hypergraph, HypergraphRule, Term, Vertex
from frontier.hypergraphs.core import order_key
from frontier.registry import Evidence, EvidenceKind, SourceRef, claim, define

from hypermatrix.arrays import ArrayError, ArrayObject, as_array
from hypermatrix.plex import PLEX_MARK, plex_contracted, plex_events, plex_free, plex_names, plex_product

Operations = Mapping[str, Hypergraph]


def _operation(operations: Operations, head: str) -> Hypergraph:
    if head not in operations:
        raise ArrayError("PlexTerm::operation", f"{head!r} is not one of the operations {sorted(operations)!r}")
    return operations[head]


def _arguments(term: Term, diagram: Hypergraph) -> dict[Any, Any]:
    names = plex_names(diagram)
    if len(term.args) != len(names):
        raise ArrayError(
            "PlexTerm::arity",
            f"{term.head} takes {len(names)} arguments (its plexes {names!r}), given {len(term.args)}",
        )
    return dict(zip(names, term.args, strict=True))


def term_operations(term: Any) -> int:
    """The number of operations in a term."""
    return 1 + sum(term_operations(a) for a in term.args) if isinstance(term, Term) else 0


def _slot_orders(term: Any, operations: Operations, found: dict) -> None:
    if not isinstance(term, Term):
        return
    diagram = _operation(operations, term.head)
    args = _arguments(term, diagram)
    for e in diagram.edges:
        a = args[e.label]
        if not isinstance(a, Term):
            if found.setdefault(a, e.arity) != e.arity:
                raise ArrayError(
                    "PlexTerm::order", f"the variable {a!r} fills slots of orders {found[a]} and {e.arity}"
                )
        _slot_orders(a, operations, found)


@define(
    "hypermatrix.plex_term_diagram",
    "Diagram of a plex term",
    r"The diagram of a term $P(t_1, \ldots, t_k)$ in plex products is $P$'s diagram with each plex "
    r"replaced by the diagram of the term that fills it: that term's output vertices are glued to the "
    r"plex's vertices in order, and its contracted vertices are renamed apart. A variable is a single "
    r"plex of its own name. The free vertices, in the order of the outermost operation's output, are the "
    r"output of the term.",
    maturity="stable",
    depends_on=("hypermatrix.plex_diagram", "hypermatrix.plex_product"),
    sources=(SourceRef("heaps-of-fish", "section:The Plex Formalism", "definition"),),
    tags=("plex", "identities"),
)
def plex_term_diagram(
    term: Any, operations: Operations, *, order: int | None = None
) -> tuple[Hypergraph, list[Hashable]]:
    """The plex diagram of ``term`` and its output vertices, in order.

    Vertices are numbered ``1, 2, ...``. A bare variable needs ``order``, the order of its array.
    Each argument sits on its slot as the slot's edge does: a variable keeps the slot's symmetry
    type (a composite argument on a symmetric slot must itself be symmetric, which is not checked).
    """
    count = itertools.count(1)
    edges: list[Edge] = []
    contracted: list[int] = []

    def expand(t: Any, out: list[int], symmetry: str) -> None:
        if not isinstance(t, Term):
            edges.append(Edge(out, symmetry, t))
            return
        diagram = _operation(operations, t.head)
        args = _arguments(t, diagram)
        free = plex_free(diagram)
        if len(free) != len(out):
            raise ArrayError("PlexTerm::order", f"{t.head} has {len(free)} free indices, its slot {len(out)}")
        name = dict(zip(free, out, strict=True))
        for c in plex_contracted(diagram):
            name[c] = next(count)
            contracted.append(name[c])
        for e in diagram.edges:
            expand(args[e.label], [name[i] for i in e.ids], e.symmetry)

    if isinstance(term, Term):
        width = len(plex_free(_operation(operations, term.head)))
    elif order is None:
        raise ArrayError("PlexTerm::order", f"the bare variable {term!r} needs the order of its array")
    else:
        width = order
    out = [next(count) for _ in range(width)]
    expand(term, out, "Directed")
    ids = list(dict.fromkeys([*out, *(i for e in edges for i in e.ids)]))
    marked = set(contracted)
    return Hypergraph([Vertex(v, PLEX_MARK) if v in marked else Vertex(v) for v in ids], edges), out


def _orderings(e: Edge) -> list[tuple]:
    if e.symmetry == "Directed" or e.arity <= 1:
        return [e.ids]
    if e.symmetry == "Cyclic":
        return [e.ids[r:] + e.ids[:r] for r in range(e.arity)]
    return list(dict.fromkeys(itertools.permutations(e.ids)))


def diagram_key(hg: Hypergraph, output: Sequence[Hashable]) -> tuple:
    """A complete invariant of a plex diagram with an output sequence, up to renaming its vertices.

    Equal keys mean an isomorphism that keeps every edge label (the variable names, not just up to
    renaming as in :func:`frontier.hypergraphs.canonical_hypergraph`), the output order and the
    contracted vertices. Vertices are numbered in order of first appearance along the output and then
    the edges sorted by label; edges sharing a label, and the vertex orders a symmetric edge allows,
    are tried in every arrangement and the least numbering is kept.
    """
    by_label: dict[Any, list[Edge]] = {}
    for e in hg.edges:
        by_label.setdefault(e.label, []).append(e)
    labels = sorted(by_label, key=order_key)
    marked = set(plex_contracted(hg))
    arrangements = [
        [list(zip(perm, choice, strict=True)) for perm in itertools.permutations(by_label[lab])
         for choice in itertools.product(*(_orderings(e) for e in perm))]
        for lab in labels
    ]  # fmt: skip
    best = None
    for combo in itertools.product(*arrangements):
        index: dict[Hashable, int] = {}
        for v in output:
            index.setdefault(v, len(index))
        body = []
        for lab, placed in zip(labels, combo, strict=True):
            for e, ids in placed:
                for v in ids:
                    index.setdefault(v, len(index))
                body.append((order_key(lab), e.symmetry, tuple(index[v] for v in ids)))
        key = (
            tuple(index[v] for v in output),
            tuple(body),
            tuple(sorted(index[v] for v in marked)),
            len(hg.vertices),
        )
        if best is None or key < best:
            best = key
    return best if best is not None else ((), (), (), len(hg.vertices))


def plex_term_key(term: Any, operations: Operations, *, order: int | None = None) -> tuple:
    """The key of a term: equal keys exactly when the terms' diagrams are isomorphic fixing names and output."""
    return diagram_key(*plex_term_diagram(term, operations, order=order))


@define(
    "hypermatrix.plex_identity_search",
    "Identity search for plex products",
    r"Terms in plex products are sorted into classes by the isomorphism type of their diagrams, the "
    r"isomorphisms fixing every variable and the output order. Terms in one class are equal as "
    r"operations on arrays; a class with several terms is an identity.",
    maturity="stable",
    depends_on=("hypermatrix.plex_term_diagram",),
    tags=("plex", "identities"),
)
def plex_identity_classes(terms: Sequence[Any], operations: Operations, *, order: int | None = None) -> list[list[Any]]:
    """The terms grouped into classes of equal operations, classes and terms in first-seen order.

    ``order`` is the order of a bare variable, needed only if one of the terms is a variable; it is
    otherwise read off the slots the variables fill."""
    if order is None:
        found: dict = {}
        for t in terms:
            _slot_orders(t, operations, found)
        orders = set(found.values())
        order = orders.pop() if len(orders) == 1 else None
    classes: dict[tuple, list[Any]] = {}
    for t in terms:
        classes.setdefault(plex_term_key(t, operations, order=order), []).append(t)
    return list(classes.values())


def plex_terms(operations: Operations, variables: Sequence[Any], *, order: int, max_operations: int) -> list[Any]:
    """Every term with at most ``max_operations`` operations in which each variable occurs exactly once.

    All variables are arrays of order ``order``; terms are kept only where every slot gets an
    argument of its order. The result has output order ``order`` too."""
    signature = {}
    for name, diagram in operations.items():
        slots = []
        for plex in plex_names(diagram):
            arities = {e.arity for e in diagram.edges if e.label == plex}
            slots.append(arities.pop() if len(arities) == 1 else None)
        signature[name] = (len(plex_free(diagram)), tuple(slots))

    @cache
    def terms(vs: tuple, want: int, budget: int) -> tuple:
        found: list[Any] = []
        if len(vs) == 1 and want == order:
            found.append(vs[0])
        if budget == 0:
            return tuple(found)
        for name, (width, slots) in signature.items():
            if width != want or None in slots or len(slots) > len(vs):
                continue
            for blocks in _ordered_partitions(vs, len(slots)):
                options = [terms(b, s, budget - 1) for b, s in zip(blocks, slots, strict=True)]
                for args in itertools.product(*options):
                    if sum(term_operations(a) for a in args) <= budget - 1:
                        found.append(Term(name, args))
        return tuple(found)

    return list(terms(tuple(variables), order, max_operations))


def _ordered_partitions(vs: tuple, k: int):
    """The ways to split ``vs`` into ``k`` nonempty labelled blocks, each block in the order of ``vs``."""
    for assignment in itertools.product(range(k), repeat=len(vs)):
        if len(set(assignment)) == k:
            yield [tuple(v for v, a in zip(vs, assignment, strict=True) if a == j) for j in range(k)]


def plex_term_value(term: Any, operations: Operations, arrays: Mapping[Any, Any]) -> ArrayObject:
    """Evaluate a term operation by operation, each with :func:`hypermatrix.plex.plex_product`."""
    if not isinstance(term, Term):
        if term not in arrays:
            raise ArrayError("PlexTerm::name", f"no array is given for the variable {term!r}")
        return as_array(arrays[term])
    diagram = _operation(operations, term.head)
    args = _arguments(term, diagram)
    return plex_product(diagram, {n: plex_term_value(a, operations, arrays) for n, a in args.items()})


@define(
    "hypermatrix.plex_normal_forms",
    "Normal forms of a plex diagram",
    r"The normal forms of a plex diagram under a set of plex rules: the diagrams reached by rewriting "
    r"with valid plex events (each contracting its match's marked vertices) in every possible order "
    r"until no rule applies. When a normal form is a single plex, its label is a term whose value is the "
    r"diagram's; distinct such terms are equal, which is associativity read as confluence.",
    maturity="stable",
    depends_on=("hypermatrix.plex_rule",),
    sources=(SourceRef("heaps-of-fish", "section:Associativity as Confluence", "definition"),),
    tags=("plex", "rewriting", "identities"),
)
def plex_normal_forms(
    hg: Hypergraph, rules: Sequence[HypergraphRule], *, max_states: int = 100_000
) -> list[Hypergraph]:
    """The distinct normal forms reachable from ``hg``, in first-reached order.

    Every rule must shrink the diagram (more plexes in its input than in its output), so rewriting
    terminates; a transposition-like rule (one plex to one plex) would rewrite forever."""
    for rule in rules:
        if len(rule.input.edges) <= len(rule.output.edges):
            raise ArrayError(
                "PlexNormalForms::rule",
                "a rule must have more plexes in its input than in its output, or it may not end",
            )
    seen: set = set()
    found: dict = {}
    stack = [hg]
    while stack:
        g = stack.pop()
        if g in seen:
            continue
        seen.add(g)
        if len(seen) > max_states:
            raise ArrayError("PlexNormalForms::size", f"more than {max_states} diagrams reached")
        successors = [ev.hypergraph for rule in rules for ev in plex_events(rule, g)]
        if not successors:
            found.setdefault(g, None)
        stack.extend(reversed(successors))
    return list(found)


claim(
    "hypermatrix.plex_identity_criterion",
    r"Two terms in plex products whose diagrams are isomorphic by a map fixing every variable and the "
    r"output order are equal as operations, over every commutative semiring and in every dimension. "
    r"Conversely, for terms in which each variable occurs once, over $\mathbb{Z}$ (or any semiring where "
    r"positive integers are nonzero) and in dimension $d \ge$ the number of vertices, non-isomorphic "
    r"diagrams give different operations: the identity search decides equality of such terms.",
    about=("hypermatrix.plex_identity_search", "hypermatrix.plex_term_diagram"),
    status="supported",
    evidence=[
        Evidence(
            EvidenceKind.ARGUMENT,
            "expansion: the value of a term is the plex product of its diagram, by plex_rewriting_evaluates read "
            "backwards (substitution of one plex product into another merges the sums over contracted indices); "
            "an isomorphism fixing names and output only renames summation indices. Converse: the coefficient "
            "of the monomial given by an injective assignment of indices to the vertices of one diagram is a "
            "positive count of assignments of the other's vertices with the same index tuples on every named "
            "plex; each such assignment is a surjective name- and output-preserving homomorphism, and "
            "homomorphisms both ways between finite diagrams are isomorphisms",
        ),
        Evidence(
            EvidenceKind.NUMERICAL,
            "every class found in the review examples agrees on random integer arrays of dimension 3, and "
            "terms in different classes disagree",
            source="themes/hypermatrix-higher-arity/questions/q02-plex-identities/experiments/2026-10-28-plex-identities.py",
        ),
    ],
)

_REVIEW = "themes/hypermatrix-higher-arity/questions/q02-plex-identities/experiments/2026-10-28-plex-identities.py"

claim(
    "hypermatrix.transpose_involutive_antihomomorphism",
    r"With $M$ the matrix product and $T$ the transposition (both plex diagrams), the identity search over "
    r"terms in two matrices with at most three operations finds $T(T(a)) = a$ and "
    r"$T(M(a, b)) = M(T(b), T(a))$, and every identity class it finds is a consequence of these two; over "
    r"three matrices it finds the associativity $M(M(a, b), c) = M(a, M(b, c))$.",
    about=("hypermatrix.plex_identity_search",),
    status="supported",
    evidence=[Evidence(EvidenceKind.NUMERICAL, "identity search with numerical check", source=_REVIEW)],
)

claim(
    "hypermatrix.fish_semiheap",
    r"The fish product $F(a, b, c)_{ijk} = \sum_{p,q,r} a_{ijp} b_{qrp} c_{qrk}$ is a semiheap: rewriting "
    r"the long fish of five plexes with the fish rule in every order gives exactly the three normal forms "
    r"$F(F(a,b,c),d,e)$, $F(a,F(d,c,b),e)$, $F(a,b,F(c,d,e))$. Among the 360 terms with two fish "
    r"operations in five distinct variables, the identity classes are exactly the instances of this axiom.",
    about=("hypermatrix.plex_normal_forms", "hypermatrix.plex_identity_search"),
    status="supported",
    evidence=[
        Evidence(EvidenceKind.LITERATURE, "heaps-of-fish, Theorem (Fish Associativity)"),
        Evidence(EvidenceKind.NUMERICAL, "confluence and identity search, checked numerically", source=_REVIEW),
    ],
    sources=(SourceRef("heaps-of-fish", "label:fishassoc", "result"),),
)

claim(
    "hypermatrix.dagger_semiheap",
    r"The ternary product $\eta(a, b, c) = a\, b^{\top} c$ of matrices (the plex diagram "
    r"$a_{ip} b_{qp} c_{qk}$) is a semiheap, found by confluence on the zigzag chain of five matrices; "
    r"its three terms equal the chain of matrix products with infix transposes "
    r"$a\, b^{\top} c\, d^{\top} e$, and the middle one rests on $T$ being an involutive antihomomorphism.",
    about=("hypermatrix.plex_normal_forms", "hypermatrix.plex_identity_search"),
    status="supported",
    evidence=[Evidence(EvidenceKind.NUMERICAL, "confluence and identity search, checked numerically", source=_REVIEW)],
    sources=(SourceRef("heaps-of-fish", "section:Heaps", "definition"),),
)
