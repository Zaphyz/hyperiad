"""Hyperiad builtins for equational identities of plex products (hypermatrix.plex_identities).

Terms are written as undefined calls, ``M(a, T(b))``: the head names an operation in a dictionary
``{"M": mm, "T": tr}`` of plex diagrams, the rest are variables. An identity displays drawn: the one
diagram its terms share, output indices named i, j, k in order and contracted ones p, q, ...
(solid dark), captioned with the equation.
"""

from __future__ import annotations

import math
from typing import Any

import sympy
from frontier.hypergraphs import Edge, Hypergraph, HypergraphRule, Term, Vertex
from frontier.hypergraphs.svg import hypergraph_svg
from hypermatrix import (
    ArrayError,
    plex_identity_classes,
    plex_normal_forms,
    plex_term_diagram,
    plex_terms,
)
from sympy.core.function import AppliedUndef

from hyperiad.display import register_display
from hyperiad.errors import HyperiadError
from hyperiad.kernel import builtin
from hyperiad.values import format_value

_NAMES = "pqrstuvwlmnoabcdefgh"


class PlexIdentity(dict):
    """Terms equal as operations, with the diagram they share: ``id["Terms"]``, ``id["Diagram"]``."""


def _term(x: Any) -> Any:
    """A term as written in Hyperiad (undefined calls are sympy functions or inert terms) as a :class:`Term`."""
    if isinstance(x, AppliedUndef):
        return Term(x.func.__name__, [_term(a) for a in x.args])
    if isinstance(x, Term):
        return Term(x.head, [_term(a) for a in x.args])
    if isinstance(x, list):
        raise HyperiadError(
            f"a term is an operation applied to terms or a variable, got the list {format_value(x, 60)}"
        )
    return x


def _operations(ops: Any, name: str) -> dict[str, Hypergraph]:
    if not isinstance(ops, dict) or not all(isinstance(k, str) and isinstance(v, Hypergraph) for k, v in ops.items()):
        raise HyperiadError(f'{name} takes the operations as {{"M": diagram, ...}}, got {format_value(ops, 60)}')
    return ops


def _order(order: Any) -> int | None:
    if order is None or (isinstance(order, sympy.Symbol) and order.name == "Automatic"):
        return None
    if isinstance(order, int) and not isinstance(order, bool) and order >= 0:
        return order
    raise HyperiadError(f"Order is the order of a bare variable's array, got {format_value(order, 40)}")


def _guard(f, *args, **kw):
    try:
        return f(*args, **kw)
    except ArrayError as e:
        raise HyperiadError(str(e)) from None


def named_diagram(hg: Hypergraph, out: list) -> Hypergraph:
    """The term diagram with its output vertices named i, j, k, ... in order and contracted ones p, q, r, ..."""
    along = dict.fromkeys(i for e in sorted(hg.edges, key=lambda e: str(e.label)) for i in e.ids)
    inner = [v for v in along if v not in out]
    outer = "ijk" if len(out) <= 3 else [f"i{k}" for k in range(1, len(out) + 1)]
    names = dict(zip(out, outer, strict=False))
    names |= {v: (_NAMES[k] if k < len(_NAMES) else f"p{k}") for k, v in enumerate(inner)}
    return Hypergraph(
        [Vertex(names[v.id], v.label) for v in hg.vertices],
        [Edge([names[i] for i in e.ids], e.symmetry, e.label) for e in hg.edges],
    )


def _identity(terms: list, ops: dict, order: int | None) -> PlexIdentity:
    hg, out = _guard(plex_term_diagram, terms[0], ops, order=order)
    return PlexIdentity(Terms=list(terms), Diagram=named_diagram(hg, out))


def _show_identity(x: PlexIdentity) -> list[dict]:
    equation = " = ".join(format_value(t, 200) for t in x["Terms"])
    return [{"type": "svg", "data": hypergraph_svg(x["Diagram"], labels=True), "caption": equation}]


register_display(PlexIdentity, _show_identity)


# -- builtins ------------------------------------------------------------------------------------


@builtin(
    "PlexTermDiagram",
    """PlexTermDiagram(t, ops) is the plex diagram of the term t in the operations ops = {"M": d1, ...} (plex
diagrams): each operation replaced by its diagram, an argument's output glued to the plex it fills, contracted
indices renamed apart. The free vertices, in vertex order, are the term's output. A bare variable needs
Order = n.""",
)
def _term_diagram(t, ops, Order=None):  # noqa: N803 - option names as written
    operations = _operations(ops, "PlexTermDiagram")
    hg, out = _guard(plex_term_diagram, _term(t), operations, order=_order(Order))
    return named_diagram(hg, out)


@builtin(
    "PlexEqualQ",
    """PlexEqualQ(t1, t2, ops) is true when the terms t1 and t2 are the same operation on arrays: their diagrams are
isomorphic by a map that fixes every variable and the output order (over every commutative semiring and in every
dimension; for terms using each variable once this decides equality over the integers).""",
)
def _equal_q(t1, t2, ops, Order=None):  # noqa: N803
    operations = _operations(ops, "PlexEqualQ")
    terms = [_term(t1), _term(t2)]
    return len(_guard(plex_identity_classes, terms, operations, order=_order(Order))) == 1


def _terms_cost(ops, variables, Order=2, MaxOperations=2, **_):  # noqa: N803
    """Ordered splittings tried: per operation node, arity^(variables) assignments, over the nodes allowed."""
    try:
        n = len(variables)
        arities = [len({e.label for e in d.edges}) for d in ops.values()]
        per_node = sum(max(a, 1) ** n for a in arities)
        return float(per_node) ** max(1, int(MaxOperations))
    except (TypeError, AttributeError):
        return 1.0


@builtin(
    "PlexTerms",
    """PlexTerms(ops, vars) is every term in the operations ops = {"M": d1, ...} in which each of the variables vars
occurs exactly once, with at most MaxOperations = 2 operations; all variables are arrays of order Order = 2, and so
is each term. These are the terms PlexIdentities searches among.""",
    cost=_terms_cost,
)
def _terms(ops, variables, Order=2, MaxOperations=2):  # noqa: N803
    operations = _operations(ops, "PlexTerms")
    if not isinstance(variables, list) or len(set(variables)) != len(variables):
        raise HyperiadError(f"PlexTerms takes a list of distinct variables, got {format_value(variables, 60)}")
    order, most = _order(Order), MaxOperations
    if order is None or not isinstance(most, int) or isinstance(most, bool) or most < 0:
        raise HyperiadError("PlexTerms needs Order = n and MaxOperations = m, nonnegative integers")
    return _guard(plex_terms, operations, variables, order=order, max_operations=most)


def _identities_cost(terms, ops, **_):
    try:
        return float(len(terms)) * 50.0
    except TypeError:
        return 1.0


@builtin(
    "PlexIdentities",
    """PlexIdentities(terms, ops) groups the terms into classes of equal operations (PlexEqualQ) and gives the
classes with more than one term: each an identity, shown as the diagram its terms share, captioned with the
equation. id["Terms"] and id["Diagram"] take it apart. All = true also gives the classes of one term.""",
    cost=_identities_cost,
)
def _identities(terms, ops, All=False, Order=None):  # noqa: N803
    operations = _operations(ops, "PlexIdentities")
    if not isinstance(terms, list):
        raise HyperiadError(f"PlexIdentities takes a list of terms, got {format_value(terms, 60)}")
    order = _order(Order)
    classes = _guard(plex_identity_classes, [_term(t) for t in terms], operations, order=order)
    return [_identity(c, operations, order) for c in classes if All or len(c) > 1]


def _rules(rules: Any, name: str) -> list[HypergraphRule]:
    items = rules if isinstance(rules, list) else [rules]
    if not all(isinstance(r, HypergraphRule) for r in items):
        raise HyperiadError(f"{name} takes plex rules (from PlexRule), got {format_value(rules, 60)}")
    return items


def _normal_cost(d, rules, **_):
    try:
        n = len(d.edges)
        return float(math.factorial(n)) * n * n * len(rules if isinstance(rules, list) else [rules])
    except (AttributeError, TypeError):
        return 1.0


@builtin(
    "PlexNormalForms",
    """PlexNormalForms(d, rules) is the diagrams reached by rewriting the plex diagram d with the plex rules (from
PlexRule) at valid events, in every possible order, until no rule applies. Each rule must shrink the diagram.
Every normal form has the value of d: when they are single plexes their labels are equal terms, and
PlexNormalTerms gives those terms.""",
    cost=_normal_cost,
)
def _normal_forms(d, rules):
    if not isinstance(d, Hypergraph):
        raise HyperiadError(f"PlexNormalForms takes a plex diagram, got {format_value(d, 60)}")
    return _guard(plex_normal_forms, d, _rules(rules, "PlexNormalForms"))


@builtin(
    "PlexNormalTerms",
    """PlexNormalTerms(d, rules) is the terms of the normal forms of d that are single plexes (PlexNormalForms): all
equal to d's product, so together an identity, associativity read as confluence. Plex names that are names
("a") appear as variables (a).""",
    cost=_normal_cost,
)
def _normal_terms(d, rules):
    forms = _normal_forms(d, rules)
    return [_as_variables(f.edges[0].label) for f in forms if len(f.edges) == 1]


def _as_variables(t: Any) -> Any:
    """A normal form's term with plex names that are names (``"a"``) written as variables (``a``)."""
    if isinstance(t, Term):
        return Term(t.head, [_as_variables(a) for a in t.args])
    return sympy.Symbol(t) if isinstance(t, str) and t.isidentifier() else t
