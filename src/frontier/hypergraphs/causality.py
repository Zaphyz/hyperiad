"""Rewriting systems and causality, ported from the reference paclet's ``src/Causality.wl``.

A rewriting system is a list of rules with a state to start from. Evolving it applies every
rule in every way it fits to every state of the last generation, for as many generations as
asked. What comes out is a **tree of state occurrences** joined by events: a state reached two
ways appears twice, since which elements it is made of depends on how it was reached. Merging
equal states is what the ``StatesGraph`` does, for display.

Elements are tracked by **token**: a counter minted when an element is created and carried for
as long as it survives. Two equal edges are still two elements, and a vertex name freed by one
event may be handed out again by a later one.

Two events are **causally related** when the second acts on a state on the path below the first
and what they touch overlaps. An event touches the edges it consumed and created and the
vertices it matched, created and destroyed. The ``CausalDependence`` option says what counts:
``"Elements"`` (the default, the paclet's: any shared vertex or edge), ``"Edges"`` (whole edges
only) or ``"Vertices"``; ``"Elements"`` is exactly the union of the other two.

States and events are **numbered from 1** in the order they occur, as the paclet numbers them;
these numbers name the vertices of the graphs. The options keep the paclet's names.
"""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from frontier.hypergraphs.canon import canonical_hypergraph
from frontier.hypergraphs.core import Hypergraph, HypergraphError
from frontier.hypergraphs.rewriting import Event, HypergraphRule, apply_rule
from frontier.registry import SourceRef, define

CAUSAL_DEPENDENCE_TYPES = ("Elements", "Edges", "Vertices")

EVOLUTION_OPTIONS = {
    "MaxStates": 2048,
    "Labeled": True,
    "LabelPatterns": True,
    "DistinctVertices": True,
    "DistinctEdges": True,
}
CAUSAL_OPTIONS = {"CausalDependence": "Elements", "TransitiveReduction": True, "CanonicalizeStates": True}
DRAWING_OPTIONS = {
    "StateRendering": None,  # None is Automatic: draw unless there are more than MaxDrawnVertices
    "EventRendering": None,
    "MaxDrawnVertices": 24,
    "PanelSize": 90,
    "HypergraphPlotOptions": {},
}
REWRITING_OPTIONS = {**EVOLUTION_OPTIONS, **CAUSAL_OPTIONS, **DRAWING_OPTIONS}

PROPERTIES = (
    "Rules",
    "RuleCount",
    "InitialState",
    "Options",
    "States",
    "StateList",
    "StateCount",
    "EventList",
    "EventCount",
    "EvolutionGraph",
    "StatesGraph",
    "BranchialGraph",
    "CausalGraph",
    "CausalEdges",
    "EvolutionCausalGraph",
    "TokenEventGraph",
)
_STATIC = ("Rules", "RuleCount", "InitialState", "Options", "Properties")


def _check_options(options: dict) -> dict:
    unknown = [k for k in options if k not in REWRITING_OPTIONS]
    if unknown:
        raise HypergraphError(
            "HypergraphRewritingSystem::opt",
            f"unknown option {unknown[0]!r}; the options are {', '.join(REWRITING_OPTIONS)}",
        )
    return dict(options)


# -- the evolution --------------------------------------------------------------------------


@dataclass
class StateOccurrence:
    hypergraph: Hypergraph
    edge_tokens: list[int]  # one per edge, in EdgeList order
    vertex_tokens: dict[Hashable, int]  # vertex id -> token
    generation: int
    parent: int | None  # state number
    in_event: int | None  # event number


@dataclass
class EventRecord:
    """An event of the evolution: the paclet's record, keyed ``Rule``, ``RuleIndex``, ..."""

    rule: HypergraphRule
    rule_index: int  # from 1
    generation: int
    input_state: int
    consumed_edges: list[int]
    created_edges: list[int]
    matched_vertices: list[int]
    created_vertices: list[int]
    destroyed_vertices: list[int]
    data: Event  # what HypergraphRuleApply gave
    output_state: int = 0
    number: int = 0  # the event's own number, from 1

    KEYS = (
        "Rule",
        "RuleIndex",
        "Generation",
        "InputState",
        "ConsumedEdges",
        "CreatedEdges",
        "MatchedVertices",
        "CreatedVertices",
        "DestroyedVertices",
        "Data",
        "OutputState",
    )

    def __getitem__(self, key: str) -> Any:
        if key not in self.KEYS:
            raise KeyError(key)
        return getattr(self, "".join("_" + c.lower() if c.isupper() else c for c in key).lstrip("_"))

    def touched(self, kind: str) -> set[tuple[str, int]]:
        """What the event touched, tagged ``("E", token)`` or ``("V", token)``."""
        edges = {("E", t) for t in [*self.consumed_edges, *self.created_edges]}
        vertices = {("V", t) for t in [*self.matched_vertices, *self.created_vertices, *self.destroyed_vertices]}
        return {"Edges": edges, "Vertices": vertices, "Elements": edges | vertices}[kind]

    def consumed(self, kind: str) -> list[tuple[str, int]]:
        edges = [("E", t) for t in self.consumed_edges]
        vertices = [("V", t) for t in sorted({*self.matched_vertices, *self.destroyed_vertices})]
        return {"Edges": edges, "Vertices": vertices, "Elements": edges + vertices}[kind]

    def produced(self, kind: str) -> list[tuple[str, int]]:
        edges = [("E", t) for t in self.created_edges]
        vertices = [("V", t) for t in sorted({*self.matched_vertices, *self.created_vertices})]
        return {"Edges": edges, "Vertices": vertices, "Elements": edges + vertices}[kind]


@dataclass
class CausalEdge:
    source: int  # event numbers
    target: int
    shared_edges: list[int]
    shared_vertices: list[int]

    KEYS = ("From", "To", "SharedEdges", "SharedVertices")

    def __getitem__(self, key: str) -> Any:
        return {
            "From": self.source,
            "To": self.target,
            "SharedEdges": self.shared_edges,
            "SharedVertices": self.shared_vertices,
        }[key]


@dataclass
class Evolution:
    states: list[StateOccurrence]
    events: list[EventRecord]
    truncated: bool = False

    def ancestry(self, state: int) -> list[int]:
        """The state occurrences from ``state`` up to the root, ``state`` included."""
        path, k = [], state
        while k is not None:
            path.append(k)
            k = self.states[k - 1].parent
        return path

    def causal_edges(self, kind: str = "Elements") -> list[CausalEdge]:
        lines = [set(self.ancestry(e.input_state)) for e in self.events]
        touched = [e.touched(kind) for e in self.events]
        out = []
        for i, ei in enumerate(self.events):
            for j in range(len(self.events)):
                if ei.output_state in lines[j]:
                    shared = touched[i] & touched[j]
                    if shared:
                        out.append(
                            CausalEdge(
                                i + 1,
                                j + 1,
                                sorted(t for k, t in shared if k == "E"),
                                sorted(t for k, t in shared if k == "V"),
                            )
                        )
        return out


class _Mint:
    def __init__(self) -> None:
        self.edges = self.vertices = 0

    def edge_tokens(self, k: int) -> list[int]:
        out = list(range(self.edges + 1, self.edges + k + 1))
        self.edges += k
        return out

    def vertex_tokens(self, ids: Sequence[Hashable]) -> dict[Hashable, int]:
        out = {}
        for v in ids:
            self.vertices += 1
            out[v] = self.vertices
        return out


def _extend(state: StateOccurrence, ev: Event, rule_index: int, generation: int, event_number: int,
            input_state: int, mint: _Mint) -> tuple[StateOccurrence, EventRecord]:  # fmt: skip
    created = mint.edge_tokens(ev.edge_origins.count(None))
    it = iter(created)
    tokens = [next(it) if o is None else state.edge_tokens[o] for o in ev.edge_origins]
    created_vertices = mint.vertex_tokens(ev.new_vertices)
    kept = {v: t for v, t in state.vertex_tokens.items() if v not in set(ev.deleted_vertices)}
    old = state.vertex_tokens
    return (
        StateOccurrence(ev.hypergraph, tokens, {**kept, **created_vertices}, generation, input_state, event_number),
        EventRecord(
            rule=ev.rule,
            rule_index=rule_index,
            generation=generation,
            input_state=input_state,
            consumed_edges=[state.edge_tokens[p] for p in ev.match_edge_positions],
            created_edges=created,
            matched_vertices=[old[v] for v in ev.match_vertices if v in old],
            created_vertices=list(created_vertices.values()),
            destroyed_vertices=[old[v] for v in ev.deleted_vertices if v in old],
            data=ev,
            number=event_number,
        ),
    )


# -- the system -----------------------------------------------------------------------------


def _rules(spec: Any) -> tuple[HypergraphRule, ...]:
    rules = [spec] if isinstance(spec, HypergraphRule) else list(spec) if isinstance(spec, (list, tuple)) else None
    if rules is None or not all(isinstance(r, HypergraphRule) for r in rules):
        raise HypergraphError("HypergraphRewritingSystem::rules", f"{spec!r} is not a rewrite rule or a list of them")
    return tuple(rules)


def _initial(spec: Any) -> Hypergraph:
    if isinstance(spec, Hypergraph):
        return spec
    try:
        return Hypergraph(spec)
    except (HypergraphError, TypeError):
        raise HypergraphError(
            "HypergraphRewritingSystem::hg", f"{spec!r} is not a hypergraph, and cannot be an initial state"
        ) from None


@define(
    "common.hypergraph_rewriting_system",
    "Hypergraph rewriting system",
    r"A list of rewrite rules with an initial hypergraph. Its evolution to generation $n$ is the tree of "
    r"state occurrences obtained by applying every rule in every way to every state of the previous "
    r"generation; elements (edges and vertices) carry tokens from creation to destruction.",
    maturity="working",
    depends_on=("common.hypergraph_rule",),
    sources=(SourceRef("hypergraph-paclet", "symbol:HypergraphRewritingSystem", "implementation"),),
    tags=("hypergraph", "rewriting", "causality"),
)
class RewritingSystem:
    """``HypergraphRewritingSystem(rules, init, options)``: rules and an initial state, with options held."""

    __slots__ = ("rules", "initial", "options", "_cache")

    def __init__(self, rules: Any, initial: Any, options: dict | None = None, **more: Any) -> None:
        object.__setattr__(self, "rules", _rules(rules))
        object.__setattr__(self, "initial", _initial(initial))
        if options is not None and not isinstance(options, dict):
            raise HypergraphError("HypergraphRewritingSystem::spec", f"{options!r} is not a set of options")
        object.__setattr__(self, "options", _check_options({**(options or {}), **more}))
        object.__setattr__(self, "_cache", {})

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("RewritingSystem is immutable")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RewritingSystem) and (self.rules, self.initial, self.options) == (
            other.rules,
            other.initial,
            other.options,
        )

    def __hash__(self) -> int:
        return hash((self.rules, self.initial, tuple(sorted(self.options.items(), key=str))))

    def __repr__(self) -> str:
        return f"RewritingSystem({list(self.rules)!r}, {self.initial!r}, {self.options!r})"

    def options_in_force(self, **call: Any) -> dict:
        """Every option with its value: the call's, else the system's, else the default."""
        return {**REWRITING_OPTIONS, **self.options, **_check_options(call)}

    def evolve(self, n: int, **call: Any) -> Evolution:
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise HypergraphError(
                "HypergraphRewritingSystem::gen", f"{n!r} is not a non-negative integer number of generations"
            )
        o = self.options_in_force(**call)
        key = (n, *(o[k] for k in EVOLUTION_OPTIONS))
        if key not in self._cache:
            self._cache[key] = _evolve(self, n, o)
        return self._cache[key]

    def prop(self, name: str, n: int = 1, **call: Any) -> Any:
        if name == "Properties":
            return list(PROPERTIES)
        if name not in PROPERTIES:
            raise HypergraphError(
                "HypergraphRewritingSystem::prop", f"unknown property {name!r}; see sys['Properties']"
            )
        if name == "Rules":
            return list(self.rules)
        if name == "RuleCount":
            return len(self.rules)
        if name == "InitialState":
            return self.initial
        o = self.options_in_force(**call)
        if name == "Options":
            return o
        return _evolution_prop(self.evolve(n, **call), name, n, o)

    def __call__(self, name: str, n: int = 1, **call: Any) -> Any:
        return self.prop(name, n, **call)

    def __getitem__(self, name: str) -> Any:
        return self.prop(name)


def _evolve(system: RewritingSystem, n: int, o: dict) -> Evolution:
    mint = _Mint()
    init = system.initial
    states = [
        StateOccurrence(init, mint.edge_tokens(len(init.edges)), mint.vertex_tokens(init.vertex_ids), 0, None, None)
    ]
    events: list[EventRecord] = []
    apply_opts = {
        "labeled": o["Labeled"],
        "label_patterns": o["LabelPatterns"],
        "distinct_vertices": o["DistinctVertices"],
        "distinct_edges": o["DistinctEdges"],
    }
    frontier, truncated = [1], False
    for g in range(1, n + 1):
        nxt = []
        for si in frontier:
            for ri, rule in enumerate(system.rules, 1):
                for raw in apply_rule(rule, states[si - 1].hypergraph, **apply_opts):
                    if len(states) >= o["MaxStates"]:
                        truncated = True
                        break
                    state, record = _extend(states[si - 1], raw, ri, g, len(events) + 1, si, mint)
                    states.append(state)
                    record.output_state = len(states)
                    events.append(record)
                    nxt.append(len(states))
                if truncated:
                    break
            if truncated:
                break
        frontier = nxt
        if truncated:
            warnings.warn(
                f"the evolution reached the limit of {o['MaxStates']} states during generation {g} and was "
                "stopped there; raise MaxStates to go further",
                stacklevel=3,
            )
            break
    return Evolution(states, events, truncated)


def dependence_type(o: dict) -> str:
    t = o["CausalDependence"]
    if t not in CAUSAL_DEPENDENCE_TYPES:
        warnings.warn(
            f"{t!r} is not a causal dependence type; it should be one of {CAUSAL_DEPENDENCE_TYPES}. Using 'Elements'",
            stacklevel=3,
        )
        return "Elements"
    return t


# -- the graphs -----------------------------------------------------------------------------


@dataclass
class SystemGraph:
    """A graph of an evolution, with what each vertex stands for, so that it can be drawn.

    ``kinds`` maps each vertex to ``"state"``, ``"event"`` or ``"token"``; ``payload`` to the
    state's hypergraph, the :class:`EventRecord`, or the token. ``edge_kinds`` marks each edge
    ``"evolution"``, ``"causal"``, ``"branchial"`` or ``"token"``. ``drawn`` says whether states
    and events are drawn as pictures or as plain vertices: past ``MaxDrawnVertices`` they are
    plain, and drawing the graph says so.
    """

    name: str
    graph: nx.Graph
    kinds: dict
    payload: dict
    edge_kinds: dict = field(default_factory=dict)
    rendering: _Rendering = field(default_factory=lambda: _Rendering(True, 0, ""))
    layered: bool = True
    options: dict = field(default_factory=dict)

    @property
    def drawn(self) -> bool:
        """Whether states and events will be drawn as pictures (drawing warns when the limit says no)."""
        return self.rendering.decide({**REWRITING_OPTIONS, **self.options}, warn=False)

    def decide_drawn(self) -> bool:
        return self.rendering.decide({**REWRITING_OPTIONS, **self.options}, warn=True)

    @property
    def vertices(self) -> list:
        return list(self.graph.nodes)

    @property
    def edges(self) -> list:
        return list(self.graph.edges)

    @property
    def vertex_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def acyclic(self) -> bool:
        return nx.is_directed_acyclic_graph(self.graph) if self.graph.is_directed() else nx.is_forest(self.graph)


@dataclass(frozen=True)
class _Rendering:
    """Whether to draw states and events as pictures: the setting, and how many there would be."""

    setting: Any  # True, False, or None for Automatic
    count: int
    what: str

    def decide(self, o: dict, *, warn: bool) -> bool:
        if self.setting is True or self.setting is False:
            return self.setting
        limit = o["MaxDrawnVertices"]
        if isinstance(limit, int) and self.count > limit:
            if warn:
                warnings.warn(
                    f"there are {self.count} {self.what} to draw, more than the {limit} allowed by MaxDrawnVertices; "
                    "plain vertices are used instead. Raise that option, or set StateRendering or EventRendering "
                    "to true, to draw them anyway",
                    stacklevel=3,
                )
            return False
        return True


def _evolution_prop(e: Evolution, name: str, n: int, o: dict) -> Any:
    states, events = e.states, e.events
    if name == "StateList":
        return [s.hypergraph for s in states]
    if name == "StateCount":
        return len(states)
    if name == "States":
        groups: dict[int, list] = {}
        for s in states:
            groups.setdefault(s.generation, []).append(s.hypergraph)
        return list(groups.values())
    if name == "EventList":
        return list(events)
    if name == "EventCount":
        return len(events)
    if name == "CausalEdges":
        return e.causal_edges(dependence_type(o))
    return _GRAPHS[name](e, n, o)


def _causal_graph(e: Evolution, n: int, o: dict) -> SystemGraph:
    g = nx.DiGraph()
    g.add_nodes_from(range(1, len(e.events) + 1))
    g.add_edges_from((c.source, c.target) for c in e.causal_edges(dependence_type(o)))
    if o["TransitiveReduction"]:
        reduced = nx.transitive_reduction(g)
        kept = nx.DiGraph()
        kept.add_nodes_from(g.nodes)
        kept.add_edges_from(edge for edge in g.edges if reduced.has_edge(*edge))  # in the order found
        g = kept
    rendering = _Rendering(o["EventRendering"], len(e.events), "events")
    return SystemGraph(
        "CausalGraph",
        g,
        dict.fromkeys(g.nodes, "event"),
        {k: e.events[k - 1] for k in g.nodes},
        dict.fromkeys(g.edges, "causal"),
        rendering,
        options=o,
    )


def _evolution_graph(e: Evolution, n: int, o: dict) -> SystemGraph:
    g = nx.DiGraph()
    g.add_nodes_from(range(1, len(e.states) + 1))
    for k, ev in enumerate(e.events, 1):
        g.add_edge(ev.input_state, ev.output_state, event=k)
    rendering = _Rendering(o["StateRendering"], len(e.states), "states")
    payload = {i: s.hypergraph for i, s in enumerate(e.states, 1)}
    return SystemGraph(
        "EvolutionGraph",
        g,
        dict.fromkeys(g.nodes, "state"),
        payload,
        dict.fromkeys(g.edges, "evolution"),
        rendering,
        options=o,
    )


def _states_graph(e: Evolution, n: int, o: dict) -> SystemGraph:
    if o["CanonicalizeStates"]:
        canon = [canonical_hypergraph(s.hypergraph) for s in e.states]
    else:
        canon = [s.hypergraph for s in e.states]
    index: dict[Hypergraph, int] = {}
    representative: dict[int, Hypergraph] = {}
    for c, s in zip(canon, e.states, strict=True):
        if c not in index:
            index[c] = len(index) + 1
            representative[index[c]] = s.hypergraph
    g = nx.DiGraph()
    g.add_nodes_from(range(1, len(index) + 1))
    for ev in e.events:
        g.add_edge(index[canon[ev.input_state - 1]], index[canon[ev.output_state - 1]])
    rendering = _Rendering(o["StateRendering"], len(index), "states")
    return SystemGraph(
        "StatesGraph",
        g,
        dict.fromkeys(g.nodes, "state"),
        representative,
        dict.fromkeys(g.edges, "evolution"),
        rendering,
        options=o,
    )


def _branchial_graph(e: Evolution, n: int, o: dict) -> SystemGraph:
    same = [i for i, s in enumerate(e.states, 1) if s.generation == n]
    g = nx.Graph()
    g.add_nodes_from(same)
    groups: dict[Any, list[int]] = {}
    for i in same:
        groups.setdefault(e.states[i - 1].parent, []).append(i)
    for members in groups.values():
        g.add_edges_from((a, b) for k, a in enumerate(members) for b in members[k + 1 :])
    rendering = _Rendering(o["StateRendering"], len(same), "states")
    payload = {i: e.states[i - 1].hypergraph for i in same}
    return SystemGraph(
        "BranchialGraph",
        g,
        dict.fromkeys(g.nodes, "state"),
        payload,
        dict.fromkeys(g.edges, "branchial"),
        rendering,
        layered=False,
        options=o,
    )


def _evolution_causal_graph(e: Evolution, n: int, o: dict) -> SystemGraph:
    g = nx.DiGraph()
    kinds, payload, edge_kinds = {}, {}, {}
    for i, s in enumerate(e.states, 1):
        g.add_node(("State", i))
        kinds[("State", i)], payload[("State", i)] = "state", s.hypergraph
    for k, ev in enumerate(e.events, 1):
        g.add_node(("Event", k))
        kinds[("Event", k)], payload[("Event", k)] = "event", ev
    for k, ev in enumerate(e.events, 1):
        for a, b in ((("State", ev.input_state), ("Event", k)), (("Event", k), ("State", ev.output_state))):
            g.add_edge(a, b)
            edge_kinds[(a, b)] = "evolution"
    for c in e.causal_edges(dependence_type(o)):
        a, b = ("Event", c.source), ("Event", c.target)
        g.add_edge(a, b)
        edge_kinds[(a, b)] = "causal"
    rendering = _Rendering(o["StateRendering"], len(e.states) + len(e.events), "states and events")
    return SystemGraph("EvolutionCausalGraph", g, kinds, payload, edge_kinds, rendering, options=o)


def _token_event_graph(e: Evolution, n: int, o: dict) -> SystemGraph:
    kind = dependence_type(o)
    edges = []
    for k, ev in enumerate(e.events, 1):
        edges += [(("Token", t), ("Event", k)) for t in ev.consumed(kind)]
        edges += [(("Event", k), ("Token", t)) for t in ev.produced(kind)]
    tokens = list(dict.fromkeys(x for edge in edges for x in edge if x[0] == "Token"))
    g = nx.DiGraph()
    g.add_nodes_from(tokens)
    g.add_nodes_from(("Event", k) for k in range(1, len(e.events) + 1))
    g.add_edges_from(edges)
    kinds = {v: "token" if v[0] == "Token" else "event" for v in g.nodes}
    payload = {v: v[1] if v[0] == "Token" else e.events[v[1] - 1] for v in g.nodes}
    rendering = _Rendering(o["EventRendering"], len(e.events), "events")
    return SystemGraph("TokenEventGraph", g, kinds, payload, dict.fromkeys(g.edges, "token"), rendering, options=o)


_GRAPHS = {
    "CausalGraph": _causal_graph,
    "EvolutionGraph": _evolution_graph,
    "StatesGraph": _states_graph,
    "BranchialGraph": _branchial_graph,
    "EvolutionCausalGraph": _evolution_causal_graph,
    "TokenEventGraph": _token_event_graph,
}


@define(
    "common.hypergraph_causal_graph",
    "Causal graph",
    r"The causal graph of a rewriting system evolved $n$ generations has one vertex per event and an edge "
    r"$i \to j$ when event $j$ acts on a state on the path below event $i$'s output and the elements they "
    r"touch overlap (by default any vertex or edge token; `CausalDependence` = `\"Edges\"` or "
    r"`\"Vertices\"` restricts the currency), transitively reduced by default.",
    maturity="working",
    depends_on=("common.hypergraph_rewriting_system",),
    sources=(SourceRef("hypergraph-paclet", "symbol:HypergraphCausalGraph", "implementation"),),
    tags=("hypergraph", "rewriting", "causality"),
)
def causal_graph(system: Any, *args: Any, **options: Any) -> SystemGraph:
    """``causal_graph(sys, n=1, **opts)`` or ``causal_graph(rules, init, n=1, **opts)``."""
    if isinstance(system, RewritingSystem):
        n = args[0] if args else 1
        return system.prop("CausalGraph", n, **options)
    if not args:
        raise HypergraphError("HypergraphCausalGraph::sys", f"{system!r} is not a rewriting system")
    n = args[1] if len(args) > 1 else 1
    return RewritingSystem(system, args[0]).prop("CausalGraph", n, **options)
