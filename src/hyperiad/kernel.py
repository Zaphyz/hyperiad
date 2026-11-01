"""Evaluating Hyperiad: sessions, scopes, builtins and the cost budget.

A :class:`Session` evaluates cells. Each cell is parsed to a tree and evaluated
against the session's global scope; its value and anything printed become the cell's
outputs (:mod:`hyperiad.display`).

Builtins are Python functions registered with :func:`builtin`. One that can be
expensive declares a ``cost`` estimator; when a session has a ``budget`` and the
estimate exceeds it, the call is refused with :class:`TooExpensive` before any work
is done, so the front end can offer to run the cell on a server.
"""

from __future__ import annotations

import inspect
import random
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Number
from typing import Any

import numpy as np
import sympy
from frontier.hypergraphs import Event, Hypergraph, HypergraphError, HypergraphRule, Term
from frontier.hypergraphs.causality import EventRecord, RewritingSystem, SystemGraph
from hypermatrix import PLEX_MARK, ArrayError, ArrayObject, FiniteField, Hypermatrix
from hypermatrix.arrays import entries_of

from hyperiad import arith
from hyperiad.errors import HyperiadError, TooExpensive
from hyperiad.expr import Expr, Symbol
from hyperiad.syntax import ParseError, parse
from hyperiad.values import FunctionValue, OptionValue, RuleValue, event_property, format_value, graph_property

CONSTANTS: dict[str, Any] = {
    "Pi": sympy.pi,
    "E": sympy.E,
    "I": sympy.I,
    "CausalDependenceTypes": ["Elements", "Edges", "Vertices"],  # the paclet's $CausalDependenceTypes
    "PlexMark": PLEX_MARK,  # the vertex label of a contracted index in a plex diagram
}


@dataclass
class Builtin:
    name: str
    fn: Callable[..., Any]
    usage: str
    cost: Callable[..., float] | None = None
    session: bool = False  # receives the session as first argument
    special: bool = False  # receives unevaluated argument trees and the scope

    def __repr__(self) -> str:
        return self.name


BUILTINS: dict[str, Builtin] = {}


def builtin(name: str, usage: str, cost: Callable[..., float] | None = None, session: bool = False):
    def register(fn):
        BUILTINS[name] = Builtin(name, fn, usage.strip(), cost, session)
        return fn

    return register


def special(name: str, usage: str):
    def register(fn):
        BUILTINS[name] = Builtin(name, fn, usage.strip(), special=True)
        return fn

    return register


@dataclass
class Scope:
    vars: dict[str, Any] = field(default_factory=dict)
    parent: Scope | None = None

    def lookup(self, name: str) -> tuple[bool, Any]:
        scope: Scope | None = self
        while scope is not None:
            if name in scope.vars:
                return True, scope.vars[name]
            scope = scope.parent
        return False, None


_OPERATORS: dict[str, Callable[..., Any]] = {
    "Plus": arith.plus,
    "Subtract": arith.subtract,
    "Times": arith.times,
    "Divide": arith.divide,
    "Minus": arith.minus,
    "Power": arith.power,
    "Equal": arith.equal,
    "Unequal": arith.unequal,
    "Less": lambda a, b: arith.compare("Less", a, b),
    "LessEqual": lambda a, b: arith.compare("LessEqual", a, b),
    "Greater": lambda a, b: arith.compare("Greater", a, b),
    "GreaterEqual": lambda a, b: arith.compare("GreaterEqual", a, b),
}
_ARITY = {"Minus": 1, "Not": 1}


class Session:
    def __init__(self, budget: float | None = None, max_depth: int = 150, seed: int = 0) -> None:
        import hyperiad.builtins  # noqa: F401 - registers the builtins

        self.globals = Scope()
        self.budget = budget
        self.max_depth = max_depth
        self.rng = random.Random(seed)
        self.depth = 0
        self.printed: list[dict] = []

    # -- cells -----------------------------------------------------------

    def run(self, source: str) -> dict[str, Any]:
        """Evaluate one cell; never raises. Returns ``{status, outputs, elapsed, ...}``."""
        from hyperiad.display import display, error_output

        self.printed = []
        start = time.perf_counter()
        result: dict[str, Any] = {"status": "ok"}
        caught: list = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                value = self.eval(parse(source), self.globals)
                outputs = display(value)
            if value is not None:
                self.globals.vars["ans"] = value
        except ParseError as e:
            result = {"status": "error", "line": e.line, "column": e.col}
            outputs = [error_output(f"Syntax error: {e}")]
        except TooExpensive as e:
            result = {"status": "too_expensive", "estimate": e.estimate, "budget": e.budget, "builtin": e.name}
            outputs = [{**error_output(str(e), kind="too_expensive"), "estimate": e.estimate, "budget": e.budget}]
        except HyperiadError as e:
            result = {"status": "error"}
            outputs = [error_output(str(e))]
        except RecursionError:
            result = {"status": "error"}
            outputs = [error_output("Recursion too deep")]
        except Exception as e:  # noqa: BLE001 - a bug here must not kill the notebook
            result = {"status": "error"}
            outputs = [error_output(f"Internal error ({type(e).__name__}): {e}")]
        notes = [{"type": "text", "kind": "warning", "data": f"Note: {w.message}"} for w in dict.fromkeys(caught, None)]
        result["outputs"] = self.printed + notes + outputs
        result["elapsed"] = time.perf_counter() - start
        return result

    # -- evaluation ------------------------------------------------------

    def eval(self, e: Any, scope: Scope) -> Any:
        if isinstance(e, Symbol):
            return self.lookup(e.name, scope)
        if not isinstance(e, Expr):
            return e
        name = e.head_name
        if name is not None:
            found, _ = scope.lookup(name)
            if not found:
                b = BUILTINS.get(name)
                if b is not None and b.special:
                    return b.fn(self, e.args, scope)
                if name in _OPERATORS and len(e.args) == _ARITY.get(name, 2):
                    return _OPERATORS[name](*(self.eval(a, scope) for a in e.args))
                core = getattr(self, f"_eval_{name}", None)
                if core is not None:
                    return core(e.args, scope)
        head = self.eval(e.head, scope)
        args = [self.eval(a, scope) for a in e.args]
        return self.apply(head, args, name or format_value(head, 40))

    def lookup(self, name: str, scope: Scope) -> Any:
        found, value = scope.lookup(name)
        if found:
            return value
        if name in CONSTANTS:
            return CONSTANTS[name]
        if name in BUILTINS:
            return BUILTINS[name]
        return sympy.Symbol(name)  # an undefined name stands for itself, symbolically

    def apply(self, f: Any, args: list[Any], name: str = "function") -> Any:
        if isinstance(f, Builtin):
            return self.call_builtin(f, args)
        if isinstance(f, FunctionValue):
            return self.call_function(f, args)
        if isinstance(f, FiniteField):
            return f(*args)
        if isinstance(f, HypergraphRule):  # rule(hg) applies the rule, as in the paclet
            return self.call_builtin(BUILTINS["HypergraphRuleApply"], [f, *args])
        if isinstance(f, RewritingSystem):  # sys("CausalGraph", 2, ...) as the paclet's sys["CausalGraph", 2]
            return self.call_builtin(BUILTINS["RewritingSystemProperty"], [f, *args])
        if isinstance(f, sympy.Symbol):  # an undefined function applied: it stays as it is
            if args and all(isinstance(a, (Number, sympy.Basic)) and not isinstance(a, bool) for a in args):
                return sympy.Function(f.name)(*(sympy.sympify(a) for a in args))
            try:
                return Term(f.name, args)  # inert, e.g. a label Row(["(", x, y, ")"])
            except TypeError:
                raise HyperiadError(f"{f.name} is not defined") from None
        raise HyperiadError(f"{format_value(f, 60)} is not a function")

    def call_builtin(self, b: Builtin, args: list[Any]) -> Any:
        if b.special:
            raise HyperiadError(f"{b.name} cannot be applied to evaluated arguments")
        pos = [a for a in args if not isinstance(a, OptionValue)]
        opts = {a.name: a.value for a in args if isinstance(a, OptionValue)}
        call_args = [self, *pos] if b.session else pos
        try:
            inspect.signature(b.fn).bind(*call_args, **opts)
        except TypeError as e:
            raise HyperiadError(f"{b.name}: wrong arguments ({e}). {b.usage.splitlines()[0]}") from None
        try:
            if b.cost is not None and self.budget is not None:
                estimate = b.cost(*pos, **opts)
                if estimate > self.budget:
                    raise TooExpensive(b.name, estimate, self.budget)
            return arith.norm(b.fn(*call_args, **opts))
        except (HyperiadError, RecursionError):
            raise
        except (ArrayError, HypergraphError) as e:
            raise HyperiadError(f"{b.name}: {e}") from None
        except (ValueError, TypeError, KeyError, ZeroDivisionError, IndexError, NotImplementedError) as e:
            raise HyperiadError(f"{b.name}: {e}") from None

    def call_function(self, f: FunctionValue, args: list[Any]) -> Any:
        if len(args) != len(f.params):
            label = f.name or "function"
            raise HyperiadError(f"{label} takes {len(f.params)} argument(s), got {len(args)}")
        if self.depth >= self.max_depth:
            raise HyperiadError("Recursion too deep")
        self.depth += 1
        try:
            return self.eval(f.body, Scope(dict(zip(f.params, args, strict=True)), f.scope))
        finally:
            self.depth -= 1

    # -- core forms ------------------------------------------------------

    def _eval_List(self, args, scope):
        return [self.eval(a, scope) for a in args]

    def _eval_Association(self, args, scope):
        out = {}
        for pair in args:
            if not (isinstance(pair, Expr) and pair.head_name == "Rule" and len(pair.args) == 2):
                raise HyperiadError("an association holds key: value pairs")
            key = self.eval(pair.args[0], scope)
            try:
                hash(key)
            except TypeError:
                raise HyperiadError(f"{format_value(key, 60)} cannot be an association key") from None
            out[key] = self.eval(pair.args[1], scope)
        return out

    def _eval_Rule(self, args, scope):
        if len(args) != 2:
            raise HyperiadError("a rule has a left and a right side")
        return RuleValue(self.eval(args[0], scope), self.eval(args[1], scope))

    def _eval_Option(self, args, scope):
        if len(args) != 2 or not isinstance(args[0], Symbol):
            raise HyperiadError("an option is written name = value")
        return OptionValue(args[0].name, self.eval(args[1], scope))

    def _eval_Not(self, args, scope):
        v = arith.norm(self.eval(args[0], scope))
        if isinstance(v, bool):
            return not v
        return arith.norm(sympy.Not(v))

    def _eval_And(self, args, scope):
        return self._logic(args, scope, sympy.And, stop=False)

    def _eval_Or(self, args, scope):
        return self._logic(args, scope, sympy.Or, stop=True)

    def _logic(self, args, scope, combine, stop: bool):
        symbolic = []
        for a in args:
            v = arith.norm(self.eval(a, scope))
            if v is stop:
                return stop
            if not isinstance(v, bool):
                symbolic.append(v)
        return arith.norm(combine(*symbolic)) if symbolic else (not stop)

    def _eval_Part(self, args, scope):
        target = self.eval(args[0], scope)
        keys = [self.eval(a, scope) for a in args[1:]]
        if isinstance(target, RewritingSystem):  # sys["StateCount", 2]: a property and its generations
            return self.call_builtin(BUILTINS["RewritingSystemProperty"], [target, *keys])
        for k in keys:
            target = part(target, k)
        return target

    def _eval_CompoundExpression(self, args, scope):
        value = None
        for a in args:
            value = self.eval(a, scope)
        return value

    def _eval_Function(self, args, scope):
        params = args[0]
        if not (
            isinstance(params, Expr) and params.head_name == "List" and all(isinstance(p, Symbol) for p in params.args)
        ):
            raise HyperiadError("function parameters must be names")
        return FunctionValue(tuple(p.name for p in params.args), args[1], scope)

    def _eval_Set(self, args, scope):
        lhs, rhs = args
        if isinstance(lhs, Symbol):
            _check_assignable(lhs.name)
            value = self.eval(rhs, scope)
            scope.vars[lhs.name] = value
            return value
        if isinstance(lhs, Expr) and isinstance(lhs.head, Symbol) and lhs.head_name not in ("Part",):
            if not all(isinstance(p, Symbol) for p in lhs.args):
                raise HyperiadError("a definition f(x, y) = ... takes plain names as parameters")
            _check_assignable(lhs.head.name)
            scope.vars[lhs.head.name] = FunctionValue(tuple(p.name for p in lhs.args), rhs, scope, lhs.head.name)
            return None
        raise HyperiadError("only names and f(x, ...) can be assigned to")


def _check_assignable(name: str) -> None:
    core = name in _OPERATORS or hasattr(Session, f"_eval_{name}")
    if name in BUILTINS or name in CONSTANTS or core:
        raise HyperiadError(f"{name} is built in and cannot be redefined")


def part(target: Any, k: Any) -> Any:
    """``target[k]``: 1-based position, a key, or an array or hypergraph property name."""
    if isinstance(target, (Hypergraph, HypergraphRule)):
        if isinstance(k, str):
            return target.prop(k)
        raise HyperiadError('hypergraph and rule properties are taken by name, e.g. hg["Arity"]')
    if isinstance(target, Hypermatrix):
        if isinstance(k, str):
            try:
                return target.prop(k)
            except KeyError as e:
                raise HyperiadError(str(e).strip("'\"")) from None
        n = len(target)
        if isinstance(k, int) and not isinstance(k, bool) and (1 <= k <= n or -n <= k <= -1):
            return target.arrays[k - 1 if k > 0 else k]
        raise HyperiadError(f"a hypermatrix of {n} arrays has parts 1..{n}, or properties by name")
    if isinstance(target, EventRecord):
        if k not in EventRecord.KEYS:
            raise HyperiadError(f"an event record has the keys {', '.join(EventRecord.KEYS)}")
        return target[k]
    if isinstance(target, SystemGraph):
        return graph_property(target, k)
    if isinstance(target, Event):
        if k not in Event.KEYS:
            raise HyperiadError(f"an event has the keys {', '.join(Event.KEYS)}")
        return event_property(target, k)
    if isinstance(target, ArrayObject):
        if isinstance(k, str):
            return target.prop(k)
        if isinstance(k, int) and not isinstance(k, bool) and target.order >= 1:
            n = target.dims[0]
            if not (1 <= k <= n or -n <= k <= -1):
                raise HyperiadError(f"position {k} is outside 1..{n}")
            sub = target.data[k - 1 if k > 0 else k]
            if isinstance(sub, np.ndarray) and sub.ndim:
                return ArrayObject._trusted(sub)
            # a single entry: NumPy gives a scalar, or the bare object for object arrays
            return arith.norm(entries_of(np.asarray(sub, dtype=object)))
        raise HyperiadError(f"cannot take part {format_value(k, 40)} of an array of order {target.order}")
    if isinstance(target, (list, str)):
        if not (isinstance(k, int) and not isinstance(k, bool)):
            raise HyperiadError("list positions are integers, counted from 1 (negative from the end)")
        n = len(target)
        if not (1 <= k <= n or -n <= k <= -1):
            raise HyperiadError(f"position {k} is outside 1..{n}")
        return target[k - 1 if k > 0 else k]
    if isinstance(target, dict):
        if k not in target:
            raise HyperiadError(f"no key {format_value(k, 40)}")
        return target[k]
    prop = getattr(target, "prop", None)
    if callable(prop) and isinstance(k, str):  # any other value with named properties (e.g. a saturation)
        try:
            return prop(k)
        except KeyError as e:
            raise HyperiadError(str(e).strip("'\"")) from None
    raise HyperiadError(f"{format_value(target, 40)} has no parts")
