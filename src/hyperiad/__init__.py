"""Hyperiad: the language and notebook system for hypergraph and hypermatrix research.

* :mod:`hyperiad.syntax` -- text syntax: parse to trees, print trees back
* :mod:`hyperiad.kernel` -- sessions, evaluation, builtins, cost budget
* :mod:`hyperiad.display` -- values as notebook outputs (pictures first)
* :mod:`hyperiad.server` -- development server: the notebook page plus an evaluation API

See docs/hyperiad.md for the design.
"""

from hyperiad.errors import HyperiadError, TooExpensive
from hyperiad.expr import Expr, Symbol
from hyperiad.kernel import Session
from hyperiad.syntax import ParseError, parse, unparse

__version__ = "0.1.0"

__all__ = ["Expr", "HyperiadError", "ParseError", "Session", "Symbol", "TooExpensive", "parse", "unparse"]
