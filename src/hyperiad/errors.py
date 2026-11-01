"""Errors raised while evaluating Hyperiad code."""

from __future__ import annotations


class HyperiadError(Exception):
    """An error in the user's code or data, reported to the user as is."""


class TooExpensive(HyperiadError):
    """A computation whose estimated cost exceeds the session's budget.

    The front end can send the cell to a server session with a larger budget.
    """

    def __init__(self, name: str, estimate: float, budget: float) -> None:
        super().__init__(
            f"{name} would take about {estimate:.3g} operations, over this session's budget of {budget:.3g}"
        )
        self.name, self.estimate, self.budget = name, estimate, budget
