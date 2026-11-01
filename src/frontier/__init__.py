"""Shared infrastructure for the frontier-math research directions.

* :mod:`frontier.registry` -- concepts, claims, evidence, notation
* :mod:`frontier.formal` -- the formalization reference engine (SymPy, Lean/Mathlib)
* :mod:`frontier.refs` -- pinned external references (reference implementations, papers)
* :mod:`frontier.wl` -- reading Wolfram Language reference sources
* :mod:`frontier.present` -- catalog export and the user-facing interface
* :mod:`frontier.themes` -- research themes, their questions and cost estimates
* :mod:`frontier.budget` -- compute tiers and run-time extrapolation
* :mod:`frontier.viz` -- figures for session summaries
"""

from frontier.keys import DIRECTIONS
from frontier.notation import Notation
from frontier.registry import (
    REGISTRY,
    Claim,
    ClaimStatus,
    Concept,
    Evidence,
    EvidenceKind,
    FormalRef,
    Issue,
    Maturity,
    Registry,
    RegistryError,
    SourceRef,
    claim,
    concept,
    define,
    symbol,
)

__all__ = [
    "DIRECTIONS",
    "REGISTRY",
    "Claim",
    "ClaimStatus",
    "Concept",
    "Evidence",
    "EvidenceKind",
    "FormalRef",
    "Issue",
    "Maturity",
    "Notation",
    "Registry",
    "RegistryError",
    "SourceRef",
    "claim",
    "concept",
    "define",
    "symbol",
]
