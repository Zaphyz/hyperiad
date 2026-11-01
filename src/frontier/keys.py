"""Keys identify every registered object: ``<direction>.<name>[.<name>...]``."""

from __future__ import annotations

import re

DIRECTIONS: tuple[str, ...] = ("common", "hypermatrix", "algebroids", "infrageometry")

_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")


def validate_key(key: str) -> str:
    if not _KEY_RE.match(key):
        raise ValueError(f"invalid key {key!r}: expected dotted lowercase segments, e.g. 'hypermatrix.cubic_product'")
    return key


def direction_of(key: str) -> str:
    return key.split(".", 1)[0]
