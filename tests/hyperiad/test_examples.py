"""Every example notebook runs: each code cell, in order, in one session, without an error.

A cell that shows an error on purpose says so with ``"expect": "error"``."""

import json
from pathlib import Path

import pytest

from hyperiad import Session

EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples").glob("*.json"))


@pytest.mark.parametrize("path", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_example_runs(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    session = Session()
    for k, cell in enumerate(notebook["cells"]):
        if cell["type"] != "code":
            continue
        result = session.run(cell["source"])
        expected = cell.get("expect", "ok")
        assert result["status"] == expected, f"cell {k}: {result['outputs'][-1].get('data', '')[:200]}"
