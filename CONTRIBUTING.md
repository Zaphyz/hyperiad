# Contributing to Hyperiad

1. Fork the repository and create a branch.
2. `uv sync`, make your change, and add or update tests.
3. Run the checks: `uv run pytest`, `uv run ruff check .` and `uv run ruff format --check .`
   (CI runs the same on every pull request).
4. Open a pull request that says what changed and why. For a new builtin, include a short notebook cell
   that shows it drawn.

What makes a good contribution:

- New builtins, faster or more general implementations, better drawings, more of the reference
  implementation's test cases ported (see `AGENTS.md`).
- Example notebooks that state a question, compute the evidence and draw it.
- Bug reports with the smallest notebook cell that shows the problem.

Definitions are research decisions. When a change needs a definition pinned down (choosing between
readings, or a convention that changes which statements hold), open an issue to discuss it before writing
the code.

## How this repository is maintained

Hyperiad is developed inside the Arity Science research workspace and exported here; every sync is a
commit titled "Sync from frontier-math …". Merged pull requests are brought back into the workspace before
the next export, so your commits are kept and your changes ship with the next sync.
