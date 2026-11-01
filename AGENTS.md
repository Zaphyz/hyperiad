# Guide for agents working on Hyperiad

Hyperiad is a language and notebook for higher-arity mathematics, and the computational framework of the
Arity Science research programme. Before changing anything, read this file and `docs/design.md`. For the
research itself (scope, status, the knowledge base of settled definitions and claims, published findings),
read https://arity.science/llms.txt and the guide it links to.

## Setup and checks

```bash
uv sync
uv run pytest                                   # all tests; they must pass
uv run ruff check . && uv run ruff format --check .
uv run hyperiad serve                           # the notebook on http://127.0.0.1:8765/
```

## Layout

- `src/hyperiad/`: `expr.py` (expression trees), `syntax.py` (parse and print; printing is exact), `kernel.py`
  (evaluation, sessions, the cost budget), `builtins.py` and `*_builtins.py` (the functions of the language),
  `display.py` (how values show), `web_kernel.py` (the browser entry point), `server.py` (serve, build),
  `jobs.py` (isolated server jobs with time and memory limits).
- `src/frontier/`: the registry of definitions, claims and notation (`registry.py`), and the hypergraph layer
  (`hypergraphs/`): core types, canonical forms, rewriting, composition, causality, and the drawing geometry
  (`geometry.py`) with its SVG renderer (`svg.py`).
- `src/hypermatrix/`, `src/infrageometry/`: the two mathematical toolkits.
- `web/`: the notebook page (plain JavaScript, no framework). `examples/`: notebooks as JSON.
- `deploy/`: the public server (Cloudflare Worker and Containers); see `deploy/README.md`.

## Conventions

- **Builtins** are Python functions registered with `@builtin(name, usage, cost=...)`. Use the reference
  implementation's name (the Hypergraphs paclet) when the function comes from it. Anything that can be
  expensive declares a `cost` estimator: the browser and the server refuse a call whose estimate is over their
  budget before doing any work.
- **Everything the browser kernel needs must be pure Python available in Pyodide** (NumPy, SymPy, NetworkX).
  `tests/hyperiad/test_server.py` checks that the bundle is self-contained.
- **Show objects drawn wherever possible**, through `display.py`. Hypergraphs follow one visual style, defined
  once in `frontier/hypergraphs/geometry.py`: one homogeneous blue hue; hyperedges as translucent concave
  regions whose sides bow inwards; binary edges as straight lines; a unary edge as a small disk on its vertex;
  highlights in a desaturated amber, never saturated red. Change the geometry, never a renderer alone.
- **Ported tests** carry `@ported("hypergraph-paclet", "<test file>", <lines>)` from `frontier.conformance`,
  citing the reference test cases they reproduce. Port the reference's cases when you bring a function in.
- **Registry**: definitions are registered with `@define`, claims with `claim(...)`, with evidence whose
  status matches it (`supported` needs evidence, `refuted` a counterexample). New concepts start at
  maturity `sketch`; only the research team marks a definition `stable`.
- **Definitions are research decisions.** When work needs a definition pinned down, stop and ask (open an issue)
  rather than choosing. Implementation choices that do not change the mathematics are yours to make.
- Mathematics in docstrings and usage strings is Markdown with `$...$` LaTeX. Positions are counted from 1.

## Contributing

Work on a branch of your fork and open a pull request; see `CONTRIBUTING.md`. Keep each pull request to one
change, with tests, and run the checks above first.
