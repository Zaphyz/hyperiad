# Hyperiad

A language and notebook for higher-arity mathematics. Compute and visualize hypermatrices and their
contractions, hypergraphs and rewrite rules with their causal structure, and plex diagrams. Code goes in
cells; the objects it computes are drawn in the output.

Hyperiad is the computational framework of [Arity Science](https://arity.science), a research programme
on the compositional structure of higher-order systems. Its findings are published as Hyperiad notebooks.

**Try it in your browser, with nothing to install:** https://arity.science/hyperiad/notebook/

## Quick start

```bash
uv sync
uv run hyperiad serve                                   # notebook on http://127.0.0.1:8765/
uv run hyperiad serve --notebook infrageometry-foundations
uv run pytest
```

A taste of the language:

```
m = [[1, 2], [3, 4]]
ArrayMultiply([m, m], "ij,jk->ik")                      # drawn as a 2×2 array
h = Hypergraph([[1, 2], [2, 3], Edge([1, 2, 3], "Unordered")])
HypergraphRule([[1, 2]], [[1, 3], [3, 2]])(h)           # the rewrite events, drawn
fact(n) = If(n <= 1, 1, n * fact(n - 1))
Help("ArrayMultiply")
```

## How it works

- **One engine, two places to run it.** The language and the mathematics are Python (NumPy, SymPy). The
  notebook runs them in the browser through Pyodide; a server runs the same code for computations too large
  for the browser. Expensive operations estimate their cost before they run.
- **Code is an expression tree.** The text syntax is one view of it; diagrammatic views are planned.
- **Objects are drawn.** Hypergraphs, arrays, rewrites and spaces display as pictures that can be checked by eye.

## Layout

| Path | What |
|---|---|
| `src/hyperiad/` | the language: syntax, evaluation, builtins, display, the server |
| `src/frontier/` | the shared layer: hypergraphs (core, canonical forms, rewriting, causality, drawing geometry) and the registry of definitions and claims |
| `src/hypermatrix/` | arrays, their contraction calculus, adjacency hypermatrices, plex diagrams |
| `src/infrageometry/` | hypergraphs as discrete spaces: regions, motions, fields |
| `web/` | the notebook page |
| `examples/` | example notebooks (JSON) |
| `tests/` | tests, including cases ported from the reference implementation |
| `deploy/` | the public server: a Cloudflare Worker in front of Cloudflare Containers |
| `docs/design.md` | the design record and the decisions behind it |

Build a static site that runs entirely in the browser:

```bash
uv run hyperiad fetch-pyodide vendor/pyodide
uv run hyperiad build site/ --notebook hypermatrix-basics --pyodide-files vendor/pyodide
python -m http.server -d site 8000
```

## For AI agents

Start with [`AGENTS.md`](AGENTS.md). The research itself (scope, status, the knowledge base of settled
definitions and claims) is described for agents at https://arity.science/llms.txt.

## Contributing

Forks and pull requests are welcome: see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Credits

Hyperiad builds on the [Hypergraphs paclet](https://github.com/Zaphyz/Hypergraph-Paclet), co-developed with
[Nikolay Murzin](https://github.com/sw1sh). That work was done in the context of our interaction at the
Wolfram Institute. The array algebra and the plex formalism follow
[Heaps of Fish](https://arxiv.org/abs/2205.05456).
