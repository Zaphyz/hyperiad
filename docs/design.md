# Hyperiad

**Hyperiad** is the language and notebook system for this research: a bespoke
language for hypergraphs, hypermatrices, rewriting and diagrammatic algebra, with a
browser front end that draws the objects it computes with.

The name joins *hyper-* and *-iad*, as in dyad, triad and polyad: relations of any
arity, which is what hyperedges and hypermatrices are. A clash check on 2026-10-26
found it free on PyPI, npm and crates.io, and hyperiad.com, .org, .io, .dev, .net,
.app and .ai all unregistered. The only near-match is *Hyperiidea*, a group of marine
amphipods.

This file records decisions made with the project owner. Update it when a decision
changes.

## What it is for

- Defining and manipulating the objects of the research: hypermatrices and their
  contractions, hypergraphs, rewrite rules and systems, causal graphs and plex
  diagrams. Later, structures from the diagrammatic-algebroid and infrageometry
  directions.
- Drawing those objects for people. Rendering is the point, not decoration: an object
  shown drawn can be checked by a reader (the same principle as the reference paclet).
- Editing large formulae that have a diagrammatic representation.
- Running the computations of the research: searching for equational identities,
  numerical examples, symbolic checks, rewriting, and searches through axiom spaces.

## Decisions

1. **Notebook front end in the browser.** It should match Mathematica's front end in
   functionality, not in visual style. Input cells hold Hyperiad code; output cells hold
   graphics of the objects and the results of computations.
2. **A language of its own.** The syntax is close to Python or Wolfram Language and
   designed for this functionality.
3. **Code is an expression tree.** The text syntax is one view of that tree. Diagram
   editors, and later a fully spatial way of writing code with no lines of text, are
   other views of the same tree. That keeps the planned visual programming an addition,
   not a rewrite.
4. **One engine, two places to run it.** The language and the mathematics are written
   once, in Python with NumPy and SymPy.
   - In the browser (Pyodide), so simple requests answer immediately.
   - On a server, for heavy work: rewriting large objects, searches through axiom
     spaces.
   - Expensive operations estimate their cost first. Large jobs go to the server, with
     progress, cancellation and cached results. With no server configured, a job over
     the limit is refused and the estimate is shown.
5. **Hosting.** During development it runs in the session's environment. Later it
   becomes a public website on the owner's domain, with the server part on the owner's
   own servers. Server jobs run in isolated containers with time and memory limits.
6. **Open source.** It must be usable for forks: browser-only with no setup, or with
   their own server from a published Docker image.
7. **No Wolfram Language at runtime.** The Hypergraphs paclet and arXiv:2205.05456 are
   pinned references (see `references/`). The paclet's test cases, ported, define the
   expected behaviour, and the record of which have been ported is kept in the
   repository.
8. **Hypergraph drawing style.** Drawings follow the reference paclet's
   `HypergraphPlot` and its published examples:
   - one homogeneous blue hue;
   - concave, spiderweb-like regions for hyperedges;
   - straight binary edges;
   - small disks centred on vertices for unary edges;
   - for directed edges of arity 3 or more, half-arrowheads of one size on the outside
     of the region, the first one hollow (described below);
   - highlights in a desaturated amber, not red; objects drawn on a space (paths,
     vector fields, field edges) in a desaturated palette of green, blue, brown, red and
     violet; an edge seen as a point of space as a four-pointed star (2026-10-29);
   - a contracted index of a plex diagram (label •) as a solid dark-blue vertex, not as
     text (2026-10-28).

   A directed edge $(v_1,\dots,v_k)$ with $k \ge 3$ is drawn with all its sides,
   including the closing side $v_k \to v_1$. Each side carries a half-arrowhead: a
   barb on the outer side of the region, pointing along the side, all of one size.
   The one on the first side $v_1 \to v_2$ is hollow (white, outlined in the arrow
   colour), and the sequence is read from it (trial set 2026-10-28; it replaced
   internal half-arrowheads growing along the sequence, convention A of 2026-10-26). The
   layout places such an edge's vertices around its region in sequence order.
   A special array (white, red, green, blue, purple) drawn as a hyperedge keeps this
   geometry and only takes a colour treatment and its label (`EDGE_TINTS`, user 2026-10-30).

   The geometry is `frontier.hypergraphs.geometry`: `hypergraph_scene` turns a
   hypergraph into shapes, positions in layout units and marks in points. Two
   renderers paint the same scene, `frontier.viz.hypergraph` (matplotlib, session
   figures) and `frontier.hypergraphs.svg` (SVG, the notebook), so they cannot drift.
   The SVG carries its own white ground, as the figures do.

9. **Mark.** The Hyperiad mark is one unordered hyperedge on three vertices, drawn by the shared
   geometry (`hyperiad.server.brand_mark`, also the favicon). Only its fill is stronger and outlined
   so that it reads at icon size (fixed 2026-10-29, at the owner's request).

## As built (0.1)

The code lives in `hyperiad/` in this repository for now; it can move to a repository of its
own once it stabilizes.

- **Trees and syntax** (`hyperiad/src/hyperiad/expr.py`, `syntax.py`): every program is an
  `Expr` tree. Operators are ordinary heads (`a + b` is `Plus(a, b)`). The printer is exact:
  every parsed tree prints back to text that parses to the same tree.
- **Evaluation** (`kernel.py`, `arith.py`, `builtins.py`): sessions with scopes, user
  functions and closures, and exact arithmetic (integers, fractions, prime fields, SymPy for
  symbolic results). Arithmetic on arrays is entry by entry. An undefined name stands for
  itself symbolically. Builtins are registered Python functions; their names follow the
  reference paclet where the function comes from it.
- **Cost budget**: an expensive builtin declares a cost estimator (for `ArrayMultiply`, the
  multiplications of the naive contraction). A session with a budget refuses a call whose
  estimate exceeds it before doing any work, and reports the estimate.
- **Hypergraphs** (`frontier.hypergraphs`, shared by every direction): `Vertex`, `Edge` with a
  symmetry type (directed, cyclic, unordered, or input–output: two nonempty blocks each of one of
  those three, `Edge([[ins], [outs]], [S, T], label)`, one input and one output being the directed
  edge; drawn as two lobes joined by a waist; `EdgeBlocks`, `EdgeInputs`, `EdgeOutputs`) and canonical vertex order, `Hypergraph` as a multiset of edges with labels,
  skeleton and simplicity, ported from the paclet. In Hyperiad: `Hypergraph`, `Edge`,
  `Vertex`, `HypergraphPlot` and the accessors, under the paclet's names. A hypergraph
  displays as its drawing unless it exceeds the paclet's thresholds (32 vertices, 128 edges,
  196 elements), when it shows a summary.
- **Isomorphism and canonical forms** (`frontier.hypergraphs.canon`): `CanonicalHypergraph`,
  `IsomorphicHypergraphQ`, `FindHypergraphIsomorphism` under the paclet's names and semantics.
  Labels count only up to renaming, vertex labels and edge labels independently; a canonical form
  renames vertices $1, \dots, n$ and labels to tokens `LabelToken(k)`, shown $\ell_k$. The method is
  individualization-refinement (colour refinement, branching, pruning by the automorphisms found),
  not the paclet's minimisation over orderings within invariant classes: both are canonical, but
  they choose different representatives, so canonical forms are comparable only within one
  implementation. `Method = "BruteForce"` tries every ordering and is kept for cross-checks.
- **Rewriting** (`frontier.hypergraphs.rewriting`): `HypergraphRule(input, output)`, applied as
  `rule(hg)` or `HypergraphRuleApply`, giving one event per match; `HypergraphRuleMatches`;
  `HypergraphRulePlot` and `HypergraphEventPlot`, the paclet's semantics and key names. A rule
  displays as input → output on a shared layout, an event as the match in amber beside the result
  with its creations in amber, and a short list of hypergraphs, rules or events displays drawn,
  one picture each. Positions in events are counted from 1, as everywhere in Hyperiad. In a
  rule's input a bare name used as a label is a variable (an undefined name is a SymPy symbol);
  note that a name assigned earlier in the notebook evaluates to its value instead. Created
  vertices are named `FormalV(k)`, shown $v_k$. A composite label holding names is a structured
  pattern (`o(x, id(X))` matches `o(f, id(A))`); vertex and edge labels share bindings; matching is
  syntactic. Options: `Labeled`, `LabelPatterns`, `DistinctVertices` (paclet default true; false
  also frees isolated input vertices) and `DistinctEdges` (default true; false lets one host edge
  fill several input edges).
- **One engine, several modes** (`frontier.hypergraphs.composition`,
  `hyperiad/src/hyperiad/rewriting_modes_builtins.py`; settled choices in
  `themes/hypergraph-rewrite-algebras/BRIEF.md`, 2026-10-30). The same matcher, applied:
  - `HypergraphRuleOnePass(rule, hg)`: every match at once, the edges they create on hg's vertices;
  - `HypergraphCompose(rule, {"a": g, "b": h})`: composition on a fixed vertex set, each input edge
    naming its factor (`a`, or `a(x)` to match that edge's own label), non-injective by default,
    `Multiplicity = "Count" | "Boolean" | "Labels"`; graph composition is
    `a:(1,2), b:(2,3) → (1,3)`. Its adjacency array equals `CompositionAdjacencyProduct(rule,
    factors)`, the plex product by `CompositionPlexDiagram(rule)` (claims
    `hypermatrix.composition_is_plex_product`, `hypermatrix.composition_variants`);
  - `HypergraphSaturate(hg, rules, Equations = [...])`: generating rules add, equation rules
    rewrite, edges as a set, until nothing changes (or `MaxRounds`); `IdentifyByLabel = true`
    merges vertices with equal labels (one vertex per object). Labels are syntax, so the unit
    and associativity laws of a category, or a commuting square, are equation rules;
    `Congruence = true` (off by default, user 2026-10-30) applies single-edge label equations inside labels too (needed for
    associativity beyond the outer bracket); `MaxRounds` and `MaxEdges` (1000) stop structures
    whose closure is infinite;
    `ParallelEdges(hg)` lists what still does not commute. Displays as frames, one per round,
    each round's additions in amber.
- **List utilities**: `Join` (lists, or hypergraphs as a union), `Flatten`, `Union`, `Tuples`,
  `Subsets`, `Fold`.
- **Rewriting systems and causality** (`frontier.hypergraphs.causality`):
  `HypergraphRewritingSystem(rules, init, options)` and `HypergraphCausalGraph`, the paclet's
  semantics, property and option names. A property is asked as `sys("CausalGraph", 2)` or
  `sys["StateCount", 2]`, options after it (`CausalDependence = "Edges"`); options given to the
  system are held by it. States and events are numbered from 1 as in the paclet. The graphs
  (evolution, states, branchial, causal, evolution-causal, token-event) display drawn in the
  Wolfram Physics Project colours the paclet uses, each state or event as a small framed picture
  up to `MaxDrawnVertices` (24), plain numbered vertices beyond. The cost estimate follows the
  evolution generation by generation, capped by `MaxStates` (2048). Warnings from the mathematics
  (a truncated evolution, a graph too large to draw with pictures) appear as notes above a cell's
  output.
- **Hypermatrices and adjacency** (`hypermatrix.hypermatrix`, `hypermatrix.adjacency`): `Hypermatrix`,
  a canonically ordered list of arrays (by order, dimensions, then Symmetric, Cyclic, Rigid), and
  `AdjacencyHypermatrix(hg, Labeled = false)`, one array per arity and symmetry type of edge, under
  the paclet's names. A hypermatrix displays as one picture per array; `hm[k]` is its k-th array.
- **Plex diagrams** (`hypermatrix.plex`): `PlexDiagram(edges, contracted)`, `PlexProduct`, `PlexRule`,
  `PlexEvents`, `PlexNames`, `PlexFreeVertices`, `PlexContractedVertices`, `PlexMark`, and
  `EnumeratePlexProducts(n, k, Equivalence = ...)` for the closed plex products of order n and arity k. Not in the
  paclet's code (only its prototype `reference/PlexAlgebra.wl`) nor fully fixed in the paper; the
  conventions were settled with the owner on 2026-10-27 and are recorded in
  `themes/hypermatrix-higher-arity/BRIEF.md`. A plex diagram is a labelled hypergraph and displays
  drawn; a contracted vertex carries the label • (`PlexMark`) and is drawn as a solid dark-blue
  vertex, without text (2026-10-28).
- **Plex identities** (`hypermatrix.plex_identities`, `hyperiad/src/hyperiad/plex_identity_builtins.py`): a
  term is written with undefined heads, `M(a, T(b))`, naming operations in a dictionary of plex diagrams
  `{"M": mm, "T": tr}`. `PlexTermDiagram(t, ops)` expands it to one plex diagram; `PlexEqualQ(t1, t2, ops)`
  compares diagrams up to isomorphisms fixing every variable and the output order; `PlexTerms(ops, vars,
  Order = n, MaxOperations = m)` lists the terms using each variable once; `PlexIdentities(terms, ops)` groups
  them, each identity displayed as the diagram its terms share (outputs i, j, k; contracted p, q, ...)
  captioned with the equation, `id["Terms"]`, `id["Diagram"]`; `PlexNormalForms(d, rules)` and
  `PlexNormalTerms(d, rules)` rewrite with plex rules in every order (associativity as confluence). Avoid
  a head that is bound to a value: `M = PlexRule(...)` makes `M(a, b)` apply the rule.
- **Inert calls**: an undefined function applied to numbers or symbols stays symbolic (`f(x) + 1`);
  applied to anything else (strings, lists) it stays as an inert term, as Wolfram Language leaves
  it. This is how labels such as `Row(["(", x, y, ")"])` or `Subscript("1", p)` are written.
- **Infrageometry** (`infrageometry`, `hyperiad/src/hyperiad/geometry_builtins.py`): spatial hypergraphs as
  discrete spaces, the foundations of question q01 of the theme *hypergraph-geometry*. `SpatialHypergraph`,
  `IncidenceSpace`, `Region`, `MinimalOpen`, `Closure`, `Interior`, `Frontier`, `OpenHull`, `Subspace`;
  `Star`, `Neighbourhood`, `Ball`, `Link`, `LinkDimension`; `TangentSpace`, `Departure`, `MotionGraph`,
  `Motion`, `Geodesic`, `ReduceMotion`, `HomotopicQ` (area homotopy of step paths), `IncidenceHomotopicQ`,
  `IncidenceGroupGenerators`, `TokenRewritingSystem`;
  `SpatialMap`, `Pushforward`, `Differential`, `ImmersionQ` and kin; `ScalarLine`, `ScalarFunction`,
  `DistanceFunction`, `CriticalPoints`, `LevelSet`, `Crossing`; `VectorField`, `ExitField`, `Transit`,
  `Flow`, `IntegralCurve`, `GradientFlow`; `PointedBall`, `Chart`, `LocallyModelledQ`; example spaces
  (`ExampleSpace`, `GridSpace`, `TorusGrid`, `KleinGrid`, `Octahedron`, `Torus7`, ...). Not from the paclet:
  the names are this project's. Points of an incidence space are vertex identifiers and edges, written as
  lists of vertices (`EdgePoint(1, 2)` where an association key is needed) and shown `{1, 2}`. Values that
  carry their space display drawn: an edge as a point of space is a four-pointed star at its centre, half-steps
  are arrows, a moving point is an amber ring. Toolkits register how their values display with
  `display.register_display`; a short list of such values displays as one picture each.
- **Rewriting calculus** (`hyperiad/src/hyperiad/calculus_builtins.py`, question q02): step paths and area
  homotopy (`StepPath`, `HomotopyRules`, `HomotopyDerivation`, `FindBarrier`, `AreaHomotopicQ`,
  `FundamentalGroup`); fields as multisets of typed edges: vector fields of departures and arrivals
  (`VectorFieldEdges`, `VectorFieldArcs`, `DirectionField`, `Cancel`), scalar fields of unary edges counted by
  multiplicity (`ScalarField`, `ScalarFieldOf`, `FieldValues`, `CompareValues`), and their rules
  (`CompositionRule`, `ClosureRules`, `BracketRules`, `Saturate`, `AnimateSaturation`). A field's arcs are drawn
  between vertices and stars, its unary edges as concentric disks, one per copy.
- **Frames** (output type `frames`): pictures shown one at a time with play/pause, step buttons and a
  slider; frames share one box so nothing jumps. `Manipulate(expr, [x, values])` draws `expr` for each
  value (a stepper), `Animate(...)` the same playing, `ListAnimate(list)` any drawn values;
  `AnimateMotion`, `AnimateReduction` and `AnimateFlow` animate infrageometry. An animation plays only
  while on screen and never when the reader asks for reduced motion.
- **Notebooks and pages**: `hyperiad serve --notebook NAME` and `hyperiad build OUT --notebook NAME` take an
  example name or a `.json` path; each notebook keeps its own slot in browser storage. A notebook with
  `"contents": true` gets a contents sidebar from its headings. `build --no-kernel` writes a read-only
  preview (no Python bundle): it shows the outputs computed at build time, never writes browser storage and
  hides the editing actions. Text cells take headings to `###`, lists, tables, fenced code and callouts: a
  block quote opening with **Definition**, **Claim**, **Choice**, **Question**, **Proposal**, **Device**,
  **Example** or **Note** is set as a coloured box of that kind.
- **Display** (`display.py`): arrays display as their pictures with a one-line description,
  symbolic results as typeset LaTeX, registry definitions as text with maths, errors as
  messages.
- **Notebook** (`hyperiad/web/`): one page with code and text cells, In/Out labels,
  Shift+Enter, a reference panel of builtins and autosave in the browser. The browser kernel
  is Pyodide 0.27 running the same Python package. A cell over the browser budget offers
  "Run on the server".
- **Server** (`server.py`): `POST /api/run {code, replay}` is stateless. It rebuilds the
  notebook's state by running the earlier cells, then runs the cell with the server's budget.
  `hyperiad serve` is the development server; `hyperiad build` writes a static site for any
  host. By default the server runs jobs in its own process; `--isolate` runs each in a child
  process with `--job-timeout`, `--job-memory` and `--max-jobs` (a busy server answers 503).

- **Hosting on arity.science** (owner, 2026-10-31): the notebook and its examples are served at
  `arity.science/hyperiad/`, and the project's findings are published there as computational papers
  (Hyperiad notebooks) under `arity.science/findings/<slug>/`. The pages are built by
  `sites/arity/scripts/build-hyperiad.py` in `Zaphyz/websites`, which calls `page()`, `precompute()` and
  `python_bundle()` from `hyperiad.server`. Shipping the Python source to visitors before the
  repository is public is accepted. The page config takes `footer` (HTML under the notebook), `home`
  (`{href, label}`, a link back to the hosting site) and `heavyMessage`. A cell over every budget
  available to it (no server, or the server refused it too) shows `heavyMessage` with the estimate and
  the limit filled in ("This task is estimated to be too heavy to be run here: …"); a refused cell's
  output carries `estimate` and `budget`. Under the notebook, a note credits the Hypergraphs paclet,
  co-developed with Nikolay Murzin in the context of the Wolfram Institute.
- **Public server** (`hyperiad/deploy/`, 2026-11-01): a Cloudflare Worker at `arity.science/api/hyperiad`
  in front of Cloudflare Containers running `hyperiad serve --isolate`. Anonymous visitors may run server
  jobs (owner, 2026-10-31), within limits sized for low traffic: a budget of 2e9 operations, 60 s and
  1 GiB per job (`hyperiad.jobs`: each job in a child process from a fork server, memory capped by
  `RLIMIT_DATA`), one job at a time on each of 2 instances, exact rate limits per visitor and overall (a Durable Object). The
  Worker answers `/info` itself, so loading a notebook never wakes a container. Deployed by Workers
  Builds on every push to `main`; see `hyperiad/deploy/README.md`.

- **Public repository** (owner, 2026-11-01): github.com/Zaphyz/hyperiad, open to forks and pull requests.
  It is an export of this repository, not a separate line of development: work continues here, where
  the toolkits change with the research, and `hyperiad/export/sync.py` maps files one to one between
  the two (`export` writes and commits the export, `import` applies merged pull requests back here as
  a three-way patch, `status` lists what changed there). The export is one installable package
  (`hyperiad`, with the `frontier` subset, `hypermatrix` and `infrageometry` under `src/`); 300 tests
  run there standalone. The bundle the browser loads is found through the installed packages, so it
  is the same in both layouts.

### Syntax (provisional)

| written | meaning |
|---|---|
| `f(x, y, k = v)` | call, with a keyword option |
| `[a, b]`, `{k: v}` | list, association |
| `a[i, j]`, `a["Order"]` | part (1-based, negative from the end), array property |
| `a -> b` | rule |
| `x => body`, `(x, y) => body` | function |
| `x = e`, `f(x, y) = body` | assignment, definition |
| `+ - * /`, `^` or `**` | arithmetic (exact on integers and fractions) |
| `== != < <= > >=`, `and or not` | comparison, logic |
| `true false null`, `# comment` | constants, comments |

Statements are separated by newlines or `;`. Newlines inside brackets continue the statement.
A cell shows its last value, and a trailing `;` suppresses it.

## Open questions

- The concrete syntax beyond the provisional table above, including notation from the
  registry (for example `⊛` for a registered product).
- The paclet's 3D array drawings (`ArrayGraphics3D`) and array themes: the parts not yet brought in.
- Running the browser kernel in a Web Worker, so that long computations do not freeze the
  page.
