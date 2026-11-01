"""Serve and build the Hyperiad notebook.

``hyperiad serve``  development server: the notebook page, the Python sources the
                    browser kernel loads, and the evaluation API (``/api``).
``hyperiad build``  a static site (one page plus ``pysrc.json``) for any web host;
                    with ``--server`` it also sends over-budget cells to that API.

The API is stateless: ``POST /api/run`` takes ``{"code", "replay"}``, rebuilds the
notebook's state by running the ``replay`` cells, then runs ``code`` with the server's
budget. By default everything runs in the server's process (development). With
``--isolate`` each job runs in a child process with time and memory limits, and jobs over
``--max-jobs`` are turned away with 503: the setting for a public server (``hyperiad/deploy/``).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import urllib.parse
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import hyperiad
from hyperiad.jobs import Busy, JobRunner, Limits

PACKAGE_DIR = Path(__file__).resolve().parent
# web/ and examples/ sit next to src/: hyperiad/ in frontier-math, the root of the hyperiad repository
WEB_DIR = PACKAGE_DIR.parents[1] / "web"
EXAMPLES_DIR = PACKAGE_DIR.parents[1] / "examples"

PYODIDE_VERSION = "0.27.7"
PYODIDE_CDN = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/"
PYODIDE_CORE = ["pyodide.js", "pyodide.asm.js", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json"]
PYODIDE_PACKAGES = ["numpy", "sympy", "mpmath", "networkx"]
BROWSER_BUDGET = 2e7
SERVER_BUDGET = 1e10
MAX_BODY = 2_000_000
BUSY = "The Hyperiad server is busy with other jobs. Try again in a minute."

# The Python the browser kernel needs: files of each package, found where the package is installed
# (the monorepo's workspace or the hyperiad repository's src/), so the bundle is the same in both.
SOURCES = [
    ("frontier", ["__init__.py", "keys.py", "notation.py", "registry.py", "hypergraphs/*.py"]),
    ("hypermatrix", ["*.py"]),
    ("infrageometry", ["*.py"]),
    ("hyperiad", ["*.py"]),
]
SERVER_ONLY = {"hyperiad/server.py", "hyperiad/jobs.py"}


def python_bundle() -> dict[str, Any]:
    import importlib

    files: dict[str, str] = {}
    for package, patterns in SOURCES:
        pkg_dir = Path(importlib.import_module(package).__file__).resolve().parent
        root = pkg_dir.parent
        for pattern in patterns:
            for path in sorted(pkg_dir.glob(pattern)):
                rel = path.relative_to(root).as_posix()
                if rel not in SERVER_ONLY:
                    files[rel] = path.read_text(encoding="utf-8")
    return {"version": hyperiad.__version__, "files": files}


def _builtin_docs() -> list[dict[str, str]]:
    from hyperiad.web_kernel import builtins

    return json.loads(builtins())


def precompute(notebook: dict[str, Any], budget: float | None) -> dict[str, Any]:
    """Run a notebook's code cells in order and store their outputs, so the page is complete before its kernel loads."""
    session = hyperiad.Session(budget=budget)
    n = 0
    cells = []
    for cell in notebook["cells"]:
        cell = dict(cell)
        if cell.get("type", "code") == "code":
            n += 1
            result = session.run(cell["source"])
            cell.update(outputs=result["outputs"], n=n, status=result["status"], where="browser")
        cells.append(cell)
    return {"cells": cells}


def _script_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def brand_mark(px: int = 34) -> str:
    """The Hyperiad mark: one unordered hyperedge on three vertices, from the shared drawing geometry.

    The concave region, the hue and the vertex disks are the drawing style's own. At icon size the
    style's faint fill would vanish, so the region is filled a little more strongly and outlined;
    the scene's span is widened so the vertex disks keep their size against the small picture.
    """
    import math
    import re

    from frontier.hypergraphs.geometry import EDGE_FILL, EDGE_LINE, Area, hypergraph_scene
    from frontier.hypergraphs.svg import scene_svg

    layout = {
        k: (math.cos(math.radians(90 + 120 * i)), math.sin(math.radians(90 + 120 * i))) for i, k in enumerate("abc")
    }
    scene = hypergraph_scene([["a", "b", "c"]], layout=layout, symmetry="unordered")
    scene.span *= 3.8
    scene.shapes = [
        Area(s.points, (*EDGE_FILL[:3], 0.6), (*EDGE_LINE[:3], 0.45), 1.6, s.z) if isinstance(s, Area) else s
        for s in scene.shapes
    ]
    svg = re.sub(r"<rect[^>]*/>", "", scene_svg(scene, width=px), count=1)  # no white ground: it sits on the bar
    return (
        svg.replace('class="hd-hypergraph"', 'class="mark"')
        .replace('role="img" ', 'aria-hidden="true" ')
        .replace(' aria-label="hypergraph drawing"', "")
    )


def page(config: dict[str, Any], notebook: dict[str, Any] | None, doctype: bool = True) -> str:
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    mark = brand_mark()
    html = re.sub(r'<svg class="mark".*?</svg>', lambda _: mark, html, count=1, flags=re.S)
    icon = urllib.parse.quote(brand_mark(64))
    html = html.replace("<title>", f'<link rel="icon" href="data:image/svg+xml,{icon}">\n<title>', 1)
    if notebook and notebook.get("title"):
        from html import escape

        html = html.replace("<title>Hyperiad Notebook</title>", f"<title>{escape(notebook['title'])}</title>", 1)
    config = {**config, "builtinDocs": _builtin_docs()}
    html = html.replace("/*STYLE*/", (WEB_DIR / "style.css").read_text(encoding="utf-8"))
    html = html.replace("/*CONFIG*/{}", _script_json(config))
    html = html.replace("/*NOTEBOOK*/null", _script_json(notebook))
    html = html.replace("/*APP*/", (WEB_DIR / "app.js").read_text(encoding="utf-8").replace("</script", "<\\/script"))
    return ("<!doctype html>\n" if doctype else "") + html


def example_notebook(name: str | Path = "hypermatrix-basics") -> dict[str, Any]:
    """A notebook by example name (``hyperiad/examples/<name>.json``) or by path."""
    path = Path(name)
    if path.suffix != ".json":
        path = EXAMPLES_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def notebook_config(notebook: dict[str, Any], name: str | Path) -> dict[str, Any]:
    """Per-notebook settings: its own storage slot in the browser, and contents when it asks for them."""
    stem = Path(name).stem
    config: dict[str, Any] = {"storage": f"hyperiad.notebook.{stem}.v1"}
    if notebook.get("contents"):
        config["toc"] = True
    return config


# -- serving ---------------------------------------------------------------


def make_handler(
    budget: float,
    pyodide: str,
    browser_budget: float,
    notebook: str | Path = "hypermatrix-basics",
    runner: JobRunner | None = None,
):
    from hyperiad.web_kernel import run_with_replay

    class Handler(BaseHTTPRequestHandler):
        server_version = f"Hyperiad/{hyperiad.__version__}"

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, value: Any) -> None:
            self._send(status, json.dumps(value).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                source = example_notebook(notebook)
                config = {"server": "/api", "pyodide": pyodide, "pysrc": "/pysrc.json", "budget": browser_budget}
                config |= notebook_config(source, notebook)
                computed = {**source, **precompute(source, browser_budget)}
                self._send(HTTPStatus.OK, page(config, computed).encode(), "text/html; charset=utf-8")
            elif path == "/pysrc.json":
                self._json(HTTPStatus.OK, python_bundle())
            elif path == "/api/info":
                self._json(HTTPStatus.OK, {"name": "hyperiad", "version": hyperiad.__version__, "budget": budget})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - http.server API
            if self.path != "/api/run":
                return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            length = int(self.headers.get("Content-Length", "0"))
            if length > MAX_BODY:
                return self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request too large"})
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                code, replay = body["code"], body.get("replay", [])
                if (
                    not isinstance(code, str)
                    or not isinstance(replay, list)
                    or not all(isinstance(r, str) for r in replay)
                ):
                    raise ValueError
            except (ValueError, KeyError, json.JSONDecodeError):
                return self._json(HTTPStatus.BAD_REQUEST, {"error": 'expected {"code": str, "replay": [str]}'})
            if runner is None:
                return self._json(HTTPStatus.OK, run_with_replay(code, replay, budget))
            try:
                self._json(HTTPStatus.OK, runner.run(code, replay))
            except Busy:
                self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": BUSY})

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write(f"[hyperiad] {self.address_string()} {fmt % args}\n")

    return Handler


def serve(
    host: str,
    port: int,
    budget: float,
    pyodide: str,
    browser_budget: float,
    notebook: str | Path,
    limits: Limits | None = None,
) -> None:
    runner = JobRunner(budget, limits) if limits else None
    httpd = ThreadingHTTPServer((host, port), make_handler(budget, pyodide, browser_budget, notebook, runner))
    where = f"isolated jobs, {limits}" if limits else "jobs in this process"
    print(f"Hyperiad notebook on http://{host}:{port}/  (server budget {budget:.3g} operations; {where})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


# -- building --------------------------------------------------------------


def fetch_pyodide(dest: Path) -> list[Path]:
    """Download the Pyodide files the notebook needs, for hosting them next to the page."""
    dest.mkdir(parents=True, exist_ok=True)

    def get(name: str) -> Path:
        target = dest / name
        if not target.exists():
            with urllib.request.urlopen(PYODIDE_CDN + name, timeout=120) as r:  # noqa: S310 - fixed https URL
                target.write_bytes(r.read())
        return target

    files = [get(n) for n in PYODIDE_CORE]
    lock = json.loads((dest / "pyodide-lock.json").read_text())
    files += [get(lock["packages"][p]["file_name"]) for p in PYODIDE_PACKAGES]
    return files


def build(
    out: Path,
    server: str | None,
    pyodide: str,
    budget: float,
    doctype: bool,
    pyodide_files: Path | None,
    kernel: bool = True,
    notebook: str | Path = "hypermatrix-basics",
    page_name: str = "index.html",
) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    if pyodide_files is not None:
        target = out / "pyodide"
        target.mkdir(exist_ok=True)
        for f in pyodide_files.iterdir():
            shutil.copy2(f, target / f.name)
        pyodide = "pyodide/"
    source = example_notebook(notebook)
    config = {"server": server, "pyodide": pyodide, "pysrc": "pysrc.json", "budget": budget}
    config |= notebook_config(source, notebook)
    if not kernel:
        config["kernel"] = "none"
    computed = {**source, **precompute(source, budget)}
    (out / page_name).write_text(page(config, computed, doctype=doctype), encoding="utf-8")
    if kernel:
        (out / "pysrc.json").write_text(json.dumps(python_bundle(), ensure_ascii=False), encoding="utf-8")
    return out / page_name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hyperiad", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("serve", help="development server: notebook page and evaluation API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--budget", type=float, default=SERVER_BUDGET, help="operations a server job may estimate")
    p.add_argument("--browser-budget", type=float, default=BROWSER_BUDGET)
    p.add_argument("--pyodide", default=PYODIDE_CDN, help="where the page loads Pyodide from")
    p.add_argument("--notebook", default="hypermatrix-basics", help="an example name or a notebook .json path")
    p.add_argument("--isolate", action="store_true", help="run each job in a child process with limits")
    p.add_argument("--job-timeout", type=float, default=Limits.timeout, help="seconds per job (with --isolate)")
    p.add_argument("--job-memory", type=int, default=Limits.memory_mb, help="MB of memory per job (with --isolate)")
    p.add_argument("--max-jobs", type=int, default=Limits.max_jobs, help="jobs at once (with --isolate)")
    p = sub.add_parser("build", help="write a static site")
    p.add_argument("out", type=Path)
    p.add_argument(
        "--server", default=None, help="URL of a Hyperiad API for over-budget cells, e.g. https://example.org/api"
    )
    p.add_argument("--pyodide", default=PYODIDE_CDN)
    p.add_argument("--pyodide-files", type=Path, default=None, help="host these Pyodide files next to the page")
    p.add_argument("--budget", type=float, default=BROWSER_BUDGET)
    p.add_argument("--fragment", action="store_true", help="omit the doctype (for hosts that wrap the page)")
    p.add_argument("--no-kernel", action="store_true", help="a read-only preview: precomputed outputs, no Python")
    p.add_argument("--notebook", default="hypermatrix-basics", help="an example name or a notebook .json path")
    p.add_argument("--page", default="index.html", help="the file name of the page")
    p = sub.add_parser("fetch-pyodide", help="download the Pyodide files the notebook needs")
    p.add_argument("dest", type=Path)
    args = parser.parse_args(argv)
    if args.cmd == "serve":
        limits = Limits(args.job_timeout, args.job_memory, args.max_jobs) if args.isolate else None
        serve(args.host, args.port, args.budget, args.pyodide, args.browser_budget, args.notebook, limits)
    elif args.cmd == "build":
        kernel = not args.no_kernel
        print(
            build(
                args.out,
                args.server,
                args.pyodide,
                args.budget,
                not args.fragment,
                args.pyodide_files,
                kernel,
                args.notebook,
                args.page,
            )
        )
    else:
        for f in fetch_pyodide(args.dest):
            print(f"{f.stat().st_size:>10,}  {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
