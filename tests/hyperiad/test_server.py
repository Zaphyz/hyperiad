import json
import subprocess
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from hyperiad.server import build, make_handler, python_bundle


def test_bundle_is_self_contained(tmp_path: Path):
    """The files the browser kernel loads must import and run on their own, as in Pyodide."""
    bundle = python_bundle()
    assert "hypermatrix/arrays.py" in bundle["files"] and "hyperiad/server.py" not in bundle["files"]
    for rel, text in bundle["files"].items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    numpy_dir = Path(__import__("numpy").__file__).parents[1]
    script = (
        "import sys, json\n"
        f"sys.path[:] = [{str(tmp_path)!r}, {str(numpy_dir)!r}] + [p for p in sys.path if 'site-packages' not in p]\n"
        "import hyperiad.web_kernel as k\n"
        "k.start(2e7)\n"
        "r = json.loads(k.run('ArrayMultiply([[[1, 2]]], \"ij->ji\")'))\n"
        "assert r['status'] == 'ok', r\n"
        "assert 'hyperiad' in k.__file__ and k.__file__.startswith(sys.path[0]), k.__file__\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=120)
    assert out.stdout.strip() == "ok", out.stderr


def test_static_build(tmp_path: Path):
    index = build(
        tmp_path, server="https://example.org/api", pyodide="pyodide/", budget=2e7, doctype=True, pyodide_files=None
    )
    html = index.read_text()
    assert html.startswith("<!doctype html>")
    for marker in ("/*STYLE*/", "/*APP*/", "/*CONFIG*/", "/*NOTEBOOK*/"):
        assert marker not in html
    assert '"server": "https://example.org/api"' in html
    # every embedded SVG and script stays inside its <script> element
    assert html.count("<script") == html.count("</script>")
    assert json.loads((tmp_path / "pysrc.json").read_text())["files"]


def test_api_runs_with_replay():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(1e10, "pyodide/", 2e7))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        info = json.load(urllib.request.urlopen(f"{base}/api/info", timeout=10))
        assert info["budget"] == 1e10
        req = urllib.request.Request(
            f"{base}/api/run",
            data=json.dumps({"code": "a + b", "replay": ["a = 2", "b = 3"]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        result = json.load(urllib.request.urlopen(req, timeout=30))
        assert result["status"] == "ok" and result["outputs"][0]["data"] == "5"
        bad = urllib.request.Request(
            f"{base}/api/run", data=b'{"code": 1}', headers={"Content-Type": "application/json"}
        )
        try:
            urllib.request.urlopen(bad, timeout=10)
            raise AssertionError("expected 400")
        except urllib.error.HTTPError as e:
            assert e.code == 400
    finally:
        httpd.shutdown()


def test_isolated_api_turns_away_jobs_when_busy():
    from hyperiad.jobs import JobRunner, Limits

    runner = JobRunner(None, Limits(timeout=3, max_jobs=1))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(1e10, "pyodide/", 2e7, runner=runner))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def post(code: str):
        req = urllib.request.Request(
            f"{base}/api/run",
            data=json.dumps({"code": code, "replay": []}).encode(),
            headers={"Content-Type": "application/json"},
        )
        return json.load(urllib.request.urlopen(req, timeout=30))

    try:
        assert post("1 + 1")["outputs"][0]["data"] == "2"
        slow = threading.Thread(target=post, args=('b = OneArray(3, 250)\nArrayMultiply([b, b], "ijk,jkl->il")',))
        slow.start()
        import time

        time.sleep(0.5)
        try:
            post("1")
            raise AssertionError("expected 503")
        except urllib.error.HTTPError as e:
            assert e.code == 503 and "busy" in json.load(e)["error"]
        slow.join()
    finally:
        httpd.shutdown()
