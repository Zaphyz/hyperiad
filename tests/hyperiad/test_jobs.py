"""Isolated server jobs: limits on time, memory and concurrency."""

import threading
import time

import pytest
from hyperiad.jobs import Busy, JobRunner, Limits

SLOW = 'big = OneArray(3, 250)\nArrayMultiply([big, big], "ijk,jkl->il")'  # about 8e9 operations: seconds


def test_job_runs_with_replay():
    r = JobRunner(1e9, Limits(timeout=30)).run("a + b", ["a = 2", "b = 3"])
    assert r["status"] == "ok" and r["outputs"][0]["data"] == "5"


def test_budget_refuses_before_running():
    r = JobRunner(1e6, Limits(timeout=30)).run(SLOW, [])
    assert r["status"] == "too_expensive" and r["outputs"][-1]["estimate"] > 1e6


def test_time_limit_stops_a_job():
    t = time.monotonic()
    r = JobRunner(None, Limits(timeout=1)).run(SLOW, [])
    assert time.monotonic() - t < 10
    assert r["status"] == "error" and "time limit of 1 s" in r["outputs"][0]["data"]
    assert r["outputs"][0]["kind"] == "limit"


def test_memory_limit_stops_a_job():
    r = JobRunner(None, Limits(timeout=30, memory_mb=1024)).run("OneArray(3, 1500)", [])
    assert r["status"] == "error" and "memory" in r["outputs"][0]["data"]


def test_too_many_cells_to_replay():
    r = JobRunner(None, Limits(max_replay=2)).run("1", ["1", "2", "3"])
    assert r["status"] == "error" and "replay" in r["outputs"][0]["data"]


def test_busy_server_turns_jobs_away():
    runner = JobRunner(None, Limits(timeout=3, max_jobs=1))
    first = threading.Thread(target=runner.run, args=(SLOW, []))
    first.start()
    time.sleep(0.3)
    with pytest.raises(Busy):
        runner.run("1", [])
    first.join()
    assert runner.run("1 + 1", [])["status"] == "ok"
