import importlib.util
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("load_test", ROOT / "scripts" / "load_test.py")
lt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lt)

QUESTIONS = [{"id": "q001", "question": "a?"}, {"id": "q002", "question": "b?"}]


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200):
        self.body, self.status = body, status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.body


def run_clients(n_clients, duration=0.3):
    out, finals, lock = [], [], threading.Lock()
    deadline = time.time() + duration
    threads = [threading.Thread(target=lt.client, args=(i, n_clients, "http://x", QUESTIONS, deadline, 5, out, lock, finals))
               for i in range(n_clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    return out, finals


def test_unexpected_exceptions_are_recorded_with_their_type(monkeypatch):
    monkeypatch.setattr(lt, "BACKOFF_S", 0.05)
    calls = {"n": 0}

    def flaky_urlopen(req, timeout):
        calls["n"] += 1
        kind = calls["n"] % 3
        if kind == 1:
            return FakeResponse(b'{"latency_ms": {"rerank": 1.0}}')   # good
        if kind == 2:
            return FakeResponse(b"<html>not json</html>")              # 200 but bad body
        raise ValueError("something nobody expected")                  # not a network error

    monkeypatch.setattr(lt.urllib.request, "urlopen", flaky_urlopen)
    out, finals = run_clients(2)

    types = {r["error_type"] for r in out if r["error"]}
    assert {"JSONDecodeError", "ValueError"} <= types
    assert any(lt.is_ok(r) for r in out)
    assert not any(lt.is_ok(r) for r in out if r["error_type"] == "JSONDecodeError")  # 200 + bad body is not ok
    assert sorted(f["client"] for f in finals) == [0, 1]
    assert all(f["requests"] == sum(r["client"] == f["client"] for r in out) for f in finals)


def test_a_client_never_vanishes_even_if_its_loop_crashes(monkeypatch):
    def broken_one_request(*args):
        return None  # rec.update() on None raises AttributeError outside one_request's own handling

    monkeypatch.setattr(lt, "one_request", broken_one_request)
    out, finals = run_clients(3, duration=0.2)
    assert sorted(f["client"] for f in finals) == [0, 1, 2]
    assert all(f["error"] and f["error"].startswith("AttributeError") for f in finals)


def test_sleep_flags():
    t0 = 1000.0
    steady = [{"start": t0 + i * 11, "latency_ms": 11000} for i in range(9)]
    assert lt.sleep_flags(steady, t0, t0 + 100, 90)["possible_sleep"] is False

    long_wall = lt.sleep_flags(steady, t0, t0 + 136, 90)
    assert long_wall["possible_sleep"] and "wall" in long_wall["reasons"][0]

    # The interrupted round-2 a-r1-c6 pattern: completions stop, then errors arrive minutes later.
    gap = steady + [{"start": t0 + 60, "latency_ms": 410_000}]
    flags = lt.sleep_flags(gap, t0, t0 + 120, 90)
    assert flags["possible_sleep"] and any("gap" in r for r in flags["reasons"])
