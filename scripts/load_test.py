"""Closed-loop load generator for POST /ask, meant to run inside the cluster as a Job.

Usage: python load_test.py --url http://rag.rag.svc.cluster.local:8000 --clients 3 --duration 90
                           --questions questions.json [--timeout 180]

Each client thread sends one request at a time and starts the next as soon as the previous one
returns, until `duration` seconds have passed; requests still in flight then are allowed to finish.
After any failed request (non-200 status or connection error) the client waits BACKOFF_S (1 s)
before its next request. Round 1 ran without this back-off; it was added for round 2.
Every request uses a NEW TCP connection: urllib opens a fresh connection per urlopen() call and
sends "Connection: close", so a Kubernetes Service (which balances per connection) can route each
request to any ready pod. Standard library only, so it runs on the service image.

Prints one line "RESULT_JSON {...}" with every request record (absolute epoch start time, latency,
HTTP status, error, server-side stage latencies) and one final record per client, for the caller
to parse. Any exception, inside a request or anywhere in a client's loop, is recorded as an error
with its type; a request counts as ok only with status 200 and no error. Every client always
writes a final record (how many requests it made, when it ended, and any exception that ended
it), so a client cannot vanish silently. sleep_flags() marks runs that look interrupted.
"""

import argparse
import json
import threading
import time
import urllib.error
import urllib.request

BACKOFF_S = 1.0


def one_request(url: str, question: str, timeout: float) -> dict:
    body = json.dumps({"question": question}).encode()
    req = urllib.request.Request(url + "/ask", data=body, method="POST",
                                 headers={"Content-Type": "application/json", "Connection": "close"})
    start = time.time()
    t0 = time.perf_counter()
    rec = {"start": start, "status": None, "error": None, "error_type": None, "server_ms": None}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            rec["status"] = resp.status
            data = json.loads(resp.read())
            rec["server_ms"] = data.get("latency_ms")
    except urllib.error.HTTPError as e:
        rec["status"], rec["error"], rec["error_type"] = e.code, f"HTTPError {e.code}", "HTTPError"
    except Exception as e:  # anything else (connection, timeout, bad body, ...) is an error, never lost
        rec["error"], rec["error_type"] = f"{type(e).__name__}: {e}"[:300], type(e).__name__
    rec["latency_ms"] = (time.perf_counter() - t0) * 1000
    return rec


def is_ok(rec: dict) -> bool:
    return rec.get("status") == 200 and not rec.get("error")


def client(idx: int, n_clients: int, url: str, questions: list[dict], deadline: float,
           timeout: float, out: list, lock: threading.Lock, finals: list) -> None:
    n, end_error = 0, None
    try:
        while time.time() < deadline:
            q = questions[(idx + n * n_clients) % len(questions)]  # clients interleave through the list
            try:
                rec = one_request(url, q["question"], timeout)
            except Exception as e:  # defensive: one_request should never raise
                rec = {"start": time.time(), "status": None, "error": f"{type(e).__name__}: {e}"[:300],
                       "error_type": type(e).__name__, "server_ms": None, "latency_ms": 0.0}
            rec.update(client=idx, question_id=q["id"])
            with lock:
                out.append(rec)
            n += 1
            if not is_ok(rec):
                time.sleep(BACKOFF_S)
    except BaseException as e:  # record why the loop stopped, then let the thread end
        end_error = f"{type(e).__name__}: {e}"[:300]
    finally:
        with lock:
            finals.append({"client": idx, "requests": n, "ended": time.time(), "error": end_error})


def sleep_flags(records: list[dict], started: float, finished: float, duration: float) -> dict:
    """Flag a run that looks paused (e.g. the Mac slept): wall > 1.5 x duration, or a gap > 60 s
    between consecutive request completions."""
    completions = sorted(r["start"] + r["latency_ms"] / 1000 for r in records)
    gaps = [b - a for a, b in zip(completions, completions[1:])]
    wall = finished - started
    max_gap = max(gaps) if gaps else None
    reasons = []
    if wall > 1.5 * duration:
        reasons.append(f"wall {wall:.0f} s > 1.5 x {duration:g} s")
    if max_gap is not None and max_gap > 60:
        reasons.append(f"gap of {max_gap:.0f} s between request completions")
    return {"possible_sleep": bool(reasons), "reasons": reasons, "wall_s": wall, "max_completion_gap_s": max_gap}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()

    with open(args.questions) as f:
        questions = json.load(f)
    records, finals, lock = [], [], threading.Lock()
    started = time.time()
    deadline = started + args.duration
    threads = [threading.Thread(target=client, args=(i, args.clients, args.url, questions, deadline,
                                                     args.timeout, records, lock, finals))
               for i in range(args.clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    finished = time.time()
    print("RESULT_JSON " + json.dumps({
        "url": args.url, "clients": args.clients, "duration_s": args.duration,
        "started": started, "finished": finished, "records": sorted(records, key=lambda r: r["start"]),
        "client_finals": sorted(finals, key=lambda f: f["client"]),
        "all_clients_reported": sorted(f["client"] for f in finals) == list(range(args.clients)),
        "flags": sleep_flags(records, started, finished, args.duration),
    }), flush=True)


if __name__ == "__main__":
    main()
