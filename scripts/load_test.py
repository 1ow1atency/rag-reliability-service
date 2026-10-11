"""Closed-loop load generator for POST /ask, meant to run inside the cluster as a Job.

Usage: python load_test.py --url http://rag.rag.svc.cluster.local:8000 --clients 3 --duration 90
                           --questions questions.json [--timeout 180]

Each client thread sends one request at a time and starts the next as soon as the previous one
returns, until `duration` seconds have passed; requests still in flight then are allowed to finish.
Every request uses a NEW TCP connection: urllib opens a fresh connection per urlopen() call and
sends "Connection: close", so a Kubernetes Service (which balances per connection) can route each
request to any ready pod. Standard library only, so it runs on the service image.

Prints one line "RESULT_JSON {...}" with every request record (absolute epoch start time, latency,
HTTP status, error, server-side stage latencies) for the caller to parse.
"""

import argparse
import json
import socket
import threading
import time
import urllib.error
import urllib.request


def one_request(url: str, question: str, timeout: float) -> dict:
    body = json.dumps({"question": question}).encode()
    req = urllib.request.Request(url + "/ask", data=body, method="POST",
                                 headers={"Content-Type": "application/json", "Connection": "close"})
    start = time.time()
    t0 = time.perf_counter()
    rec = {"start": start, "status": None, "error": None, "server_ms": None}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            rec["status"] = resp.status
            data = json.loads(resp.read())
            rec["server_ms"] = data.get("latency_ms")
    except urllib.error.HTTPError as e:
        rec["status"], rec["error"] = e.code, f"HTTPError {e.code}"
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["latency_ms"] = (time.perf_counter() - t0) * 1000
    return rec


def client(idx: int, n_clients: int, url: str, questions: list[dict], deadline: float,
           timeout: float, out: list, lock: threading.Lock) -> None:
    n = 0
    while time.time() < deadline:
        q = questions[(idx + n * n_clients) % len(questions)]  # clients interleave through the list
        rec = one_request(url, q["question"], timeout)
        rec.update(client=idx, question_id=q["id"])
        with lock:
            out.append(rec)
        n += 1


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
    records, lock = [], threading.Lock()
    started = time.time()
    deadline = started + args.duration
    threads = [threading.Thread(target=client, args=(i, args.clients, args.url, questions, deadline,
                                                     args.timeout, records, lock))
               for i in range(args.clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    finished = time.time()
    print("RESULT_JSON " + json.dumps({
        "url": args.url, "clients": args.clients, "duration_s": args.duration,
        "started": started, "finished": finished, "records": sorted(records, key=lambda r: r["start"]),
    }), flush=True)


if __name__ == "__main__":
    main()
