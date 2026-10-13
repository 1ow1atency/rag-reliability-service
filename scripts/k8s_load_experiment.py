"""Run the Kubernetes load-test experiments from the host and save raw data.

Usage: python scripts/k8s_load_experiment.py {smoke,a,b} [replicas ...] [--round=round2]
(results go under "round1" unless --round is given)

Load comes from scripts/load_test.py running INSIDE the cluster as a Job (service image, script and
questions mounted from a ConfigMap) against the Service DNS name, one new connection per request.
The host only orchestrates: scales the Deployment, creates Jobs, and samples kubectl every 15 s.

  smoke: 1 client for 15 s, to check the Job works (not an experiment cell)
  a:     HPA deleted; replicas {1, 3} x clients {1, 3, 6}, 90 s per cell (optionally only the given replicas)
  b:     HPA from k8s/hpa.yaml; start at 1 replica, 4 clients for 300 s, then wait for scale-down

Assumes LLM_STUB=1 is already set in the live ConfigMap (reranking only, no Groq calls).
Raw data goes to results/metrics/k8s_load_test.json after every cell; nothing is smoothed or dropped.
"""

import json
import random
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "metrics" / "k8s_load_test.json"
NS = "rag"
URL = "http://rag.rag.svc.cluster.local:8000"
SAMPLE_EVERY = 15
N_QUESTIONS = 10
SEED = 42
ROUND = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--round=")), "round1")


def kubectl(*args: str, input: str | None = None, check: bool = True) -> str:
    r = subprocess.run(["kubectl", "-n", NS, *args], input=input, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(s: str | None) -> float | None:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp() if s else None


def load_results() -> dict:
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save_results(data: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2) + "\n")


# ---------- setup ----------

def pick_questions() -> list[dict]:
    with (ROOT / "data" / "eval" / "questions.jsonl").open() as f:
        dev = sorted((q for q in map(json.loads, f) if q["split"] == "dev" and q["answerable"]), key=lambda q: q["id"])
    return [{"id": q["id"], "question": q["question"]} for q in random.Random(SEED).sample(dev, N_QUESTIONS)]


def setup_configmap(questions: list[dict]) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(questions, f)
        qpath = f.name
    manifest = subprocess.run(
        ["kubectl", "-n", NS, "create", "configmap", "load-test", f"--from-file=load_test.py={ROOT / 'scripts' / 'load_test.py'}",
         f"--from-file=questions.json={qpath}", "--dry-run=client", "-o", "yaml"],
        capture_output=True, text=True, check=True).stdout
    kubectl("apply", "-f", "-", input=manifest)


def job_manifest(name: str, clients: int, duration: int) -> str:
    return json.dumps({
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": name, "namespace": NS, "labels": {"app": "load-test"}},
        "spec": {"backoffLimit": 0, "ttlSecondsAfterFinished": 600, "template": {
            "metadata": {"labels": {"app": "load-test"}},
            "spec": {
                "restartPolicy": "Never",
                "securityContext": {"runAsNonRoot": True, "runAsUser": 10001, "runAsGroup": 10001},
                "volumes": [{"name": "load-test", "configMap": {"name": "load-test"}}],
                "containers": [{
                    "name": "load-test", "image": "rag-reliability-service:latest", "imagePullPolicy": "Never",
                    "command": ["python", "/loadtest/load_test.py", "--url", URL, "--clients", str(clients),
                                "--duration", str(duration), "--questions", "/loadtest/questions.json"],
                    "resources": {"requests": {"cpu": "100m", "memory": "64Mi"},
                                  "limits": {"cpu": "500m", "memory": "256Mi"}},
                    "volumeMounts": [{"name": "load-test", "mountPath": "/loadtest"}],
                    "securityContext": {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}},
                }],
            }}},
    })


# ---------- observation ----------

def top_pods() -> dict[str, dict]:
    out = kubectl("top", "pod", "-l", "app=rag", "--no-headers", check=False)
    pods = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            pods[parts[0]] = {"cpu_m": int(parts[1].rstrip("m")), "mem_mi": int(parts[2].rstrip("Mi"))}
    return pods


def rag_pods() -> list[dict]:
    pods = json.loads(kubectl("get", "pods", "-l", "app=rag", "-o", "json"))["items"]
    out = []
    for p in pods:
        cs = (p["status"].get("containerStatuses") or [{}])[0]
        ready = next((c for c in p["status"].get("conditions", []) if c["type"] == "Ready"), {})
        last = (cs.get("lastState") or {}).get("terminated") or {}
        out.append({"name": p["metadata"]["name"], "phase": p["status"].get("phase"),
                    "created": parse_ts(p["metadata"]["creationTimestamp"]),
                    "ready": ready.get("status") == "True", "ready_since": parse_ts(ready.get("lastTransitionTime")),
                    "restarts": cs.get("restartCount", 0), "last_terminated_reason": last.get("reason"),
                    "deleting": "deletionTimestamp" in p["metadata"]})
    return out


def hpa_status() -> dict | None:
    out = kubectl("get", "hpa", "rag", "-o", "json", check=False)
    if not out:
        return None
    s = json.loads(out)["status"]
    cpu = next((m["resource"]["current"].get("averageUtilization") for m in s.get("currentMetrics") or []
                if m.get("type") == "Resource"), None)
    return {"current_replicas": s.get("currentReplicas"), "desired_replicas": s.get("desiredReplicas"), "cpu_pct": cpu}


def deployment_status() -> dict:
    d = json.loads(kubectl("get", "deploy", "rag", "-o", "json"))
    return {"spec_replicas": d["spec"]["replicas"], "ready_replicas": d["status"].get("readyReplicas", 0)}


def sample(with_cgroup: bool = False) -> dict:
    s = {"t": time.time(), "deployment": deployment_status(), "hpa": hpa_status(), "top": top_pods(), "pods": rag_pods()}
    if with_cgroup:
        s["cgroup"] = cgroup_memory()
    return s


def problems_since(t0: float) -> dict:
    events = json.loads(kubectl("get", "events", "-o", "json"))["items"]
    bad = []
    for e in events:
        raw = e.get("lastTimestamp") or e.get("eventTime")  # "...Z" or "...<fraction>Z"
        ts = parse_ts(raw[:19] + "Z") if raw else None
        if ts and ts >= t0 - 1 and (e["type"] == "Warning" or e["reason"] in ("Unhealthy", "Killing", "OOMKilling", "BackOff")):
            bad.append({"t": ts, "object": e["involvedObject"]["name"], "reason": e["reason"],
                        "count": e.get("count"), "message": e.get("message", "")[:200]})
    pods = rag_pods()
    return {"warning_events": bad, "restarts": {p["name"]: p["restarts"] for p in pods},
            "oom_killed": [p["name"] for p in pods if p["last_terminated_reason"] == "OOMKilled"]}


def cgroup_memory() -> dict[str, dict]:
    """memory.current, memory.peak and anon (bytes) per running rag pod, read from the cgroup files."""
    cmd = ('c=/sys/fs/cgroup; echo "$(cat $c/memory.current) $(cat $c/memory.peak) '
           '$(awk \'/^anon /{print $2}\' $c/memory.stat)"')
    out = {}
    for p in rag_pods():
        if p["phase"] != "Running" or p["deleting"]:
            continue
        r = subprocess.run(["kubectl", "-n", NS, "exec", p["name"], "--", "sh", "-c", cmd], capture_output=True, text=True)
        vals = r.stdout.split()
        out[p["name"]] = ({"current": int(vals[0]), "peak": int(vals[1]), "anon": int(vals[2])}
                          if r.returncode == 0 and len(vals) == 3 else {"error": r.stderr.strip()[:200]})
    return out


def requests_per_pod(since: float) -> dict[str, int]:
    counts = {}
    for p in rag_pods():
        logs = kubectl("logs", p["name"], f"--since-time={iso(since)}", check=False)
        counts[p["name"]] = sum(1 for line in logs.splitlines() if '"POST /ask' in line)
    return counts


# ---------- running a Job ----------

def run_job(name: str, clients: int, duration: int, samples: list | None = None, stop: threading.Event | None = None) -> dict:
    kubectl("delete", "job", name, "--ignore-not-found", "--wait=true")
    kubectl("apply", "-f", "-", input=job_manifest(name, clients, duration))
    t_submit = time.time()
    deadline = t_submit + duration + 600
    while time.time() < deadline:
        status = json.loads(kubectl("get", "job", name, "-o", "json"))["status"]
        if status.get("succeeded") or status.get("failed"):
            break
        if samples is not None and (not samples or time.time() - samples[-1]["t"] >= SAMPLE_EVERY):
            samples.append(sample())
        time.sleep(1)
    logs = kubectl("logs", f"job/{name}", check=False)
    line = next((l for l in logs.splitlines() if l.startswith("RESULT_JSON ")), None)
    if line is None:
        raise RuntimeError(f"job {name} produced no result: {logs[-2000:]}")
    result = json.loads(line[len("RESULT_JSON "):])
    result["job"] = name
    kubectl("delete", "job", name, "--wait=false", check=False)
    return result


def summarize(result: dict) -> dict:
    recs = result["records"]
    ok = [r for r in recs if r["status"] == 200 and not r.get("error")]
    lat = sorted(r["latency_ms"] for r in ok)
    rerank = sorted(r["server_ms"]["rerank"] for r in ok if r.get("server_ms"))
    wall = result["finished"] - result["started"]

    def pct(v, p):
        return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))] if v else None

    n503 = sum(1 for r in recs if r["status"] == 503)
    return {"requests": len(recs), "ok": len(ok), "errors": len(recs) - len(ok),
            "n_503": n503, "other_errors": len(recs) - len(ok) - n503,
            "error_kinds": sorted({r["error"] or f"HTTP {r['status']}" for r in recs if r["status"] != 200}),
            "p50_ms": pct(lat, 50), "p95_ms": pct(lat, 95), "max_ms": lat[-1] if lat else None,
            "server_rerank_p50_ms": pct(rerank, 50), "wall_s": wall, "rps": len(ok) / wall if wall else None,
            "error_types": sorted({r.get("error_type") or f"HTTP {r['status']}" for r in recs if r not in ok}),
            "possible_sleep": result.get("flags", {}).get("possible_sleep"),
            "possible_sleep_reasons": result.get("flags", {}).get("reasons"),
            "all_clients_reported": result.get("all_clients_reported")}


# ---------- experiments ----------

def wait_ready(replicas: int, timeout: int = 600) -> None:
    kubectl("scale", "deploy/rag", f"--replicas={replicas}")
    end = time.time() + timeout
    while time.time() < end:
        pods = [p for p in rag_pods() if not p["deleting"]]
        if len(pods) == replicas and all(p["ready"] for p in pods):
            return
        time.sleep(2)
    raise RuntimeError(f"deployment did not reach {replicas} ready replicas")


def cell(label: str, clients: int, duration: int) -> dict:
    samples = []
    restarts_before = {p["name"]: p["restarts"] for p in rag_pods()}
    mem_before = cgroup_memory()
    t0 = time.time()
    result = run_job(f"load-{label}", clients, duration, samples=samples)
    summary = summarize(result)
    peak = {}
    for s in samples:
        for pod, v in s["top"].items():
            cur = peak.setdefault(pod, {"cpu_m": 0, "mem_mi": 0})
            cur["cpu_m"], cur["mem_mi"] = max(cur["cpu_m"], v["cpu_m"]), max(cur["mem_mi"], v["mem_mi"])
    return {"label": label, "clients": clients, "duration_s": duration, "summary": summary,
            "per_pod_requests": requests_per_pod(result["started"]), "top_peak": peak,
            "restarts_before": restarts_before, "cgroup_memory_before": mem_before,
            "cgroup_memory_after": cgroup_memory(), "samples": samples, "problems": problems_since(t0), "raw": result}


def new_oom(c: dict) -> list[str]:
    """Pods that restarted during the cell and whose last termination was OOMKilled."""
    after = {p["name"]: p for p in rag_pods()}
    return [n for n, p in after.items()
            if p["restarts"] > c["restarts_before"].get(n, 0) and p["last_terminated_reason"] == "OOMKilled"]


def fmt(v, spec: str) -> str:
    return "n/a" if v is None else format(v, spec)


def experiment_a(data: dict, replicas_list: tuple[int, ...] = (1, 3)) -> None:
    kubectl("delete", "hpa", "rag", "--ignore-not-found")
    exp = data.setdefault(ROUND, {}).setdefault("experiment_a", {"hpa": "deleted for this experiment", "cells": []})
    for replicas in replicas_list:
        wait_ready(replicas)
        for clients in (1, 3, 6):
            c = cell(f"a-r{replicas}-c{clients}", clients, 90)
            c["replicas"] = replicas
            c["new_oom_killed"] = new_oom(c)
            exp["cells"].append(c)
            save_results(data)
            s = c["summary"]
            mib = lambda m: {k: round(v / 1048576) for k, v in m.items()} if "error" not in m else m
            print(f"   flags: possible_sleep={s['possible_sleep']} {s['possible_sleep_reasons']} "
                  f"all_clients_reported={s['all_clients_reported']} error_types={s['error_types']}", flush=True)
            print(f"   503={s['n_503']} other_errors={s['other_errors']} cgroup before={ {p: mib(m) for p, m in c['cgroup_memory_before'].items()} } "
                  f"after={ {p: mib(m) for p, m in c['cgroup_memory_after'].items()} }", flush=True)
            if c["new_oom_killed"]:
                raise SystemExit(f"STOP: OOMKilled during {c['label']}: {c['new_oom_killed']}")
            print(f"A replicas={replicas} clients={clients}: ok={s['ok']} err={s['errors']} p50={fmt(s['p50_ms'], '.0f')} "
                  f"p95={fmt(s['p95_ms'], '.0f')} rps={fmt(s['rps'], '.3f')} per-pod={c['per_pod_requests']} top_peak={c['top_peak']} "
                  f"problems={len(c['problems']['warning_events'])} restarts={c['problems']['restarts']} "
                  f"oom={c['problems']['oom_killed']}", flush=True)


def experiment_b(data: dict) -> None:
    kubectl("delete", "hpa", "rag", "--ignore-not-found")
    wait_ready(1)
    subprocess.run(["kubectl", "apply", "-f", str(ROOT / "k8s" / "hpa.yaml")], check=True, capture_output=True)
    end = time.time() + 180
    while time.time() < end and (hpa_status() or {}).get("cpu_pct") is None:
        time.sleep(5)
    samples, result_box = [], {}
    t_load = time.time()

    def load():
        result_box["result"] = run_job("load-b", 4, 300)

    th = threading.Thread(target=load)
    th.start()
    while th.is_alive():
        samples.append(sample(with_cgroup=True))
        print(f"B t+{samples[-1]['t'] - t_load:5.0f}s {samples[-1]['deployment']} hpa={samples[-1]['hpa']} "
              f"top={samples[-1]['top']}", flush=True)
        th.join(timeout=SAMPLE_EVERY)
    t_load_end = time.time()
    # Load stopped: keep sampling until back to 1 ready replica (scale-down window 300 s), max 20 min.
    while time.time() < t_load_end + 1200:
        samples.append(sample(with_cgroup=True))
        s = samples[-1]
        print(f"B post t+{s['t'] - t_load_end:5.0f}s {s['deployment']} hpa={s['hpa']}", flush=True)
        if s["deployment"]["spec_replicas"] == 1 and len([p for p in s["pods"] if not p["deleting"]]) == 1 \
                and s["t"] - t_load_end > 60:
            break
        time.sleep(SAMPLE_EVERY)
    result = result_box["result"]
    data.setdefault(ROUND, {})["experiment_b"] = {"clients": 4, "duration_s": 300, "t_load_submitted": t_load, "t_load_job_end": t_load_end,
                            "summary": summarize(result), "per_pod_requests": requests_per_pod(result["started"]),
                            "samples": samples, "problems": problems_since(t_load),
                            "hpa_events": [e for e in json.loads(kubectl("get", "events", "-o", "json"))["items"]
                                           if e["involvedObject"]["kind"] == "HorizontalPodAutoscaler"],
                            "raw": result}
    save_results(data)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    mode = args[0] if args else ""
    if mode not in ("smoke", "a", "b"):
        raise SystemExit(__doc__)
    cm = json.loads(kubectl("get", "configmap", "rag-config", "-o", "json"))["data"]
    if cm.get("LLM_STUB") != "1":
        raise SystemExit("LLM_STUB is not 1 in the live ConfigMap; refusing to spend Groq tokens")
    questions = pick_questions()
    setup_configmap(questions)
    data = load_results()
    data.setdefault("meta", {
        "note": "stub LLM (LLM_STUB=1): retrieval + reranking only, not end-to-end; "
                "load generator and cluster share the Mac's CPUs (OrbStack VM)",
        "url": URL, "questions": questions, "sample_every_s": SAMPLE_EVERY,
        "new_connection_per_request": "urllib opens a fresh TCP connection per request and sends Connection: close",
    })
    if mode == "smoke":
        c = cell("smoke", 1, 15)
        data.setdefault("round1", {})["smoke_test"] = c
        save_results(data)
        print(json.dumps(c["summary"]), c["per_pod_requests"])
    elif mode == "a":
        experiment_a(data, tuple(int(x) for x in args[1:]) or (1, 3))
    else:
        experiment_b(data)


if __name__ == "__main__":
    main()
