# Load testing details

Detailed results moved unchanged from the project README, plus a summary of the memory diagnosis. Back to the [README](../README.md).

## Round 1 (stub LLM: reranking only, not end-to-end)

Round 1 on OrbStack's single-node Kubernetes, with `LLM_STUB=1` (retrieval and reranking only, no Groq calls). The load generator (`scripts/load_test.py`) runs inside the cluster as a Job and calls the Service by DNS with a new connection per request. Each pod is limited to 2 CPU / 2.5 Gi; HPA off; 90 s per cell. The load generator and the cluster share the Mac's CPUs. Raw data: `results/metrics/k8s_load_test.json` (`round1`).

| Replicas | Clients | OK | Errors | p50 | p95 | Requests/s | Peak working set (max pod) | Restarts / OOMKilled |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 9 | 0 | 11.0 s | 11.4 s | 0.091 | 2,373 Mi | 0 / no |
| 1 | 3 | 6 | 0 | 49.7 s | 51.3 s | 0.060 | 2,467 Mi | 0 / no |
| 1 | 6 | 0 | 2,308* | n/a | n/a | 0 | killed at the 2.5 Gi limit | 3 / **OOMKilled, all requests failed** |
| 3 | 1 | 9 | 0 | 11.0 s | 11.7 s | 0.091 | 2,191 Mi | 0 / no |
| 3 | 3 | 13 | 0 | 18.9 s | 34.3 s | 0.134 | 2,205 Mi | 0 / no |
| 3 | 6 | 16 | 0 | 59.6 s | 89.2 s | 0.099 | 2,372 Mi | 0 / no |

\* Inflated by immediate retries: the load generator retries without backoff, so most errors are "Connection refused" while the OOM-killed pod restarted. Every request in that cell failed; it was not rerun.

**Correction (2026-10-13):** the "Peak working set" column above is `memory.current` as reported by `kubectl top` on OrbStack, which includes page cache. It overstates the memory a request actually needs: per-rerank process (anon) memory is about 1.2 GiB (see `results/metrics/k8s_memory_diagnosis.json`). The round-1 numbers are left as recorded.

**Experiment B (HPA scaling) was skipped in round 1:** a-r1-c3 and a-r1-c6 showed that 1 replica cannot hold 3 or more concurrent reranks within the 2.5 Gi limit, so starting the HPA at 1 replica with 4 constant clients would mostly measure OOM kills, not scaling.

## Round 2: one rerank at a time per pod

Same setup and grid as round 1 (stub LLM, 2 CPU / 2.5 Gi per pod, HPA off, 90 s per cell), after adding a concurrency guard to the service: each pod reranks one request at a time. Retrieval and the LLM call run outside the guard; up to `MAX_WAITING` (8) requests queue, each for up to `WAIT_TIMEOUT_S` (120 s), and beyond that `/ask` returns 503 with `Retry-After` and a JSON error body. `/health` never waits on the guard, and `latency_ms` includes `queue_wait_ms`. Memory is per-pod anon (process) memory read from the cgroup before and after each cell, not `memory.peak`, which includes page cache. Raw data: `results/metrics/k8s_load_test.json` (`round2`).

| Replicas | Clients | Round 1 req/s | Round 1 p50 | Round 1 result | Round 2 req/s | Round 2 p50 / p95 | Round 2 OK / 503 / other | Round 2 max anon |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.091 | 11.0 s | ok | 0.053* | 10.7 / 12.7 s | 5 / 0 / 1* | 1,119 MiB |
| 1 | 3 | 0.060 | 49.7 s | ok | 0.091 | 32.8 / 33.9 s | 11 / 0 / 0 | 1,119 MiB |
| 1 | 6 | 0 | n/a | **OOMKilled, all failed** | 0.087† | 68.4 / 69.4 s | 13 / 0 / 0 | 1,167 MiB |
| 3 | 1 | 0.091 | 11.0 s | ok | 0.085 | 12.1 / 12.5 s | 8 / 0 / 0 | 1,159 MiB |
| 3 | 3 | 0.134 | 18.9 s | ok | 0.111 | 24.5 / 36.0 s | 13 / 0 / 0 | 1,159 MiB |
| 3 | 6 | 0.099 | 59.6 s | ok | 0.113 | 48.9 / 58.7 s | 15 / 0 / 0 | 1,174 MiB |

\* The cell started 2 s after scaling from 3 replicas to 1, and its first request hit a departing pod ("No route to host" after 38.8 s). The recorded 0.053 req/s is a driver artifact; a recomputation over the 5 OK requests (55.1 s from first start to last end) gives 0.091 req/s.
† Flagged `possible_sleep` by the load generator's wall-time rule, but a false positive: requests completed steadily every 11-12 s, and the 150 s wall time includes the queue draining after the deadline.

- **No OOM kills, no restarts and no 503s in any round-2 cell.** Anon memory stayed at or below 1,174 MiB per pod, including 1 replica with 6 clients, which was OOM-killed in round 1.
- **1 replica is flat at about 0.09 req/s** from 1 to 6 clients; the extra clients wait in the queue, so latency grows by about one rerank (~11 s) per waiting request instead of every request slowing down.
- **3 replicas plateau at about 0.11 req/s**, only about 1.25x one pod: the pods and the load generator share the Mac's CPUs, and one pod already saturates its 2-CPU quota.
- **Experiment B (HPA scaling) was skipped in round 2 as well:** with 3 replicas adding only about 1.25x on the shared CPUs, the scaling measurement would say little. `k8s/hpa.yaml` is included, but its autoscaling behaviour was not measured.

Caveats: the Mac's CPUs are shared by all pods and the load generator, so throughput does not scale as it would on separate nodes; samples are small (5-15 requests per cell, so p95 rests on few points); and the round-2 load generator differs from round 1's (1 s back-off after errors, 503s counted separately, every exception and every client recorded, a per-cell `possible_sleep` flag). An earlier round-2 run interrupted by the Mac sleeping is kept separately as `round2_interrupted` in the JSON. Known tool gaps: the driver does not wait for departing pods to leave the Service after a scale-down, and the `possible_sleep` wall-time rule counts queue drain time.
