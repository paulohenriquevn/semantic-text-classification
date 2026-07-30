# Blueprint: M8 Pilot Hardening & V1 Ship

**Slug:** `m8-pilot-hardening`
**Date:** 2026-07-30
**Plan reference:** `knowledge-base/discoveries/plans/m8-pilot-hardening-plan.md`
**Edge-cases reference:** `knowledge-base/reviews/m8-pilot-hardening-edge-cases-2026-07-30.md`

## Executive summary

M8 is the capstone: prove every V1 ship criterion, add operational self-monitoring, and
instrument the supervisor alert-engagement north-star proxy. Two cloned peers supply the
prior art:

- **chatwoot** gives a production `/health` endpoint (`ActionController::Base`, renders
  `{ status: 'woot' }` — liveness-only, zero deps) with a two-line request spec, and a
  Redis-backed engagement/presence tracker (sorted-set-by-timestamp activity in a rolling
  window with stale-flush + a DB fallback for status).
- **livekit-agents** gives a self-metrics collection pattern: pydantic value objects with a
  `type` discriminator, a stateful accumulator collector (`collect(...) → += → get_summary()`
  returning a defensive deepcopy), and a discriminated `AgentMetrics` union.

The four M8 decisions synthesized below are: (D1) a **readiness** probe set that goes beyond
chatwoot's liveness-only example — checking session/queue/DB; (D2) a **self-metrics value
object + accumulator collector** modeled on livekit; (D3) a **V1-acceptance-report contract
that RE-RUNS live checks / reads fresh metrics artifacts — never hard-codes PASS** (EC-1);
(D4) an **alert-engagement metric** reusing the M5 `labels` table, framed honestly as
instrumented-and-tested with its production rate pending a real pilot (EC-2); (D5) a
**failure-recovery posture** (liveness vs readiness split + existing backpressure/pool
queueing); (D6) the **synthetic-load honesty caveat** for real 8 kHz drift (EC-3).

All 7 research questions were fully answered from the cited reference files; the only honest
gap is that livekit's `UsageCollector`/`UsageSummary` are marked deprecated upstream (the
structural pattern still applies — see `## Honest gaps`).

## Context

M8 (`ROADMAP.md:278` — `### M8 — [ ] Pilot hardening & V1 ship`) depends on all prior
milestones, whose V1 evidence already exists: alert p95 < 2 s + critical-alert precision 0.89
(M3), retrieval p95 = 160.95 ms (`experiments/results/m5_hybrid_bench.json`), sentiment
macro-F1 0.856 (M2), purge/rollup-survives-drop (M6). The monitor is a running FastAPI app
whose composition root already exposes `app.state.session`, `app.state.channel`,
`app.state.kpi_repo`, `app.state.label_repo` and a read `MonitoringPool`
(`src/talkex/monitoring/interface/app.py:117-121`). M8 adds a `/health` route to that same
interface adapter and a self-metrics + engagement surface, plus a single acceptance report
that aggregates the V1 criteria from **live** checks. Honest scope: there is no real 8 kHz
call-center pilot in this environment, so V1 criteria are validated on synthetic load (as
M2-M6 did) and the acceptance report carries an explicit "real-drift pending" caveat.

Constraints: `.claude/rules/architecture.md § 1` (the health endpoint is an interface
adapter — the composition root wires concretes, `app.py:88-130`); `.claude/rules/testing.md`
(the acceptance harness re-runs real checks, never a hard-coded PASS); Unbreakable Rule 3
(synthetic-load validation is labeled; the engagement rate is a proxy on synthetic data).

## Objective

Decide the M8 hardening architecture — health-probe set, self-metrics shape, V1-acceptance
contract, engagement metric, failure-recovery posture — from evidence in the two in-scope
peers, so `/to-plan` can turn this blueprint into an M8 implementation plan.

## Coverage Corner 1 — Integration Tests

**Question mapped: Q5 — How does chatwoot test the `/health` endpoint?**

Chatwoot's health test is a Rails **request spec** (integration-level: real routing + real
response body), not a controller unit test:

| Example | Request | Assertions | Citation |
|---|---|---|---|
| "returns success status" | `get '/health'` | `have_http_status(:success)` **and** `response.parsed_body['status']).to eq('woot')` | `knowledge-base/references/chatwoot/spec/controllers/health_controller_spec.rb:5-9` |

Structure: `RSpec.describe 'Health Check', type: :request` (line 3), one `describe 'GET /health'`
block (line 4), one `it` that hits the real route and asserts **both** the status code and a
named body field (lines 5-9). The contract under test is deliberately tiny: *status code +
one JSON field*.

**Synthesis for M8's `/health` integration test:** mirror this shape with FastAPI's
`TestClient` — one integration test that `GET /health` and asserts (a) HTTP 200 when the
session is running and deps reachable, plus (b) a named JSON body (e.g.
`{"status": "ok", "checks": {...}}`). Because TalkEx's probe is a *readiness* probe (D1),
add negative-case tests the chatwoot example does not have: session-not-running and
DB-unreachable each return a degraded status (503 / `"status": "not_ready"`) — this is the
edge-vs-negative discipline of `.claude/rules/testing.md § 4.1`. The chatwoot spec proves the
happy-path contract shape; TalkEx extends it with the failure branches its richer probe
introduces.

**EC-1 acceptance-harness integration test:** the V1-acceptance harness (D3) is itself an
integration surface and MUST have a test asserting it **sources each criterion from a live
check or a freshly-produced metrics artifact** — e.g. by pointing it at a temp metrics JSON
with a deliberately failing value and asserting the report flips to FAIL. A harness that
passes against a hard-coded constant is acceptance theatre; the test is what forbids it.

## Coverage Corner 2 — Dependencies

**Questions mapped: Q3 (chatwoot health/engagement path deps), Q4 (livekit metric data structures).**

### Q3 — chatwoot health + engagement dependencies

| Dependency | Role | Citation |
|---|---|---|
| `ActionController::Base` (NOT `ApplicationController`) | Health controller inherits the bare base **to skip all middleware, authentication, and callbacks** — so a health check never touches app deps | `knowledge-base/references/chatwoot/app/controllers/health_controller.rb:1-3` |
| (none) | `show` renders `json: { status: 'woot' }` — no DB, no cache, no version read | `knowledge-base/references/chatwoot/app/controllers/health_controller.rb:4-6` |
| `::Redis::Alfred` (sorted sets + hashes) | Engagement/presence store: `zadd`/`zscore`/`zrangebyscore`/`zremrangebyscore` for presence; `hset`/`hget` for status | `knowledge-base/references/chatwoot/lib/online_status_tracker.rb:11,15,33,37,48-49` |
| `Account` / `account_users` (ActiveRecord) | DB fallback when a status is absent from Redis | `knowledge-base/references/chatwoot/lib/online_status_tracker.rb:67,73` |
| `ENV['PRESENCE_DURATION']` / `ENV['CONTACT_PRESENCE_DURATION']` | The rolling activity window is configurable | `knowledge-base/references/chatwoot/lib/online_status_tracker.rb:3-5` |

**Insight:** chatwoot deliberately makes the **health path depend on nothing** (liveness) while
the **engagement path depends on a fast store (Redis) with a durable fallback (DB)**. For
TalkEx the analog store is Timescale/Postgres behind the existing `MonitoringPool`
(`src/talkex/monitoring/infrastructure/pool.py:17-40`) and the M5 `labels` table — engagement
reads go through the pooled read path, health reads probe those deps for *readiness*.

### Q4 — livekit metric data structures

All metric types are pydantic `BaseModel` subclasses of `_BaseMetrics`
(`knowledge-base/references/livekit-agents/livekit-agents/livekit/agents/metrics/base.py:5,13`):

| Class | Representative fields | Citation |
|---|---|---|
| `LLMMetrics` | `type` (Literal discriminator), `label`, `timestamp`, `duration`, `ttft`, `*_tokens`, `tokens_per_second`, `cancelled` | `base.py:20-34` |
| `STTMetrics` | `duration`, `audio_duration`, `streamed`, `acquire_time`, `connection_reused` | `base.py:37-56` |
| `TTSMetrics` | `ttfb`, `duration`, `audio_duration`, `characters_count` | `base.py:59-81` |
| `VADMetrics` | `idle_time`, `inference_duration_total`, `inference_count` | `base.py:84-91` |
| `EOUMetrics` | `end_of_utterance_delay`, `transcription_delay` (latency deltas) | `base.py:94-112` |
| `RealtimeModelMetrics` | nested `InputTokenDetails`/`OutputTokenDetails` value objects | `base.py:131-179` |
| `Metadata` | `model_name`, `model_provider` (optional nested) | `base.py:8-10` |
| `AgentMetrics` | **discriminated union** of all metric types | `base.py:212-222` |

**Pattern:** every metric carries a `type: Literal[...]` **discriminator** (e.g.
`base.py:21,38,60`) + a `timestamp` + typed counters/latencies, and the whole family is a
tagged union. This is the exact shape TalkEx should adopt for its own latency/health metric
value objects (frozen pydantic, per `CLAUDE.md` "models/ frozen, strict"): a small set of
metric records (e.g. `AlertLatencyMetric`, `QueueDepthMetric`, `IngestThroughputMetric`) each
with a `type` discriminator + `timestamp` + measured value, aggregated by a collector (D2).

## Coverage Corner 3 — Tools

**Questions mapped: Q6 (chatwoot engagement/activity tracker), Q7 (livekit collector structure for reuse).**

### Q6 — chatwoot online-status tracker (engagement / activity)

`OnlineStatusTracker` is a stateless module of class methods over two Redis structures
(`knowledge-base/references/chatwoot/lib/online_status_tracker.rb`):

1. **Record activity (write):** `update_presence` does `zadd(key, Time.now.to_i, obj_id)` —
   a **sorted set scored by timestamp**, value = object id (line 10-12). Recording an activity
   = writing the current timestamp.
2. **Query activity (read):** `get_presence` compares `zscore` against `now - duration`
   (lines 14-18); `get_available_contact_ids` / `get_available_user_ids` do a
   `zrangebyscore(key, range_start, '+inf')` over the rolling window (lines 44-49, 72-79).
3. **Stale-flush:** before reading, `zremrangebyscore(key, '-inf', "(#{range_start}")` evicts
   entries older than the window so the set does not clog (line 46-48).
4. **Status + DB fallback:** discrete status (online/busy/offline) lives in a Redis hash
   (`set_status`/`get_status`, lines 32-38); on a cache miss it falls back to
   `get_availability_from_db` and back-fills Redis (lines 62-70).

The spec proves exactly these behaviors: only in-window ids are returned
(`online_status_tracker_spec.rb:15-18`), DB fallback populates Redis
(`online_status_tracker_spec.rb:25-34`), and stale records are flushed after the duration
(`online_status_tracker_spec.rb:52-55`).

**Synthesis for M8's alert-engagement metric:** the "acted-on rate" is a *windowed activity
count over a total*. TalkEx already has the two event streams: **alerts** (M3, persisted via
the alert repo) and **acts** — a supervisor label IS an act-on-alert, recorded through
`POST /label` into the M5 `labels` table (`src/talkex/monitoring/interface/app.py:169-180`).
So engagement = `labels_in_window / alerts_in_window`, computed with the same
window-and-count shape chatwoot uses (a time-bounded range query), read through the existing
pooled KPI/label read path rather than a new store. **EC-2 (honesty):** the plumbing (record
an act → compute a rate) is real and unit-testable, but on synthetic data the *rate itself is
not a production signal* — there is no real supervisor acting. The metric is
instrumented-and-tested; its production value awaits a live pilot.

### Q7 — livekit collector structure (collect + summarize)

`UsageCollector`'s public surface is three members
(`knowledge-base/references/livekit-agents/.../metrics/usage_collector.py`):

| Member | Role | Citation |
|---|---|---|
| `__init__` | creates the accumulator state `self._summary = UsageSummary()` | `usage_collector.py:65,72` |
| `__call__(metrics)` | delegates to `collect` so the collector is usable **as an event callback** | `usage_collector.py:74-75` |
| `collect(metrics)` | type-dispatches (`isinstance`) and accumulates with `+=` into summary fields | `usage_collector.py:77-116` |
| `get_summary()` | returns `deepcopy(self._summary)` — a **defensive copy**, so readers cannot mutate internal state | `usage_collector.py:118-119` |

**Pattern:** stateful accumulator + type-dispatched increment + callback ingress + defensive-copy
read. This is the reusable structure for TalkEx's self-metrics collector (D2): a long-lived
object the pipeline pushes metric records into (callable / `record(...)`), which accumulates
counters + latency samples, and exposes a `summarize()` returning an immutable snapshot the
`/health` route and the acceptance report can read without racing the writer.

## Coverage Corner 4 — Techniques

**Questions mapped: Q1 (chatwoot readiness probing), Q2 (livekit collect+aggregate technique).**

### Q1 — what chatwoot's health controller probes

Chatwoot's health controller probes **nothing** — it is a pure **liveness** signal:

- It inherits `ActionController::Base` (not `ApplicationController`) **specifically to skip all
  middleware, authentication, and callbacks** (`health_controller.rb:1-3`).
- `show` renders a static `json: { status: 'woot' }` (`health_controller.rb:4-6`). No DB
  query, no cache ping, no version string, no dependency check.

**Technique takeaway:** chatwoot draws a sharp line — the health endpoint answers only "is the
process up and routing?", intentionally decoupled from every dependency so the check can never
be blocked by a slow/broken dep. For M8 this is the *liveness* half; TalkEx's `/health`
additionally needs a *readiness* half (D1/D5) that DOES probe deps, and the two must be
distinguishable (a healthy-but-not-ready process should not be killed by an orchestrator, only
kept out of rotation).

### Q2 — how livekit collects + aggregates self-metrics

The aggregation technique in `usage_collector.py`:

1. **Single ingress, type-dispatched:** `collect(metrics)` branches on the metric subtype via
   `isinstance` (`usage_collector.py:78,83,111,115`) — one entry point handles the whole
   discriminated union.
2. **In-place accumulation:** each branch does `self._summary.<field> += metrics.<field>`
   (e.g. tokens `usage_collector.py:79-81`, TTS characters/duration `usage_collector.py:112-113`,
   STT audio duration `usage_collector.py:116`). Aggregation is a running sum in a mutable
   summary object, not a re-scan of stored events.
3. **Snapshot read:** `get_summary()` returns a `deepcopy` (`usage_collector.py:118-119`) so a
   reader gets a consistent, immutable view.

**Technique takeaway:** aggregate incrementally as metrics arrive (O(1) per event), keep the
running total in one value object, and hand out defensive copies. For latency-type metrics
TalkEx will additionally need percentiles (p95, per V1 criteria) — that requires keeping a
bounded sample buffer or a streaming-quantile estimator rather than a plain sum, which is the
one place TalkEx must go beyond livekit's counter-only accumulator (noted in Recommendations).

## Cross-cutting Comparison

| Dimension | chatwoot | livekit-agents | M8 decision |
|---|---|---|---|
| Health/readiness surface | Liveness-only; static body; zero deps (`health_controller.rb:4-6`) | (none — metrics, not health) | Readiness probe: session + queue + DB (D1), keep a liveness split (D5) |
| Test shape | Request spec: status + body field (`health_controller_spec.rb:5-9`) | Unit accumulation (implied) | Integration test happy + negative branches (D1, Corner 1) |
| Metric data model | Redis sorted set + hash (`online_status_tracker.rb:11,33`) | pydantic tagged union w/ `type` discriminator (`base.py:212`) | Frozen pydantic metric records w/ discriminator (D2, Q4) |
| Aggregation | Windowed range query + stale-flush (`online_status_tracker.rb:44-49`) | In-place `+=` accumulator + deepcopy read (`usage_collector.py:77-119`) | Accumulator for counters/latency (D2) + windowed count for engagement (D4) |
| Engagement/activity | Timestamped events in a rolling window → available set (`online_status_tracker.rb:44-55`) | (none) | acted-on rate = labels/window ÷ alerts/window, proxy (D4, EC-2) |
| Dependency posture | Health depends on nothing; engagement on Redis + DB fallback | Metrics purely in-memory | Health probes deps for readiness; engagement via pooled read path (D3, Q3) |
| Honesty | — | Deprecation warnings on `UsageSummary`/`UsageCollector` (`usage_collector.py:12-14,60-63`) | Synthetic-load + proxy caveats explicit (D6, EC-2/EC-3) |

## ADRs

### D1 — Health-probe set: a readiness probe, not chatwoot's liveness-only

**Decision:** M8 adds `GET /health` to the interface layer
(`src/talkex/monitoring/interface/app.py`, alongside the existing routes) returning a JSON
body `{"status": "ok"|"not_ready", "checks": {"session": ..., "queue_depth": ..., "db": ...}}`.
The probe checks three things already reachable from the composition root: the monitoring
session is running (`app.state.session`, `app.py:118`), the ingest channel depth
(`app.state.channel.qsize()`, `channel.py:37`), and DB reachability via a trivial query on a
pooled connection (`MonitoringPool.connection()`, `pool.py:37-40`). HTTP 200 when ready, 503
when a required dep is down.

**Rationale:** chatwoot proves the *shape* of a health endpoint but its probe is liveness-only
(`health_controller.rb:4-6`) — it intentionally checks nothing. TalkEx's operational risk is a
*dependency* failure (DB down, queue saturated, consumer dead), which a liveness probe hides.
`.claude/rules/architecture.md § 1` puts the endpoint in the interface layer with concretes
injected at the composition root (`app.py:88-130`) — the probe reads `app.state.*`, it does
not import infra directly.

**Alternatives considered:** (a) copy chatwoot's static liveness body verbatim — rejected: it
would report "ok" while the DB is unreachable, the exact failure ops needs to see. (b) A deep
probe that runs a real search/ingest — rejected (KISS + it would make the health check itself a
load source and could block on the very dep it tests).

**Consequence:** a readiness route the orchestrator/load-balancer can poll; queue depth and DB
state become externally visible. Honesty (Rule 3): the probe reports *reachability*, not
correctness — a reachable-but-wrong DB still returns ready, documented as a known limit.

### D2 — Self-metrics shape: pydantic value objects + an accumulator collector

**Decision:** model the monitor's own metrics as a small set of frozen pydantic records, each
with a `type` discriminator + `timestamp` + measured value (mirroring `base.py:20-34,212`), and
aggregate them with a stateful collector exposing `record(metric)` / `summarize()` where
`summarize()` returns an immutable snapshot (mirroring `usage_collector.py:77-119`).

**Rationale:** livekit's discriminated-union + in-place-accumulator + deepcopy-read pattern is a
proven, dependency-free self-metrics design; frozen pydantic aligns with the project's
"embeddings represent, models decide … models/ frozen, strict" convention (`CLAUDE.md`).
Value objects with a discriminator make each metric self-describing for the `/health` body and
the acceptance report.

**Alternatives considered:** (a) a Prometheus client — rejected for V1 as heavier than needed
and out of scope here (revisit when ops adds a scrape target; noted in Recommendations). (b)
plain dict counters — rejected: no typing, no discriminator, easy to mis-key.

**Consequence:** a reusable collector feeding both `/health` and the acceptance report. Rule 3
caveat: counters aggregate as `+=` like livekit, but latency percentiles (p95, required by V1)
need a sample buffer/streaming quantile — the one deliberate extension over livekit's
counter-only summary.

### D3 — V1-acceptance-report contract: RE-RUN live checks, never hard-code PASS (EC-1)

**Decision:** the acceptance harness emits a single pass/fail report where **each V1 criterion
is sourced from a live check or a freshly-produced metrics artifact** at run time — never from a
historical constant. Concretely: retrieval p95 is read from a fresh
`experiments/results/m5_hybrid_bench.json` (re-generated by the M5 benchmark, not the committed
number quoted in Context); alert p95 + critical-alert precision by re-running the M3 alert
tests; sentiment macro-F1 by re-running the M2 evaluation (or reading its freshly-emitted
metrics JSON); purge/rollup-survives-drop by re-running the M6 `test_kpi_rollup`. The report
carries, per criterion: the criterion, the threshold, the **measured** value, the source
(command/artifact path), and PASS/FAIL. An explicit banner states "no hard-coded PASS — every
value came from a live check or a freshly-produced artifact this run".

**Rationale:** `.claude/rules/testing.md` — the acceptance harness re-runs real checks; a
hard-coded PASS is acceptance theatre, the same failure the plan-confidence golden rule guards
against. EC-1 makes this a mandatory synthesis checkpoint. chatwoot's own health spec models
the discipline in miniature: it asserts the *actual* response, not a constant
(`health_controller_spec.rb:6-8`).

**Alternatives considered:** (a) embed the M2-M6 numbers from Context as constants and print
PASS — rejected outright (theatre; violates EC-1 and Rule 3). (b) re-run everything from
scratch every invocation including model training — rejected as impractical; reading a
freshly-produced metrics artifact (regenerated by the existing benchmark) is the sanctioned
middle path.

**Consequence:** the report is trustworthy — a regression in any criterion flips it to FAIL.
Cost: the harness must orchestrate real check runs / artifact regeneration, so it is slower
than printing constants (accepted). The harness itself gets an integration test (Corner 1)
asserting a deliberately-failing artifact flips the report to FAIL.

### D4 — Engagement metric: acted-on rate as a north-star PROXY, reusing the labels table (EC-2)

**Decision:** define alert-engagement as `acted_on_rate = labels_in_window / alerts_in_window`,
where an act-on-alert reuses the existing `POST /label` path into the M5 `labels` table
(`src/talkex/monitoring/interface/app.py:169-180`) and alerts are the M3 alert stream. Compute
it with the windowed-count technique chatwoot uses for presence
(`online_status_tracker.rb:44-49`) through the existing pooled read path
(`app.state.kpi_repo` / `label_repo`, `app.py:120-121`) — no new store. Expose it as a metric
value object (D2) and surface it on the dashboard.

**Rationale:** the north-star is supervisor engagement; a label IS a supervisor acting on what
they saw, so the signal already exists without new instrumentation (KISS / don't-reinvent —
`.claude/rules/parsimony-ladder.md` rung 4: reuse an installed capability). chatwoot shows the
window-and-count shape is the right primitive for an activity rate.

**Alternatives considered:** (a) a brand-new "ack" endpoint + table — rejected: the labels
table already captures the act (YAGNI/duplication). (b) infer engagement from SSE connection
time à la chatwoot presence — rejected: connection ≠ action; a label is a stronger act signal.

**Consequence (EC-2 honesty, Rule 3):** the plumbing (record an act → compute a rate) is real
and unit-testable, but on synthetic data **the rate is not a production signal** — there is no
real supervisor. The blueprint frames the metric as *instrumented-and-tested, production value
pending a live pilot*. The acceptance report lists engagement as "instrumented" not as a passed
numeric criterion.

### D5 — Failure-recovery posture: liveness/readiness split over existing backpressure

**Decision:** V1 failure-recovery is documented, not newly built: (1) the `/health` readiness
probe (D1) reports `not_ready` (503) when the session is down or the DB is unreachable, so an
orchestrator keeps the instance out of rotation without killing a merely-not-ready process;
(2) ingest overload is already handled — `TurnChannel.put` awaits when full
(`src/talkex/monitoring/domain/channel.py:40-44`, backpressure) and `MonitoringPool.connection`
**queues rather than errors** under saturation (`src/talkex/monitoring/infrastructure/pool.py:37-40`);
(3) the runbook documents restart/recovery and what each `/health` state means for operators.

**Rationale:** `.claude/rules/error-handling.md` fail-fast/fail-clear — readiness makes a
dependency failure *loud and externally visible* instead of silently serving a broken monitor.
The backpressure + pool-queueing primitives already exist from M0/M5; M8's job is to expose
and document their recovery semantics, not to add new machinery (YAGNI).

**Alternatives considered:** (a) add retry/circuit-breaker layers now — rejected: no measured
failure mode yet demands them (YAGNI; revisit post-pilot). (b) a single combined health signal
with no liveness/readiness distinction — rejected: it would let a transient dep blip trigger a
process kill (the chatwoot liveness-only lesson inverted).

**Consequence:** operable V1 with visible degradation and a runbook. Rule 3 caveat: recovery is
validated against synthetic failure injection, not a real outage.

### D6 — Honesty caveat: synthetic-load validation + unvalidated 8 kHz drift (EC-3)

**Decision:** the acceptance report and the M8 README/CHANGELOG language carry an explicit
caveat: "All V1 criteria validated on synthetic load; real 8 kHz call-center speech (ROADMAP
risk 1) is unvalidated until a live pilot." No unqualified "production-ready" claim ships
without dogfood evidence.

**Rationale:** `.claude/rules/public-copy.md § 3` bans unqualified production claims pre-v1.0
with real evidence; Unbreakable Rule 3 demands the synthetic-vs-real boundary be stated. EC-3
accepts this as a documented pilot risk, not a defect.

**Alternatives considered:** claim V1 as production-proven on synthetic evidence — rejected
(dishonest; `/dogfood` gate would block it).

**Consequence:** an honest V1 ship — criteria proven, real-world drift flagged as the first
pilot-phase validation task.

## Recommendations

One concrete proposal per research question:

- **Q1 → `/health` probe set:** implement a readiness route returning `{status, checks:
  {session, queue_depth, db}}` (D1), 200/503. Split liveness (process up) from readiness (deps
  reachable) so orchestrators do not kill a not-ready-but-alive process.
- **Q2 → aggregation technique:** aggregate self-metrics incrementally (in-place `+=` per event,
  livekit-style) for counters; for the V1 latency criteria keep a bounded sample buffer /
  streaming-quantile so `/health` and the acceptance report can emit p95 (the one extension
  beyond livekit's counter-only summary).
- **Q3 → dependency posture:** back health readiness + engagement reads on the existing
  `MonitoringPool` and M5 labels/KPI repos; keep the *liveness* signal dependency-free
  (chatwoot's lesson).
- **Q4 → metric value objects:** frozen pydantic records with a `type` discriminator +
  `timestamp` + measured value; a discriminated union like `AgentMetrics` (`base.py:212`).
- **Q5 → tests:** one happy-path integration test (chatwoot shape) + negative-case tests
  (session down, DB unreachable → 503); plus a harness test asserting a failing artifact flips
  the acceptance report to FAIL (EC-1).
- **Q6 → engagement metric:** `acted_on_rate = labels/window ÷ alerts/window` reusing
  `POST /label` + the pooled read path; label it a proxy, production value pending a pilot
  (EC-2).
- **Q7 → collector structure:** a long-lived collector with `record(metric)` (callable ingress)
  + `summarize()` returning an immutable snapshot (deepcopy/`model_copy`), feeding both
  `/health` and the acceptance report.

**Cross-cutting:** defer Prometheus/OpenTelemetry export and retry/circuit-breaker layers to
post-pilot (YAGNI); V1 needs visible health + an honest acceptance report, not a full
observability stack.

## Honest gaps

- **livekit `UsageCollector`/`UsageSummary` are deprecated upstream.** Both carry deprecation
  warnings pointing to `ModelUsageCollector`/`LLMModelUsage`
  (`knowledge-base/references/livekit-agents/.../metrics/usage_collector.py:12-14,60-63,66-71`).
  The *structural* pattern (collect → `+=` → deepcopy summarize; discriminated union) is still
  the reference and is what D2/Q7 synthesize; TalkEx is not adopting the deprecated classes
  themselves, only the shape. The non-deprecated `ModelUsageCollector` was not in the plan's
  cited path set, so its API is not inspected here.
- **Percentile aggregation has no direct peer example.** livekit's collector sums counters; it
  does not compute p95. TalkEx's V1 latency criteria need percentiles, so D2/Recommendations
  extend the pattern with a sample buffer — this extension is reasoned from first principles,
  not copied from a reference.
- **No health/engagement surface in the other cloned peers.** Per plan ADR D3,
  `ai-powered-call-center-intelligence`, `portuguese-bert`, `portuguese-nlp` were excluded;
  all four corners are covered by chatwoot + livekit, so no corner relied on an absent peer.
- **Engagement rate is unvalidated on real data** (EC-2/EC-3) — plumbing tested, production
  value pending a live pilot. This is a documented scope boundary, not a missing answer.

## discover-confidence verdict

**SHIPPABLE_WITH_CAVEATS** — all 7 research questions are fully answered with real `path:line`
citations to the in-scope reference files; all four coverage corners have non-placeholder,
evidence-dense content; six ADRs synthesize the health-probe / self-metrics / V1-acceptance /
engagement / failure-recovery / honesty decisions, each citing ≥1 project principle plus Rule 3.
The caveats are honest scope boundaries (livekit's reference classes are deprecated — pattern
adopted, not code; percentile aggregation extends the peer pattern; engagement rate is a
synthetic-data proxy), not structural gaps.
