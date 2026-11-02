# M2A: bounded prefix scheduling on native AsyncScheduler

Status: implementation in progress; no M2A native acceptance yet. Design recorded
before implementation on 2026-11-02 (Asia/Hong_Kong). Baseline is accepted M1
`fc23bc380dc82f5b57922b26d072470397184267`; M1 mechanism evidence remains attached
to `0288eb4`, not its later validation-entry fixes. PR #4 is open.

## Fixed-source investigation

All references below are to vLLM 0.32.0, commit
`5df63a7203cdb9328d824a27b682e77c7b9928be`, Apache-2.0, copyright vLLM contributors.
Existing source snapshots were checked against the SHA-256 values frozen in
`configs/native-bc-replay.json`. M2A freezes its own extended source list.

| Source and location | Finding |
| --- | --- |
| `vllm/config/scheduler.py:228–253` | `get_scheduler_cls()` resolves the fully qualified custom class; class/MRO and effective async settings must also be checked. |
| `vllm/v1/engine/core.py:161–182` | Reference for real scheduler construction; no model runner is needed for host-side metadata replay. |
| `vllm/v1/core/kv_cache_manager.py:249–310` | `prefix_cache_lookup_enabled(request)` preserves APC/skip-read semantics. `get_computed_blocks` emits cached-block events in full report mode, so it is not a scoring probe. |
| `vllm/v1/core/kv_cache_coordinator.py:574–592` | Unitary coordinator delegates native longest-prefix lookup using Request block hashes. |
| `vllm/v1/core/single_type_kv_cache_manager.py:732–828` | Full-attention lookup preserves block alignment. Passing `request.num_tokens - 1` preserves last-token recomputation. |
| `vllm/v1/core/block_pool.py:62–72,199–225` | The supported lookup uses dictionary reads, without allocation, touch, metrics or events. |
| `vllm/v1/request.py:84–241` | Progress, output, preemption, placeholder, in-flight, stale and streaming fields distinguish fresh requests. Initial PrefillStats and QUEUED events are not computation history. |
| `vllm/v1/core/sched/scheduler.py:657–939,1571–1644,2457–2463` | Running precedes waiting; kv_holding_waiting has priority. deferred_waiting is a set. Real chunks stay running. Preemption resets computed progress but increments num_preemptions. |
| `vllm/v1/core/sched/request_queue.py:74–103` | FCFS is a deque; pop removes the head and prepend_requests uses extendleft, reversing its input. |
| `vllm/v1/core/sched/scheduler.py:2077–2103,2371–2385,2915–2936` | Output updates legitimately consume events/statistics; observation must retain native returned stats and published batches. Aborted historical objects may keep counters after removal. |

Browse the [fixed upstream tree](https://github.com/vllm-project/vllm/tree/5df63a7203cdb9328d824a27b682e77c7b9928be).
Source inspection is not native-object validation.

## Implementation contract

Three production modules: standard-library configuration, length-only read-only
probe, and a direct AsyncScheduler subclass. Configuration lives in
additional_config.prefix_scheduler; defaults are off/16/500 ms. guarded raises
NotImplementedError. max_wait_ms is validated but never used for protection or
timing. off delegates directly; supported prefix/window=1 also delegates directly.
The schedule signature preserves throttle_prefills and calls super exactly once.

Prefix supports only audited single-CUDA-device semantics (all parallel degrees
one), effective async/V2, FCFS, APC, ordinary autoregressive text generation,
one causal exact FullAttentionSpec and identical hash/physical/scheduler block
sizes. Distributed, hybrid, offload, speculative, LoRA, multimodal, auxiliary
output, KV and EC connector configurations are rejected. Precheck before native
construction where possible; postcheck failure shuts down a successfully
constructed native scheduler. off does not inherit prefix-only restrictions.

The probe calls the native lookup-enabled predicate, then the coordinator lookup
with existing Request hashes and num_tokens-1. It returns only the hit length.
It never passes probe blocks to allocation, disables events, pins blocks, stores
cross-step results or reads a waiting clock.

The window is the contiguous fresh ordinary WAITING prefix of self.waiting,
bounded by window. Reject membership in deferred/in-flight sets, any computation,
output, preemption, placeholder, stale, chunk, scheduling or resumption history,
and unsupported request types. Stop at the first barrier; never scan a suffix to
filter candidates. Stable descending score retains current order on ties.

Read and score before any mutation. If the sorted order changes, save the original
window and tail length, pop only that window, then prepend reversed(sorted).
On a pre-super transaction exception, remove the current head portion determined
by len(queue)-tail_len with native deque primitives, restore reversed(original),
and rethrow. The queue is the real FCFS queue in the serialized engine thread.
Super is outside the transaction catch: native scheduling is not rolled back.
Running may change cache state after scoring; normal native relookup and allocation
remain authoritative. No production per-request metadata or counters are added.

## Independent verification contract

M1 code, factory, trace auditor, frozen config, evidence and locks remain read-only.
M2A owns separate construction, finite FIFO synthetic-output driver, snapshots,
matrix and standard-library trace auditor. Synthetic outputs are the only execution
substitution; no model weights or KV tensors are loaded/computed.

Six test families: BC causal controls; native/off equivalence with APC on/off;
read-only probe boundaries; bounded windows/rollback; real lifecycle; fail-closed
entry gates. Every formal stateful native case is independently rebuilt at least
three times. Deterministic configuration/CLI negatives use real subprocesses.
The required scenario set is fixed in code, not freely shrinkable through config.
Missing, empty, extra, duplicate, skipped or incomplete cases cannot verify.

Snapshots use explicit native field/type handling and identity/alias records,
never draining methods or descriptor evaluation. Compare same-object probe state
exactly, including Python random state. Enable log_stats, full KV events, metrics
with sample=1.0, and the real null publisher. Transparently record native publish
batches and returned EngineCoreOutputs stats. Sensitivity controls must detect
real touch, full-event and statistics changes. Independent native/off comparisons
list allowed class/config/identity and clock-field differences explicitly.

BC input/geometry remains M1: pressure=5, ample=10 total blocks (including null),
block=16, prompts A/B/C=33/49/33, common prefix32, budget64, active2, pending2.
All modes enqueue B then C; only class policy may reorder. Pressure native/off
must dispatch B49+C33=82 prefill tokens; prefix must dispatch B49+C1=50. Ample
must dispatch50 in all modes. Full trace establishes eviction/relookup/allocation
and dispatch intervals, not just summary totals.

Chunk fixture: capacity10, budget16, active1, warm/drain A, cold65-token D,
then hot C after D's first dispatch. Natural preemption fixture: cold initial
cache, capacity5, budget64, active2, two distinct32-token prompts, max_tokens3,
watermark0, speculative/lookahead0, no connectors, V2/async/PP1 and pending2.
Keep the first output pending while the next decode grows to slot33; native
allocation must preempt the trailing request. After FIFO consumption enqueue a
hot request behind it and prove the preempted barrier remains excluded. These
parameters are source-derived and must be demonstrated by native calibration.

Also verify scoring followed by running-stage eviction, waiting/running cancellation,
normal completion and ID reuse after old pending outputs drain. Generation IDs
exist only in test traces. For cancellation, distinguish discarded historical
object counters from live state; ordinary completion/preemption must fully drain
placeholders/in-flight/stale/references. No manually set status substitutes for
real chunk/preemption evidence.

## Execution and handoff

Stage commits: A integration/off; B probe; C bounded ordering; D independent
matrix/auditor. D freezes code before formal collection, then documents results
in a separate delivery commit. Failures and calibration use new private runs;
fixes require a new frozen commit and a new accepted full matrix.

Mac runs lightweight checks and full make check, never Linux GPU installs. The
coordinator alone owns serial Linux allocations of at most20 minutes, using
private local connection configuration and the existing fixed environment.
Record explicit release. The schema2 collector, command-success gate and semantic
acceptance are separate. Keep full traces private; review any public fields and
never reuse the readiness exporter for this schema. Preserve failed/unexecuted
items. No performance, TTFT, GPU-work or inference-speedup claims.

M1 and M2A PRs remain unmerged. Hand off commits, exact commands, private evidence
index, independent review, resource release and workspace protection to the
project coordinator. Waiting clocks/protection metadata remain M2B work.
