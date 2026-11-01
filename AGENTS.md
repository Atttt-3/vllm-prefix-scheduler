# Project working agreement

## Read before continuing

Read `docs/state.md`, `docs/project-plan.md`, `docs/architecture.md`, and relevant files under `configs/`. If `.local/workspace.json` exists, use its machine-specific connection and storage locations. It is private, ignored configuration and must never be committed.

## Current scope

The custom scheduler is not implemented. The completed native run validates Python/vLLM installation, Qwen3-4B BF16 inference, APC reuse, and async configuration. Do not describe this as a scheduler optimization result. Begin scheduler work by reproducing the B/C cache-eviction counterexample in the pinned native Scheduler.

## Runtime and implementation boundaries

- Pin vLLM 0.32.0 / `5df63a7203cdb9328d824a27b682e77c7b9928be` and the model revision in `configs/upstream.json`.
- Development uses `uv.lock` in `.venv`; native tests and inference use the Linux GPU lock in `.venv-gpu`. Never sync the development lock into the GPU environment or install Linux GPU dependencies on the Mac.
- Retain `AsyncScheduler`; do not silently fall back to a synchronous scheduler.
- Reorder only a bounded, contiguous head window of fresh ordinary `WAITING` requests. Stop at blocked, resumed, or partially computed requests.
- Probe native prefix state without allocation, reference changes, free-queue mutations, statistics pollution, or KV events. Do not copy client-side hashing.
- Use a monotonic clock for waiting protection, clear metadata on terminal paths, and test request-ID reuse and deque prepend ordering.
- Preserve unsupported requests in native handling. Reject incompatible global configurations before enabling the extension.
- Do not implement CPU offload, distributed scheduling, speculative decoding, LoRA, multimodal or hybrid attention in the first milestone.

## Working and verification

Use a focused branch for each implementation task. Run `make check` for changed local code and records. Add native replay/regression tests when scheduler behavior changes; local evidence tests alone are not native scheduler validation. GPU work requires a scheduler allocation. Keep heavy installation, compilation and inference off login gateways. Release allocations after bounded tasks and never cancel another task's jobs.

Record the exact command, commit, config, workload seed, runtime versions, raw trace location and result interpretation for experiments. Timing from the minimal validation is not a performance benchmark. Keep negative results and failed/unfinished requests.

Update `docs/state.md` when a milestone or blocker changes, and update the relevant documentation alongside behavior changes. Do not mark planned functionality complete based on a placeholder, mocked result, or skipped test.

## Public data and upstream attribution

Do not commit credentials, SSH material, personal accounts, internal addresses, machine paths, raw cluster logs, environments or model weights. Public evidence must use the explicit exporter and retain numerical values and provenance. Preserve existing public evidence; add a new run directory for new records.

Attribute any adapted implementation to its upstream repository, file/PR and commit. Preserve applicable license notices. Do not create new user-facing chats, recurring work or external messages without the user's request.
