# Project working agreement

## Read before continuing

Read `docs/state.md`, `docs/project-plan.md`, `docs/architecture.md`, and relevant files under `configs/`. If `.local/workspace.json` exists, use its machine-specific connection and storage locations. It is private, ignored configuration and must never be committed. Remote HKU SSH requires the user to connect Cisco Secure Client to the school VPN first.

## Current scope

M4 execution and the explicitly authorized audit-only revision are complete and await project acceptance; see docs/research/m4-handoff.md. Actual execution remains 5525bda3eff2dad20f614eaca7718383c31448e4 and independent analysis is 34fa4eabb846b8095fe7aa9ecdcc05d2a6bf27a7. The original frozen F failure is retained. All three M4 allocations are consumed; the user explicitly authorized no additional GPU for the audit revision. No further allocation, automatic merge, or M5 is authorized.

This isolated worktree is now explicitly authorized for M4-scheduler-stability from M3 delivery 12705a8e392eaa90c019de8e347ee31134ab3ca4. Follow docs/research/m4-design.md and configs/m4-stability.json. Earlier M3 stop-at-M3 language below records historical scope and is superseded only for this isolated M4 task. M4 has its own maximum three serialized single-3090 allocations, twenty minutes each, sixty requested minutes total; no M3 budget transfer or unknown-terminal exception. Preserve all historical records and worktrees. No automatic PR merges or M5.

M1, M2A and M2B have passed project acceptance at their recorded commits. This worktree is authorized for M3 real-model integration on the fixed RTX 3090 / Qwen3-4B BF16 platform. M2B delivery a1e57ef and actual native tested 342f908 remain distinct. Preserve all historical worktrees and evidence. Do not merge stacked PRs or start M4/M5 automatically. M3 uses dedicated passive diagnostic subclasses and real engine smoke; it does not change accepted production policy by default. The original M3 implementation contract is docs/research/m3-design.md. The user subsequently authorized four additional allocations, for a cumulative limit of eight serialized, single-GPU allocations of at most 20 minutes each, including failures and retries. This prospective authorization is recorded in configs/m3-closeout-authorization.json and docs/research/m3-closeout.md; it does not rewrite the original four-allocation runtime configuration or excuse missing historical terminal evidence. Do not allocate GPU merely while waiting for the formal-02 platform record. Before any new allocation, freeze a separate supplemental runtime configuration and compatible budget auditing/export tools, preserving all historical allocations.

On 2026-11-03 the user explicitly authorized a replacement M3 campaign, treating the formal-02 terminal as unrecoverable. Execute docs/research/m3-campaign.md and configs/m3-campaign-v1.json under configs/m3-campaign-authorization.json. The original four records and their unknown terminal remain immutable historical evidence; the original full-history gate remains blocked. New campaign acceptance is independent and must disclose that limitation. Every new allocation still requires complete terminal/resource proof, all attempts count against the cumulative eight-allocation cap, and there is no generic unknown-terminal exemption. Start with one new 20-minute allocation and retain unused budget. Administrator recovery is no longer a prerequisite for this newly authorized scope.

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

### Reusable project workflows

Use the project-local skill at `.agents/skills/project-workflow/SKILL.md` for bounded multi-agent work, upstream mechanism research, and collecting/reviewing experiment evidence. Its procedures and runnable examples live in `docs/workflows.md`; task, research and handoff templates live in `docs/templates/`. Routine small edits do not require delegation or a new research document.

- For independent investigations or reviews, delegate bounded questions with explicit file ownership and return evidence to one coordinator. Subagents share the checkout unless an isolated worktree is explicitly created; coordinate writes and serialize experiments on the single GPU. Do not change the shared branch or absorb another task's uncommitted work. Use a focused branch/worktree when the baseline is clean or the existing changes belong to this task.
- Research starts with a project question. Open primary source files/PRs, record access date and commit/head/status, compare upstream behavior with this project's invariants, and preserve attribution/license boundaries. A search snippet, moving `main`, or open PR does not establish pinned-runtime behavior. Keep failed retrievals and unverified claims explicit.
- Collect repeatable commands with `make run-evidence ARGS='...'` (or `scripts/run_evidence.py` in the appropriate environment). The collector records private logs, configs, source fingerprints and artifacts; GPU/performance collection runs inside an existing allocation. `kind` is a declared layer and collector runtime is not necessarily the child runtime.
- Run `make check-run-evidence ARGS='RUN_DIR'` to check integrity. Command success and accepted behavior are separate gates. Review `acceptance.json` against actual criteria/artifacts; skipped, mocked and missing validation remain visible and cannot satisfy that layer. `--require-command-success` and `--require-accepted` provide distinct automated gates. Record reviewed technical conclusions and limitations in the handoff or relevant technical documentation; collection never certifies a milestone or publishes raw data.

Use a focused branch for each implementation task. Run `make check` for changed local code and records. Add native replay/regression tests when scheduler behavior changes; local evidence tests alone are not native scheduler validation. GPU work requires a scheduler allocation. Keep heavy installation, compilation and inference off login gateways. Release allocations after bounded tasks and never cancel another task's jobs.

Record the exact command, commit, config, workload seed, runtime versions, raw trace location and result interpretation for experiments. Timing from the minimal validation is not a performance benchmark. Keep negative results and failed/unfinished requests.

Update `docs/state.md` when verified technical capabilities or their limitations change. Link claims to code, tests, or experimental records. Do not mark planned functionality complete based on a placeholder, mocked result, or skipped test.

Sync accepted changes through GitHub and update the cluster checkout with `git pull --ff-only`.

## Public data and upstream attribution

Do not commit credentials, SSH material, personal accounts, internal addresses, machine paths, raw cluster logs, environments or model weights. Public evidence must use the explicit exporter and retain numerical values and provenance. Preserve existing public evidence; add a new run directory for new records.

Attribute any adapted implementation to its upstream repository, file/PR and commit. Preserve applicable license notices. Do not create new user-facing chats, recurring work or external messages without the user's request.
