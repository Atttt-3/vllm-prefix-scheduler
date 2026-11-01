# 前缀感知准入：开源实现调研样例

研究日期：2026-11-01，Asia/Hong_Kong。对应任务：`M1-call-chain` 的研究输入；本记录不完成该任务的原生验收。

## 问题与范围

问题：固定 vLLM 0.32.0 下，哪些前缀排序机制可作为参考，哪些假设必须重新验证？本轮选择 SGLang 与两个已有 vLLM PR，提取机制和迁移差异；没有实现调度器、运行 native replay 或测量性能。

结论：缓存命中长度可作为排序信号，但必须分别验证查询副作用、屏障语义与异步兼容性。项目继续采用连续队头窗口、monotonic 等待保护和 `AsyncScheduler`；研究结果不改变这些范围。

## 来源与版本

| 来源 | 固定定位 | 本次证据与边界 |
|---|---|---|
| vLLM 项目底座 | [`5df63a7203cdb9328d824a27b682e77c7b9928be`](https://github.com/vllm-project/vllm/tree/5df63a7203cdb9328d824a27b682e77c7b9928be)，与 `configs/upstream.json` 一致 | 阅读已有源码快照及获取元数据；不是本轮运行证据 |
| SGLang | [`baa9befa381511cd4d56f83980951e5a45fffc6d` 的 schedule_policy.py](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/managers/schedule_policy.py) | 本次打开固定源码，检查 `SchedulePolicy._sort_by_longest_prefix`、`_sort_by_hrrn`、`_determine_active_policy`；原参考版本 `3edea93…` 本次获取失败，未声称两版等价 |
| [vLLM PR #54625](https://github.com/vllm-project/vllm/pull/54625) | API 快照 head `a9317c9a9bd08b71316ce95c0f047aacd56105eb`，base `2a537887de7a103baa49bbfdce93e99d48d19441` | 2026-11-01 04:29:44 UTC 的已有 API 快照为 open、未合并；本次主页仍显示 Open。可读取的 [Files changed](https://github.com/vllm-project/vllm/pull/54625/files) 是缓存页面，未核实其与该 head 完全一致 |
| [vLLM PR #58837](https://github.com/vllm-project/vllm/pull/58837) | API 快照 head `5d243031fdb46640a0be8901b6e72fb848468fcf`，base `e5f3d08d722870a5301560fad0cfa65b7e333900` | 2026-11-01 04:29:43 UTC 的已有 API 快照为 open、未合并；本次主页仍显示 Open。其 [Files changed](https://github.com/vllm-project/vllm/pull/58837/files) 显示较旧的两提交版本，不作为当前 head 的逐行验证 |

检索使用仓库限定词、PR 号和函数名；采用已打开的源码/PR 页面，未采用搜索摘要作为实现证据。两个 PR 的 API 和固定 head 源码本次无法重新获取，进入代码适配前应重新取得并检查完整 diff。

已有 API 快照的 SHA-256：#54625 为 `ba3ab9808a3421cb25e0d9b3922ca09fe5ead50b46c46e5a9fb2503c1406fbe0`；#58837 为 `84796555dc5c6ac59158efe208d7eeeccbdaf6f31a0696ac6c4a2883da534434`。它们定位历史输入，不保证未来 PR 状态。

## 机制卡

**SGLang：信号与等待轴要分别检查。** LPM 使用匹配 token 数降序，并处理暂时降权请求；HRRN 使用已处理 prefill-token 计数构造 aging，平分按请求 ID。大队列会切换策略控制成本。这些行为来自上述固定文件，不能视作本项目的 monotonic 阈值与同分 FIFO。可借鉴排序信号和成本观测；cache/request 对象、等待轴与 tie-break 需自行适配。

**#54625：窗口定义也属于算法。** PR 提议有界缓存感知准入和独立的只读 `KVCacheManager.get_num_cached_tokens`。可读 diff 的 `Scheduler._reorder_waiting_by_cached_prefix` 及 `test_cache_aware_admission_keeps_blocked_and_started_requests_in_place` 固定屏障自身位置，却允许普通请求在屏障两侧交换。这与本项目“遇屏障即停止”不同；迁移前需在固定 head 复核。

**#58837：risk 估计与异步支持是独立问题。** PR 描述在分配前检查 free queue 的缓存暴露风险，再触发窗口排序；其公开配置要求同步调度。可读旧 diff 的 `RiskPerfScheduler.enable_risk_policy` 拒绝 async。风险估计和推广后重查可作为思路；它不提供本项目所需的异步或等待保护保证，作者历史性能结果也不能转为本项目结论。

## 与固定底座对照

在底座 [`kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/kv_cache_manager.py) 中，`get_computed_blocks` 会在满足事件开关等条件时生成缓存复用事件，不能仅凭方法名当作无副作用评分。底座 [`block_pool.py`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/block_pool.py) 的 `get_new_blocks` 从 free queue 取块并调用 `_maybe_evict_cached_block`；这支持继续调查淘汰时序，不证明特定 B/C 场景已发生。

| 适配项 | 项目决策 | 下一项需要的证据 |
|---|---|---|
| 候选范围 | 连续队头窗口，遇 blocked/恢复/部分计算即止 | 真实队列的窗口前后快照 |
| 评分 | 原生 hash/查找语义，保持引用、free queue、统计与事件不变 | 真实对象查询前后快照与事件/统计对照 |
| 等待保护 | monotonic 时间，受保护项按原始到达顺序 | 可控时钟、多个保护项、ID 复用与清理回归 |
| 异步 | 保持 `AsyncScheduler` | 异步占位与原生调用链验证 |
| 性能 | 单卡项目独立配对实验 | 相同到达表、缓存初态、实际 prefill、冷热等待与失败分母 |

## 下一轮验证任务

先给源码调查 agent 一个只读任务：在固定底座中标注 `Scheduler.schedule`、`KVCacheManager.get_computed_blocks` / `allocate_slots`、`BlockPool.get_new_blocks` 的调用关系，并注明 C 的查找发生在 B 分配之前还是之后。独立审查 agent 检查候选是否处于同一准入循环、A 的引用状态、chunked prefill 与准入上限条件。

在已检出固定 commit 的上游仓库根目录，可先执行以下定位命令；它只产生源码定位结果：

```bash
git rev-parse HEAD
rg -n 'get_computed_blocks|allocate_slots|while self.waiting|def schedule' vllm/v1/core/sched/scheduler.py
rg -n 'def get_computed_blocks|emit_cached_block_events|def allocate_slots' vllm/v1/core/kv_cache_manager.py
rg -n 'def get_new_blocks|popleft_n|_maybe_evict_cached_block|def touch' vllm/v1/core/block_pool.py
```

再构造真实 Scheduler fixture，分别记录缓存压力与充足缓存负对照。此时才新增并验收原生重放入口；当前没有可宣称通过的 B/C replay 命令。执行与证据归档遵循[工作流](../workflows.md)和[实验规范](../experiments.md)。

## 许可与个人贡献

[SGLang 固定版本 LICENSE](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/LICENSE) 和 [vLLM 固定版本 LICENSE](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/LICENSE) 均为 Apache-2.0；PR 中可读新增文件也包含相应 SPDX 头。实际引入代码时仍须逐文件核查许可，保留适用 notice，写明来源、commit 与适配 diff。

本轮个人交付是来源复核、机制差异和验证任务；尚无独立调度实现与性能结果。后续可主张的贡献须对应自己的适配、等待保护、原生回归和实测记录。
