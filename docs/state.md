# 技术状态与验证边界

截至 2026-11-02，已完成固定版本原生环境最小验证、M1 原生 AsyncScheduler B/C 基线，以及 M2A 的真实异步扩展、只读前缀探测和稳定窗口排序。M2A 完成专项验证，交回总项目验收；M1/M2A PR 均保持未合并。等待保护、真实模型调度闭环和正式性能实验尚未实现。

## 已验证能力

- 单张 RTX 3090 24 GiB 上，Python 3.12.16、vLLM 0.32.0、Qwen3-4B BF16 完成 7 个请求。重复的 2,230-token 提示命中 2,224 tokens，运行配置与 worker 均确认 APC 和原生 `AsyncScheduler` 开启，见[原生验证记录](../evidence/2026-11-01-rtx3090/validation.json)。
- KV cache 容量为 88,848 tokens（5,553 blocks × 16 tokens），约 12.20 GiB；0.5 秒采样的显存峰值为 20,802 MiB。测量定义及来源见[验证汇总](../evidence/2026-11-01-rtx3090/summary.json)与[来源清单](../evidence/2026-11-01-rtx3090/provenance.json)。
- 从版本化 checkout 复用独立环境、模型和缓存后再次通过原生 7 请求验证；引擎退出后 GPU 占用 1 MiB，作业已释放，见[重放验收记录](../evidence/2026-11-01-checkout-replay/workspace-verification.json)。

- M1 在真实 `AsyncScheduler` / Request / hash / KVCacheManager / BlockPool 上完成四组 × 三次重放：压力 B→C 的 C prefill 派发33，测试 C→B 为1；充足容量两序均为1。A自然完成释放后被B原生分配淘汰的因果 trace、独立审查与证据见 [M1记录](research/m1-native-bc-replay.md)。这只是合成 runner 驱动的 host-side 元数据和 scheduled/dispatched prefill tokens，没有真实 KV tensor 计算。

- M2A 在固定 vLLM 0.32.0 / `5df63a7` 上实际执行 24 组 × 3 次原生矩阵，被测代码为 `1091037dcc2a3a9c31d9692841faebff1f063808`。native/off 压力组 B+C prefill 派发均为82，prefix 自动保留 C 的32-token复用后为50；充足容量三模式均为50。full KV events、统计与 metrics 开启的探测无副作用、APC开关下的 off 等价、19种窗口子例、真实多步 chunk、自然抢占、取消、ID复用和 FIFO 排空均通过。完整来源、失败保留、独立审查及复现入口见 [M2A交接](research/m2a-handoff.md)。这些仍是合成 runner 驱动的 host-side 元数据验证，不代表实际 GPU 工作量或推理加速。

## 下一项技术工作

M2B 需另行设计 monotonic 等待保护及保护元数据生命周期；当前 `guarded` 明确报未实现，`max_wait_ms` 仅保留配置契约。M2A 完成不等于整个 M2 完成。真实模型闭环与正式性能、公平性实验留给后续阶段。实现边界见[架构说明](architecture.md)，验收标准见[项目方案](project-plan.md)。

## 结果解释

早期 readiness 证据证明 BF16 推理、APC 复用、异步配置和工作区重放可用；其中16K是上下文上限，实际输入约2.2K tokens，CUDA Graph与torch.compile关闭。M2A另行证明了固定合成场景下scheduled/dispatched tokens的机制差异；两类证据均不支持调度优化的真实模型端到端性能收益结论。性能结论须按[实验规范](experiments.md)另行验证。
