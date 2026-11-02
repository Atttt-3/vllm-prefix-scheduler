# 技术状态与验证边界

截至 2026-11-02，已完成固定版本原生环境最小验证、M1 原生 AsyncScheduler B/C 基线，以及 M2A 的真实异步扩展、只读前缀探测和稳定窗口排序。M2A 已通过总项目验收；M1/M2A PR 均保持未合并。本分支已完成 M2B monotonic 初始等待保护与对象绑定清理，完整本地检查、固定 Linux 原生矩阵及独立审查通过，现交回总站验收；PR #6 为未合并草稿。真实模型调度闭环和正式性能实验不在本轮。

## 已验证能力

- 单张 RTX 3090 24 GiB 上，Python 3.12.16、vLLM 0.32.0、Qwen3-4B BF16 完成 7 个请求。重复的 2,230-token 提示命中 2,224 tokens，运行配置与 worker 均确认 APC 和原生 `AsyncScheduler` 开启，见[原生验证记录](../evidence/2026-11-01-rtx3090/validation.json)。
- KV cache 容量为 88,848 tokens（5,553 blocks × 16 tokens），约 12.20 GiB；0.5 秒采样的显存峰值为 20,802 MiB。测量定义及来源见[验证汇总](../evidence/2026-11-01-rtx3090/summary.json)与[来源清单](../evidence/2026-11-01-rtx3090/provenance.json)。
- 从版本化 checkout 复用独立环境、模型和缓存后再次通过原生 7 请求验证；引擎退出后 GPU 占用 1 MiB，作业已释放，见[重放验收记录](../evidence/2026-11-01-checkout-replay/workspace-verification.json)。

- M1 在真实 `AsyncScheduler` / Request / hash / KVCacheManager / BlockPool 上完成四组 × 三次重放：压力 B→C 的 C prefill 派发33，测试 C→B 为1；充足容量两序均为1。A自然完成释放后被B原生分配淘汰的因果 trace、独立审查与证据见 [M1记录](research/m1-native-bc-replay.md)。这只是合成 runner 驱动的 host-side 元数据和 scheduled/dispatched prefill tokens，没有真实 KV tensor 计算。

- M2A 在固定 vLLM 0.32.0 / `5df63a7` 上实际执行 24 组 × 3 次原生矩阵，被测代码为 `1091037dcc2a3a9c31d9692841faebff1f063808`。native/off 压力组 B+C prefill 派发均为82，prefix 自动保留 C 的32-token复用后为50；充足容量三模式均为50。full KV events、统计与 metrics 开启的探测无副作用、APC开关下的 off 等价、19种窗口子例、真实多步 chunk、自然抢占、取消、ID复用和 FIFO 排空均通过。完整来源、失败保留、独立审查及复现入口见 [M2A交接](research/m2a-handoff.md)。这些仍是合成 runner 驱动的 host-side 元数据验证，不代表实际 GPU 工作量或推理加速。

- M2B 最终 Linux 被测 `342f9082e05cd1bebee93a760dd02219eaabae34` 完成636个完整原生case、独立432个兼容case、87个正负子进程和12个M1回归case；每个命名叶子独立重建三次。guarded在500ms使冷B优先于新热H4，热流prompt派发为118（prefix86），如实增加32；不推导GPU性能。原始独审确认保护记录最终归零、原生资源分别排空；状态机、迁移、失败保留与复现见[M2B交接](research/m2b-handoff.md)。

## 下一项技术工作

M2B 的冻结原生矩阵与独立原始 trace 复算已完成，等待总站独立验收，不自动合并任何 PR。当前 guarded 实现合同见 [M2B 设计](research/m2b-wait-guard.md)，本地生产审查见 [独立审查记录](research/m2b-production-review.md)。历史完整 M2A 入口仍属于其已验收版本；当前兼容回归使用 M2B 新入口，不改写旧 guarded 拒绝证据。真实模型闭环与正式性能、公平性实验留给后续阶段。

## 结果解释

早期 readiness 证据证明 BF16 推理、APC 复用、异步配置和工作区重放可用；其中16K是上下文上限，实际输入约2.2K tokens，CUDA Graph与torch.compile关闭。M2A另行证明了固定合成场景下scheduled/dispatched tokens的机制差异；两类证据均不支持调度优化的真实模型端到端性能收益结论。性能结论须按[实验规范](experiments.md)另行验证。
