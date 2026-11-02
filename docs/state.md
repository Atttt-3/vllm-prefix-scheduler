# 技术状态与验证边界

截至 2026-11-02，M1/M2A/M2B 已通过总站验收；历史 PR 均未合并。M3 已在固定 RTX 3090 / Qwen3-4B BF16 链路运行生产三模式 smoke、完整受控机制与取消场景，最终 GPU 被测为 `403b7838e46ecf17a97cc14de0a1f048938dfbd8`，同提交 M2B compat432 与 M1原入口12回归成功。完整四次分配台账仍缺一项已被清除的历史 Slurm 终态，严格资源与公开导出门槛 blocked，不能标记 M3 全项 accepted。详见[M3交接](research/m3-handoff.md)。没有进入 M4/M5。

## 已验证能力

- 单张 RTX 3090 24 GiB 上，Python 3.12.16、vLLM 0.32.0、Qwen3-4B BF16 完成 7 个请求。重复的 2,230-token 提示命中 2,224 tokens，运行配置与 worker 均确认 APC 和原生 `AsyncScheduler` 开启，见[原生验证记录](../evidence/2026-11-01-rtx3090/validation.json)。
- KV cache 容量为 88,848 tokens（5,553 blocks × 16 tokens），约 12.20 GiB；0.5 秒采样的显存峰值为 20,802 MiB。测量定义及来源见[验证汇总](../evidence/2026-11-01-rtx3090/summary.json)与[来源清单](../evidence/2026-11-01-rtx3090/provenance.json)。
- 从版本化 checkout 复用独立环境、模型和缓存后再次通过原生 7 请求验证；引擎退出后 GPU 占用 1 MiB，作业已释放，见[重放验收记录](../evidence/2026-11-01-checkout-replay/workspace-verification.json)。

- M1 在真实 `AsyncScheduler` / Request / hash / KVCacheManager / BlockPool 上完成四组 × 三次重放：压力 B→C 的 C prefill 派发33，测试 C→B 为1；充足容量两序均为1。A自然完成释放后被B原生分配淘汰的因果 trace、独立审查与证据见 [M1记录](research/m1-native-bc-replay.md)。这只是合成 runner 驱动的 host-side 元数据和 scheduled/dispatched prefill tokens，没有真实 KV tensor 计算。

- M2A 在固定 vLLM 0.32.0 / `5df63a7` 上实际执行 24 组 × 3 次原生矩阵，被测代码为 `1091037dcc2a3a9c31d9692841faebff1f063808`。native/off 压力组 B+C prefill 派发均为82，prefix 自动保留 C 的32-token复用后为50；充足容量三模式均为50。full KV events、统计与 metrics 开启的探测无副作用、APC开关下的 off 等价、19种窗口子例、真实多步 chunk、自然抢占、取消、ID复用和 FIFO 排空均通过。完整来源、失败保留、独立审查及复现入口见 [M2A交接](research/m2a-handoff.md)。这些仍是合成 runner 驱动的 host-side 元数据验证，不代表实际 GPU 工作量或推理加速。

- M2B 最终 Linux 被测 `342f9082e05cd1bebee93a760dd02219eaabae34` 完成636个完整原生case、独立432个兼容case、87个正负子进程和12个M1回归case；每个命名叶子独立重建三次。guarded在500ms使冷B优先于新热H4，热流prompt派发为118（prefix86），如实增加32；不推导GPU性能。原始独审确认保护记录最终归零、原生资源分别排空；状态机、迁移、失败保留与复现见[M2B交接](research/m2b-handoff.md)。

## 下一项技术工作

交回总站审查 M3 实现、原始推理证据及历史资源缺口。四次 GPU 分配预算已用完，不再扩大采集，不自动合并。M2B 历史契约和记录保持原版本；完整资源台账、失败查询与导出拒绝均保留，不能删去未知项来宣称通过。M3 的执行证据与完整验收状态必须区分。正式性能、公平性和其他稳定性扩展仍未开展。

## 结果解释

早期 readiness 证据证明 BF16 推理、APC 复用、异步配置和工作区重放可用；其中16K是上下文上限，实际输入约2.2K tokens，CUDA Graph与torch.compile关闭。M2A另行证明了固定合成场景下scheduled/dispatched tokens的机制差异；两类证据均不支持调度优化的真实模型端到端性能收益结论。性能结论须按[实验规范](experiments.md)另行验证。

M3 新增真实模型闭环证据，但受控暂停积队与被动观察具有干预和开销，不用于推导自然TTFT、公平性或性能收益。派发区间、实际runner处理与可观测输入token数分开记录；它们不等于FLOPs。
