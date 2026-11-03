# 技术状态与验证边界

## M4执行中（2026-11-03）

已获授权从M3交付12705a8开始独立稳定性/局部CPU成本专项，设计与冻结候选矩阵见[设计](research/m4-design.md)。实现与首次真实校准已完成：校准冻结c58e6fb的native105叶/1200轮及CPU原始样本通过独审；GPU五场景机制成立，但原始事件因launcher标识覆盖被严格拒绝。压力参数已有真实校准依据，已修记录与诊断问题，正在冻结两角色正式全量验证；尚未宣称M4验收通过。M3旧formal-02终态与原历史gate限制保持。M4预算独立3×20分钟，不进入M5。

截至 2026-11-03，M1/M2A/M2B 已通过总站验收；相关 PR 均未合并。M3 新批次在 `77844c447e7917264bb6f20fdb41e7056a7181b4` 完成真实GPU完整矩阵、同提交compat432/M1原入口12、完整审查、原30控件与新增完整正控/28负控、专用白名单导出及独立A–F审查，当前待总站验收。原四次历史及formal-02未知终态保持不变，原全历史门槛仍blocked；不能把新批次通过冒称历史缺口恢复。累计使用6/8次分配，余2次未申请。详见[M3新批次交接](research/m3-campaign-handoff.md)和[原历史交接](research/m3-handoff.md)。该段为M3历史状态；当前M4已单独获授权，未进入M5。

## 已验证能力

- 单张 RTX 3090 24 GiB 上，Python 3.12.16、vLLM 0.32.0、Qwen3-4B BF16 完成 7 个请求。重复的 2,230-token 提示命中 2,224 tokens，运行配置与 worker 均确认 APC 和原生 `AsyncScheduler` 开启，见[原生验证记录](../evidence/2026-11-01-rtx3090/validation.json)。
- KV cache 容量为 88,848 tokens（5,553 blocks × 16 tokens），约 12.20 GiB；0.5 秒采样的显存峰值为 20,802 MiB。测量定义及来源见[验证汇总](../evidence/2026-11-01-rtx3090/summary.json)与[来源清单](../evidence/2026-11-01-rtx3090/provenance.json)。
- 从版本化 checkout 复用独立环境、模型和缓存后再次通过原生 7 请求验证；引擎退出后 GPU 占用 1 MiB，作业已释放，见[重放验收记录](../evidence/2026-11-01-checkout-replay/workspace-verification.json)。

- M1 在真实 `AsyncScheduler` / Request / hash / KVCacheManager / BlockPool 上完成四组 × 三次重放：压力 B→C 的 C prefill 派发33，测试 C→B 为1；充足容量两序均为1。A自然完成释放后被B原生分配淘汰的因果 trace、独立审查与证据见 [M1记录](research/m1-native-bc-replay.md)。这只是合成 runner 驱动的 host-side 元数据和 scheduled/dispatched prefill tokens，没有真实 KV tensor 计算。

- M2A 在固定 vLLM 0.32.0 / `5df63a7` 上实际执行 24 组 × 3 次原生矩阵，被测代码为 `1091037dcc2a3a9c31d9692841faebff1f063808`。native/off 压力组 B+C prefill 派发均为82，prefix 自动保留 C 的32-token复用后为50；充足容量三模式均为50。full KV events、统计与 metrics 开启的探测无副作用、APC开关下的 off 等价、19种窗口子例、真实多步 chunk、自然抢占、取消、ID复用和 FIFO 排空均通过。完整来源、失败保留、独立审查及复现入口见 [M2A交接](research/m2a-handoff.md)。这些仍是合成 runner 驱动的 host-side 元数据验证，不代表实际 GPU 工作量或推理加速。

- M2B 最终 Linux 被测 `342f9082e05cd1bebee93a760dd02219eaabae34` 完成636个完整原生case、独立432个兼容case、87个正负子进程和12个M1回归case；每个命名叶子独立重建三次。guarded在500ms使冷B优先于新热H4，热流prompt派发为118（prefix86），如实增加32；不推导GPU性能。原始独审确认保护记录最终归零、原生资源分别排空；状态机、迁移、失败保留与复现见[M2B交接](research/m2b-handoff.md)。

## 下一项技术工作

M3新批次技术闭环已完成并交总站待验收。保留所有失败：首次新冻结 `9dd9de4a9b71149f8102ad26eb8e9b3da48a8ded` 的组合审查错误地要求原M1替代声明为空，修复为精确版本化声明后在 `77844c447e7917264bb6f20fdb41e7056a7181b4` 全矩阵重跑。两次新分配分别09:16与09:36、COMPLETED/0:0、单卡20分钟以内、严格串行，独立监视保存提交前ready、终态、全队列和进程退出。原生产策略、模型、推理参数和历史记录未改；不自动合并，不开始M4/M5。

## 结果解释

早期 readiness 证据证明 BF16 推理、APC 复用、异步配置和工作区重放可用；其中16K是上下文上限，实际输入约2.2K tokens，CUDA Graph与torch.compile关闭。M2A另行证明了固定合成场景下scheduled/dispatched tokens的机制差异；两类证据均不支持调度优化的真实模型端到端性能收益结论。性能结论须按[实验规范](experiments.md)另行验证。

M3 新增真实模型闭环证据，但受控暂停积队与被动观察具有干预和开销，不用于推导自然TTFT、公平性或性能收益。派发区间、实际runner处理与可观测输入token数分开记录；它们不等于FLOPs。
