# 技术状态与验证边界

截至 2026-11-02，已完成固定版本原生环境最小验证，以及 M1 原生 AsyncScheduler B/C 淘汰反例与派发工作基线。自定义调度器、完整原生调度回归和正式性能实验尚未实现。

## 已验证能力

- 单张 RTX 3090 24 GiB 上，Python 3.12.16、vLLM 0.32.0、Qwen3-4B BF16 完成 7 个请求。重复的 2,230-token 提示命中 2,224 tokens，运行配置与 worker 均确认 APC 和原生 `AsyncScheduler` 开启，见[原生验证记录](../evidence/2026-11-01-rtx3090/validation.json)。
- KV cache 容量为 88,848 tokens（5,553 blocks × 16 tokens），约 12.20 GiB；0.5 秒采样的显存峰值为 20,802 MiB。测量定义及来源见[验证汇总](../evidence/2026-11-01-rtx3090/summary.json)与[来源清单](../evidence/2026-11-01-rtx3090/provenance.json)。
- 从版本化 checkout 复用独立环境、模型和缓存后再次通过原生 7 请求验证；引擎退出后 GPU 占用 1 MiB，作业已释放，见[重放验收记录](../evidence/2026-11-01-checkout-replay/workspace-verification.json)。

- M1 在真实 `AsyncScheduler` / Request / hash / KVCacheManager / BlockPool 上完成四组 × 三次重放：压力 B→C 的 C prefill 派发33，测试 C→B 为1；充足容量两序均为1。A自然完成释放后被B原生分配淘汰的因果 trace、独立审查与证据见 [M1记录](research/m1-native-bc-replay.md)。这只是合成 runner 驱动的 host-side 元数据和 scheduled/dispatched prefill tokens，没有真实 KV tensor 计算。

## 下一项技术工作

M1 调用链与反例已验收；下一阶段另行设计只读评分、连续等待窗口及等待保护，并建立相应原生回归。本轮不进入 M2。实现边界见[架构说明](architecture.md)，验收标准见[项目方案](project-plan.md)。

## 结果解释

现有证据证明 BF16 推理、APC 复用、异步配置和工作区重放可用，不支持调度优化收益结论。16K 是上下文上限，当前输入约 2.2K tokens；CUDA Graph 与 torch.compile 在这次功能验证中关闭。性能结论须按[实验规范](experiments.md)另行验证。
