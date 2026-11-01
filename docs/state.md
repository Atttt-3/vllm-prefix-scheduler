# 技术状态与验证边界

截至 2026-11-01，已完成固定版本原生环境的最小验证；自定义调度器、B/C 淘汰反例、原生调度回归和正式性能实验尚未实现。

## 已验证能力

- 单张 RTX 3090 24 GiB 上，Python 3.12.16、vLLM 0.32.0、Qwen3-4B BF16 完成 7 个请求。重复的 2,230-token 提示命中 2,224 tokens，运行配置与 worker 均确认 APC 和原生 `AsyncScheduler` 开启，见[原生验证记录](../evidence/2026-11-01-rtx3090/validation.json)。
- KV cache 容量为 88,848 tokens（5,553 blocks × 16 tokens），约 12.20 GiB；0.5 秒采样的显存峰值为 20,802 MiB。测量定义及来源见[验证汇总](../evidence/2026-11-01-rtx3090/summary.json)与[来源清单](../evidence/2026-11-01-rtx3090/provenance.json)。
- 从版本化 checkout 复用独立环境、模型和缓存后再次通过原生 7 请求验证；引擎退出后 GPU 占用 1 MiB，作业已释放，见[重放验收记录](../evidence/2026-11-01-checkout-replay/workspace-verification.json)。

## 下一项技术工作

在固定版本真实 Scheduler 中定位等待队列、前缀命中查找、free queue 分配与异步占位的调用顺序，然后重现 B/C 缓存淘汰反例并记录实际 prefill 工作量。实现边界见[架构说明](architecture.md)，验收标准见[项目方案](project-plan.md)。

## 结果解释

现有证据证明 BF16 推理、APC 复用、异步配置和工作区重放可用，不支持调度优化收益结论。16K 是上下文上限，当前输入约 2.2K tokens；CUDA Graph 与 torch.compile 在这次功能验证中关闭。性能结论须按[实验规范](experiments.md)另行验证。
