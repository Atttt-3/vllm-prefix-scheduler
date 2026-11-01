# 开发与贡献

开始前阅读 [当前状态](docs/state.md) 和 [项目方案](docs/project-plan.md)。当前可复现成果为原生 GPU 环境验证；调度器实现是下一阶段工作。

1. 从更新后的 `main` 建立范围明确的 `feat/`、`fix/` 或 `docs/` 分支。
2. 保留上游固定版本。只修改当前任务涉及的代码与文档。
3. 执行 `uv sync --locked --dev` 和 `make check`；调度行为变化需补真实 Scheduler replay 或原生回归证据。
4. 需要 GPU 时先申请资源，记录运行配置、模型 revision、seed 和完成情况，结束后释放作业。
5. 提交说明写清具体行为、参考来源、验证方法和限制。PR 使用仓库模板。

升级 vLLM、替换模型、改变 Graph/backend 或测量定义会改变对照条件，应单独记录决策并重新建立基线。

请勿将 `runs/`、`.local/`、环境、模型或未处理的集群日志加入 Git。公开记录由 `scripts/export_readiness.py` 生成；导出后检查差异与来源清单。
