# 日常开发流程

## 本地环境

使用 Python 3.12.16 和 uv 0.13.6。`uv.lock` 只管理 `.venv` 中的轻量开发依赖；Linux GPU 依赖在独立的 `.venv-gpu` 中，见 GPU 锁文件。不要将开发依赖锁同步到 GPU 环境。首次执行：

```bash
uv sync --locked --dev
make check
```

工作区若已准备 `.tools/bin/uv`，`make` 会自动使用它；也可显式执行 `.tools/bin/uv sync --locked --dev`。本地命令不会安装 vLLM、PyTorch 或模型。

```bash
make format    # 自动整理 Python 格式与可修复检查
make check     # 检查、测试、证据核对、打包
make evidence  # 只核查已归档记录
```

## 开始一项任务

读取 `docs/state.md` 的下一步和验收门槛，确认当前工作区无未处理变更，更新 `main` 后建立一个范围明确的分支。每个任务围绕一个可观测行为，例如“真实 Scheduler 中重现 B/C 淘汰”，不要同时扩张到新的引擎或另一条技术路线。

本地源代码是开发入口，GitHub 保存版本，集群 checkout 通过 `git pull --ff-only` 更新。模型和 GPU 环境位于忽略目录；同步代码不复制大文件或 SSH 配置。机器连接与已有资源的位置可以保存到忽略的 `.local/workspace.json`。

修改后先跑当前必要检查。调度行为变化需 native replay 和原生回归；GPU 集成变化需在获准资源上验证。不要为了增加测试数量编写只重复实现的断言，也不要用跳过的 GPU 测试冒充通过。

## 提交与状态

提交说明写清问题、行为、上游依据、验证和限制。默认流程为分支加 PR；初始仓库建立后，后续功能通过可审阅的增量推进。CI 自动运行本地检查，不自动申请 GPU、下载模型或执行性能实验。

发生里程碑变化、测量定义变化或阻塞时更新 `docs/state.md`。变更运行配置时同步配置文件与运行说明，保留旧 evidence。不要在没有真实引擎结果时更新“已完成”的性能或原生回归状态。

## 原生环境

安装、模型下载与运行步骤见 [集群说明](cluster-runbook.md)。固定版本核心源码可以在计算节点安装目录中阅读，或单独 clone 到忽略的 `upstream/` 后检出指定 commit。不要将完整上游仓库、编译产物或模型加入本项目历史。

适配上游代码时在文件和文档中标明来源与 commit。官方 scheduler 接口仍有版本兼容风险，版本升级应单独迁移、重新回归，不追随 `main` 自动更新。
