# 当前状态与下一步

更新日期：2026-11-01，Asia/Hong_Kong。

## 已完成

- 在单张 RTX 3090 上建立 Python 3.12.16 独立环境，安装 vLLM 0.32.0 / PyTorch 2.13.0+cu129。
- 固定 Qwen3-4B revision，真实 BF16 参数验证通过。
- 原生 APC 和 async 配置在 worker 中确认；7 个请求完成，重复提示命中 2,224/2,230 tokens。
- 单 full-attention group、5,553 blocks、88,848-token KV 容量，约 12.20 GiB。
- 运行参数为 16K 最大上下文、最多 4 条活跃序列、2,048 batched tokens 和 eager；显存采样峰值 20,802 MiB。
- 原始环境、模型与完整记录保留在私有工作区，公开版结构化证据保留数值与来源哈希。

- 本地独立 Git 仓库、公开 GitHub 仓库、开发依赖锁、GPU 依赖锁、贡献与工作约定已建立。
- 本地记录与导出检查、脚本语法和打包检查通过；[首次 GitHub CI](https://github.com/Atttt-3/vllm-prefix-scheduler/actions/runs/41302896213) 全部成功。
- 开发环境使用 `.venv`，GPU 脚本使用 `.venv-gpu`，两套依赖独立。

## 工作副本状态

本地与 [GitHub](https://github.com/Atttt-3/vllm-prefix-scheduler) 已同步。尝试建立集群新 checkout 时，登录网关在已有会话、全新会话和 IPv4 重试中均连接超时；新 checkout 尚未创建。此前已验证的环境、模型和原始记录仍保留在原工作目录。

连接恢复后，先按忽略的 `.local/workspace.json` 中的机器位置 clone 同一仓库，以 `.venv-gpu`、`models` 和 `.cache` 链接复用既有资源，再从新 checkout 重放最小验证。仓库化脚本已完成静态检查，新的路径布局尚待集群复跑。不要重复下载模型，也不要把开发锁同步到已有 GPU 环境。

## 下一项工作

**M1：在锁定版本的真实 Scheduler replay 中重现 B/C 缓存淘汰反例。**

需要先定位等待队列、原生命中查找、free queue 分配与异步占位的实际调用顺序。构造 A 已缓存、B 冷、C 可命中的资源状态，证明原生 FCFS 下 B 的分配会破坏 C 的复用机会。保留可重放输入、队列/缓存快照及实际 prefill 工作量。

验收时需要一条原生重放命令和失败/成功路径说明。若该反例无法在所支持范围成立，先记录原因并调整机制定义，不能绕过反例直接进入性能报告。

## 尚未完成

自定义 `scheduler_cls`、`off/prefix/guarded`、无副作用评分、等待保护、原生调度回归、取消/抢占清理、自定义调度器 GPU 闭环、正式到达回放、TTFT p95 与公平性实验均待实现。当前 `make check` 检查的是工作区和归档证据，不是上述功能的验收。

## 后续更新约定

完成状态必须链接到代码、原生测试或运行记录。新运行创建新目录，更新配置时说明对对照条件的影响。已完成与待实现各维护一次，避免在 README、工作记录和 Issue 中出现互相冲突的进度。
