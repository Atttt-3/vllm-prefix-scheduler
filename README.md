# vLLM 共享前缀调度优化与等待保护

[![checks](https://github.com/Atttt-3/vllm-prefix-scheduler/actions/workflows/checks.yml/badge.svg)](https://github.com/Atttt-3/vllm-prefix-scheduler/actions/workflows/checks.yml)

在固定版本 vLLM 中研究小窗口缓存感知排序：让等待队列中能够复用现有前缀的请求先执行，减少缓存淘汰引起的重复 prefill，并通过等待保护限制冷请求持续被插队的代价。

**当前阶段：RTX 3090 上的原生运行环境和模型验证已完成；自定义调度器、B/C 淘汰反例与优化实验待实现。** 当前结果证明 BF16 推理、APC 和异步调度可用，不代表调度优化收益。

## 已完成的原生验证

2026-11-01，单张 RTX 3090 24 GiB，Python 3.12.16，vLLM 0.32.0，PyTorch 2.13.0+cu129，Qwen3-4B BF16。

| 项目 | 观测结果 |
|---|---|
| 请求 | 7 个完成，包含四请求批次 |
| APC | 重复的 2,230-token 提示命中 2,224 tokens；首次请求命中 0 |
| 异步调度 | 运行配置与 worker 均确认开启，解析为原生 `AsyncScheduler` |
| KV cache | 88,848 tokens；5,553 blocks × 16 tokens；约 12.20 GiB |
| 显存 | 初始化后 20,618 MiB；0.5 秒采样峰值 20,802 MiB，约 20.31 GiB |
| 运行配置 | 16K 上下文上限，最多 4 条活跃序列，chunked prefill 2,048，eager 模式 |

原始数值保存在[结构化验证记录](evidence/2026-11-01-rtx3090/validation.json)，导出范围与原始文件哈希见[来源清单](evidence/2026-11-01-rtx3090/provenance.json)。集群账号、地址、绝对路径、作业号与 GPU UUID 已从公开记录移除。CUDA Graph 与 torch.compile 在这次功能验证中关闭，正式性能实验需另行冻结配置。

## 要研究的问题

先缓存长前缀 A。冷请求 B 先进入等待队列，稍后的请求 C 可以复用 A。缓存压力下，先准入 B 可能淘汰 A，让 C 重算前缀；先执行 C 则可能先兑现这次复用。持续优先热请求又会增加冷请求的等待，因此需要同时观察性能与等待代价。

计划通过现有 `scheduler_cls` 接入 `AsyncScheduler` 子类，在每个 step 开始时最多重排一次队头连续候选窗口，再调用原生调度。原生 KV 分配、抢占、异步占位和模型执行继续由 vLLM 负责。

| 计划模式 | 行为 |
|---|---|
| `off` | 使用原生顺序 |
| `prefix` | 按当前可复用前缀 token 数降序，同分维持 FIFO |
| `guarded` | 窗口内等待超阈值的请求先按原始到达顺序排列，其余按前缀命中排序 |

默认窗口 16，等待阈值 500 ms。等待保护约束候选窗口中的排序行为，不承诺过载或显存不足时的绝对 TTFT。首版范围为单 GPU、单 full-attention group、FCFS 基础队列和普通文本请求。

## 开始工作

开发依赖安装到 `.venv`，GPU 运行依赖安装到 `.venv-gpu`，避免开发环境同步清除 GPU 依赖。Mac 上只安装开发依赖；原生 vLLM 安装与推理在获得 Slurm 分配的 Linux NVIDIA 节点执行。

```bash
git clone https://github.com/Atttt-3/vllm-prefix-scheduler.git
cd vllm-prefix-scheduler
uv sync --locked --dev
make check
```

`make check` 执行代码检查、记录一致性测试、公开证据核查和打包。它不运行模型，也不证明自定义调度器已经完成。开发依赖锁定在 `uv.lock`，已验证的 Linux GPU 依赖锁定在 `requirements/gpu-cu129.lock`。

在有 Slurm 的 GPU 集群，从仓库根目录提交：

```bash
# 首次环境准备，在获准的计算作业内执行
bash scripts/bootstrap_gpu.sh
.venv-gpu/bin/python scripts/download_model.py

# 后续最小验证；先确认分区与资源参数适用于所在集群
sbatch scripts/validate_gpu.sbatch
```

完整步骤见[集群运行说明](docs/cluster-runbook.md)。模型、环境、下载缓存和未经导出的运行记录由忽略规则排除，公开证据通过显式导出生成。

## 工作区导航

| 位置 | 用途 |
|---|---|
| [`docs/state.md`](docs/state.md) | 当前完成状态、下一步和最近验证 |
| [`docs/project-plan.md`](docs/project-plan.md) | 问题定义、范围、里程碑和验收标准 |
| [`docs/architecture.md`](docs/architecture.md) | 原生调用链、计划接口与实现边界 |
| [`docs/experiments.md`](docs/experiments.md) | 工作负载、对照、指标和运行记录规范 |
| [`docs/development.md`](docs/development.md) | 分支、检查和日常工作流程 |
| [`docs/references.md`](docs/references.md) | 上游固定版本、参考实现和归属 |
| `src/vllm_prefix_scheduler/` | 可安装的项目包；当前提供证据一致性检查 |
| `scripts/` | 环境准备、模型下载、只读诊断和 GPU 验证 |
| `configs/` | 上游版本、模型 revision、运行基线与待实现调度配置 |
| `tests/` | 当前的记录一致性检查；原生调度回归将在实现阶段加入 |
| `evidence/` | 经过身份信息处理、可公开核查的实测记录 |
| `runs/` | 每次运行的完整记录，默认不进入 Git |

## 验收方向

工程验收要求真实引擎扩展可运行、原生回归通过、资源清理正确、异步模式保持、实验可重放。优化结论以 TTFT p95 为主，同时报告实际 prefill 与重算 tokens、冷热请求分布、吞吐、TPOT、排序成本和等待代价。关键正式场景每轮至少 500 请求、五组交替配对运行，以运行作为统计重复单位；收益需超过运行波动。

## 参考与许可

窗口排序和 LPM 思路参考 SGLang 与 vLLM 的公开实现，归属见[参考资料](docs/references.md)。本项目的后续贡献聚焦固定版本适配、等待保护、原生回归和可复现实验；不将已有机制或直接调用缓存功能认作原创算法。

本仓库自有代码采用 [Apache-2.0](LICENSE)。模型权重、上游软件和未来可能引入的第三方代码分别遵守各自许可证，见 [NOTICE](NOTICE)。
