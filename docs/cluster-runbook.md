# Slurm GPU 环境与重放

## 工作区与现有资源

从本地通过已配置的 SSH 别名连接登录网关。网关用于 Git、文件管理和提交作业；依赖安装、编译、模型下载与推理在分配到的计算节点进行。仓库不保存学校账号、主机地址、SSH 密钥或内部挂载信息。

建议集群 checkout 为 `~/vllm-prefix-scheduler`。如果已有验证通过的环境和模型，可以在 checkout 下建立忽略的符号链接复用，避免复制权重；机器实际位置记录在 `.local/workspace.json`。`git pull` 只更新代码，不管理这些链接目标。

目录约定：`.venv-gpu/` 为 GPU 环境，`models/Qwen3-4B/` 为固定模型，`.cache/` 为下载/编译缓存，`.tmp/` 为临时文件，`runs/` 为完整运行输出。用户盘有配额，安装前检查剩余空间；不把大模型或 wheel 放到容量不足的系统 `/tmp`。

## 新环境准备

先确认所在集群允许的分区名称与资源申请方式。以下资源已经用于本项目的 3090 验证，但不是对其他集群的通用配置：

```bash
cd ~/vllm-prefix-scheduler
srun --partition=q-3090 --gres=gpu:rtx_3090:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G \
  --time=01:00:00 --pty bash

bash scripts/bootstrap_gpu.sh
.venv-gpu/bin/python scripts/download_model.py
```

安装入口使用 Python 3.12.16、固定的 Linux GPU 依赖锁和 CUDA 12.9 对应索引。预编译 FlashInfer 内核也在锁中。安装后检查依赖、CUDA 可见性、BF16 支持和 vLLM 版本，并输出环境记录。

有些集群的 `srun --pty` 会改变工作目录，必要时重新进入 checkout。不要覆盖调度器设置的 `CUDA_VISIBLE_DEVICES`。退出交互 shell 后分配会结束；只取消自己为当前任务创建的作业。

## 最小验证

从 checkout 根目录提交：

```bash
sbatch scripts/validate_gpu.sbatch
```

脚本申请一张 3090、8 CPU、32 GiB 内存，时长上限 20 分钟。配置由 `configs/baseline-3090.json` 读取，不在脚本中另存一套参数。每次创建独立 `runs/readiness/<UTC-time>-<job-id>/`，包含日志、JSON 结果、显存采样和进程退出后的 GPU 记录。最大上下文 16K 是运行上限，本次功能输入约 2.2K tokens。

已经处于获准的计算作业时，可执行 `bash scripts/run_readiness.sh`，无需再申请嵌套作业。单次验证检查实际 worker BF16、APC/async 配置、KV 分配、冷热请求缓存差异和四请求批次，结束后关闭引擎。eager 结果不代替 CUDA Graph 或正式性能验收。

## 记录与公开导出

```bash
# 将已完成的私有运行导出到新的公开证据目录；不要覆盖旧目录
.venv-gpu/bin/python scripts/export_readiness.py \
  --run-dir runs/readiness/<run-id> \
  --output evidence/<new-run-id>
```

导出脚本使用字段白名单保留数值与测量条件，不提交完整集群日志。检查导出 diff、来源哈希和 `make evidence` 后再提交。正式调度实验有不同的 trace 需求，使用其专门的导出流程。

## 已知条件与排查

- 当前驱动实测为 580.178.04，Torch CUDA runtime 为 12.9；`nvidia-smi` 的 CUDA Version 是驱动支持信息。
- 首次 Python 导入和安装在 NFS 用户盘上可能较慢，先检查进程和文件写入，再判断是否阻塞。
- worker RPC 使用具名只读方法；不要为传递 Python 函数开启不安全序列化。
- vLLM 0.32.0 的 `KVCacheConfig` 字段是 `kv_cache_tensors`。诊断代码应依照锁定版本读取。
- 调度器分区的可见时长上限不等于账户可用配额或计费承诺。检查当前分配与排队情况，按任务设置时限。
