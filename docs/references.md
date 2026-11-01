# 上游参考与贡献归属

## 固定底座

- [vLLM v0.32.0](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)，commit `5df63a7203cdb9328d824a27b682e77c7b9928be`。
- [固定版本 scheduler 配置](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/config/scheduler.py)：自定义类解析、异步基类和兼容性说明。
- [固定版本 GPU 安装说明](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/docs/getting_started/installation/gpu.cuda.inc.md)：预编译依赖、Python-only 开发安装和内核缓存。
- [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)，本项目模型 revision `1cfa9a7208912126459214e8b04321603b3df60c`。[配置](https://huggingface.co/Qwen/Qwen3-4B/blob/1cfa9a7208912126459214e8b04321603b3df60c/config.json)中的层数、KV heads 与 head dimension 用于容量估算。

已安装的 scheduler、cache、async scheduler 和 engine core 四个文件与固定 commit 的逐文件哈希一致，见 `evidence/2026-11-01-rtx3090/pinned-source-verification.json`。这不等于逐一核对了所有二进制编译输入；依赖包版本另由 GPU 锁记录。

## 直接机制参考

| 来源 | 参考内容 | 贡献边界 |
|---|---|---|
| [SGLang schedule_policy.py](https://github.com/sgl-project/sglang/blob/3edea93cbc0d857026253df0535a4d89791032a3/python/sglang/srt/managers/schedule_policy.py) | LPM 缓存感知排序及独立 HRRN aging 策略 | 思路和权衡参考，不宣称原创 LPM/aging |
| [vLLM PR 54625](https://github.com/vllm-project/vllm/pull/54625) | 有界等待窗口中的缓存感知准入 | 直接技术参考；后续适配应记录具体来源与差异 |
| [vLLM PR 58837](https://github.com/vllm-project/vllm/pull/58837) | risk-aware admission 与窗口排序 | 对照原生接口与副作用边界，不将整套机制认作个人原创 |

2026-11-01 核查时两个 PR 均为 open，尚未合并；其后状态可能改变。异步支持已有上游工作，不单独作为本项目的新贡献。当前仓库尚未复制或实现这些 PR 的调度代码。

## 其他取舍

[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 可辅助理解 scheduler、block manager 和 model runner；主成果仍在固定版本 vLLM 上验证。LMCache/CacheBlend、CPU offload 和投机解码不进入首版依赖，避免把项目扩展成多个独立主线。

后续引入上游代码时，应在对应文件和变更说明中补充仓库、路径、commit/PR、适配内容和原许可证说明，保留必要 notice。公开机制来源、个人代码和本人实测结果分别陈述。

## 可复用调研记录

[2026-11-01 前缀感知准入调研](research/2026-11-01-prefix-scheduling.md) 是本项目首份机制卡样例：区分固定源码、PR 历史 head、当前页面状态和未核实 diff，并列出连续窗口、等待轴与异步适配差异。本次可读取并固定的 SGLang 对照版本为 `baa9befa381511cd4d56f83980951e5a45fffc6d`；上表原参考版本保留，未假定两版一致。

新调研使用[研究模板](templates/research.md)，按[项目工作流](workflows.md)输出来源、迁移差异及下一项验证；研究记录不替代原生运行和性能验收。
