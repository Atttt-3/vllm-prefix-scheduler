# M2B 任务包：monotonic 等待保护与元数据生命周期

记录日期：2026-11-02（Asia/Hong_Kong）。本文件定义执行归属与验收，不是验收结果。
技术合同与固定源码依据见 [M2B 设计](m2b-wait-guard.md)。

## 输入、范围与工作区

- 任务 ID：`M2B-wait-guard`；目标是 guarded 的 monotonic 初始等待保护、对象身份安全清理和可信原生验收。
- 已验收起点：M2A 交付 `a776cd3467ce1697f9dfd6bf8b858f1952a34de2`；M2A 实际 Linux 被测 `1091037dcc2a3a9c31d9692841faebff1f063808`。
- 固定底座：vLLM 0.32.0 / `5df63a7203cdb9328d824a27b682e77c7b9928be`；复用既有原生环境，不改锁文件。
- 独立分支：`feat/m2b-wait-guard`；独立工作区名称 `vllm-m2b-wait-guard`；stacked PR base 为 `feat/m2a-prefix-scheduler`。
- 原工作区用户拥有的 workflow 修改不得改动、stash、提交或还原；SHA-256 必须仍为 `c3dca495872357a584ecc2464dc92dafa5e4467d16237f4c11dfdf9ab096bb82`。
- M1/M2A 原工作区、分支、历史证据、取证工具和依赖锁只读。若远端基线变化，先审查差异，不自动吸收未验收实现。
- 本轮层级为 `local` 与 `native`；不加载真实模型、不计算真实 KV tensor、不做 M3/M5 或性能实验，不自动合并任何 PR。

## 分工与阶段提交

协调者负责最终接口、工作区保护、提交/PR、串行 Linux 作业与逐项交接。子任务仅写明确拥有的文件，
共享接口修改由协调者整合；独立审查者不以实现者摘要代替源码和原始证据。

| 子任务 | 文件归属 | 输出与限制 |
| --- | --- | --- |
| 固定源码调查与设计 | `docs/research/m2b-wait-guard.md`、本文件 | 完整状态机、时钟/数值/故障合同、不可变来源、迁移边界；不写生产，不提交，不申请资源 |
| 保护实现与轻量测试 | 新保护模块、`scheduler.py`/`config.py` 窄改，以及对应 `tests/` | 先统一私有接口与状态机；off/prefix 无保护时钟与请求记录；生产不保留无限历史或异常对象 |
| 原生夹具与重放 | `tests/native/m2b_harness.py`、`m2b_snapshot.py`、`test_m2b_native.py`、`scripts/replay_m2b.py` | 原生对象与有限 FIFO synthetic-output 驱动；按冻结 clock/seed/case 执行；不申请并行计算资源 |
| 独立合同与 trace 审查 | `configs/native-m2b.json`、`m2b_contract.py`、新 M2B 审查器、失败封闭脚本及合同测试 | 从原始事件重算，不导入生产保护判断作为 oracle；检查 tamper/完整矩阵/重复/配置失败 |
| 协调整合与交接 | 新入口 Make target、阶段状态与架构/AGENTS 的过时描述、最终 research 交接 | 不重写历史证据；保留实际失败和未执行项；本地与 Linux 原生结果分开报告 |

独立提交顺序：

1. 固定源码、状态机、时钟/配置合同与任务包。
2. 保护排序、身份绑定、生命周期和本地合同测试。
3. 原生矩阵、独立审查与修复；冻结实际被测代码并采集；最后交付证据/状态/PR 说明。

阶段 3 若修复生产或验证代码，重新确定被测提交并完整重跑适用矩阵。交付提交与实际被测提交分别记录。
历史 M2A guarded 拒绝不再是 M2B 当前契约；新入口必须保留适用的 off/prefix 兼容覆盖及真实配置负例。

## 验收清单与证据

| ID | 必须观察的行为 | 原始证据要求 |
| --- | --- | --- |
| A：时钟/排序 | 阈值前、等于、超过；零、有限浮点、超大整数；同 tick、多 protected、曾被改序、窗口外年龄 | 原始入队 clock、独立序号、对象代际、每次共同 now、窗口前后；审查重算阈值与排序 |
| B：等待/派发 | 原生资源不足/重复失败/重复 step 不刷新；最终正 token/首 chunk 清理；持续热流量中冷请求在窗口达到阈值后得到保护 | 实际查找/分配结果、最终返回派发、pending FIFO、逐请求 token 区间与越过次数；无 sleep 替代 |
| C：生命周期 | 自然完成、等待取消、批量/未知/重复 ID、str/list/generator/None、异常、shutdown、旧代排空后 ID 复用 | 逐对象清理原因与代际；自有表存活重算；与原生 map/队列/pending/引用/资源独立核对 |
| D：原生继承 | 真实 chunk、自然抢占、pending/stale、取消/FIFO；连续窗口/屏障/window=1/同分/后缀/prepend/恢复；probe 无副作用 | 原生状态及源码/hash；stats/metrics/full events 开启前后快照；占位与 cache_blocks 保留证据 |
| E：对照/代价 | native/off/prefix 兼容；未到阈值 guarded 与 prefix 决策相同；保护生效后的命中或派发代价如实记录 | 实际准入、命中、prompt/decode/重复派发、被越过；不能把 host-side tokens 转成 GPU 性能结论 |
| F：可信度 | 完整非空唯一矩阵、至少三次独立重建；缺/多/重复/错标/不足重复/优化解释器/缺依赖/无效配置均失败封闭 | 冻结 case 清单、实际 subprocess 返回/错误、完整 trace、独立审查及篡改敏感性结果 |
| G：工作区/资源/交接 | 原工作区保护；无共享环境覆盖；作业释放；所有 PR 未合并；交总项目验收 | 前后 Git/hash、起始/被测/交付提交、PR base/state、资源终态与私有失败材料索引 |

异常验收明确区分：profiling context 退出失败、add 部分插入后失败、clock 错类型/回退、preempt 状态转换前后失败、
free 异常、generator 物化失败、populated shutdown。保护表清零不能替代原生排空，也不能把故障注入当自然抢占证据。

## 执行约束与交回

仅协调者从原工作区忽略配置读取连接与存储位置，私有账号、内部地址和资源路径不进入公共文档。
原生使用既有固定 Linux 环境；单卡任务串行，每次分配最多 20 分钟，命令超时及最大 step 有界，结束确认释放。
环境缺失、VPN/连接失败、依赖不匹配或原生 skip 都保留为未完成/失败，不能提升为通过；不得在 Mac 安装 GPU 依赖。

所有采集使用已有 schema 2 工具和新的私有 `runs/` 目录，不修改历史 runs/trace 或工具语义。
采集起始/被测/交付 commit、实际源码和环境哈希、冻结输入/clock/seed/case、原始 trace、manifest、逐项 acceptance、
独立 review、失败/超时/跳过/未执行项及释放记录。完整 `make check` 与实际 Linux native 结果分别列出。

交接需给出可复制命令与逐项证据映射、三模式加 native 对照和保护代价、元数据及原生资源结果、独立审查发现/修复/复核、
当前不支持范围、原工作区保护结果，以及所有 PR 未合并状态。总项目对话负责最终验收，不自动推进后续阶段。
