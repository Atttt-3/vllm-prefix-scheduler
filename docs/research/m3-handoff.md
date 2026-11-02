# M3 真实模型集成：草稿交接，完整验收仍阻断

任务 `M3-real-model-integration`，2026-11-02。真实模型执行、机制与原生回归已完成；
**完整四次分配台账存在一项当前未能核实的历史 Slurm 终态，严格资源门槛及公开导出未通过。**
本交接不是 M3 全项 accepted 声明。保留所有失败，交回总站判断，不合并 PR，不进入 M4/M5。

## 分支、基线和被测版本

分支 `feat/m3-real-model-integration`，堆叠 base 为 `feat/m2b-wait-guard`。
起点精确为 `a1e57efc6a72811c53a8d259bdebcafd9b9c8c2e`；M2B 实际 Linux 被测仍为
`342f9082e05cd1bebee93a760dd02219eaabae34`，两者仅五个文档/阶段文件不同，未冒称另一次 Linux 运行。

| 运行 | 完整提交 | 结果和边界 |
| --- | --- | --- |
| calibration-01 | `a139049cf47af5c89c683483ef663aa7853932e7` | 权重/真实 KV 加载成功；新入口误把 tokenizer 返回对象的字段名当 token IDs，生成前失败；资源释放有证据 |
| formal-01 | `d04092c34908a800850e8360f94265fb81109cfe` | 真实全矩阵成功；随后审查器、输入绑定和导出接口加固，保留为被最终重放替代的记录 |
| formal-02 | `403b7838e46ecf17a97cc14de0a1f048938dfbd8` | 全 GPU 矩阵、M2B 432-case compat、M1 原入口 12-case 均成功；即时 Slurm 记录被清除且历史记账禁用，不能补写终态 |
| formal-03 | `403b7838e46ecf17a97cc14de0a1f048938dfbd8` | 同一冻结提交完整 GPU 重放；同步监视捕获真实 COMPLETED/0:0，进程、显存与全队列释放有证据 |

交付后续提交仅更新本文及项目文档；完整交付 SHA、逐文件差异和保护证明在私有交接及 PR 中记录。
不能把文档提交称为新的 GPU 被测版本。生产 scheduler/config/probe/wait_guard、历史配置、上游及锁均未修改。

## 实现与已实际运行的范围

[实现前合同](m3-design.md)固定了入口、源码锚、文件归属、时钟域、观测开销和预算。
新入口使用 AsyncLLM、真实 tokenizer/权重和原生异步输出链；生产类 off/prefix/guarded 与原生
AsyncScheduler 各使用全新引擎进程完成 smoke。诊断类与生产 smoke 分开记录。

功能 profile 只把活跃序列配置从4改为1，实际 scheduler.max_num_active_reqs 另行读取。
每个机制 case 独立 reset、真实预热、排空、原生 pause/keep、B/C 实际 add 确认和 resume。
三个重复均观察到 prefix 的 C 优先；guarded 未过期一致，真实500ms保护使较早冷B优先。
真实客户端 abort、首个成功正派发后的对象绑定清理、原生通知消费、请求排空与 shutdown 分别观测。

观察层只包原调用一次，保留返回/异常；旁路关联 scheduler output、V2 execute/sample、原生
异步 materialization 与 FIFO update。零 token 通知不要求 GPU forward。没有合成 runner 输出、
fake clock、人工 waiting 改序、保护记录写入或原生输出替代，也未增加逐 batch GPU synchronize。
无 observer smoke 的 scheduler 选择由固定构造源码、实际配置和加载模块来源支持；直接 self/MRO
证据属于诊断实例，worker 的 get_scheduler_cls 不冒充 EngineCore 对象反射。

## A–F 当前结论

| 项 | 结论 |
| --- | --- |
| A 固定来源与真实实例 | 功能证据通过：固定环境/上游内容、模型与 tokenizer 文件身份、三模式生产 smoke、诊断实际对象、BF16/APC/async/V2均核验 |
| B 真实执行闭环 | 功能证据通过：全部预期请求与真实返回、批次、FIFO消费、明确结束关联成立 |
| C 非平凡策略动作 | 功能证据通过：prefix、未过期 guarded 和精确500ms保护均有三个独立初态重复 |
| D 生命周期与资源 | 最终运行本身排空/退出/Slurm释放成立；完整历史台账仍因 formal-02 的当前未能核实的 Slurm 终态而 **blocked** |
| E 对照与口径 | 功能证据通过：native/off逐请求greedy对照相同；派发与runner输入计数分列，不推导加速、公平性或FLOPs |
| F 证据可信度 | 严格拒绝缺口有效；旧完整真实 trace 的30个正负子进程和本地契约测试通过。最终同版本完整正控因四次台账缺口未通过，公开 accepted 导出 **未完成** |

最后一次完整审查在已检查真实矩阵、来源、批次、请求、保护和对照之后，于原生资源总闸拒绝。
八个引擎分区审查成功但均为 `partial / accepted=false`，不替代完整验收。
完整四次分配全部列在台账中，没有删去不利项、伪造终态或放宽冻结合同。四次上限已用完，停止 GPU 工作。
有效的全队列查询没有本任务活跃作业；这与历史终态不可复核是两个不同事实。

## 回归、独审与公开边界

- 最终被测提交完整 `make check`：406 passed、3个明确 native opt-in skip；lint、shell语法、历史证据及wheel/sdist成功。skip不计native/GPU。
- 同提交固定 Linux：当前 M2B compat432、M1原入口12均实际执行，原始trace再次独审；生产策略未变，不借用旧提交运行冒充此次回归。
- 独审发现并修复了异步输出包装引用环、实际生产模块来源缺口、tokenizer对象误用、batch/输入绑定不足、导出契约漂移，以及负控变异器遇到null case的崩溃；原始失败保留。
- 专用白名单导出器拒绝本次 blocked audit，未创建公开 evidence 目录。本文是工程状态交接，不是经验证导出的数值证据。原始日志、内部路径、主机/账号和权重均未提交。

私有证据均在本 worktree 忽略的 `runs/m3-real-model/`：

| 材料 | 相对路径 |
| --- | --- |
| 最终GPU schema2、来源、输入、原始跨进程trace | `remote-formal-03/runs/m3-real-model/formal-03*` |
| 同提交原生回归及原始审查 | `remote-formal-02/runs/m3-real-model/formal-02-regressions/` |
| 完整四次预算与资源，保留unknown | `final-resources-complete-ledger.json` |
| 最终trace逐case行号与数值定义 | `final-execution-facts.json`、`final-engine-audits/` |
| 完整总闸、正控和公开导出拒绝 | `final-full-ledger-gate/`、`final-fail-closed-gate/`、`final-public-export-guard/` |
| 独立审查和总站完整私有交接 | `independent-final-review.md/json`、`final-handoff.md` |
| 正负控及校准/旧尝试 | `formal-01-fail-closed-current*`、`remote-calibration/`、`remote-formal-01/` |
| 本地检查与打包保护 | `local-check-frozen-02/`、`package-privacy-check.json` |

## 可复制入口（本轮不再申请资源）

在独立 Linux checkout 切到被测提交；环境、模型及cache位置仍由原工作区的私有workspace配置读取。
不安装依赖、不同步开发锁、不覆盖CUDA_VISIBLE_DEVICES；一个资源所有者在已有单卡分配内运行：

```bash
# NATIVE_PYTHON / MODEL_CONFIG_DIR / M3_CACHE 使用已有固定资源。
export M3_RUN=runs/m3-real-model/new-replay
bash scripts/run_m3.sh --repeats 3 --engine-timeout 240

export M3_REGRESSION_RUN=runs/m3-real-model/new-regressions
bash scripts/run_m3_regressions.sh

# 完整resources须包含所有分配的实际终态和最终有效全队列查询。
python scripts/check_m3_trace.py --config configs/m3-real-model.json \
  --inputs RUN-inputs.json --trace-dir RUN-data/trace.jsonl \
  --resources complete-resource-ledger.json --output audit.json
python scripts/check_m3_fail_closed.py --config configs/m3-real-model.json \
  --inputs RUN-inputs.json --trace-dir RUN-data/trace.jsonl \
  --resources complete-resource-ledger.json --output NEW-negative-dir
```

采集命令、clean提交、源码前后指纹和原始文件哈希在schema2 manifest中；完整性、命令成功与行为验收分别判断。
Mac只运行本地检查。原工作区及M1/M2A/M2B保持原分支/修改；用户文档SHA-256仍为
`c3dca495872357a584ecc2464dc92dafa5e4467d16237f4c11dfdf9ab096bb82`。

本轮仍不支持TTFT p95改善、吞吐收益、自然排队公平性、真实淘汰/压力矩阵、全局等待上界或其他模型/并行路径。
受控暂停与等待不是自然流量；同ID旧pending未排空的并发代际仍不支持。等待总站验收与对证据缺口的决定。
