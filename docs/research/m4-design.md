# M4：调度稳定性与有限窗口 CPU 成本

任务 `M4-scheduler-stability`，2026-11-03。人类授权本专项实际执行，精确起点
`12705a8e392eaa90c019de8e347ee31134ab3ca4`。M3 replacement campaign 实际被测
`77844c447e7917264bb6f20fdb41e7056a7181b4`，交付差异仅文档/白名单数据；不声称交付另跑 GPU。
旧 formal-02 仍为 `UNVERIFIED_PURGED`，原全历史 gate 仍 blocked；M4 无此终态例外。

## 来源、分工与提交阶段

项目约定、project-workflow、M2B/M3设计与交接为输入。先核对本地/远端 M3 SHA、PR #4–#7、
全部既有 worktree 未提交修改。原工作区分支和用户 workflows 原件受保护；M1–M3源文件不改。
仅新 worktree `feat/m4-scheduler-stability` 工作，草稿PR base为M3分支，不合并、不进入M5。
能力库当前无active，周期PAUSED；不调用草稿，不将设备结论作为通用能力。

| 阶段 | 必须先完成的交付 |
|---|---|
| A | 本设计、固定源码调查、矩阵/接口、独立预算；压力profile是待真实校准候选 |
| B | 最小新增观察与fixture、独立raw审查/正负控、原生组合与长循环、成本；独审后冻结源码 |
| C | 同冻结完整native/GPU矩阵、全部终态与预算、失败封闭、白名单导出与草稿PR |

主执行者拥有配置、M4 launcher/controller、成本脚本、预算台账、Makefile和文档，独占GPU。
观察agent拥有 `m4_observer.py`、`tests/native/m4_*.py`、`test_m4_observer.py`、
`scripts/m4_engine_fixture.py`；独立契约agent拥有 `m4_contract.py`、`scripts/check_m4*.py`、
`test_m4_contract.py`、`m4_trace_fixture.py`，不得导入生产策略作oracle。另设只读审查。
共享接口先协调，agent不独立提交、申请GPU或修改其他文件。

## 固定源码与支持边界

vLLM0.32.0/upstream `5df63a7203cdb9328d824a27b682e77c7b9928be`，Python3.12.16，
Qwen3-4B BF16/model与tokenizer revision `1cfa9a7208912126459214e8b04321603b3df60c`，
RTX3090 24GiB；APC/AsyncScheduler/V2/eager/既有attention后端保留。
37份已安装上游源码按M2B/M3固定SHA重新读取校验。后续新调查文件亦绑定内容哈希。

| 源码位置（固定版本） | M4设计依据 |
|---|---|
| 本项目 scheduler.py:241–301 | freshness检查实际对象全状态；最多读取连续window，遇屏障停止 |
| scheduler.py:151–181、217–239 | 完整父schedule成功后定向删guard；终止/关闭按既有fault合同 |
| wait_guard.py:66–118 | 实际对象绑定，原年龄/seq不刷新；无历史扫描，关闭不是native排空 |
| 上游 scheduler.py:644–671、715–800 | 长prefill分片/运行优先；实际allocate失败触发原生抢占 |
| 上游 kv_cache_manager.py:522–540 | full-ISL准入只作容量检查；不提前分配整个prompt |
| 上游 kv_cache_utils.py:2605–2659 | override派生真实tensor预算，扣null块后检查max_model_len |
| 上游 async_scheduler.py:26–96 | 输出placeholder与stale交付语义；不能丢旧FIFO输出 |
| M3 observer.py:169–247、events.py | 可复用真实batch/execute/sample/materialize/consume桥；补抢占通知与代际 |

正式仅支持旧代输出、通知和资源完全排空后的同ID新对象。旧pending未排空时同ID并发代际不支持，
不改原生ID路由。computed归零不清除preempt/stale/inflight屏障。手改字段/故障注入只证明具名native分支。
无已证实生产缺陷时不改策略；发现反例先保留失败，然后最小修复并扩大同提交回归至M2B636和M3完整20case。

## 状态转换与原始接口

成功入队 → 实际对象/代际绑定 → fresh WAITING连续窗口 → 首次完整成功正派发清guard →
partial/running → 正常完成或真实pending取消 → FIFO（含零token终止/抢占通知）排空 → native资源排空 →
同ID新对象新年龄/新seq。抢占保持同代，记录num_preemptions、stale和重复prompt区间；恢复不再是fresh。

复用M3事件存储和真实runner桥，不复用其按ID汇总的guard/区间oracle。M4附加事件以实际weakref对象注册，
单调registration token + generation作为身份；裸id只作私有辅助。每次dispatch保存当时对象身份，不能消费时
按当前ID映射回填。observer只存活跃对象/批次；历史记录写入raw，不保留强引用历史ledger。

最小事件接口：`request_bound`；`request_state`；`preempt_start/success`；`cancel_start/success`；
`generation_drained`；`round_begin/end`；`probe_before/after`；`cpu_sample`。
保留M3原事件和完整batch_id链。新增事件包含layer、case、seed/repeat、mode、进程内seq和时钟域；
取消/抢占入口包含当时真实request状态、pending batch IDs/interval身份；schedule包含preempted和finished通知。
trace schema与collector schema2分开。具体字段以M4独立contract固定实现为准，变更需同步fixture/负控。

## 冻结矩阵与实际前提

`configs/m4-stability.json`逐项枚举GPU60case：四模式native/off/prefix/guarded × 三次重复 × 五场景。
每case先完全排空并原生reset缓存建立独立空闲初态；复用引擎不冒称进程重启。native/off是相同输入成对基线。

| GPU场景 | 原始事件必须证明 |
|---|---|
| chunk_complete | 同代至少两个正prefill区间、中间partial状态、最终正常完成 |
| capacity_preempt_resume | 真实36层KV容量压力；原生抢占计数增加；恢复重复prompt区间且最终完成 |
| waiting_cancel_reuse | pause/keep入队确认，实际WAITING且未正派发时client abort；排空后同ID复用并完成 |
| pending_cancel_reuse | 长prompt真实执行后client abort；终止入口RUNNING/partial且对应pending非零；排空后复用并完成 |
| full_events_probe | 同一次EngineCore只读probe前后完整纯存储快照相等，stats/metrics/full事件开启 |

取消优先正常client abort；观察已有文件事件不阻塞EngineCore，不增GPU同步或替换时钟。请求抢跑完成、
取消入口pending为零均为无效case，不转标签。需要受控原生取消fixture时先另冻结并明确干预，不能暗放observer。
每种完成/取消之后继续提交新请求证明可用。压力运行禁止活请求reset、手改状态、替换allocate、fake runner和单层KV。

原生模式/seed固定为四模式×[17,29,43]；每组>=100轮，每轮包含完成、等待取消、pending取消、安全同ID复用，
共至少1200轮；有限steps/超时。组合覆盖blocked、partial、preempt/resume、placeholder/inflight/stale、窗口外等待、
同分FIFO；prefix/guarded具名prepend失败恢复，guarded具名fault/closed/shutdown。注入项明确层级。
轮末核对真实请求/全部队列/通知/pending/非null引用/free queue/guard；测试强引用ledger清除后GC+weakref检查。
保留单调sequence作为标量，不要求其数值恒定；不把APC hash残留当泄漏。

## 压力校准与独立预算

候选 `capacity1025-v1`：max_model_len16384、真实num_gpu_blocks_override1025、block16、active2、batch2048、
long_prefill_threshold1024且adaptive=false；两条不同8192-token prompt，输出32。1024非null块恰容纳两prompt，
decode增长应触发原生容量抢占。full-ISL准入与零spec lookahead源码允许两条初态同入，但只有实测可确认。
实际KV tensor层数/块数/bytes重新读取；所有模式使用同一个校准后冻结profile。候选不冒称已触发。

独立上限3次、每次单3090最长20分钟、总申请<=60分钟；校准/失败/超时/重试全部计数。不转入M3余量。
先一次校准和原生验证，再按结果使用剩余预算，预留最终同冻结重复。每次提交前watcher ready，立即绑定receipt/
job/freeze/config；包括非零sbatch尝试。所有成功提交必须有权威终态/exit/start/end/limit、GPU前中后、子进程退出、
有效全队列查询和watcher退出；无终态阻断。服务器时间和controller monotonic分域。
VPN中断后不盲重提；只查询/清理本任务。无嵌套申请、不在网关推理/安装、不改CUDA_VISIBLE_DEVICES或共享环境。

## 副作用与CPU局部成本

诊断probe纯存储快照包括refcounts、hash映射、双链free queue、请求字段、事件缓冲、stats/metrics；快照本身不得调用
get_computed_blocks、allocate、touch、drain/take事件方法。probe仅一次，保留full事件开启的正向活动证明。
M2B native快照与真实EngineCore证据分层。诊断IO/快照不进入成本测量。

成本主window16，对照8/32；窗口外长度0/128/4096；off、全冷、同热分、冷热交错，5重复×100样本，20预热。
分别测取窗、评分、排序（扣嵌套评分时间并报告误差）、窗口提交/失败恢复；计调用数、候选/queue访问和hash查找。
恢复是具名native注入分支，耗时不可当正常成功路径。off/native另测实际schedule基线，不宣称off零开销。
原生对象/原生queue为操作对象；包装计数与计时本身空调用成本单列，全部样本先内存记录后写盘；无JSONL observer。
报告机器/时钟/采样单位/p50/p95及重复分布，仅算法局部CPU成本，非TTFT/吞吐/M5，也不要求加速。

## 独立验收与交付

A来源/实际实例/完整可重复矩阵；B生命周期/代际/guard与长循环有界；C真实chunk/容量抢占/pending取消/FIFO通知；
D连续窗口/屏障/prepend及full事件probe无副作用；E局部成本/有界访问/off-native基线；F失败封闭/全部尝试终态/独审/导出。
M4-lifecycle/barriers/async-state/score-side-effects/overhead分别判定，任何必需项缺失不宣称完整验收。

`make check`的native/GPU opt-in skip不计实际验证。生产不变时同冻结实际compat432+M1原12；生产改动时完整636+
M3完整20模型闭环。失败留存且不跨冻结拼case。raw独审不信passed汇总，真实子进程正负控包括空/缺/重复/错标、
不足重复、未chunk/preempt/pending、早复用、probe变化、资源残留、漏作业/缺终态和优化解释器；缺依赖/GPU明确拒绝。

新原件位于 `runs/m4-scheduler-stability/<run-id>/`；复用schema2采集器，原始pending acceptance不事后重绑。
白名单导出绑定原件hash，排除连接/账号/内网/路径/PID/job身份/原始文本/权重。最终交接提供A–F行号、成本表、
基线/实际被测/交付SHA及差异、失败、复放、预算释放、工作区保护与全部PR未合并证明，停在总站待验收。
