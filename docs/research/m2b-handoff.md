# M2B monotonic 等待保护与元数据生命周期交接

2026-11-02：实现、完整本地检查、固定 Linux 原生验收及独立审查完成。交回“vllm项目总站”验收，不自动合并或推进后续阶段。
分支 `feat/m2b-wait-guard`；[草稿 PR #6](https://github.com/Atttt-3/vllm-prefix-scheduler/pull/6)
以 `feat/m2a-prefix-scheduler` 为 base。PR #4、#5、#6 均保持未合并。

## 基线、版本和实现归属

起点为已验收 M2A `a776cd3467ce1697f9dfd6bf8b858f1952a34de2`；其实际 Linux 被测版本仍归属
`1091037dcc2a3a9c31d9692841faebff1f063808`。固定上游为 vLLM 0.32.0 /
`5df63a7203cdb9328d824a27b682e77c7b9928be`，没有吸收其他未验收实现。

| 阶段 | 提交 | 内容 |
| --- | --- | --- |
| A | `d26920a` | 固定源码、状态机、精确数值和异常合同 |
| B | `11e6040dcc07d32e509cdc663dc82e50c67175d0` | 保护策略、生命周期窄包装和轻量单测 |
| C 初始冻结 | `5e6d83c` | 独立原生夹具、完整叶子矩阵、审查器及入口 |
| 审查加固一 | `ea038ea01294d0966ac40a8266f8b8024cfcf06e` | 拒绝不完整并行配置观测；第一次实际采集 |
| 审查加固二 | `fc86a6fe0b0a0dfde80e79ce88269e02db50df42` | 真实唯一重试、ID复用前通知显式核验；第二次实际采集 |
| 最终 Linux 被测 | `342f9082e05cd1bebee93a760dd02219eaabae34` | 完整原始 trace、FIFO 与原生资源残留失败子进程 |

本交接所在后续提交只更新研究记录、技术状态、架构及 scheduler-plan 阶段描述。交付 HEAD 的完整 SHA
由 PR 和总站交接消息记录，可用 `git rev-parse HEAD` 获取；不是一次新的 Linux 运行。
最终被测提交到交付 HEAD 的生产代码、测试、脚本、Makefile、原生矩阵及锁文件保持相同。

生产实现、原生夹具、独立审查器分属不同执行者。协调者实现保护模块及 scheduler/config 窄改；
夹具作者负责真实原生对象、受控时钟和被动 observer；审查器作者不导入生产保护模块作为 oracle。
另一路源码审查者独立检查生产和原始正常生命周期，其本人编写的故障 helper 由其他审查者复核。
固定来源、原始行号、SHA-256、Apache-2.0 归属和完整状态转换表见 [设计与源码依据](m2b-wait-guard.md)；
实现文件哈希与数值反证见 [生产独立审查](m2b-production-review.md)。

## 实际保护合同

默认仍为 `off / 16 / 500`。只有 guarded 创建自有状态，以实际 Request 对象为 key，保存首次成功原生
add 后的 monotonic 纳秒时间和独立递增序号。off/prefix 不创建保护实例字段、不读保护时钟。
构造前与构造后支持检查均启用，包括 APC、async/V2、FCFS、单因果 full-attention group 和所有并行维度为1。

- 每次 schedule 最多读取一次连续候选窗口；先核对记录，window=1/单候选也不能绕过身份检查。
  只有实际排序读一次共同 now；受保护者按原始入队序号，其他候选按当前只读 prefix 长度稳定降序。
- 独立私有时钟默认 monotonic_ns，exact int、允许负 epoch 和同 tick、必须非递减；时钟非法或倒退会锁定。
  等待阈值对整数或 float.as_integer_ratio 使用整数 ceil；`0.1ms` 为 `100001ns`，`10**400` 不溢出、无限化或限幅。
- 时间不从候选窗口开始，失败分配、重复 step、prefix 改序和窗口进出不刷新时间/序号。
  完整 `super().schedule()` 成功返回后才按正 token map 的实际对象清理，首个 chunk 已到边界。
- 抢占按实际状态/计数转换清理，恢复不重新登记；终止在 `_free_request` 的 finally 按实际终止对象清理。
  不包装 `finish_requests`、`_enqueue_waiting_request`、最终更新或异步输出消费。
- 新 ID add、原生 schedule/preempt/free 的异常锁定；首个原生异常原样传播，之后 add/schedule 在父类调用前拒绝。
  可恢复的 M2A 评分/队头事务异常仍可重试。fault 只存错误码，不存异常、traceback 或历史请求。
  shutdown 在 finally 清表并 closed；关闭后不能再 add/schedule，已有输出和终止清理仍可进行。

保护只约束当前连续候选窗口，不能跨 blocked/resumed/部分计算屏障，也不覆盖 running 优先级或解决任意过载。
不承诺全局 TTFT 上界。生产正常清理只处理受影响对象，不每步扫描全部等待队列或元数据表。

## 冻结矩阵与 A–F 验收

最终完整矩阵为 **50 个顶层 scenario/mode 组合、212 个具名叶子 × 3 = 636 cases**。
每个 `(repeat, scenario, mode, variant)` 独立重建输入和测试上下文；实际调度叶子各自新建 scheduler、clock、observer，构造拒绝叶子重新执行对应构造/支持检查。不以三次重复列表排序替代原生状态。
单独 `native-m2b-compat` 使用固定 **432 cases**，输入仍须通过完整冻结配置校验。
独立 M1 回归保持既有四条件 × 三次，即 **12 cases**。完整、兼容、失败封闭及 M1 均在同一最终提交实际运行，无必需 case 跳过、缺失或超时。99 个采集源文件在各运行前后保持 clean 且哈希一致，29 个原生模块与冻结源码哈希一致。完整 trace 共96,843行、1,181,899,245 bytes，SHA-256为 `b5e42d9de051774aec85cda995688944b2f453d7e5e17eb4b875eee2c40f4f89`。

| 标准 | 实际证据与检查 | 主要叶子/产物 |
| --- | --- | --- |
| A 时钟与保护顺序 | 前/等于/超过阈值、零/负零、有限/次正规/最大浮点、巨大整数、负 epoch、同 tick；多个 protected、改序后原序恢复、窗口外年龄；按精确有理数重算 | `clock_thresholds`、`protected_order`、`window_age`；原始 clock/binding/decision |
| B 等待和首次派发 | 两次真实 allocate_slots 失败保留年龄；首个真实 chunk 清理；固定热流量和阈值保护；不使用 sleep 或人工列表替代 | `allocation_retry`、`chunk`、`hot_flow`；native return、dispatch 和清理事件 |
| C 生命周期 | str/list/generator/None/未知/重复 ID；生成器中途异常；自然完成、取消、异常、shutdown、weakref；旧代通知/输出/资源排空后复用 | `metadata_terminal`、`enqueue_identity`、`id_reuse`、`repeat_cycles`、`shutdown` |
| D 原生继承 | 真实 chunk、自然抢占、pending/stale/FIFO；连续窗口和屏障、window=1、同分、后缀、prepend、异常恢复；full events/stats/metrics 同开只读 probe | `window`、`preemption`、`cancellation`、`probe`、`score_eviction`；M1独立回归 |
| E 对照与代价 | 24 组成对 native/off 完整归一化事件流相同；99 组未过期 guarded/prefix 决策和派发一致；压力和热流保护代价分开报告 | `trace-acceptance.json` comparisons；原始命中、准入、dispatch 区间 |
| F 可信度 | 完整 raw trace 正控；缺/多/重复/错标/skip/缺整轮；clock/seq/generation/清理篡改、FIFO错序/重复、KV残留；唯一重试 call→raise；配置/缺依赖/优化解释器失败 | `negative-data/results.json`、mutation plans、逐项 stdout/stderr；87个实际子进程：16正控、71负例 |

审查器按叶子流式重建 ledger，不信 protected 标签或 summary。原生暂存更新和成功返回的派发区分处理；
异常调用必须有对应 raise，拒绝故障/closed 后重试还必须有唯一真实 call→匹配的 WaitGuardError。
配置目录、结果目录和完整原始 trace 三层失败检查分别记录，不把结果目录测试冒充 raw trace 敏感性。
完整 trace 变异使用保留的原始 SHA-256 与 mutation plan，在子进程中逐个写临时副本、调用正式审查器并删除；
原始 trace 不修改。六个完整 trace 变异的子进程记录均确认临时副本已删除；原始 trace 的 SHA 保持一致。

## 四模式对照与保护代价

以下结果须在最终三个独立重复中一致。B/C 表只统计 B+C，A 预热另有33 prompt token；所有 B/C 请求 decode=2，
没有重复 prompt/decode 派发。数值定义为 host-side scheduled/dispatched tokens。

| 场景 | native | off | prefix | guarded 未到阈值 | guarded 已到阈值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 压力5块，B+C prompt | 82 | 82 | 50 | 50 | 82 |
| 充足10块，B+C prompt | 50 | 50 | 50 | 50 | 50 |

固定热流量：总块5、block16、budget64、两个并发上限均1；A33自然完成后，B49在0ns入队，
H1—H4在100/200/300/500ms到达，每轮一次 schedule 和 FIFO consume，max_tokens=1，中途不通用 drain。

| 模式 | 四轮实际准入 | B 被更新热请求越过 | H1/H2/H3/H4 实际命中 | B+H prompt（含A总量） |
| --- | --- | ---: | --- | ---: |
| native/off | B、H1、H2、H3 | 0 | 0/32/32/32 | 85（118） |
| prefix | H1、H2、H3、H4 | 4 | 32/32/32/32 | 53（86） |
| guarded | H1、H2、H3、B | 3 | 32/32/32/0 | 85（118） |

B 的实际对象、0ns入队时间和sequence=1保持不变，500ms恰达到阈值时窗口包含B/H4，guarded先准入B。
H4随后命中丢失，guarded比prefix增加32个prompt派发token；decode及重复工作均0。停止到达后有限步排空。
这明确允许保护牺牲命中，不强迫guarded仍保持82→50，也不将token数量转成GPU性能或全局公平性结论。

## 元数据与原生资源

完整原始事件额外独审确认：414个具化实例的最终原生 map、队列、pending、非null块引用、无重复free queue、request-block映射、in-flight集合和deferred frees排空；其中预期故障的显式测试侧恢复单列。258个实际带guard实例最终closed且记录为0，观测同时记录峰值为3，这不是任意负载的固定上界。8轮完成/取消/复用每次回零，总登记8、首次派发清理4、终止清理4。

54个复用/循环边界的通知与资源均为空；独立重算串联84次真实带finished_ids的零token输出、FIFO消费及后续新代边界。固定源码的has_requests包含has_finished_requests（Scheduler 2808–2830），既有drain已承担这些真实通知派发。新增防御helper本轮全部attempts=0，只验证显式后置条件，没有声称其备用flush分支被执行。

B在step2/3连续实际分配失败后仍为同一对象、0ns、sequence=1，最终派发49 prompt。D的65-token prompt分为16/16/16/16/1，首chunk返回后清除保护；自然抢占R1的computed归零、stale/in-flight32经FIFO消费归零，恢复不重新登记，真实重复prompt为16。

取消后 observer 保留的历史 aborted Request 可以仍有 in-flight/placeholder 计数，不能据此冒称活跃资源未释放，
也不能宣称所有历史对象字段已清零。原生拥有的 map、队列、pending、引用、块映射、in-flight 集合和 deferred frees
分别核对。正常实例在 shutdown 前排空；populated shutdown 先如实记录保护表清零与原生请求/pending/KV仍存活。

预期故障和恢复材料单列：失败原生 add 可能已入队但尚无 ID 绑定，该孤立项的测试侧 detach 明确标为
`native_terminal=False`，不伪造原生成功终止。释放异常只重试同一个当前实际终止对象；不调用旧代危险原生回调。
fault/closed 实例不重新调度来清通知，其残余终止/抢占通知字符串单独记录，不与活跃 Request/KV 混为一谈。

正式支持的是旧代输出、原生通知和资源排空后的同 ID 复用，新对象从零年龄和新序号开始。
旧对象清理隔离还由纯保护层对象测试验证；**旧代 pending 未排空时重名复用不受支持**，不重写原生按ID输出路由。

## 独立审查、修复与历史材料

| 发现 | 修复与复核 |
| --- | --- |
| 完成 native 最终更新并不等于完整 schedule 已返回 | 实现前将清理放在完整父类成功返回后；更新前/更新后 context 退出异常均实测保留记录并锁定 |
| 单候选/window=1 不能略过记录一致性 | 早退前核对对象与记录；真实缺记录反例不读排序时钟且不进入原生 schedule |
| 空 parallel map 可让 all(...) 真空通过 | 精确字段集合和严格整数1；缺字段真实子进程反例 |
| 仅证明故障后零工作，不证明实际拒绝了重试 | 各故障/closed场景真正调用add/schedule；审查唯一call→同step/object的WaitGuardError；删除、重复、错错误类型反例 |
| 复用边界尚缺显式通知后置条件 | 固定原生has_requests包含finished通知，既有drain已实际发出空输出并FIFO消费；新增边界检查及防御helper。全部54个边界通过，helper的补充flush分支本轮attempts=0，未宣称该分支已执行 |
| 目录层失败测试不能替代完整raw trace敏感性 | 增加完整raw正控及6类变异、FIFO错序/重复和最终KV残留；原始source hash与错误原因逐项核验 |

最终无未修复阻断项的结论以 `attempt03-independent-native-review.md/json`、`final-negative-independent-review.md/json`及最终来源/证据核验 为依据。生产实现自 `11e6040` 后未变，后续修复集中在验证可信度。

| 保留运行 | 版本 | 结果和归属 |
| --- | --- | --- |
| attempt-01 | `ea038ea` | 636主矩阵、47子进程和12 M1原生用例成功；独立审查发现重试/复用通知边界尚需加固，acceptance标blocked/superseded |
| attempt-02 | `fc86a6f` | 636主矩阵、77子进程和12 M1原生用例成功；完整raw矩阵/FIFO/资源残留子进程覆盖尚需补齐，acceptance标blocked/superseded |
| attempt-03 | `342f908` | 636主矩阵及trace、87正负子进程、12 M1用例与独立审查均成功；逐项acceptance通过 |
| compat-01 | `342f908` | 通过真实Make入口独立执行432 cases和trace审查；21组native/off、99组未到阈值guarded/prefix对照通过 |

旧运行的原始trace、命令结果和manifest保持原义，未拿新审查器覆盖旧命令结果，也未拼接不同版本验收。
Mac `local-check-01/03` 的打包步骤因沙箱网络阻止hatchling索引而失败，其日志保留；允许网络后的独立全量重跑成功。
早期工具路径/依赖准备失败在preflight setup notes注明哪些只有当时工具输出摘要，未伪造原始日志。
最终被测提交的完整 `make check` 为 **364 passed、3个明确native opt-in skip**；lint/format、shell语法、历史证据检查、wheel/sdist均通过。
这些skip不计作native；交付文档提交后另做完整本地检查，不冒称一次新的Linux矩阵。构建产物另行核对，不包含runs、.local、.cache、.venv、.tools或.tmp。

## 私有证据索引

以下均以新worktree的 `runs/m2b-wait-guard/` 为根；不提交原始runtime、连接信息、内部地址或资源路径。

| 材料 | 路径 |
| --- | --- |
| 最终schema2主采集与A–F acceptance | `remote/attempt-03/` |
| 来源/输入/clock plans/seed/case全集/trace/summary/独立审查 | `remote/attempt-03-data/` |
| 完整失败封闭采集与逐项子进程产物 | `remote/attempt-03-negative/`、`remote/attempt-03-negative-data/` |
| M1独立回归及原始trace | `remote/attempt-03-m1/`、`remote/attempt-03-m1-data/` |
| 独立兼容入口 | `remote/compat-01/`、`remote/compat-01-data/` |
| 资源前后、作业输出和释放查询 | 对应`attempt-03-resources/`、`compat-01-resources/` |
| 最终逐case行号、来源对照、人工独审和交付核验 | `attempt03-independent-native-review.*`、`final-negative-independent-review.*`、`final-trace-line-index.json`、`final-source-verification.json`、`resource-release-final.json`、`final-review/` |
| 被测提交完整本地检查 | `local-check-05/` |
| 交付文档提交后的完整检查 | `local-check-06/` |
| 旧采集/失败/工作区保护 | `attempt-01/02*`、`local-check-01`至`04`、`preflight/` |

完整性、命令成功、行为验收是三重门槛。schema2工具只核对哈希与acceptance结构，行为由原始trace独立重算和交叉审查建立。

## 可复制入口和契约迁移

历史 `make native-m2a` / `native-m2a-trace-check` 的完整入口适用于 M2A 已验收提交及其实际被测提交，
其中 guarded“未实现”拒绝属于旧版本；不要求M2B继续满足这一过时语义。历史配置、审查器、trace、失败和验收均未改写。
当前配置单测只最小更新 guarded 可解析；M2B用受支持guarded正例和APC-off等真实配置负例代替旧拒绝。
新 `native-m2b-compat` 覆盖当前全部适用兼容子集，M1四条件三次另行运行。

Linux在独立checkout切到最终被测提交，继续使用既有固定解释器和模型**配置**目录。
`NATIVE_PYTHON`、`MODEL_CONFIG_DIR`按原工作区忽略配置设置，不在公开文档披露实际值。
仅单资源所有者在已分配节点执行，单次≤20分钟；目录必须新建，不能覆盖既有采集。

```bash
export PYTHONHASHSEED=42 VLLM_USE_V2_MODEL_RUNNER=1 HF_HUB_OFFLINE=1
unset PYTHONOPTIMIZE
M2B_RUN=runs/m2b-wait-guard/replay-01
M2B_DATA=runs/m2b-wait-guard/replay-01-data
"$NATIVE_PYTHON" scripts/run_evidence.py --task-id M2B --kind native \
  --output "$M2B_RUN" --timeout 900 --seed 42 --config configs/native-m2b.json \
  --artifact "$M2B_DATA/runtime.json" --artifact "$M2B_DATA/inputs.json" \
  --artifact "$M2B_DATA/clock-plans.json" --artifact "$M2B_DATA/case-index.json" \
  --artifact "$M2B_DATA/trace.jsonl" --artifact "$M2B_DATA/summary.json" \
  --artifact "$M2B_DATA/trace-acceptance.json" \
  --criterion 'A monotonic clocks and original sequence ordering' \
  --criterion 'B native allocation retries first dispatch and continuous hot arrivals' \
  --criterion 'C object identity terminal cleanup fault shutdown and drained ID reuse' \
  --criterion 'D native async chunk preemption probe and bounded window inheritance' \
  --criterion 'E native off prefix equivalence and honest guarded dispatch cost' \
  --criterion 'F complete unique independently rebuilt leaf matrix and raw trace audit' \
  -- make native-m2b NATIVE_PYTHON="$NATIVE_PYTHON" \
  ARGS="--config configs/native-m2b.json --model-config '$MODEL_CONFIG_DIR' --output '$M2B_DATA'"

make native-m2b-trace-check NATIVE_PYTHON="$NATIVE_PYTHON" \
  ARGS="--suite full --config configs/native-m2b.json --trace '$M2B_DATA/trace.jsonl'"

M2B_NEG=runs/m2b-wait-guard/replay-01-negative-data
"$NATIVE_PYTHON" scripts/run_evidence.py --task-id M2B-negative --kind native \
  --output runs/m2b-wait-guard/replay-01-negative --timeout 420 --seed 42 \
  --config configs/native-m2b.json --artifact "$M2B_NEG/results.json" \
  --criterion 'Real malformed inputs and raw trace tampering fail for their specific reasons' \
  -- "$NATIVE_PYTHON" scripts/check_m2b_fail_closed.py --config configs/native-m2b.json \
  --trace "$M2B_DATA/trace.jsonl" --output "$M2B_NEG"

# 在下一次串行分配中运行独立兼容入口；同样包在schema2 collector中。
make native-m2b-compat NATIVE_PYTHON="$NATIVE_PYTHON" \
  ARGS="--config configs/native-m2b.json --model-config '$MODEL_CONFIG_DIR' --output runs/m2b-wait-guard/replay-compat-data"
make native-m2b-trace-check NATIVE_PYTHON="$NATIVE_PYTHON" \
  ARGS="--suite compat --config configs/native-m2b.json --trace runs/m2b-wait-guard/replay-compat-data/trace.jsonl"

# M1保持原入口与冻结配置，使用另一个新输出目录。
make native-replay NATIVE_PYTHON="$NATIVE_PYTHON" \
  ARGS="--config configs/native-bc-replay.json --model-config '$MODEL_CONFIG_DIR' --output runs/m2b-wait-guard/replay-m1-data"
make native-trace-check NATIVE_PYTHON="$NATIVE_PYTHON" \
  ARGS="--config configs/native-bc-replay.json --trace runs/m2b-wait-guard/replay-m1-data/trace.jsonl"

.venv/bin/python scripts/check_run_evidence.py "$M2B_RUN" --require-command-success --require-accepted
```

完整实际命令及采集参数在各manifest和私有job脚本。采集命令成功不会自动将acceptance改为accepted。
Mac运行 `make check`；不安装GPU依赖、不同步开发锁到原生环境。

## 工作区、资源和未支持范围

四次计算分配严格串行、每次上限20分钟。最终完整分配实际10分38秒、兼容分配3分26秒，均COMPLETED、ExitCode=0:0，GPU前后均1 MiB；最终有效活动队列查询无本任务四个作业。早期第二次作业的scontrol记录在后续查询时已过期，sacct不可用；该查询限制保留，另以退出记录、GPU结束记录和成功的活动队列查询确认释放，不伪造终态。

一次归档传输观察到了仍运行的兼容目录，其本地暂存副本隔离保留于pre-completion-mirror，未用于验收；兼容结束后按attempt过滤重新采集，正式schema2哈希核验通过。此过程未修改远端原始采集。

原工作区仍为 `feat/project-workflows` / `e71dfc11135d1f86b4cf28400ab603e268ea1a57`，仅有用户原本的
`docs/workflows.md` 修改；最终 SHA-256 保持
`c3dca495872357a584ecc2464dc92dafa5e4467d16237f4c11dfdf9ab096bb82`。
M1仍为 `fc23bc380dc82f5b57922b26d072470397184267`、M2A仍为`a776cd3467ce1697f9dfd6bf8b858f1952a34de2`，均干净。
历史实现、原始证据、取证工具和锁文件不变。没有覆盖共享Linux checkout、解释器环境或模型资源。

本轮只证明固定vLLM、合成ModelRunnerOutput、单层FullAttentionSpec几何下的host-side元数据及派发语义。
没有加载模型、tokenizer、真实KV tensor；没有TTFT、吞吐或GPU性能实验。特殊请求继续交由原生处理；
并行、offload、连接器、投机、LoRA、多模态等未核查全局配置仍拒绝。故障/closed实例须新建，不宣称原生事务回滚。
旧pending未排空的同ID并发代际、全局等待上界、真实模型闭环与M3/M5均不属于本交付。
