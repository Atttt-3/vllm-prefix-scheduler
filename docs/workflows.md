# 可复用项目工作流

本页把多 agent 协作、开源实现调研和实验验收接成同一条工作链：**定义任务 → 并行调查 → 统一方案 → 实现 → 留存证据 → 独立审查 → 记录技术结论**。按需要选择其中的环节；普通小改动不增加无关流程。

项目技能位于 [project-workflow](../.agents/skills/project-workflow/SKILL.md)。在支持仓库技能发现的 Codex 会话中，可用 `$project-workflow` 显式调用；也可直接要求遵循本页。项目约定与最新用户指令决定执行边界。

## 开始与任务包

从仓库目录开始，读取 `AGENTS.md`、`docs/state.md`、相关方案和配置，检查现有分支与未提交修改。版本、范围和技术验收以这些文件及本轮任务要求为准。已有工作由原作者保留，不能为“清理工作区”覆盖它。

使用 [任务模板](templates/task.md) 写清目标、输入、允许修改的文件、验收标准和资源预算。为每个任务定义稳定 ID，用于关联分工、运行记录和审查结论；关联技术阶段时注明项目方案中的里程碑。

任务包限定本轮执行范围，交接记录保存已验证结论、未解决条件和下一动作。短任务可直接在分工消息里填写相同字段；需要跨轮交接时再保存任务包。技术能力变化以相应代码、测试和实验记录为依据。

## 多 agent 协作

主 agent 负责范围、依赖、冲突处理与最终验收。独立的源码调查、反例设计、测试或日志分析可并行；先确定接口与状态语义，再拆并行实现。建议根据问题从 2–3 个子 agent 起步，简单任务由一个 agent 完成。

| 任务角色 | 以 M1 为例的调查范围 | 交付 |
|---|---|---|
| 源码调查 | 固定版本的等待队列、前缀查找、分配、淘汰、异步占位 | 调用顺序、源码位置与状态变化 |
| 反例设计 | A 已缓存、B 冷、C 可复用时的必要条件 | 请求时序、缓存/引用条件、最小输入与预期 trace |
| 独立审查 | 从原始任务、源码、实际改动与记录检查假设 | 反证、遗漏条件、验收缺口或有依据的通过结论 |

分工时明确读写边界、输出路径和资源所有者。默认子 agent 与主 agent 共享文件系统；自然语言中的“独立 agent”不代表独立工作副本。读源码可共享 checkout；并行写代码时使用独立 worktree/分支，或明确互不重叠的文件归属。共享接口及技术记录指定单一维护者，跨文件修改先协调归属。

每个子任务返回简短结论、改动文件、运行命令、证据位置及未验证项，使用 [交接模板](templates/handoff.md)。主 agent 必须读取相关差异和证据，处理不同 agent 的分歧；多数意见不替代运行结果。审查者尽量只接收原始要求与原始材料，减少沿用实现者结论的偏差。

同一张 GPU 的测量任务统一排队，由单一资源所有者提交限时作业并确认释放。其他 agent 可同时做轻量开发和分析。子 agent 不创建用户侧新聊天，也不自行发起定时工作、外部消息、合并或发布；这些行为遵循用户授权与现有项目约定。分工本身不增加权限。

## 开源实现调研

先提出本轮要解决的一个问题，再搜索。建议先限时 20–30 分钟，筛选两三个候选后深入最相关的实现；已有足够源码证据时直接研究，不必重复联网。

可使用的检索示例是 `repo:sgl-project/sglang longest prefix schedule policy`、`repo:vllm-project/vllm prefix cache admission eviction`。关键词只是入口：打开一手源码、相关测试和 PR 讨论，固定实际阅读的 commit；搜索摘要、文档中的方法名或 PR 标题不构成实现证明。

每次研究使用 [调研模板](templates/research.md)，保留问题、仓库、不可变链接、函数、阅读日期、当时的 PR 状态、关键机制、测试证据、代价、当前固定版本的迁移差异、许可与归属。`main` 链接可帮助发现实现，最终支撑结论的链接须落到固定 commit。PR 未合并时标为候选参考；访问失败时保留失败范围，不能补写成已核实。

迁移结论要回答三个问题：

1. 上游机制在什么状态下成立，查询或评分有哪些副作用？
2. 与本项目 `AsyncScheduler`、连续队头窗口、只读评分及等待保护有什么差异？
3. 哪个最小原生测试能验证这项借鉴？

已有入口见 [references.md](references.md)。源代码适配需保留来源、commit/PR 和适用许可；研究文档区分上游已有能力、个人实现与本人实测。M1 未验收前不将调研结论写成调度优化结果。

## 实验与验收证据

现有 `make check` 是本地代码、记录和打包检查。下列层级分别验收，不相互代替：

| 层级 | 需要实际执行的内容 | 结果能支持什么 |
|---|---|---|
| `local` | 本地检查、记录契约与工具测试 | 开发工具和材料一致性 |
| `native` | 固定版本真实 Scheduler replay/回归 | 调度机制与状态语义 |
| `gpu` | 已分配计算节点上的真实模型运行 | 自定义扩展集成与资源释放 |
| `performance` | 冻结负载、对照、重复配对和完整 trace | 指定条件下的性能与等待代价 |

`kind` 是执行者声明的层级，采集工具不能根据标签认证真实执行环境。原生 replay 尚待 M1 实现；下述命令只示范本地归档。GPU 工作继续遵守 [集群说明](cluster-runbook.md)，完整实验条件见 [实验规范](experiments.md)。

在仓库根目录执行一次不会启动模型的本地检查并归档：

```bash
make run-evidence ARGS='--task-id workflow-local-check --kind local --output runs/workflow-local-check/first-check --timeout 300 --config configs/upstream.json --config configs/scheduler-plan.json --criterion "本地测试实际执行且通过" -- .venv/bin/python -m pytest -q'

make check-run-evidence ARGS='runs/workflow-local-check/first-check'
```

Makefile 使用工作区内缓存和已有开发环境；已安装依赖时也可用 `.venv/bin/python scripts/run_evidence.py` 与 `.venv/bin/python scripts/check_run_evidence.py`。每次使用新的运行目录；工具拒绝覆盖已有记录。具体参数可通过两个脚本的 `--help` 查询。合成负载另传 `--seed`，用 `--config` 登记负载/到达表等实际输入，用可重复的 `--artifact` 登记请求 trace 或额外结果，用可重复的 `--criterion` 定义本轮待核验标准。超时取适合任务和资源分配的上限，不沿用本地示例来覆盖长实验。

登记配置在启动前归档、命令结束后逐项重算 SHA-256，包括仓库外或被 Git 忽略的配置，以及配置符号链接当前指向的内容。配置变化、删除或读取失败会写入 `config_checks` 和采集错误，令 `config_inputs_stable=false`，阻止成功及验收门槛通过。该检查比较运行边界，不能发现中途修改后又恢复原内容的情况；实验仍须冻结输入、避免并行改写。

Make 入口把 `ARGS` 原样交给执行命令的 shell，避免 make 自身展开 `$5`、`$(...)` 或 `${VAR}`。`ARGS` 仍是一段需要 shell 引用的参数文本：外层引号保护调用 make 的 shell，参数内部的引号保护 recipe shell；字面美元符号和其他 shell 元字符应在后者使用单引号保护。自动化调用可用 `shlex.join(argv)` 构造 `ARGS`，再以参数列表调用 make。直接调用 Python 脚本时只经过调用者的 shell 引用；以 `subprocess.run([...])` 调用脚本则无需 shell 引用。复杂程序也可保存为脚本再作为子命令执行。

采集工具执行分隔符后的参数列表，不通过 shell 拼接命令；它保存实际 argv、项目 commit、执行前后源码指纹、固定上游配置、配置快照、产物哈希、seed、采集进程的运行时信息、stdout/stderr 和退出结果，并生成 `report.md`。`--seed` 只记录元数据，负载命令须实际采用相同 seed；子命令若切换解释器或远程运行，还需归档它自己的真实运行时记录。`command_status` 区分 `succeeded`、`failed`、`timed_out`、`interrupted` 与 `launch_failed`；失败、超时、SIGINT/SIGTERM 中断也会留下记录。

POSIX 系统中，工具会终止本轮命令遗留的同进程组子进程，并把后台遗留写入采集错误；超时或中断也清理该进程组。CLI 在命令运行、清理和归档阶段只记录 SIGINT/SIGTERM 请求，由主流程有序退出；重复信号不打断清理，首次信号决定 `termination_signal` 与退出码。若超时后的清理又收到信号，记录同时保留 `timeout_expired=true` 与 `interrupted` 状态；封存期间收到信号也会重写最终状态并重新绑定验收哈希。直接从 Python 非主线程调用 API 时不能安装进程级信号处理，调用者负责该线程之外的终止策略。

不能依靠此行为释放 Slurm 分配，也不能覆盖自行脱离进程组的服务。SIGKILL、机器断电或采集进程在封存前退出可能留下 `running` 记录；核验器拒绝将其作为完成证据。仓库内的原始记录只允许写入 `runs/`；工具不改写已有运行目录。

`gpu` 与 `performance` 仅在已有 `SLURM_JOB_ID` 的环境中执行。这个环境变量只是资源分配前提，不证明真实硬件、模型或原生路径已经验证。工具不申请 GPU。日志和 manifest 留在忽略的 `runs/`，不会自动公开、提交或改写项目文档。

每轮同时产生 `acceptance.json`，初始 `decision=pending` 并绑定本轮 manifest 哈希。原始验收标准的 ID 与描述保存在 manifest 中，审查时不能删减或替换；标准变化需新建一轮记录。审查者核对原始记录后，填写 reviewer、每项标准的 status、归档 evidence 路径和 note；顶层 limitations 记录结论边界。核验工具检查契约、绑定关系和已登记文件哈希，默认只报告命令与验收状态；可按用途加门槛：

```bash
make check-run-evidence ARGS='runs/workflow-local-check/first-check --require-command-success'
make check-run-evidence ARGS='runs/workflow-local-check/first-check --require-accepted'
```

后者要求命令成功、采集无错误、源码指纹和登记配置在运行边界保持不变、审查者非空、原始非空标准逐项通过且关联归档证据。新采集的 manifest 使用 schema 2；核验器仍可读取旧 schema 1 的完整性与历史审查结果，但缺少配置核验信息的旧记录不能通过新的 `--require-accepted` 门槛，需要按新流程重新采集，不能手动补写为已核验。

存在未提交代码时，commit 不能单独复现该轮源码；两份 source 指纹用于定位变更，分享复现结果前还需保留对应代码版本。**命令退出成功表示执行成功，记录完整表示材料可核对；工具无法判断标准的语义是否被真实满足。** 被测命令如果跳过测试、使用 mock 或只做 readiness，不因退出码为 0 或人工填写 accepted 而升级为原生或性能证据。

失败、超时与未执行项均保留。环境未满足时记录原因和所缺验证，不把跳过写成通过，也不删除失败请求来改善分母。等待时间和性能指标的定义、缓存初态与预热方式须在实验前冻结。

## 收口与交接

协调者按“验收条件 → 原始证据 → 实际观察 → 结论边界”逐项核查，独立审查有分歧时继续最小验证。使用 [交接模板](templates/handoff.md) 留下下一步可执行动作。

核查后将结论、证据与限制写入交接记录；已验证的技术能力或限制变化时，手动更新 `docs/state.md`。运行项目检查并同步已验收改动。只有适用层级的真实证据满足前置条件与标准，才能声明相应技术成果通过验收。工具维护完成不会推进 M1/M2 的调度验收。

公开材料继续经显式导出生成到新的 `evidence/` 目录；私有 argv、完整日志和机器路径不直接进入公开材料或仓库。现有 readiness 导出器只适用于其既定 schema，不能将任意新运行目录交给它并假定已经正确处理。
