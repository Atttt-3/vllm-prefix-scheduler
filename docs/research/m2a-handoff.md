# M2A 原生 AsyncScheduler 专项交接

2026-11-02：实现及专项验证完成，交回“vllm项目总站”验收。分支为
`feat/m2a-prefix-scheduler`，堆叠 PR base 为 `feat/m1-native-bc-replay`。
交付不合并任何 PR。M2A 不代表整个 M2 完成。

## 基线、提交与来源

起点为已验收 M1 `fc23bc380dc82f5b57922b26d072470397184267`；PR #4 在交付前复核仍为
OPEN、未合并且 head 未变。M1 原生机制证据继续归属 `0288eb4`，不把后续入口及门槛修复
描述为新的 M1 Linux 矩阵。固定上游为 vLLM 0.32.0 /
`5df63a7203cdb9328d824a27b682e77c7b9928be`。

| 阶段 | 提交 | 内容 |
| --- | --- | --- |
| A | `8bba47424ed43cd65a487a0cba063d21eac53a16` | 真实 AsyncScheduler 子类、严格配置、off 直通 |
| B | `4d9f4bf71a05255d7781ab7c743ae47d7badaf99` | 原生 hash、只返回长度的只读查找 |
| C | `803397270e3b10f4dd01ca6492248b32b1dac2eb` | 连续新鲜窗口、稳定排序及异常恢复 |
| D 冻结一 | `4ceb5a0f8fb861956612c54c234543c384f04527` | 独立工厂、快照、72-case 矩阵、trace 审查器和入口 |
| D 修复一 | `47041dae1197c7bb890f07cbc9efb53d9451f86d` | 负配置夹具真实生效、post-probe 证据隔离和比较核验 |
| D 最终被测 | `1091037dcc2a3a9c31d9692841faebff1f063808` | 独立验证目标拒绝输入、错误及顺序；guarded 审查修复 |

本交接所在的后续提交只更新研究记录、技术状态、架构及 scheduler-plan 的分阶段状态。
交付 HEAD 用 `git rev-parse HEAD` 获取；它不是一次新的 Linux 原生运行。
从上述被测提交到交付 HEAD，生产代码、测试、脚本和 `configs/native-m2a.json` 均不变。

正式运行先固定提交再采集；运行前后 83 个采集源文件和配置一致，无 dirty。
29 个原生模块的实际哈希全部等于冻结配置，Python 为3.12.16。
实际实例为 `vllm_prefix_scheduler.scheduler.PrefixAsyncScheduler`，MRO 为：

```text
PrefixAsyncScheduler → vllm.v1.core.sched.async_scheduler.AsyncScheduler
→ vllm.v1.core.sched.scheduler.Scheduler → SchedulerInterface → ABC → object
```

原生 `get_scheduler_cls()` 实际解析、构造上述实例；生效 async=true、V2=1，各并行维度为1。
类来源 SHA-256 为 `ba6a5042a3567f05b151290024c4aef287f855ccb6f108ca0930bc53cac352e8`。
其他自定义模块及原生来源完整清单在私有 runtime 和冻结配置中。
编码前的逐函数、行号、版本及机制调查见[设计记录](m2a-prefix-scheduler.md)。

## 最终行为证据

正式成功运行 `attempt-03`：24 个 scenario/mode 组合各独立重建3次，共72 cases。
主入口和独立完整 trace 审查成功；没有缺项、跳过或超时。以下数值每次重复一致。

| 容量 | native | off | prefix |
| --- | --- | --- | --- |
| 压力：5块（含 null） | B49+C33 = **82** | B49+C33 = **82** | B49+C1 = **50** |
| 充足：10块（含 null） | B49+C1 = **50** | B49+C1 = **50** | B49+C1 = **50** |

所有原始入队均为 B→C；prefix 由类内策略自动将 C 提到 B 前，保留32-token复用机会。
没有 M1 夹具人工改序。完整 trace 记录预热状态、淘汰、原生再次命中、准入及每次派发区间；
审查器独立重算，不仅检查 summary。指标仅为 **scheduled/dispatched prefill tokens**。

| 条目 | 结果及证据内容 | 最终 trace 首次重复行范围 |
| --- | --- | --- |
| A 接入/off | 实际类/MRO、异步配置；21组 native/off 成对事件流等价，覆盖APC开关；off窗口/探测次数0 | 1–1569，后续生命周期对照 |
| B 探测 | stats、full KV events、metrics/sample=1 同时启用；7个边界/skip-read分数为0、16、32、0、16、0、0；同对象完整快照精确相等 | 1570–1846 |
| C 自动prefix | 压力82/82/50，充足50/50/50；同容量跨模式/重复预热状态一致 | 1–1247 |
| D 窗口 | 每次重复19个独立子例：同分/全冷、window=1/上限、队头/中间屏障、后缀、真实prepend、评分/排序/提交异常恢复；逐步计数 | 1847–4151 |
| E 生命周期 | 真正chunk、自然抢占、FIFO、取消、完成、ID复用、评分后缓存失效 | 4152–7275 |
| F 失败封闭 | 每次重复33项整体配置拒绝，核验真实生效值、类型、目标错误及顺序；完整固定case合同；另有7个真实失败子进程 | 7276–7343及独立negative材料 |

其余两次重复及全部72个case边界在私有 `final-trace-line-index.json`。完整 trace 共22,029条事件，
SHA-256 为 `816af7e8cebb400490c07cd2fe6274ad649b36bcf8ab579cc61eec64d80f4661`。

- 探测之后的正常原生路径保留1 request / 33 queries / 32 hits和对应的 BlockStored事件。
  该证明发生在full-event/touch/stats敏感性正控制之前。快照不调用drain/reset或时钟getter；
  独立对象比较只归一化列明的原生时间和对象身份，同对象探测不做归一化。
- 冷65-token请求实际派发 `[0,16)`、`[16,32)`、`[32,48)`、`[48,64)`、`[64,65)`，
  维持running优先级，后到热C不越过它。
- 两个不共享的32-token请求占满4个可用块，pending保留时原生decode扩块自然抢占R1。
  抢占时computed=0、preemptions=1、in-flight=32、stale=32、placeholders=0；
  队列 `[R1,H]` 的插件候选/探测为0。实际恢复与最终排空均有trace。
- 评分失效场景中C评分32，随后running扩块改变缓存，原生重新查找得到16；
  分配从未使用探测返回块或缓存分数。
- 66个主生命周期实例加57个窗口子例，共123个生命周期核验FIFO、引用和终态；
  pending峰值2且最终0。正常完成/抢占的placeholder、in-flight、stale归零。
  取消对象的历史字段原样保留，活跃状态及引用排空。ID复用仅在旧代pending排空后进行，
  generation只存在于测试trace，生产类没有请求级元数据或跨步分数。

## 审查、失败与修复

独立交叉审查按文件归属进行：生产实现者没有编写native harness、snapshot或trace审查器，
逐区间/FIFO/生命周期独立核验原始证据；验证实现者独立完整阅读生产3模块和对应固定上游源码。
最终均无阻断项。最终原始22,029条事件与已独立验算的第二轮在明确时间/身份白名单归一化后
逐结构相等，零差异；另外核对29个native、4个custom及83个采集源文件哈希。
正式 review 记录在私有 `review-final.json`。

| 历史运行 | 被测提交 | 原始结果 | 修复与处理 |
| --- | --- | --- | --- |
| attempt-01 | `4ceb5a0` | 失败，首轮重复完成23个case | encoder-decoder负例更改HF字段却未失效原生cached property；只修测试夹具，记录并核验全部33项实际输入；同时隔离正常post-probe证据。保留原始失败、reject acceptance |
| attempt-02 | `47041da` | 72 cases执行完，最终审查失败 | guarded合法“未实现”文本没有冒号，审查器错误要求统一冒号前缀；修正目标错误核验并增加7项回归。保留原始verified=false及reject acceptance |
| attempt-03 | `1091037` | 主矩阵及negative命令均成功 | 全新冻结提交完整重跑，verified=true，逐项acceptance通过 |

前两次失败trace及manifest从未重写；新审查器只读复核第二轮通过不改变其历史命令结果。
失败修复没有修改M1或生产调度机制。最终无尚未执行的M2A必需case。

Mac完整 `make check` 在最终被测提交通过：217 passed、2个明确native opt-in skips，
lint/format、shell语法、原有证据检查、wheel及sdist构建成功。两个skip不充当原生验收；
Linux矩阵独立实际执行。构建产物检查未包含runs、.local、.cache或.venv私有路径。
交付文档提交后另做本地完整检查，保存在新目录，不冒称新的Linux矩阵。

## 私有证据索引与三重门槛

以下相对路径均以新worktree为根，全部位于忽略的 `runs/m2a-prefix-scheduler/`；
不进入Git或PR。公开本文只手工核查技术事实、commit、哈希及case行号，不导出原始runtime，
不使用readiness导出器，不公开连接账号、内部地址或资源实际路径。

| 材料 | 路径 |
| --- | --- |
| 最终schema2主采集、manifest和A–F acceptance | `remote/attempt-03/` |
| 实际运行来源、冻结输入、case全集、完整trace、summary和审查结果 | `remote/attempt-03-data/` |
| 7项实际失败子进程的独立采集及原始结果 | `remote/attempt-03-negative/`、`remote/attempt-03-negative-data/` |
| GPU前后状态及作业日志 | `remote/attempt-03-resources/` |
| 独立审查、逐case行号、最终资源释放核验 | `review-final.json`、`final-trace-line-index.json`、`resource-release-final.json` |
| 最终被测代码完整本地检查 | `local-check-03/` |
| 前两次原始失败、negative和本地检查 | 对应`attempt-01`、`attempt-02`、`local-check-01/02`目录 |

材料完整、命令成功、行为验收分别判断。`check_run_evidence.py`验证哈希与acceptance结构，
不自行证明行为；行为由完整trace独立复算和人工交叉审查建立。

```bash
.venv/bin/python scripts/check_run_evidence.py runs/m2a-prefix-scheduler/remote/attempt-03
.venv/bin/python scripts/check_run_evidence.py runs/m2a-prefix-scheduler/remote/attempt-03 --require-command-success
.venv/bin/python scripts/check_run_evidence.py runs/m2a-prefix-scheduler/remote/attempt-03 --require-accepted
```

## 可复制入口

本地在新worktree运行 `make check`。Linux继续使用既有固定环境和模型**配置**目录，
`NATIVE_PYTHON`、`MODEL_CONFIG_DIR`由本地忽略配置解析，不写入公开文档。
先切到上述被测提交的独立checkout；从代码及配置冻结到采集结束不再修改。
仅在单一资源所有者控制的已分配节点执行，单次分配不超过20分钟。

```bash
export PYTHONHASHSEED=42 VLLM_USE_V2_MODEL_RUNNER=1 HF_HUB_OFFLINE=1
M2A_DATA=runs/m2a-prefix-scheduler/replay-01-data
M2A_RUN=runs/m2a-prefix-scheduler/replay-01

"$NATIVE_PYTHON" scripts/run_evidence.py --task-id M2A --kind native \
  --output "$M2A_RUN" --timeout 900 --config configs/native-m2a.json \
  --artifact "$M2A_DATA/runtime.json" --artifact "$M2A_DATA/inputs.json" \
  --artifact "$M2A_DATA/case-index.json" --artifact "$M2A_DATA/trace.jsonl" \
  --artifact "$M2A_DATA/summary.json" --artifact "$M2A_DATA/trace-acceptance.json" \
  --criterion 'A real AsyncScheduler resolution and off equivalence' \
  --criterion 'B native read-only probe with full instrumentation' \
  --criterion 'C automatic prefix BC causal 82 to 50 and ample50' \
  --criterion 'D bounded stable windows and exception restoration' \
  --criterion 'E real chunk preemption cancellation ID reuse and FIFO drain' \
  --criterion 'F complete nonempty three-repeat native matrix and fail-closed gates' \
  -- make native-m2a NATIVE_PYTHON="$NATIVE_PYTHON" \
  ARGS="--config configs/native-m2a.json --model-config '$MODEL_CONFIG_DIR' --output '$M2A_DATA'"

make native-m2a-trace-check NATIVE_PYTHON="$NATIVE_PYTHON" \
  ARGS="--config configs/native-m2a.json --trace '$M2A_DATA/trace.jsonl'"

M2A_NEG=runs/m2a-prefix-scheduler/replay-01-negative-data
"$NATIVE_PYTHON" scripts/run_evidence.py --task-id M2A-negative --kind native \
  --output runs/m2a-prefix-scheduler/replay-01-negative --timeout 120 \
  --config configs/native-m2a.json --artifact "$M2A_NEG/results.json" \
  --criterion 'Actual missing-native and invalid-entry subprocesses fail for their expected reasons' \
  -- "$NATIVE_PYTHON" scripts/check_m2a_fail_closed.py --output "$M2A_NEG"
```

上述目录必须是新目录。缺依赖或未执行native会失败，不退回mock。采集结束仍需逐项审查，
命令成功不会自动把acceptance改为accepted。

## 资源、工作区与剩余边界

三次计算分配串行、各上限20分钟，实际分别56秒、57秒、66秒；均已结束释放。
终态分别FAILED/FAILED/COMPLETED，最终退出码0:0；GPU退出后1 MiB。
保存了当时scontrol终态证据，最终完整队列查询无本任务作业。
稍后按旧ID查询因作业被清理返回invalid ID，sacct不可用；该后续查询限制另行记录，
不替换已有释放证据。没有覆盖共享checkout、环境或锁文件。

原工作区仍在 `feat/project-workflows` / `e71dfc11135d1f86b4cf28400ab603e268ea1a57`，
仅保留用户已有的 `docs/workflows.md` 修改。其SHA-256复核仍为
`c3dca495872357a584ecc2464dc92dafa5e4467d16237f4c11dfdf9ab096bb82`。
M1 worktree仍为 `feat/m1-native-bc-replay` / `fc23bc3` 且干净。
未修改M1实现、工厂、审查器、冻结配置、原始证据、取证工具或依赖锁。

本次是原生host-side元数据机制验证：合成ModelRunnerOutput与单层FullAttentionSpec几何，
不加载模型、tokenizer、真实KV tensor，不测TTFT、吞吐或GPU工作量。
只支持已经核查的固定版本和配置范围；运行时特殊请求继续交给原生处理。
`guarded`仍报未实现；`max_wait_ms`只有校验契约。monotonic等待保护、保护元数据生命周期
属于M2B，真实模型闭环和正式性能实验属于后续阶段。
