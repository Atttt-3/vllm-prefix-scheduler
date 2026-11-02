# M3 新批次交接：A–F 通过，待总站验收

任务 `M3-real-model-integration`，2026-11-03。按用户批准的新批次范围完成真实模型重跑、完整审查、正负控和专用白名单导出。**原四次分配及 formal-02 未知终态原样保留，原全历史门槛仍为 blocked；新批次通过不恢复或豁免旧终态。** 不合并 PR，不进入 M4/M5。

## 版本与交付

- 分支 `feat/m3-real-model-integration`；[草稿 PR #7](https://github.com/Atttt-3/vllm-prefix-scheduler/pull/7)，base `feat/m2b-wait-guard`。
- 起始 M2B 交付：`a1e57efc6a72811c53a8d259bdebcafd9b9c8c2e`；M2B 实际 Linux 被测仍为 `342f9082e05cd1bebee93a760dd02219eaabae34`。
- 原 M3 GPU/native：`403b7838e46ecf17a97cc14de0a1f048938dfbd8`，只作为保留历史。
- 新批次首次运行：`9dd9de4a9b71149f8102ad26eb8e9b3da48a8ded`。命令成功、终态完整；组合审查因新审查器误要求 M1 substitutions 为空而失败。失败归档保留，不取其中 case 拼入最终矩阵。
- **最终 GPU/native 冻结提交：`77844c447e7917264bb6f20fdb41e7056a7181b4`。** 修复仅严格匹配原 M1 的三项版本化 fixture 声明，并增加两项拒绝测试；生产策略、推理参数、上游、模型和锁不变。
- 最终交付在此之后只追加项目文档及两份白名单 JSON；完整交付 SHA 和逐文件差异列于 PR 与私有交付记录。该交付提交没有另跑 GPU。

固定平台仍为单 RTX 3090、Python 3.12.16、vLLM 0.32.0 / `5df63a7203cdb9328d824a27b682e77c7b9928be`、Qwen/Qwen3-4B BF16，模型/tokenizer revision `1cfa9a7208912126459214e8b04321603b3df60c`。实际加载模块、37份固定上游源码、10份模型/tokenizer文件及84份项目实现来源逐项绑定；不是只信版本字符串或目录名。

## A–F 验收及原始导航

下列行号均指私有 `remote-attempt-02/attempt-02/gpu-data/trace.jsonl`，共4781行，SHA-256 `74fb4fa2f76c255e1128e762957d8974875ff5e5d5d55d6c64580bec86209c18`。物理拼接顺序不是跨进程时间线；事件保留 PID、序号、时钟种类及 boot 时钟域。

| 项 | 新批次结论与证据 |
| --- | --- |
| A 固定来源/实例 | 通过。生产 off/prefix/guarded 及 native 各一个全新无 observer smoke；4个诊断引擎。smoke worker 事实见4642、4688、4732、4777行；guarded EngineCore 实例/MRO见263行，活跃V2 runner及BF16参数见266行。 |
| B 真实执行闭环 | 通过。20case、58请求全部明确结束（57 length、1 abort），诊断46个原生请求；439批次、363正工作批次，真实返回/异步物化/FIFO消费完整。prefix首批链见3879、3880–3883、3890–3892行。 |
| C 非平凡策略 | 通过。native/off三次均B→C；prefix及未过期guarded三次均C→B；真实500ms保护三次均B→C。精确决策行见下表。每个重复独立空闲/reset/真实预热并排空；实际enqueue确认先于resume。 |
| D 生命周期/资源 | 通过。首次正派发清除对应对象保护记录见1177行；真实取消完成257行、原生终态1914行、排空1930行；shutdown前1932行，shutdown 1933–1934行，8个进程退出4593–4600行。两个新作业均真实COMPLETED/0:0，独立监视和全队列查询无本任务活动作业。 |
| E 对照/口径 | 通过。native/off对应输入token IDs、输出token IDs和结束原因在smoke及三个机制重复完全一致；只陈述本次对照，不推广为任意batching的数值等价。派发、runner处理和可观测输入量分列。 |
| F 证据可信度 | 通过。完整正控+原30项控件全部通过，新增完整campaign正控+28项负控全部通过。逐子进程核对具体拒绝原因，覆盖历史遗漏/冒用/例外扩大/新终态缺失/资源残留/优化解释器。导出重新计算完整审查、来源和间接原件哈希。 |

生产 smoke 不安装诊断子类：其 scheduler 选择由实际配置/加载模块及固定 EngineCore 构造源码支撑，worker 的配置解析不冒称实际 EngineCore 对象反射。直接 `self/type/MRO` 证据来自诊断子类，父类仍为生产 `PrefixAsyncScheduler`。所有引擎保持 BF16、APC、async、chunked prefill、eager、原 attention backend 和 V2 runner。

功能 profile 保留16K/2048 tokens/0.85/TP1，仅把 max_num_seqs 从4改为1；固定上游实际活跃上限字段 `max_num_active_reqs=1`，见263行。`window=16` 与此上限不同。真实完成A并保留缓存后用原生 pause/keep积队，B、C在 EngineCore 的实际 add 后事件得到确认；无手改 waiting、fake clock或合成GPU输出。

| 机制 | 三次决策行 | 实际结果 |
| --- | --- | --- |
| prefix | 3878 / 4148 / 4418 | B分数0、C分数608；三个连续窗口均C→B |
| guarded未过期 | 366 / 636 / 906 | 相同评分与C→B；B年龄55.532 / 89.782 / 110.771ms |
| guarded到期 | 1176 / 1446 / 1716 | B年龄910.221 / 927.926 / 922.475ms，C年龄50.861 / 52.123 / 53.064ms；B序号10/13/16早于C的11/14/17，因此B→C |

保护年龄从原始对象绑定 `enqueued_ns` 与共同 `decision.now_ns` 重算，阈值严格500,000,000ns。表内毫秒只用于阅读，原件保留整数纳秒。到期B通过保护分支优先；不额外评分受保护B。暂停和人为等待是受控功能干预，不是自然排队公平性或TTFT实验。

## 计数、KV与资源

诊断计数为439次成功execute、363次sample和439次真实输出物化/原生消费，其中76批是合法空通知。scheduled/dispatched为18,517 tokens（prompt区间18,200，decode区间317，无重复区间）；直接观测的V2 runner输入有效/填充tokens均为18,517。此计数不覆盖无observer smoke的批次观测，也不等于物理FLOPs、速度收益或所有GPU工作。

实际参数4,022,468,096个，全部BF16；FullAttentionSpec涵盖36层、每块16 tokens。4个smoke各5553 blocks、KV storage去重后13,101,170,688 bytes；4个诊断各5562 blocks、13,122,404,352 bytes。gross槽位分别88,848和88,992，包含保留/null块，不冒称全部可分配容量。诊断排空后5561个非null块均回到free queue，引用总和/请求块映射/保护记录/原生队列/pending/通知/in-flight/stale/延迟释放归零；缓存hash可保留。

最终分配577份GPU样本，0.5秒采样的设备used峰值20,676 MiB；分配前/全部推理与native结束后实际均为1 MiB。它们是本次nvidia-smi设备口径，不复用readiness数值；torch allocator的allocated/reserved/peak bytes另存于worker快照。请求排空、scheduler shutdown、OS进程退出及Slurm释放分别审查。

新增预算实际用2次，原4次+新增2次=6/8，余2次未申请。两个新作业分别09:16、09:36，均单卡/20分钟上限、串行、COMPLETED/0:0。每次watcher先ready再submit，真实终态随后全队列查询，再核watcher退出；本地monotonic与Slurm服务器时间分域，不跨机器相减。第一次审查失败不从分母移除；历史formal-02未知也未被新运行替代。

## 验证、独审与导出

冻结提交完整 `make check`：482 passed、3个Linux native opt-in skip，lint/format/shell语法/历史证据/wheel与sdist通过。skip不计native/GPU。同一冻结提交实际Linux执行compat432及M1原入口12；生产策略未变，未要求或冒称636矩阵重新执行。M1原生回归的合成输出/单层KV/一次排列边界严格保留，不混同M3真实GPU链。

独审读取真实diff、每个来源Git blob、原始trace、全部资源原件、失败记录与子进程结果；执行者和独审都承认首次未在冻结前发现M1声明误设。修复后另冻结重跑完整矩阵，没有在原结果上改标签。完整性、命令成功、行为验收分别保存；GPU/native原始collector的默认pending保持原样，最终行为结论由组合audit和独立审查给出，避免事后更改已绑定原件。

公开证据仅为专用导出器生成的[summary](../../evidence/2026-11-03-m3-campaign/summary.json)和[provenance](../../evidence/2026-11-03-m3-campaign/provenance.json)。两者都保留new campaign范围、历史终态不可核查及原全历史blocked字段。不公开原始日志、输入文本、内部路径、账号、主机、进程/作业身份或权重。交接文字不是新的性能证据。

私有根目录为本worktree的 `runs/m3-real-model/campaign-20261103/`：

| 材料 | 相对该私有根的路径 |
| --- | --- |
| 历史封存/首轮失败 | `history-index.json`、`attempt-01-audit-failure.json`、`submission-ledger-version-01.json`；旧索引SHA固定为`c79c635723fb161e98b01b26c58241fbc7757709089eaf260cbbe0f6d2cea8e5` |
| 最终原始GPU/native | `remote-attempt-02/attempt-02/`；GPU trace见上方SHA |
| 2次新分配原件 | `control/attempt-01/`、`control/attempt-02/`；终态分别是`watch.jsonl`第517、533行 |
| 完整审查 | `attempt-02-audit.json`，SHA `8d56982d8c570389d7467bdf92432773ccab6d4b3a0aaa53280fe2cffae6d2b3` |
| 正负控 | `controls30-02/results.json`及全部子目录；`controls-campaign-02/result.json`及全部子目录 |
| schema2命令/门槛 | `offline-02/{local,audit,controls30,controls_campaign,export}/` |
| 独审/导航/保护 | `independent-final-review.json`及`.md`；`metrics-02.json`及`.md`；`workspaces-before/`、最终交付记录 |

## 从冻结提交重放

离线验收不申请GPU。在独立重放checkout中检出冻结提交，不切换受保护worktree的分支；已有私有归档可复用，实际环境与存储位置仍只从原工作区忽略的`.local/workspace.json`读取。以下路径均相对M3 worktree：

```bash
git checkout --detach 77844c447e7917264bb6f20fdb41e7056a7181b4
M3_EVIDENCE=runs/m3-real-model/campaign-20261103
M3_DATA="$M3_EVIDENCE/remote-attempt-02/attempt-02"
M3_ARGS=(--campaign-config configs/m3-campaign-v1.json
  --authorization configs/m3-campaign-authorization.json
  --history-index "$M3_EVIDENCE/history-index.json"
  --config configs/m3-real-model.json --inputs "$M3_DATA/gpu-inputs.json"
  --trace-dir "$M3_DATA/gpu-data/trace.jsonl"
  --resources "$M3_EVIDENCE/attempt-02-resources.json"
  --native-index "$M3_EVIDENCE/attempt-02-native-index.json")
.venv/bin/python scripts/check_m3_campaign.py "${M3_ARGS[@]}" --output /tmp/m3-reaudit.json
.venv/bin/python scripts/check_m3_fail_closed.py --config configs/m3-real-model.json \
  --inputs "$M3_DATA/gpu-inputs.json" --trace-dir "$M3_DATA/gpu-data/trace.jsonl" \
  --resources "$M3_EVIDENCE/attempt-02-resources.json" --output /tmp/m3-controls30-new
.venv/bin/python scripts/check_m3_campaign_fail_closed.py "${M3_ARGS[@]}" --output /tmp/m3-controls-campaign-new
.venv/bin/python scripts/export_m3_campaign.py "${M3_ARGS[@]}" \
  --sources "$M3_DATA/gpu-data/sources.json" --audit /tmp/m3-reaudit.json --output /tmp/m3-public-new
UV_OFFLINE=1 make check
```

正式命令的完整参数、开始/结束源码指纹和退出码已由schema2 collector保存。输出目录必须新建；不要覆盖现有原件。若另开GPU重放，须先追加预算/提交台账、先启动 `scripts/watch_m3_campaign.py` 再提交单卡≤20分钟作业，在现有分配内以既有解释器调用 `scripts/run_m3_campaign.py --campaign-config configs/m3-campaign-v1.json --history-index <私有历史索引> --output <全新私有输出>`。原件中的`control/*/job.sbatch`记录实际命令；不在登录网关推理，不重装共享环境。

原工作区用户`docs/workflows.md`保护SHA仍为`c3dca495872357a584ecc2464dc92dafa5e4467d16237f4c11dfdf9ab096bb82`；M1/M2A/M2B worktree不动。PR #4–#7保持未合并，#7保持draft。阶段停在M3总站待验收。当前结论不覆盖自然公平性、TTFT p95、吞吐提升、真实缓存淘汰/自然抢占压力、容量矩阵、排序成本、M4故障矩阵或M5正式性能实验。
