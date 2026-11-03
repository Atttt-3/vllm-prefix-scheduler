# M5-paired-performance 实验设计

M5 从总站验收的 M4 交付 `f4d94d3d22a967efeeb3a00bc13f6a6b8591835b` 起步。M4 的执行 `5525bda3eff2dad20f614eaca7718383c31448e4`、独立分析 `34fa4eabb846b8095fe7aa9ecdcc05d2a6bf27a7` 和交付各自保留。总站接受的是获批的离线审查修订，原冻结 F 为 FAILED。M3 formal-02 为 UNVERIFIED_PURGED，原全历史门槛为 blocked。生产调度策略沿用 M4 交付。

## 冻结条件

[配置](../../configs/m5-performance.json) 固定实际 GPU、引擎、上游、模型及 tokenizer、采集条件、正式矩阵和判定门槛。正式模式为原生 APC-on FCFS、插件 off、prefix、guarded，窗口 16，保护阈值 500 ms。每组四模式同卡、同节点、同到达表、同采样、同预热和缓存初态，依冻结顺序启动四个新引擎。每次绑定实际物理 KV 块数和 FLASH_ATTN。关键场景各五组，负对照各两组，共 16 组、64 个模式轮次、32,000 条计划请求。16 个正式 seed 与校准 seed 分离。

| 场景 | 输入与到达 | 校验 |
| --- | --- | --- |
| burst | 500 条，25 批，批内 2 ms 间距。64 个共享前缀族轮转，半热半冷。 | 真实新鲜候选竞争、缓存淘汰和后续同族复用。 |
| mixed | 500 条，按固定速率和确定性有界抖动开放环到达。冷热各半，前缀族有短距复用和长距回访。 | 真实新鲜候选竞争、淘汰、窗口及整队等待。 |
| ample | 500 条，129-token 提示和四个共享族。 | 工作集上界可容纳，无后续复用相关淘汰。 |
| all-cold | 500 条，与关键负载相同的输入和输出长度，首完整块唯一。 | 实际零命中。 |
| serial | 500 条，129-token 提示，8 个输出 token，固定间距开放环。 | 实际最大未终态请求数为一。 |
| mechanism | A 先真实执行，B 冷，C 复用 A 的 1,024 tokens。按 B→C 入队，暂停积队后恢复。129/257 物理块为压力/充足配置。 | 在原始事件中核对真实候选、分配、淘汰、命中和 runner。诊断单列。 |

## 指标

主指标为成功请求从计划到达至客户端首个真实生成 token 的 TTFT p95。采用 nearest-rank，另报实际提交起算的 TTFT。所有计划请求进入互斥终态账本。每请求截止为计划到达后 30 秒。成功 TTFT 和 TPOT 各有自己的分母。完成率和按时首 token 率以计划总数为分母。提交偏差 p99 ≤5 ms、最大 ≤25 ms，拒绝提前、跳过或合并提交。跨进程用相同 CLOCK_MONOTONIC、boot ID、时间命名空间和实测往返界限绑定。吞吐使用首计划到达至末计划到达后 30 秒的共同窗口，计入空闲与排空，分列完成请求、已生成 token 和实际排空时长。

冷和热由冻结输入定义。实际命中单列。较晚入队请求首次正派发时，较早入队请求仍在等待首派发且同批未派发，给较早请求计一次整队越过。两者还须同处该决策的新鲜队头窗口才计窗口内越过。另列实际热请求造成的越过、已过阈值的候选及保护顺序。

GPU 工作以成功物化的原生 forward、实际 runner 请求切片和 CUDA 位置为依据。有效位置按原始 prompt 边界划分 prefill 与 decode，按同一代以前已执行位置的交集识别重算。派发、物化、消费、丢弃、padding 和缺证分列。采集在内存中有界，排空后落盘。实际位置以同流异步复制和既有同步点获取。C3 单独比较 endpoint 与 light，校验 TTFT p95 和吞吐的扰动。profile 单列取窗口、评分、排序、提交与整次调度的计时范围。阶段分位数不可相加。

新冻结使用 M5 测量 schema 2，外层 collector 仍沿用既有 schema 2。审查从 seed 和场景配置复算到达、请求标签、提示形状和前缀族。每个有候选的批次须有对应的决策记录。仅客户端采集的公平性指标为不可观测。全冷逐请求核对客户端报告和同代、同批的首次分配与 runner 起点。淘汰明细绑定实际物理块和原生分配事件，并与分配事件的淘汰计数核对。具有未来热需求的移除键只用于标记可能受影响的复用。实际损失仍依赖后续命中与 runner 证据。

正式配对主吞吐使用共同计划窗口，另列实际运行窗口和排空。C3、D1 和 D2 的扰动比较继续使用各轮末终态窗口，容限为 TTFT p95 绝对差至多 `max(2 ms, 参考轮 p95 × 5%)`，请求与 token 吞吐各在 ±5% 内。profile 中 `schedule` 包住调度父类调用，`schedule_envelope` 另计外层观测、影子窗口及记录成本。两者都包含其区间内启用的钩子，均需结合采集档位解释。

## 作业和预算

校准、正式均使用明确的 M5 授权文件。每次请求 20 分钟、单卡、严格串行，工作截止 1,050 秒。提交前验证 watcher ready，提交后保存回执、权威终态、退出码、全队列和 watcher 退出。逐作业记录 GPU UUID、前中后状态、子进程所有权和退出残留。失败和重试计入已授权额度，未知终态会阻断后续提交。

| 角色 | 作业内容 | 工作估算，含余量 |
| --- | --- | --- |
| C1 | 四模式 × 两种容量的真实 A/B/C 诊断 | 约 11 分钟 |
| C2 | 两个关键负载，各 5 档原生 offered rate，每档 60 条 | 约 16 分钟 |
| C3 | 两场景各两对 endpoint/light，另加 guarded profile | 约 16 分钟 |
| C4 | burst，四模式 × 500 条 | 约 17 分 8 秒 |
| C5 | mixed，四模式 × 500 条 | 约 17 分 8 秒 |
| C6 | 两个负对照各 500 条、并发 1 pilot 和全长核验、生成 16 张正式表 | 约 14–16 分钟，含单次 tokenizer 加载批生成的预留 |
| 正式 | 每组一个作业，四模式各 500 条 | 16 次，合计 320 申请分钟 |

校准建议 6 次、120 申请分钟。上表为本地规划估算。正式分钟和可行性以校准后的工时与资源证据核定。C2 从 3、4、5、6、8 请求/秒选满足提交门槛、至少 99% 按时完成及可见新鲜候选竞争的最低原生速率。C4/C5 各要求至少 20 次冷热窗口竞争和 5 次可复用块淘汰，四模式同组含余量 ≤1,050 秒。C6 的并发 1 间距取 pilot 最大提交至终态时长的 1.25 倍加 25 ms，再向上取整到 10 ms，验证全长无重叠。若估算或资格失败，保留原件及阻断理由，再评估新预算或冻结版本。

## 本地与集群入口

```bash
make check
python scripts/plan_m5_calibration.py plan --config configs/m5-performance.json --stage C1 --output runs/c1-plan.json
python scripts/check_m5_trace.py run --config configs/m5-performance.json --inputs INPUT --run RAW --pair PAIR --mode native --phase formal
python scripts/check_m5_resources.py runs/M5-paired-performance/attempts --history
```

真实校准前由主负责人从原仓库忽略的 `.local/workspace.json` 读取连接、现有 GPU 环境、模型与缓存位置。在其远端原 checkout 同级部署独立的 `vllm-m5-paired-performance`，检出获批 SHA，保证 Git 干净。控制端使用原配置中的 SSH 别名，登录节点仅做 Git、私有材料传输和调度。每份私有授权记录绑定用户原话所在的记录、完整执行 SHA、配置哈希、phase、获批 roles、单次 20 分钟和单卡、累计次数和总申请分钟。正式授权再绑定 `formal-freeze.json` 哈希。两个阶段各用一份独立的授权。

```bash
python scripts/submit_m5.py --authorization PRIVATE_CALIBRATION_AUTH --phase calibration --role C1 --attempt c1-01
python scripts/freeze_m5.py --config configs/m5-performance.json --history runs/M5-paired-performance/attempts --execution-commit EXEC_SHA --output runs/M5-paired-performance/frozen
python scripts/submit_m5.py --authorization PRIVATE_FORMAL_AUTH --phase formal --role burst-01 --attempt burst-01-a1 --freeze runs/M5-paired-performance/frozen/formal-freeze.json
```

每次调用只提交一个获批角色。完成前一校准作业后方可顺序启动下一角色。可执行命令、控制端账本、计算节点结果和重算文件在新的忽略目录 `runs/M5-paired-performance/` 下。若终态、下载或清理证据缺失，保留目录并核对该作业，后续提交由门槛阻断。

新作业在调用 `sbatch` 前持久化提交意图。遇到终态查询或证据传输中断时，用 `python scripts/submit_m5.py --resume --attempt ATTEMPT_ID` 按原回执恢复同一作业的证据。恢复过程保留原 watcher 的提交前 ready 证据。提交意图存在但回执不确定时继续阻断后续作业。新资源记录按采样时间和引擎进程组核对活跃 GPU 进程的归属。

### D2-R 热状态与同档诊断

[D2-R 冻结方案](../../configs/m5-d2r.json) 使用封存 D1 mixed 输入表，按 8 请求/秒重放。热试跑先冻结本次设备目标，随后以四个新引擎按 E1–E4 顺序执行 guarded endpoint，每轮计划 60 条。保持 D2 的 20 秒热窗口、温差、频率、采样和首次合格窗口规则。每个测量引擎的热准入时间从 75 秒延长至 95 秒，记录固定准入截止与实际结束。热准入失败时保留已执行波次和缺组，按原方案停止。

用户只授权 D2-R 新增一次串行单 RTX 3090 分配，申请最多 20 分钟。预算为 150 秒热试跑、325 秒引擎开销、380 秒测量引擎热准入和 160 秒核验清理，共 1,015 秒，距作业内限 1,050 秒余 35 秒。独立复算 E1/E2、E3/E4、E1/E4，分别判断测量期间的热可比性和同档 TTFT、吞吐。原 C3 与 D2 的结果保留，资格复验单独安排。

### D2-R110 与拆分 C3

[D2-R110 候选冻结](../../configs/m5-d2r110.json) 绑定 D2-R 的未完成报告、原 C3 阻断报告及原输入表。保留每波 60 条、8 请求/秒、完整排空和 APC 重置，只将测量轮热准入上限改为 110 秒。预算为 150 秒热试跑、325 秒引擎开销、440 秒热准入和 120 秒核验清理，共 1,035 秒。硬上限为 1,050 秒。新增分配只在专属 `D2-R110` 授权后提交，单 RTX 3090、最多 20 分钟。历史账本累计上限为 9 次、180 申请分钟，其中本批新增一次。每个尝试的测量轮从同一原始日志复算 95 秒的准入资格，按截止前可见的完整排空、重置和判定记录处理。

```bash
python scripts/submit_m5.py --authorization PRIVATE_D2_R110_AUTH --phase calibration --role D2-R110 --attempt d2-r110-01
python scripts/check_m5_d2.py --attempt runs/M5-paired-performance/attempts/d2-r110-01 --proposal configs/m5-d2r110.json --output runs/M5-paired-performance/d2-r110-review/analysis.json
```

拆分 C3 的历史资格入口以 D2-R110 稳定为前提，该轮独立复算为不稳定。新版资格入口等待受控同条件 E/E 的后继角色及原始证据，当前允许的后继角色清单为空。资格批仍规划七次。角色顺序为 C1、C2、C3-burst、C3-mixed、C4、C5、C6，新增 140 申请分钟。七个角色使用同一执行 SHA。C3 每场景固定两张 60 条到达表，分别按 E L L E、L E E L 启动新引擎。同块比较须复用同一输入哈希。每场景复算四对 E/L、两对 E/E、两对 L/L，沿用原门槛。C3-mixed 另带阶段计时。常规预热、APC 预置和带外采样与正式轮一致。温度和频率用于解释波动。任一同档或跨档对照越界时保留原始结果，后续阶段受阻。

正式冻结独立复算两个 C3 作业，再将其原始客户端与引擎记录、输入、collector、分配和释放绑定到逻辑 C3 的同一摘要。提交和发布重新验证此摘要。原 C3 仍保存在各自历史版本。C4–C6 继续以原自然压力、负对照和单组耗时门槛验收。正式 16 组、64 轮的预算在校准后单独审批。

### D3 提交后分段诊断

[D3 冻结方案](../../configs/m5-d3.json) 使用单独获批的一次串行单 RTX 3090 分配，申请最多 20 分钟。复用 D2-R110 的热试跑、110 秒准入、同一张 mixed 表、8 请求/秒及四轮各 60 条。作业内估算 1,035 秒，硬上限 1,050 秒。提交前核验固定 vLLM 原生事件、输出统计、调度和引擎源文件哈希。计算节点在逐轮测量前后核对客户端和引擎的单调时钟域及往返界限。

四轮采用同一采集方式。客户端先记真实 token 到达，再从既有 `RequestOutput.metrics` 复制原生时间戳，排空后写盘。首次接纳时间指调度轮次起点。原生首末输出时间来自引擎输出封包。独立复算逐请求分解提交到原生入队、入队到首次接纳、首次接纳到原生首输出、原生首输出到客户端首 token，并另列首 token 后的输出和终态。跨进程区间须有已核验的同域证据。缺失时间戳保留在分母内，缺证区间单列。

独立核对 E1/E2、E3/E4、E1/E4 的原热状态、TTFT 和吞吐门槛。额外报告匹配请求的各区间差值、p95 的实际请求身份和最后 15 条，避免用不同请求的分位数相加。若 E3/E4 再次同方向同时越过两项门槛，报告复现；未满足则按本次观测范围报告。D3 证据保留原 C3 的 `blocked`，资格复验单独安排。

```bash
python scripts/submit_m5.py --authorization PRIVATE_D3_AUTH --phase calibration --role D3 --attempt d3-01
python scripts/check_m5_d2.py --attempt runs/M5-paired-performance/attempts/d3-01 --proposal configs/m5-d3.json --output runs/M5-paired-performance/d3-review/analysis.json
```

## 独立审查与交付

`freeze_m5.py` 从 schema 2 collector 和 Git 对象核对源文件与原始输入，再独立复算六个逻辑校准阶段并绑定正式表，拆分 C3 版本使用七次实际分配。分析使用同一冻结代码，实际执行、分析和文档交付 SHA 分别记录。正式导出先用 `export_m5.py review` 核对完整配对、来源和全作业释放，再用 `check_m5_release_controls.py` 从封存原件运行子进程正负控。负控按指定的退出码和拒绝原因核验。独立审查人核对 A–F、审查包与控制记录，并通过协调渠道交付签认文件 SHA。发布时重新从封存原件复算，核对签认和控制记录绑定后再写出白名单。逐轮表包含相对原生的配对差以及 guarded−prefix。报告中位数、范围、方向和失败分母，负对照为描述性结果。最终区分命令、证据、行为与性能状态，给出实际执行、分析和交付 SHA、原件索引及资源账本。

```bash
python scripts/export_m5.py review --config configs/m5-performance.json --freeze runs/M5-paired-performance/frozen/formal-freeze.json --history runs/M5-paired-performance/attempts --analysis-commit ANALYSIS_SHA --output runs/M5-paired-performance/formal-review
python scripts/check_m5_release_controls.py --config configs/m5-performance.json --freeze runs/M5-paired-performance/frozen/formal-freeze.json --history runs/M5-paired-performance/attempts --review-dir runs/M5-paired-performance/formal-review --output runs/M5-paired-performance/formal-controls
python scripts/export_m5.py release --config configs/m5-performance.json --freeze runs/M5-paired-performance/frozen/formal-freeze.json --history runs/M5-paired-performance/attempts --review-dir runs/M5-paired-performance/formal-review --decision PRIVATE_INDEPENDENT_DECISION --approved-decision-sha256 DECISION_SHA_FROM_REVIEWER --controls runs/M5-paired-performance/formal-controls/controls.json --output evidence/M5-paired-performance
```

## D4 到达率对照

[D4 冻结方案](../../configs/m5-d4.json) 检验 D3 尾部积压是否随到达压力变化。固定生产 guarded 策略、端点采集、4 个序列槽位及同一张 mixed 已分词输入。沿用 D3 绑定的原生时间戳、时钟域验证、缓存预置和 8 请求/秒热试跑。四个测量引擎都沿用 110 秒热准入、20 秒热窗口和原采样与准入门槛。按 8、6、6、8 请求/秒启动四个新引擎，每轮 60 条。8 请求/秒使用封存表原字节。6 请求/秒按原 mixed 绝对时间及有界抖动算法重算到达偏移，保留请求顺序、token、冷热标签、采样和预热。两表的哈希和末到达时刻在方案内固定。

E1/E4 和 E2/E3 分别做同速率配对，均以较早轮为参考，沿用原 TTFT p95 与末终态请求和 token 吞吐容限。逐轮报告最后 15 条的原生入队至首次接纳中位数、待接纳峰值、等待累计和与四个原生输出区间重合的比例。四轮完整、提交与时钟证据合格、所有测量窗口两两热可比，才判读跨速率信号。只有两轮 6 请求/秒的尾部等待中位数均低于两轮 8 请求/秒且待接纳峰值均不更高，才记为本批支持到达压力解释。原生接纳至末输出区间仅用于近似占用，真实准入原因与 GPU 执行仍需各自证据。历史 C2 的原生 6 请求/秒未形成新鲜候选竞争，因此 D4 只用于诊断，正式负载继续由 C2/C4/C5 判定。

申请一次串行单 RTX 3090，最多 20 分钟。D3 的估算为 1,035 秒，两轮降速增加的绝对到达窗口共约 4.917 秒，向上取整后估算 1,040 秒，硬上限仍为 1,050 秒。保留原 watcher、权威终态、全队列和清理协议。热准入或边界核验失败时保存原件并停止，额度内无重试。控制器执行角色绑定冻结 SHA。提交前用节点环境复核已上传计划和两张表，确认独立 watcher 就绪后才提交。D4 两个同速率配对的结果只解释本次压力对照。C3 资格需同条件的新证据独立复算通过。

离线校验与私有输入准备命令使用 D3 原报告或字节一致的独立复算文件。

```bash
python scripts/prepare_m5_d4.py --plan configs/m5-d4.json --config configs/m5-performance.json \
  --history runs/M5-paired-performance \
  --d3-analysis runs/M5-paired-performance/d3-review/analysis.json \
  --output runs/M5-paired-performance/d4-prepared
```

## D5 隔离活跃准入上限

[D5 冻结方案](../../configs/m5-d5.json) 从 D4 的到达压力证据出发，在同一张原始 mixed 8 请求/秒表上比较活跃准入上限。五个新引擎均固定 5 个 runner 槽位、2,048 token 批预算、5,553 个实际物理 KV 块、原生产 guarded 策略及原生端点采集。热试跑使用 4 个活跃槽，随后四轮依次使用 4、5、5、4 个活跃槽，各 60 条。全部复用原 8 请求/秒预热输入、APC 重置与预置，以及 D3 热准入、时钟和源码核验。

固定版本的 vLLM 从不同字段读取 runner 槽位与活跃准入上限。D5 仅在诊断回放中传入显式 KV 块覆盖。覆盖会替代容量自动估计，五槽位能否用此容量启动由热试跑实测核验。每个引擎在首个预热波次前核验实际槽位、活跃上限、token 批预算和 KV 块数。若启动、配置、热准入或资源边界失败，保存已取得的原件并按冻结流程停止。生产与正式配置仍使用原 M5 基线。

E1/E4 与 E2/E3 分别按原热状态、TTFT p95 与末终态请求及 token 吞吐门槛检查同档重复。四轮证据完整、全部六对热状态可比、两组同档均通过，且两个 4 槽轮次的末 15 条接纳前等待中位数均至少 300 ms、待接纳峰值均至少 4，才能判读优化。该复现门槛来自 D4 两轮 8 请求/秒的 345.136 和 444.593 ms、峰值均 4。D5 的 4 槽组共用 5 槽 runner，复现条件将这项布局差异纳入解释。

两个 5 槽轮次的末 15 条接纳前等待中位数与 TTFT p95 都须低于两个 4 槽轮次，且任一跨档对照的请求与 token 吞吐降幅各不超过 5%，才记为支持增加一个准入槽。另报告全部四个跨档差值和各档波动。原生接纳至末输出区间用于估计占用。复现、重复或证据前提不足记为不可判定，证据和前提完整但收益规则失败记为不支持。D5 只诊断准入，C3 后继资格仍按原流程判断。

预算为一次串行单 RTX 3090，最多 20 分钟。沿用 D3 的 1,035 秒预算包络，给五槽位启动增加 5 秒余量，估算 1,040 秒，硬上限 1,050 秒。D4 的最大单引擎非预热开销为 59.809 秒，低于该预算按每引擎 65 秒给出的包络。作业须沿用提交前节点端材料复核、独立 watcher、权威终态、全队列和资源清理。无失败重试额度。

在对应执行 SHA 和新增单次授权文件就绪后，按以下入口运行和复算。

```bash
python scripts/submit_m5.py --authorization PRIVATE_D5_AUTH --phase calibration --role D5 --attempt d5-01
python scripts/check_m5_d2.py --attempt runs/M5-paired-performance/attempts/d5-01 --proposal configs/m5-d5.json --output runs/M5-paired-performance/d5-review/analysis.json
```

## D5-R 热目标修复

[D5-R 冻结方案](../../configs/m5-d5r.json) 继承 D5 的 8 请求/秒输入、4/5/5/4 顺序、实际容量、判定规则及预算。热试跑仍使用原 150 秒，在每个完整波次排空并清 APC 后，按原采样可见边界检查 20 秒平台。选取第一个通过原平台检查且温度中位数在 **64–66°C（含边界）**的窗口，将该窗口实际温度、SM 和显存时钟冻结为目标。超过热试跑截止时间仍无合格目标，保留原件并停止。测量各轮的 110 秒热准入、±2°C 目标温差、时钟、缓存预置和性能门槛沿用 D5。

原 D5 的热试跑在 62°C 冻结，E1 有 17 个基础合格平台的中位数为 65–66°C。用同一份封存 E1 按 64、65、66°C 的假设目标分别复算，均在第八个完整波次的 70.764 秒通过。该复算只使用截止前已可见且完成排空与清 APC 的波次。原热试跑停止后的温度轨迹仍需实测。

D5-R 执行与独立审查分别实现热试跑的带宽过滤与首个窗口选择。D5 封存审查逐字节复现。四个测量轮的六对热窗口、两组同档及四组跨档均按原规则复算。四轮不齐、热不可比、同档失败或两个 4 槽轮次未复现积压时，优化记为 `unverified`。前提完整后按 D5 的尾部等待、TTFT 和吞吐规则判断。资格入口明确排除 D5-R，原 C3 保持 `blocked`。

D5-R 单独授权一次串行单 RTX 3090，最多 20 分钟，预计 1,040 秒，硬上限 1,050 秒。累计上限为第 13 次分配、260 申请分钟。候选提交只允许绑定原 D5 已核验释放的第 12 次分配。提交前用计算环境复核已上传计划及原件，独立 watcher 就绪后才提交。任一轮热准入失败即停止，保留缺组。此批无重试。热状态与两组同档重复均通过后，才设计并审核单独的同条件资格前置试验。

在 D5-R 执行 SHA 和新增单次授权文件就绪后，按以下入口运行及复算。

```bash
python scripts/submit_m5.py --authorization PRIVATE_D5R_AUTH --phase calibration --role D5-R --attempt d5-r-01
python scripts/check_m5_d2.py --attempt runs/M5-paired-performance/attempts/d5-r-01 --proposal configs/m5-d5r.json --output runs/M5-paired-performance/d5-r-review/analysis.json
```

## D6 最早出包慢段诊断

[D6 冻结方案](../../configs/m5-d6.json) 绑定 D5-R 的实际执行与分析原件。热试跑继承 D5-R 的 64–66°C 首个合格目标选择。三轮按端点 E1、逐步 E2、逐步 E3 执行，各启用新引擎。三轮均用原 guarded 策略、同一张 8 请求/秒表、60 请求、4 活跃槽、5 runner 槽、5,553 个实际 KV 块、2,048 token 批预算和原 110 秒热准入。为给逐步采集 arm 留出明确边界，三轮共用 250 ms 的计划起点提前量，仍按原遥测新鲜度和温差核验。逐步 arm 未在计划时点前返回即停止。

保存每轮全部请求的客户端和原生端点。候选筛查只使用 T0 后 0–3 秒，按后一个真实 token 到达时刻落入 500 ms 半开窗。同请求相邻 token 才构成间隔，同封包内多个 token 共用到达时间。每窗至少 10 个间隔、两个请求，连续两个合格窗中位数至少 30 ms。D5-R 封存 E1 的 1.5–2.0 和 2.0–2.5 秒窗分别为 35.042 和 35.075 ms，E4 均约 22 ms。候选判定最早在 2.5 秒成立，逐步追溯从约 1.3–1.7 秒的在途请求和实际首个异常事件开始。

逐步观察绑定固定 UniProc、EngineCore、scheduler、V2 runner 与异步输出源码。记录入队处理、调度与活跃状态、实际 runner 批次有效和填充 token、原生执行调用、采样、future 消费、结果物化、scheduler 消费和原生输出。原生调度输出身份将提交的新批和消费的旧批分别绑定。0–3 秒中发出的批次保留至原生消费完成，终态或空批通知单列。诊断热路径沿用原调用顺序和消费点，使用有界记录，在排空后落盘。超时、溢出、断链、跨域或不完整捕获停止推进并保留已产生原件。真实调度原因无分支证据时记 `unknown`。执行、future 与物化间隔是主机可见计时。D6 暂无独立设备时钟采集。

独立复算从封存请求、批次和时钟域追溯最早异常。只有端点参照复现、逐步轮有热可比的快慢对照、异常先于持续积压且位于同请求或同构批次出包的关键路径，并能解释观察到的间隔量级，才归入相应阶段。槽位放大以实际释放、下一次接纳和待接纳增长串联。缺少对照、区分不了候选机制或只在逐步轮出现时，结论为 `unresolved`，另记可核对的最窄阶段。端点参照只用于发现采集扰动迹象。

预算为 **1 次串行单 RTX 3090 分配，最多 20 分钟**。预计作业内 860 秒，包含试跑 150 秒、三轮热准入 330 秒、四引擎其他工作 260 秒及核验清理 120 秒。封存 D5-R 单引擎实测非预热开销上限约 57.147 秒。D6 冻结作业内硬上限 **900 秒**，为采集导出和运行偏差留 40 秒。提交前核验已上传输入和源码，独立 watcher 就绪后才提交，无重试额度。慢段未复现时在当前作业内保留完整对照，结论记 `unresolved`。D6 独立于 C3 同条件前置与资格复验。

```bash
python scripts/m5_d6_signature.py --attempt runs/M5-paired-performance/attempts/d5-r-01 --output NEW_PRIVATE_REPORT
python scripts/submit_m5.py --authorization PRIVATE_D6_AUTH --phase calibration --role D6 --attempt d6-01
python scripts/check_m5_d2.py --attempt runs/M5-paired-performance/attempts/d6-01 --proposal configs/m5-d6.json --output NEW_PRIVATE_AUDIT
```

## D7 晚段接纳诊断

[D7 冻结方案](../../configs/m5-d7.json) 从 D6 交付 `f8c2049` 起步，绑定原 D6 方案、实际分析及七轮 420 条请求的封存晚段复算。D6 三轮首次显著接纳等待在 T0 后约 3.76 秒，持续待接纳从 4.77–4.84 秒开始。E3 的 6.0–7.0 秒晚段有两个连续慢窗，早于该时点的积压仍需原生批次证据。D7 固定观察 **[3.5, 7.5) 秒**。三阶段分别保留首次显著等待、持续积压、晚段出包变化的证据。

热试跑后依次运行端点 E1、逐步 E2、逐步 E3、端点 E4。每轮新建引擎，沿用同一 8 请求/秒、60 条表，guarded、4 活跃槽、5 runner 槽、5,553 实际 KV 块和 2,048 token 批预算。继续使用 D5-R 的 64–66°C 试跑平台选择和原 110 秒准入。六对热可比性、E1/E4 和 E2/E3 同档，以及相邻 E1/E2 和 E4/E3 的 TTFT 与两项吞吐均独立复算。相邻两对均以端点轮为参考。原容限为 TTFT p95 的 5% 或 2 ms 中较大值、请求和 token 吞吐各 ±5%。

在 T0 前空闲时 arm。窗前仅保存有界入队与真实原生输出序号前缀，详细事件从第一个合格的原生循环开始。入口记录运行、等待和 FIFO 快照。固定 EngineCore 的两深度 FIFO 在步入口最多一项，已有批次单列为跨入。使用其原生 future 及消费路径核对输出，跨入批次的派发、执行和 runner 输入均为缺证。相邻请求序号按实际原生输出递进，与客户端同 ID 的真实 token 顺序核对。窗内派发的在途批次跟踪到原生消费。原生时间与客户端时间只有通过时钟域核验才相减。排空后导出，仍使用有界记录，无设备计时归因。

等待起点从真实入队至首次接纳区间复算，记录首个 ≥50 ms 的请求和首段连续 ≥500 ms 待接纳。慢窗保持相对于 T0 的 500 ms 对齐，每窗至少 10 个间隔和两个请求，连续两窗中位数 ≥30 ms。将积压形成和后续放大分别判定。同请求、批次构成、缓存及队列状态对齐后，主机阶段需要至少两个可比批次同向证据，且领先于其解释的等待变化。晚段定位要求端点也复现、逐步轮有可比快慢对照。只有观测到槽位持有延长、释放和后续接纳的原生链，才支持等待放大。证据不足记 `unresolved`。捕获缺失、热准入失败或作业内到期时保留原件和未执行轮次。

预算为 **一次串行单 RTX 3090，最多 20 分钟，无重试**。按试跑 150 秒、四轮热准入 440 秒、五引擎非预热工作 325 秒、核验与清理 120 秒、额外导出余量 5 秒估算 **1,040 秒**。D6 实测单引擎非预热上限 60.990 秒，D7 按每引擎 65 秒包络。作业内硬上限 1,050 秒，仍需 D7 独立授权及提交前 watcher 就绪。原 C3 和资格前置流程继续适用。

```bash
python scripts/submit_m5.py --authorization PRIVATE_D7_AUTH --phase calibration --role D7 --attempt d7-01
python scripts/check_m5_d2.py --attempt runs/M5-paired-performance/attempts/d7-01 --proposal configs/m5-d7.json --output NEW_PRIVATE_D7_AUDIT
python scripts/check_m5_resources.py runs/M5-paired-performance/attempts --history
```

## D8 同条件端点资格前置

[D8 冻结方案](../../configs/m5-d8.json) 从 D7 交付 `171eef2` 起步，绑定原 D2-R110 方案、D7 实际审查与释放记录、原 mixed 到达表和原 C3 `blocked`。D7 的晚段采集、五个 runner 槽及暖目标选择仍属 D7 诊断。D8 继承 D2-R110 原 `target`、`conditioning`、`stability_gate` 与预算，采用原基线四个 runner 槽、原生容量配置、guarded、8 请求/秒和仅端点采集。试跑按原热规则选首个合格平台。之后四轮新引擎按 E1–E4 顺序，各 60 条。

四轮均须通过原热准入和端点有效性，独立复算 E1/E2、E3/E4、E1/E4。TTFT p95 容限为 `max(2 ms, 参考轮 p95 × 5%)`，请求与 token 末终态吞吐各 ±5%。独立审查核对五个阶段、原始客户端与设备记录、实际基线、相同的实测 KV 块数和资源释放。缺轮、热状态不合格、请求证据不完整或任一配对越界均阻断资格前置。D8 原件按执行 SHA 重放，只有完整通过且独立报告摘要一致才满足 `same-condition-ee-v1`。

本批只规划一次串行单 RTX 3090，最多 20 分钟，无重试，新增为第 16 次分配。预计 1,035 秒，包括热试跑 150 秒、四轮热准入 440 秒、五引擎其他工作 325 秒、核验和清理 120 秒。作业内硬上限 1,050 秒。实际 GPU 提交须有 D8 独立授权、原件上传核验及 watcher 就绪。

D8 通过后另行申请七次资格分配。提交侧按阶段顺序检查前置报告，并从封存输入、collector、审查原件和资源记录重新判定其资格。C3 任一场景未通过即阻断后续作业。原 C3 保持 `blocked`，正式 16 组另行冻结和审批。

```bash
python scripts/submit_m5.py --authorization PRIVATE_D8_AUTH --phase calibration --role D8 --attempt d8-01
python scripts/check_m5_d2.py --attempt runs/M5-paired-performance/attempts/d8-01 --proposal configs/m5-d8.json --output NEW_PRIVATE_D8_AUDIT
```

## D9 原四槽双窗诊断

[D9 冻结方案](../../configs/m5-d9.json) 从 D8 交付 `63c84db` 起步。D3、D6、D7、D8 共 15 轮、900 条请求使用同一张输入表。D8 E1 在 0–3 秒、E3 在 6.5 秒后触发慢窗。D3 的同 ID 证据定位其自身尾部接纳前差异，D7 直接核对四槽占满后的释放与接纳。D6/D7 的 runner 配置有五槽，D8 只有客户端端点。D9 在 D8 原生四槽上定位早段慢出包，并独立检查晚段慢出包能否进一步积累接纳等待。

热试跑后依次运行 **端点 → 双窗 → 双窗 → 端点**，各轮新建引擎。固定同一张 8 请求/秒、60 条输入表、guarded、四个原生 runner 槽、2,048 token 批预算，核对 5,553 个实际 KV 块。无活跃槽或 KV 容量覆盖。热试跑按 D8 原规则选首个合格平台，测量轮保持原 110 秒热准入。四轮使用相同 T0 提前量并在实际 T0 检查设备样本。A 只收集客户端端点，B 补充原生请求端点及两段逐步事件。

B 在 T0 前空闲时只 arm 一次。逐步采集为 **[0, 3) 和 [6, 8.5) 秒**。单一有界记录保留同一请求代际和真实 token 序号。两段之间保留原生请求端点和序号，窗内发出的批次沿原生路径跟到消费。晚窗入口核对原生 FIFO，跨入批次的派发、执行和 runner 输入均标为缺证，只记录实际可见的消费与出包。排空后导出，若排空早于 8.5 秒，等到观察边界再导出。该等待独立于末终态和吞吐窗口。完整性审查核对原生调用顺序、时钟域、跨窗批次、token 顺序和排空状态。

六对热状态、两对同档和两对相邻 A/B 均独立复算。相邻性能对照以 A 为参考，另保留 D8 原配对方向。沿用 60/60 按时完成、原提交门槛、TTFT p95 的 2 ms 或参考轮 5% 中较大值、末终态请求和 token 吞吐各 ±5%。主配对越界时记采集等价性未建立。按后一个真实 token 到达时间分入 T0 对齐的 500 ms 窗，每窗至少 10 个间隔、两个请求，连续两窗中位数至少 30 ms。两窗分别核对 A 复现、B 快慢反差和同 ID 相邻 token 序号。至少两个不同批次具有可比的实际 runner 构成、缓存及队列状态，并由同一主机阶段解释相应的时间差，才支持阶段定位。积压起因要求该阶段早于持续待接纳。晚段放大还须有慢批延长槽位持有、实际释放、下一次接纳和等待增长的连续证据。关键逐步证据落在空档、采集等价性未建立或候选无法区分时，相关因果结论为 `unresolved`。设备归因需要独立关联的设备时间。

预算为 **1 次串行单 RTX 3090 分配，最多 20 分钟，无重试**。作业内预计 1,040 秒，硬上限 1,050 秒。预计由热试跑 150 秒、四轮热准入 440 秒、五个引擎其余开销 325 秒、核验清理 120 秒和双窗额外开销 5 秒构成。集群 CPU 预检核对上传输入、固定源码、原生调用以及双窗额外开销。每次提交仍需要 D9 对应批次授权与独立 watcher 就绪。原 C3 仍为 `blocked`，D9 仅用于诊断。

```bash
UV_OFFLINE=1 make check
python scripts/check_m5_d9.py --attempt PRIVATE_D9_ATTEMPT --proposal configs/m5-d9.json --output NEW_PRIVATE_AUDIT
python scripts/preflight_m5_d9_cpu.py --proposal configs/m5-d9.json --input PRIVATE_FROZEN_TABLE --scratch PRIVATE_CPU_OUTPUT_DIR
```

集群 CPU 入口使用现有 Linux Python 3.12.16，核对 47 个已安装 vLLM 源码哈希。它以源代码固定的 EngineCore/UniProc 方法和合成请求执行 ABBA，比较原生调用顺序和 960 个输出 token，计量双窗导出、MsgPack 往返编解码、JSON 写入和客户端标量复制。以更慢的 B 与更快的 A 估算两轮增量，再加上四轮共用 T0 提前量和两轮最大静默等待。该预检检验 CPU 开销预算。真实推理下的扰动由正式诊断的 ABBA 判定。

## D10 原四槽主机 CPU 隔离

从 D9 交付 `c3ac0c2` 起步。封存 D8、D9 两批 E1 的首个请求在实际提交至首 token 上与各自 E4 相差约 2–3 ms，首至末 token 分别延长约 152 和 127 ms。两批快轮均在第二个请求提交前已收到第 2 个 token，E1 在该时点仍未收到。到计划到达 3–6 秒的 24 个同 ID 请求，D8 E1/E4 提交至首 token 中位数为 850.507/222.230 ms，D9 为 547.685/210.374 ms。D9 两个快出包轮也在 4.825/4.773 秒出现持续原生待接纳。前段出包损失先出现，接纳等待会累积延迟。E1 缺少原生阶段数据，历史原件无法定位初始损失发生在 GPU 执行、主机推进还是传输。早段低功率出现在相近 SM 频率和 P2 下，1 Hz 样本无法证明因果。

只改变**作业内的 CPU 角色放置**。A 继承原 Slurm CPU 掩码。B 将客户端和采样器放在既有分配内的两个物理核心，EngineCore 和 UniProc worker 使用其余物理核心，禁止跨角色共用 SMT 兄弟。部署前核验亲缘、PID 代际、物理拓扑和可用 CPU。所有轮次保持原生四槽、8 请求/秒、60 请求、实际 5,553 个 KV 块、2,048 token 批预算、原 guarded 策略、既有运行设置与热准入。热试跑采用原 D8 协议。测量四轮使用相同原生请求级标量端点与同一到达表，保存轮前轮后角色掩码及可用的进程调度计数。没有双窗逐步观察。采集的标量原生端点用于复算 3–6 秒的首次显著等待、持续待接纳及等待深度。

同一张卡热试跑后依次执行 B→A→A→B，四轮均使用新引擎。D8、D9 的 E1 都出现早段慢窗，先跑 B 可检验隔离能否覆盖这一位置，A 若未复现波动则证据不足。诊断有效性要求四轮 60/60、原提交及热状态门槛通过、CPU 放置证据完整。A/A 须复现原 TTFT 越界，至少一个 A 在 0–3 秒触发原双慢窗规则。两个 B 须在对应的合格窗低于 30 ms，B/B 通过原 TTFT p95 和两项吞吐容限。两个 B 对每个慢 A 的 TTFT p95 改善均须超过原参考轮容限，对任一 A 的 TTFT 和两项吞吐均须满足原回退容限。原参考轮 TTFT 容限为 2 ms 或 p95 的 5% 中较大值，两项吞吐容限各 5%。对照未复现、采集缺证或未达到预登记效果时保留原件和结论。端点同条件资格前置单独冻结，在采用 B 布局后的四个新端点轮上复核三对原门槛，并独立绑定原始证据后才可进入资格复验。D10 ABBA 本身不能放行。

最小 D10 预算为 **一次串行单 RTX 3090、最多 20 分钟、无重试**。按热试跑 150 秒、四轮热准入 440 秒、五个引擎其他工作 325 秒、核验清理 120 秒估算 1,035 秒，硬上限 1,050 秒，保留 15 秒余量。D10 通过后才安排同条件资格前置，预计另需一次同规格分配。原 C3 保持 `blocked`，原性能及资源门槛继续适用。
