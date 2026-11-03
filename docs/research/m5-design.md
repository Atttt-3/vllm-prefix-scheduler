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

独立复算验证 D2-R110 受控 E/E 稳定后，资格批另行授权七次。角色顺序为 C1、C2、C3-burst、C3-mixed、C4、C5、C6，新增 140 申请分钟。七个角色使用同一执行 SHA。C3 每场景固定两张 60 条到达表，分别按 E L L E、L E E L 启动新引擎。同块比较须复用同一输入哈希。每场景复算四对 E/L、两对 E/E、两对 L/L，沿用原门槛。C3-mixed 另带阶段计时。常规预热、APC 预置和带外采样与正式轮一致。温度和频率用于解释波动。任一同档或跨档对照越界时保留原始结果，后续阶段受阻。

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

`freeze_m5.py` 从 schema 2 collector 和 Git 对象核对源文件与原始输入，再独立复算六个校准作业并绑定正式表。分析使用同一冻结代码，实际执行、分析和文档交付 SHA 分别记录。正式导出先用 `export_m5.py review` 核对完整配对、来源和全作业释放，再用 `check_m5_release_controls.py` 从封存原件运行子进程正负控。负控按指定的退出码和拒绝原因核验。独立审查人核对 A–F、审查包与控制记录，并通过协调渠道交付签认文件 SHA。发布时重新从封存原件复算，核对签认和控制记录绑定后再写出白名单。逐轮表包含相对原生的配对差以及 guarded−prefix。报告中位数、范围、方向和失败分母，负对照为描述性结果。最终区分命令、证据、行为与性能状态，给出实际执行、分析和交付 SHA、原件索引及资源账本。

```bash
python scripts/export_m5.py review --config configs/m5-performance.json --freeze runs/M5-paired-performance/frozen/formal-freeze.json --history runs/M5-paired-performance/attempts --analysis-commit ANALYSIS_SHA --output runs/M5-paired-performance/formal-review
python scripts/check_m5_release_controls.py --config configs/m5-performance.json --freeze runs/M5-paired-performance/frozen/formal-freeze.json --history runs/M5-paired-performance/attempts --review-dir runs/M5-paired-performance/formal-review --output runs/M5-paired-performance/formal-controls
python scripts/export_m5.py release --config configs/m5-performance.json --freeze runs/M5-paired-performance/frozen/formal-freeze.json --history runs/M5-paired-performance/attempts --review-dir runs/M5-paired-performance/formal-review --decision PRIVATE_INDEPENDENT_DECISION --approved-decision-sha256 DECISION_SHA_FROM_REVIEWER --controls runs/M5-paired-performance/formal-controls/controls.json --output evidence/M5-paired-performance
```

## D4 到达率对照

[D4 冻结方案](../../configs/m5-d4.json) 检验 D3 尾部积压是否随到达压力变化。固定生产 guarded 策略、端点采集、4 个序列槽位及同一张 mixed 已分词输入。沿用 D3 绑定的原生时间戳、时钟域验证、缓存预置和 8 请求/秒热试跑。四个测量引擎都沿用 110 秒热准入、20 秒热窗口和原采样与准入门槛。按 8、6、6、8 请求/秒启动四个新引擎，每轮 60 条。8 请求/秒使用封存表原字节。6 请求/秒按原 mixed 绝对时间及有界抖动算法重算到达偏移，保留请求顺序、token、冷热标签、采样和预热。两表的哈希和末到达时刻在方案内固定。

E1/E4 和 E2/E3 分别做同速率配对，均以较早轮为参考，沿用原 TTFT p95 与末终态请求和 token 吞吐容限。逐轮报告最后 15 条的原生入队至首次接纳中位数、待接纳峰值、等待累计和与四个原生输出区间重合的比例。四轮完整、提交与时钟证据合格、所有测量窗口两两热可比，才判读跨速率信号。只有两轮 6 请求/秒的尾部等待中位数均低于两轮 8 请求/秒且待接纳峰值均不更高，才记为本批支持到达压力解释。原生接纳至末输出区间仅用于近似占用，真实准入原因与 GPU 执行仍需各自证据。历史 C2 的原生 6 请求/秒未形成新鲜候选竞争，因此 D4 只用于诊断，正式负载继续由 C2/C4/C5 判定。

申请一次串行单 RTX 3090，最多 20 分钟。D3 的估算为 1,035 秒，两轮降速增加的绝对到达窗口共约 4.917 秒，向上取整后估算 1,040 秒，硬上限仍为 1,050 秒。保留原 watcher、权威终态、全队列和清理协议。热准入或边界核验失败时保存原件并停止，额度内无重试。控制器执行角色、提交前检查和实际作业冻结需随 D4 运行材料绑定。D4 两个同速率配对的结果只解释本次压力对照。C3 资格需同条件的新证据独立复算通过。

离线校验与私有输入准备命令使用 D3 原报告或字节一致的独立复算文件。

```bash
python scripts/prepare_m5_d4.py --plan configs/m5-d4.json --config configs/m5-performance.json \
  --history runs/M5-paired-performance \
  --d3-analysis runs/M5-paired-performance/d3-review/analysis.json \
  --output runs/M5-paired-performance/d4-prepared
```
