# M5 部分校准交接

任务 ID 为 `M5-paired-performance`。已获批 6 次串行单 RTX 3090 分配，每次最多 20 分钟，合计 120 申请分钟。原六次额度用完后，D2 获批新增一次单卡 20 分钟。累计使用 7 次、140 申请分钟，Slurm 实际运行合计 46 分 36 秒。七次作业均已核对权威终态、全队列和资源清理。正式矩阵等待完成校准后另行授权。

| 记录 | 状态 |
| --- | --- |
| 起点 | M4 交付 `f4d94d3d22a967efeeb3a00bc13f6a6b8591835b` |
| 预检交付 | `9f68fdff372b78d784661703f7473567ad321219` |
| C1–C3 实际执行与独立分析 | `cb82627ef4b2f8d0a9f167b1ba5510781f22f29c` |
| D1 实际执行与独立分析 | `9b8a2e1ee84337e5998dbcfd2547735eab9cde58` |
| D2 实际执行与独立分析 | `3fcb3a53192ddf9e5d0d59003c69f027d73b28da` |
| 草稿交付 | 本页所在提交，使用草稿 PR 的完整 head SHA |
| 命令 | D2 执行候选的完整 `make check` 为 698 通过、3 项固定 Linux 原生 opt-in 跳过 |
| 证据 | C1–C3 的 27 个真实运行叶及 D1 的 8 个叶均已独立核对原始 collector、源码和复算结果。D2 的热试跑和三个实际启动的叶也已复核 |
| 行为与性能 | C1 机制通过，C2 到达率选择通过，C3 采集扰动门槛失败。D1 同档波动越界，D2 因第三轮热准入未完成，资格复验仍受阻。C4–C6、正式冻结与正式运行待完成 |
| M4 / M3 历史 | M4 原冻结 F 维持 FAILED，获批离线修订在总站范围内通过。M3 formal-02 为 UNVERIFIED_PURGED，原全历史门槛为 blocked |

| A–F | 当前已具备 | 尚需原件 |
| --- | --- | --- |
| A | C1–C3 实际固定 Linux 来源、模型和 KV 配置，独立绑定冻结 Git 树 | 正式输入表、正式冻结和实际执行 |
| B | 受控 A/B/C 因果链，原生短开放环筛选 | 全长自然缓存压力和负对照 |
| C | 16 组、64 模式轮次、32,000 请求的候选矩阵与配对审查器 | 完整同节点原始配对 |
| D | 真实 runner 区间、受控 prefill 及 C3 阶段计时的独立复算 | 合格轻量采集与完整正式逐轮原件 |
| E | 校准配对结果、失败分母和保留的扰动反例 | 正式性能、公平性、波动和成本结论 |
| F | 本地子进程控制和全部七次分配的终态及释放证明 | 正式原件控制、白名单发布与独立 A–F 签认 |

命令、角色预算、测量定义与重放步骤见 [M5 设计](m5-design.md)。生产调度代码由 M4 继承。原仓库用户修改的 `docs/workflows.md` 保护 SHA-256 为 `c3dca495872357a584ecc2464dc92dafa5e4467d16237f4c11dfdf9ab096bb82`。M1–M4 工作树及历史审查保留，PR #4–#8 与 M5 草稿保持未合并。

## C1 启动反例与复验范围

首个 C1 分配在预检交付 `9f68fdff372b78d784661703f7473567ad321219` 上运行 79 秒，Slurm 终态为 FAILED/1:0。首个引擎在 ZeroMQ 绑定时超过 107 字节 UNIX 套接字路径上限。诊断尚未进入模型执行。原始 stderr SHA-256 为 `bd6fdf1a8a67f719566c910d77629c318b1905d68c1ef2bc984c4f0c7d3a21ab`。独立监视、全队列、设备前后边界和进程清理均通过核查。

修订将固定版本 vLLM 的 IPC 套接字定向到计算节点上的短私有目录，在清理引擎进程后移除。离线用真实 UNIX 套接字验证长路径拒绝、短路径绑定和目录清理，并验证资源门槛拒绝 IPC 残留。实际复验从完整 C1 开始，按原校准矩阵和门槛执行。首次失败保留在六次授权额度内。

C1 第二次在 `9b81f953daea20650f40af8379af263b20843359` 上运行 69 秒，Slurm 终态为 FAILED/1:0。短路径修复使首个 native 压力叶完成真实生成并采齐。该叶的 `pending_batches` 是空列表，旧边界审查误要求整数零。原始 client 和 engine SHA-256 分别为 `f85f35ad1d8b906547bb776f0b98017b5907a3c0840ff2b8cd29bb06ff45a2d6`、`f993dd11b4ea9cf60c2bfa94b7bebbb00a44b46c1b49b27bcd4aa4aedab949f0`。

审查修订按生产者的列表契约验证空值，以非空列表作为负控。封存首叶离线通过边界、runner、运行配置和机制链复算，仍保留原分配的失败状态。新冻结需重新完成全部 C1 叶。

## 已完成校准和阻断

第三次 C1 完成全部八个诊断叶并通过，随后 C2 完成全部十个短到达率 pilot 并通过。C3 的九个原始运行和逐叶审查均完成，整阶段因三组配对的 TTFT p95 扰动超过冻结容限而阻断。两组 mixed 的差值方向相反，现有数据不足以分离采集扰动和运行波动。独立复核从原始数据重算得到同一判定。C4–C6 保持待执行，正式验收与数值发布继续受原门槛约束。

| 校准阶段 | 阶段状态 | 私有阶段结果 SHA-256 |
| --- | --- | --- |
| C1 | qualified | `ca2ee18703757e708ee571565306f6a5210f036ad74c449d07079865efd27883` |
| C2 | qualified | `9ad8fce6fca69314873682d8fd07416995cba037f2704cafb56c54dbc18e945a` |
| C3 | blocked | `e8618fde6793d2c2d749e32b7f8052a495f695ab1ddc68dc015ac58012495866` |

私有交接入口为 `runs/M5-paired-performance/calibration-partial/summary.json`，包括逐次资源账本、校准数值与原件哈希。正式冻结和正式公开数值出口等待校准通过。

## D1 诊断完成，资格复验受阻

D1 使用最后一次获批分配，独立核验 Slurm 终态为 COMPLETED/0:0，实际运行 442 秒。封存的同一张 mixed 表由 guarded 模式重放八次，完成全部 480 条请求。两组 E/E 和两组 L/L 均超出原 TTFT 容限，四组 E/L 有一组同时通过 TTFT 与吞吐门槛。相同采集档位已出现足以越界的波动。边界设备读数与顺序共变，测量窗口内的根因仍待验证。按预定规则，D1 的进入资格复验判定为否，原 C3 的 blocked 结论和门槛保留。

私有原始结果入口为 `runs/M5-paired-performance/attempts/d1-01/`。独立复算、逐对分母、资源账本和带外设备证据在 `runs/M5-paired-performance/d1-independent-review/analysis.json`，SHA-256 为 `2e2824a94e58bea9eb9c044bbc63e882e91f10d8d1046fba90edf0035d377b13`。执行和分析可从 D1 冻结 SHA 运行 `scripts/check_m5_d1.py`，传入私有作业、既有方案和新的输出路径。

## D2 部分完成，热状态与 E/E 整体验证缺失

D2 使用单独获批的一次单卡分配。Slurm 终态为 COMPLETED/0:0，实际 510 秒，释放审查通过。独立热试跑成功冻结目标。E1/E2 通过热状态可比性和原 TTFT、吞吐门槛。E3 在预定的 75 秒内没有满足 20 秒稳定窗口的温差门槛，按方案停止计时，E4 未启动。E3/E4 与 E1/E4 缺少比较数据，整组热状态与受控 E/E 稳定性均未验证。原 C3 的 blocked、原 TTFT 和吞吐门槛保留，资格复验未启动。

D2 私有原件在 `runs/M5-paired-performance/attempts/d2-01/`，独立结果在 `runs/M5-paired-performance/d2-independent-review/analysis.json`，SHA-256 为 `0f369e956dd62373a310d467054d684e371b63303662563b7eba790ed12400ba`。E3 热窗口补充复算 SHA-256 为 `097b91cdfaf6a6311250141abb25a9b6cb6f622db3fa8ad75c647f546f997eeb`。从 D2 执行冻结运行 `scripts/check_m5_d2.py`，传入私有作业、原方案和新的输出路径，可重放同一审查。

## 离线修订验收

离线实现与验证 SHA 为 `b641d64a283af5318a9bcf628717194f5fd60112`。新冻结启用 M5 测量和资源 schema 2。候选决策、淘汰账本、全冷零命中和资源归属按逐条证据核验。正式吞吐配对使用共同计划窗口，调度外层计时只在 profile 采集。发布重新审查正式原件并绑定独立签认、正负控和资源记录，按显式白名单生成材料。校准原件在正式冻结阶段独立复算，发布阶段绑定校准报告与冻结表哈希。

| 核验 | 结果 |
| --- | --- |
| 完整本地命令 | `UV_OFFLINE=1 make check` 通过。759 项测试通过，3 项固定 Linux 原生 opt-in 跳过。Ruff、脚本语法、已有证据校验和打包通过。默认缓存刷新在本地网络权限下失败，复跑使用已有缓存。 |
| 真实子进程 | 封存输入专项为 3 项正控、5 项负控，按预期退出。另有 27 份 schema 2 派生正控覆盖 C1–C3 的 1,156 条请求。派生字段用于本地验证审查契约。正式发布控制仍需完整正式原件。 |
| 封存原件 | 按各自执行 SHA 重放 C1–C3、D1、D2，五份结果 SHA-256 与历史记录相同。1,012 份原件哈希一致，7 次分配的终态与释放证据仍通过。 |
| 实验资格 | 原 C3 仍为 `blocked`。TTFT p95 绝对差仍须在 `max(2 ms, 参考轮 p95 × 5%)` 内，请求与 token 吞吐仍各在 ±5% 内。D2 整组仍未验证。D2-R 待单独审批，正式运行仍为 0/64 轮。 |

私有验收入口为 `runs/M5-paired-performance/offline-repair-review/make-check-offline/`。`manifest.json` SHA-256 为 `66ec13f1ec9714e372a6a1ef6a4ed686100da720d1ab9b6dec9445a63b83b98c`，`acceptance.json` SHA-256 为 `b0735bba7146c5778eee3441c5f7ee701a71c246ee09164224d5d892058e9bf8`。同级目录保存历史复算脚本、逐条原件哈希、子进程退出和私有记录。可在该离线实现 SHA 上复查。

```bash
UV_OFFLINE=1 make check
python scripts/check_run_evidence.py runs/M5-paired-performance/offline-repair-review/make-check-offline --require-command-success --require-accepted
python scripts/check_m5_resources.py runs/M5-paired-performance/attempts --history
```

本次新增 GPU 作业为 0，累计资源账本仍为 7 次、140 申请分钟。草稿 PR #9 保持待总站验收。D2-R 仍按前述热状态门槛另行审批，诊断完成前资格复验保持受阻。
