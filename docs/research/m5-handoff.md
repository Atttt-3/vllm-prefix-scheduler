# M5 部分校准交接

任务 ID 为 `M5-paired-performance`。原校准获批 6 次串行单 RTX 3090 分配，每次最多 20 分钟。原六次额度用完后，D2、D2-R、D2-R110 与 D3 各获批新增一次单卡 20 分钟。累计使用 10 次、200 申请分钟，Slurm 实际运行合计 72 分 37 秒。十次作业均已核对权威终态、全队列和资源清理。正式矩阵等待完成校准后另行授权。

| 记录 | 状态 |
| --- | --- |
| 起点 | M4 交付 `f4d94d3d22a967efeeb3a00bc13f6a6b8591835b` |
| 预检交付 | `9f68fdff372b78d784661703f7473567ad321219` |
| C1–C3 实际执行与独立分析 | `cb82627ef4b2f8d0a9f167b1ba5510781f22f29c` |
| D1 实际执行与独立分析 | `9b8a2e1ee84337e5998dbcfd2547735eab9cde58` |
| D2 实际执行与独立分析 | `3fcb3a53192ddf9e5d0d59003c69f027d73b28da` |
| D2-R 实际执行与独立分析 | `e4cfe667f937d09652ea9f06ec06e7d66a69c3e9` |
| D2-R110 实际执行与独立分析 | `9f8637dd23dbb4b26eea71bbb7b5f842aeaab7ca` |
| D3 实际执行与独立分析 | `f4a492872f7aac917b5d01f37a3f0de97c814f47` |
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
| F | 本地子进程控制和全部十次分配的终态及释放证明 | 正式原件控制、白名单发布与独立 A–F 签认 |

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

离线修订新增 GPU 作业为 0。其验收时累计资源账本为 7 次、140 申请分钟。

## D2-R 热准入未通过

D2-R 单独授权的新增分配已使用。作业 `140048` 为单 RTX 3090、申请 20 分钟，Slurm 终态 `COMPLETED/0:0`，实际运行 289 秒。提交前 watcher ready、全队列查询、独立终态及 watcher 退出均有记录。GPU 边界占用 1→1 MiB，进程、采样器和引擎 IPC 已清理。固定执行提交的本地 `UV_OFFLINE=1 make check` 为 773 项通过、3 项固定 Linux 原生 opt-in 跳过。

热试跑在 80.139 秒冻结目标 64°C、SM 1875 MHz、显存 9501 MHz。E1 在 95 秒内完成 10 个完整预热波次。至最后一次排空和 APC 重置，可用的 61 个采样窗口中，最小温差为 3°C，高于 2°C 上限。另有两段温差 2°C 且满足其余热采样条件的窗口，结束于约 93.957、94.958 秒，均出现在尚未完成排空及重置的第 11 波。按冻结规则未纳入准入。实际清理结束晚于固定截止约 2.608 ms。E1 未进入计时，E2–E4 未启动。

| 配对 | 实测请求数 | 测量期热可比性 | TTFT p95 与吞吐 |
| --- | --- | --- | --- |
| E1/E2 | 0 / 0 | 缺数据 | 无实测分母 |
| E3/E4 | 0 / 0 | 缺数据 | 无实测分母 |
| E1/E4 | 0 / 0 | 缺数据 | 无实测分母 |

独立脚本从逐条客户端与设备记录复算，与冻结审查器的配对缺项和总体判定一致。诊断状态为 `diagnostic-incomplete`，测量期间热状态控制和 E/E 稳定性均为 `unverified`。原 C3 保持 `blocked`，原热门槛、TTFT p95 的 `max(2 ms, 参考轮 p95 × 5%)` 容限和请求及 token 吞吐的 ±5% 容限保留。资格复验未启动，正式矩阵为 0/64 轮。

私有原件索引为 `runs/M5-paired-performance/attempts/d2-r-01/`。运行根目录中的 `d2-r-review/` 保存预检、释放账本及独立复算。`analysis.json` SHA-256 为 `67ef91b0052f59289ec648f84ed88ce161046e737cc54fdf2668ff51062beb72`，`raw-independent-compared.json` 为 `b44da57b53e9a860ae16e1f6ea5f314ed731670fa5941b90603f541dba584201`，`e1-thermal-windows.json` 为 `f1abcb98b588ef3e37c8286546b9e807765d8d79db0d9b00c90bc0657530d930`。在执行 SHA 上复放时使用新的输出路径。

```bash
python scripts/check_m5_d2.py --attempt runs/M5-paired-performance/attempts/d2-r-01 --proposal configs/m5-d2r.json --output runs/M5-paired-performance/d2-r-review/replayed.json
python scripts/check_m5_resources.py runs/M5-paired-performance/attempts --history
```

## D2-R110 热可比性通过，同档性能越界

作业 `140049` 使用本批唯一一次单 RTX 3090 分配，申请 20 分钟，实际 670 秒，Slurm 终态为 `COMPLETED/0:0`。提交前独立 watcher ready、权威终态、全队列与 watcher 退出均已核验。GPU 边界为 1→1 MiB，进程、采样器与 IPC 清理通过。执行及审查 SHA 为 `9f8637dd23dbb4b26eea71bbb7b5f842aeaab7ca`。预检的完整 `UV_OFFLINE=1 make check` 为 801 项通过、3 项固定 Linux 原生 opt-in 跳过。

热试跑冻结目标为 64.5°C、SM 1,860 MHz、显存 9,501 MHz。四轮共 240 条全部完成并按时取得首 token。测量窗口温度中位数为 65–66°C，SM 中位数为 1,860–1,890 MHz，显存众数均为 9,501 MHz。三对热状态均可比。

| 轮次 | 实测请求 | 95 秒准入检查点 | TTFT p95 | 末终态请求/秒 | 末终态 token/秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| E1 | 60/60 | 60.423 秒 | 635.009 ms | 7.282 | 116.511 |
| E2 | 60/60 | 72.195 秒 | 651.420 ms | 7.264 | 116.227 |
| E3 | 60/60 | 55.358 秒 | 790.618 ms | 6.875 | 109.996 |
| E4 | 60/60 | 81.979 秒 | 621.604 ms | 7.343 | 117.491 |

| 配对 | 热状态 | TTFT p95 差值 | TTFT 容限 | 请求及 token 吞吐差 | 判定 |
| --- | --- | ---: | ---: | ---: | --- |
| E1/E2 | 可比 | +16.412 ms | 31.750 ms | −0.244% | 通过 |
| E3/E4 | 可比 | −169.014 ms | 39.531 ms | +6.814% | 两项越界 |
| E1/E4 | 可比 | −13.405 ms | 31.750 ms | +0.841% | 通过 |

95 秒复算仅使用截止前完整排空、清 APC 和判定的波次。四轮都通过此预热准入检查。独立脚本与冻结审查器对热状态、配对统计和 95 秒判定完全一致。诊断完整，受控 E/E 仍不稳定。原 C3 保持 `blocked`，原热状态、TTFT p95 和吞吐门槛保留。用本次原件验证了资格入口拒绝继续。资格复验未启动，正式矩阵为 0/64 轮。

私有原件在 `runs/M5-paired-performance/attempts/d2-r110-01/`。`runs/M5-paired-performance/d2-r110-review/` 保存资源账本、冻结审查和独立复算。`analysis.json` SHA-256 为 `27b389a03dfa8df4f9a6b531663e9767910de6f51ffbceef8006f5039cfdd844`，`raw-independent-compared.json` 为 `a75ed5a74734f1962382ae77d9ed7db28d42cbf6acaa39348bd92047ff1c9498`，`resource-ledger.json` 为 `7e886b98ad98c0a87d47681c91bfda06358dc3d845003f05ec1a08c2d91a7d58`。在执行 SHA 上复放时使用新的输出路径。

```bash
python scripts/check_m5_d2.py --attempt runs/M5-paired-performance/attempts/d2-r110-01 --proposal configs/m5-d2r110.json --output runs/M5-paired-performance/d2-r110-review/replayed.json
python scripts/check_m5_resources.py runs/M5-paired-performance/attempts --history
```

## D3 同档积压复现，入队至首次接纳为主要差异区间

作业 `140050` 使用本批唯一一次单 RTX 3090 分配，申请 20 分钟，实际 602 秒，Slurm 终态 `COMPLETED/0:0`。独立 watcher 提交前就绪、权威终态、全队列及退出记录均已核验。GPU 边界为 1→1 MiB，进程、采样器与 IPC 清理通过。执行与审查 SHA 为 `f4a492872f7aac917b5d01f37a3f0de97c814f47`。本地 `UV_OFFLINE=1 make check` 为 819 项通过、3 项固定 Linux opt-in 跳过。登录节点的完整导入探针超时，已安装源码的隔离 CPU 语义试验及同域时钟往返试验通过，计算节点四轮均通过实测来源绑定和前后时钟核验。

四轮共 240 条请求全部完成并按时取得首 token。240 条均有完整原生首 token 和终态分段，三个配对的热状态均可比。表中差值为后者减前者。

| 配对 | TTFT p95 | TTFT 差值 / 容限 | 请求及 token 吞吐变化 | 原门槛 |
| --- | ---: | ---: | ---: | --- |
| E1/E2 | 1,073.902 → 640.868 ms | −433.034 / 53.695 ms | +6.851% | 两项越界 |
| E3/E4 | 653.955 → 757.014 ms | +103.059 / 32.698 ms | −4.733% | TTFT 越界 |
| E1/E4 | 1,073.902 → 757.014 ms | −316.888 / 53.695 ms | +0.993% | TTFT 越界 |

E1–E4 的客户端未终态峰值为 11、8、8、9，已原生入队且尚未首次接纳的峰值为 7、4、4、5。E1/E2 最后 15 条同 ID 请求中，E2 的 TTFT 差值中位数为 −476.559 ms，入队至首次接纳的差值中位数为 −485.618 ms。这两个分位数分别计算。E1 的 p95 请求 `000059` 按同 ID 对照 E2，TTFT 多 667.359 ms，其中入队至首次接纳多 620.489 ms。

D3 的 E3/E4 方向与 D2-R110 相反，旧 E3/E4 同方向双越界形态未复现。D3 中 E4 的 p95 请求 `000057` 按同 ID 对照 E3，TTFT 多 609.826 ms，其中入队至首次接纳多 536.711 ms。两轮最后结束的请求均为 `000059`，E4 排空多 410.607 ms，其中入队至首次接纳多 175.013 ms、原生首至末输出多 173.967 ms。首次接纳和输出封包之间的区间用于定位经过时间。此次证据未采集实际 GPU runner 工作。

独立脚本与冻结审查器对全部配对和逐请求分段一致。受控热状态下同档波动仍存在，原 C3 保持 `blocked`，原热状态、TTFT 和吞吐门槛保留。资格复验未启动，正式矩阵为 0/64 轮。

私有原件在 `runs/M5-paired-performance/attempts/d3-01/`。`runs/M5-paired-performance/d3-review/` 保存原生源码预检、审查和释放账本。`analysis.json` SHA-256 为 `d7f18b84ac663ce7d6463cdf1ee8547bac0381b414216ba87faaa0db19b401a5`，`raw-independent-compared.json` 为 `08239f07d4c8aabd7c205d6a8982caecd163d247cc4b74e2415bb8e97bfee04a`，`resource-ledger.json` 为 `fcee4fd791c00aa6c6072eb3ef5104e6f9faaa1dcf516a77c51bea2e22b10506`。在执行 SHA 上复放时使用新的输出路径。

```bash
python scripts/check_m5_d2.py --attempt runs/M5-paired-performance/attempts/d3-01 --proposal configs/m5-d3.json --output runs/M5-paired-performance/d3-review/replayed.json
python scripts/check_m5_resources.py runs/M5-paired-performance/attempts --history
```

## D3 后的离线修订与 D4 方案

实现和本地预检 SHA 为 `dfc37b0e79bd9b19d51f223abf474a6f22be3497`。资格入口版本 2 固定已核验释放的历史前缀，累计预算为前缀实际分配加本批获准次数，失败仍占额度。新版只认事先登记的同条件后继角色，从其冻结执行 SHA 用真实子进程复算封存原件，核对独立摘要、三对原始判定及后继释放与资格批的时序。当前清单为空。D3 原结果、重算哈希后的虚假稳定结论及 D2-R110 的相同反例均被拒绝。正式冻结、提交、节点重放与发布共同验证实际到达表，绑定 C2/C6 参数及 C6 输入清单。

轻量采集在入队时保存有界原值，在排空后生成提示哈希和 KV 键文本。runner 使用原来的同流异步 GPU 至主机复制和原生物化同步点，在 pinned slot 复用前将真实位置保存到 16 MiB 有界主机缓冲，排空后核对每个位置、代际、切片和输出顺序。纯 CPU 专项测量使用原 C3 的 60 条请求、101 次交替重复。入队数据准备中位数由 1.169 ms 降至 0.112 ms，排空后格式化的结果一致。端到端采集扰动仍需重新校准。

| 本地门槛 | 结果 |
| --- | --- |
| `UV_OFFLINE=1 make check` | 841 通过，3 项固定 Linux 原生 opt-in 跳过。证据清单 SHA-256 `5ececa558dfd9f7b70a4ec63ec219f06e3d6f072b16970b0f02ce16d0cf4941b` |
| 真实子进程 | 正式表 8 项，封存资格 5 项，D4 方案 6 项。重算摘要后的语义错误仍被目标门槛拒绝 |
| 封存复算 | C1–C3、D1、D2、D2-R、D2-R110、D3 的 8 份报告逐字节一致。10 次分配的资源账本也一致 |
| 原结论及资源 | 原 C3 仍为 `blocked`，原热状态、TTFT、请求与 token 吞吐门槛保留。本轮 GPU 分配为 0，正式矩阵为 0/64 |

D3 四轮累计排队约 99.86%–99.88% 的时间与四条请求的原生首次接纳至末输出区间重合。这个区间只近似槽位占用。D4 冻结 8/6/6/8 请求/秒，每轮 60 条，一张卡串行，申请上限 20 分钟，估算 1,040 秒，硬上限 1,050 秒。方案 SHA-256 为 `d3b200bcda5a4017b40095c741d27b584eac9b1da6fc0a70b6a6806ecd5f075f`。两张表已离线验证。D4 在此次交付中处于方案冻结与输入预检状态，计算节点执行接线和实际测量待完成。私有索引为 `runs/M5-paired-performance/offline-repair2.9SCPGp/`。 [D4 设计与准备命令](m5-design.md#d4-到达率对照)

## D4 到达压力对照结果

D4 执行与冻结审查 SHA 为 `ddc1c5e4e70305fe6315151102d88b90959f6722`。作业 `140051` 使用一次串行单 RTX 3090 分配，申请 20 分钟，实际 658 秒，权威终态 `COMPLETED/0:0`。提交前集群 Python 用已上传材料复算五个叶的顺序，独立 watcher 已就绪。全队列查询、设备前中后记录、进程、采样器和 IPC 清理均通过，GPU 边界为 1→1 MiB。累计 11 次、220 申请分钟，实际合计 5,015 秒。完整 `UV_OFFLINE=1 make check` 为 866 项通过、3 项固定 Linux opt-in 跳过。集群预检核对七个固定 vLLM 文件和两个进程的 CLOCK_MONOTONIC 域，五次往返中最短 0.654 ms。计算节点每轮重新绑定源码、实际配置及时间域。

四轮各 60 条全部完成并按时取得首 token。提交偏差 p99 为 1.930–2.358 ms，六对测量窗口全部符合原热状态门槛。四轮各有 5,553 个实际 KV 块和四个有效序列槽位。每轮 22 条命中 512 tokens，38 条零命中，原生抢占计数均为零。吞吐以各轮首计划到达至末终态为分母。

| 轮 | 请求/秒 | TTFT p95 | 末 15 条接纳前等待中位数 | 待接纳峰值 | 末终态请求/秒 | 末终态 token/秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 | 8 | 652.367 ms | 345.136 ms | 4 | 7.323 | 117.171 |
| E2 | 6 | 188.658 ms | 36.762 ms | 1 | 5.831 | 93.290 |
| E3 | 6 | 127.016 ms | 0.020 ms | 1 | 5.878 | 94.042 |
| E4 | 8 | 640.718 ms | 444.593 ms | 4 | 7.257 | 116.113 |

| 同速率配对 | TTFT p95 差值 / 容限 | 请求及 token 吞吐变化 | 判定 |
| --- | ---: | ---: | --- |
| E1/E4，8 请求/秒 | −11.649 / 32.618 ms | −0.903% | 通过 |
| E2/E3，6 请求/秒 | −61.642 / 9.433 ms | +0.806% | TTFT 越界 |

两轮 6 请求/秒的尾部等待中位数均低于两轮 8 请求/秒，待接纳峰值也均更低，支持该冻结到达表的压力解释。8 请求/秒两轮各约 99.87% 的等待累计量与四条原生首次接纳至末输出区间重合，此区间用于近似占用。6 请求/秒仍有同档 TTFT 波动。E2 的 p95 请求 `000056` 按同 ID 对照 E3，TTFT 多 58.801 ms，其中入队至首次接纳多 60.504 ms。两轮 p95 请求 ID 不同，不能把该同 ID 分段当作整体 p95 的分解。D4 受控同档稳定性仍为 `unstable`，原 C3 保持 `blocked`，原热状态、TTFT 和吞吐门槛保留。资格未启动，正式矩阵为 0/64。

下一次诊断可在同一张 8 请求/秒表上隔离准入上限。固定 `max_num_seqs=5`、相同实际 KV 块数和 2,048 token 批预算，仅让 `max_num_active_seqs` 按 4/5/5/4 变化。固定版本 vLLM 分别配置 runner 槽位和准入上限。先核验两档的同档重复满足原热状态、TTFT 和吞吐门槛，再判断两个 5 槽轮次的尾部等待和 TTFT 是否均低于两个 4 槽轮次，且吞吐降幅在 5% 以内。4 槽未复现积压时保留诊断的解释边界。生产与正式基线继续使用现有配置。

私有原件在 `runs/M5-paired-performance/attempts/d4-01/`，预检与审查在 `runs/M5-paired-performance/d4-review/`。主审查 SHA-256 为 `47b27cdb2ca50066ae0a18940fb07930417345611aae8dea9e3ca09add5c90fe`，独立对照复算为 `0f6793014d0a6b36cfcbb73143992aa500a33e326a42ef1af7b08070c48a5d36`，资源账本为 `89ba81c5e827a644939ef02c3f8a1de3478261123f1b13a237e7235ed93c773c`。两条审查路径对全部逐请求分段、两组同速率统计、六对热状态和压力判定一致。在执行 SHA 上复放时使用新的输出路径。

```bash
python scripts/check_m5_d2.py --attempt runs/M5-paired-performance/attempts/d4-01 --proposal configs/m5-d4.json --output runs/M5-paired-performance/d4-review/replayed.json
python scripts/check_m5_resources.py runs/M5-paired-performance/attempts --history
```

## D5 准入上限对照预检

D5 执行候选 SHA 为 `581c50b7d43480f825b7948f04064fa59e3dbce2`，方案 SHA-256 为 `dc4817bb421c774e2730aba541bf4752d946cb4a3d05a0edce093fdb1df85d1f`。D4 的压力信号支持在固定 8 请求/秒下隔离准入上限。D5 热试跑用 4 个活跃槽，测量按 4/5/5/4，每轮 60 条。所有引擎固定 5 个 runner 槽位、2,048 token 批预算与显式覆盖的 5,553 个 KV 块。执行入口会在预热前核验四项实际配置。D3 热准入和原 TTFT、吞吐门槛保留。

完整 `UV_OFFLINE=1 make check` 为 896 项通过、3 项固定 Linux opt-in 跳过。预检清单 SHA-256 为 `726f8fc612916c903f9a9185f22551ce5961167c987b74679c48e7b553d64640`。真实子进程接受封存输入和 4/5/5/4 顺序，拒绝篡改的方案、继承热计划、到达输入和前序报告。离线模拟引擎控制验证槽位、活跃上限、token 预算及实际 KV 块不匹配时在预热前停止。集群已安装源码哈希与 D4 一致，CPU 子进程用冻结 SHA 和实际上传材料核对热试跑及四轮配置。D4 在其执行 SHA 上逐字节复算，11 次历史资源释放账本与原件一致。

D5 预算为一次串行单 RTX 3090，最多 20 分钟，预计 1,040 秒，作业内硬上限 1,050 秒。显式 KV 覆盖替代自动容量估计，真实五槽位容量须在热试跑中验证。本轮新增 GPU 分配为 0，D5 尚无实测数据，资格复验未启动，正式矩阵为 0/64，原 C3 继续 `blocked`。私有材料位于 `runs/M5-paired-performance/d5-review/`。 [冻结规则与命令](m5-design.md#d5-隔离活跃准入上限)

## D5 热准入停止与资格准备

D5 作业 `140052` 在执行 SHA `581c50b7d43480f825b7948f04064fa59e3dbce2` 使用一次串行单 RTX 3090 分配，申请 20 分钟，实际 292 秒，权威终态 `COMPLETED/0:0`。独立 watcher 提交前就绪，终态、全队列查询、GPU 前中后、采样器、进程与 IPC 清理均核验。GPU 边界为 1→1 MiB，累计 12 次、240 申请分钟、5,307 秒实跑。冻结启动检查在热试跑和 E1 均核对了实际 5 个 runner 槽位、4 个活跃准入槽、2,048 token 批预算与 5,553 个实际 KV 块。

热试跑在约 37°C 启动预热，8 个波次后选中 62°C、SM 1,875 MHz、显存 9,501 MHz 的平台。E1 在约 50°C 启动预热，110 秒内完成 12 个预热波次，末次完整判定在 106.987 秒。第 13 波于 107.007 秒开始，在截止前未完成排空和清 APC。独立复算按截止前实际可见的 77 个候选窗口检查。60 个窗口的温度跨度超过原 2°C 门槛，其余 17 个满足基础平台、采样和时钟条件，温度中位数均为 65–66°C，超过试跑目标允许的 60–64°C。没有合法窗口通过完整准入。

E1 未开始正式计时，E2–E4 未启动。两组同档和四组跨档对照均缺测，TTFT、尾部等待及吞吐的 4→5 槽变化为 `unverified`。独立脚本与冻结审查器一致判为 `diagnostic-incomplete`。本次观察支持优先复核热试跑与后续引擎的起始热状态是否可匹配，再冻结新的运行方案。原 20 秒热窗口、±2°C 目标温差、110 秒准入及性能门槛仍按本次冻结执行和审查。

资格侧已生成 C1、C2、C3-burst、C3-mixed、C4、C5、C6 的七份离线作业清单。旧 C2 速率只用于排程，正式资格仍须重测。前置条件仍为预登记、独立复算通过的同条件 E/E 证据，四轮各 60 条，三个原配对全部通过热状态、TTFT 和两项吞吐门槛。封存 D3 的原始配对仍被拒绝，D5 诊断角色也被资格门槛排除。资格 GPU 作业未提交，原 C3 保持 `blocked`，正式矩阵仍为 0/64。若一次新的同条件前置试验和七次资格作业均一次通过，最少还需 8 次单卡分配、160 申请分钟。此前需冻结可审查的热状态控制方案与对应预算。

私有原件位于 `runs/M5-paired-performance/attempts/d5-01/`。主审查、独立对照、资源账本和热窗口分解位于 `runs/M5-paired-performance/d5-review/`。主审查 SHA-256 为 `b321f08ab51d16ffedf0cbeb0cf5bb9fe2f8e926eb712586ecd5e7c398bf9aa6`，独立对照为 `c913bfff3bc2ddde9f6192fa266efe16b726c86e10fb16a08dc456aca92e4320`，资源账本为 `9bc32f3c8ff796ba690a7075912ad4f9e428264f08bfafe17506460183bba2f6`。资格离线预检与 D5 释放后的前缀复核位于私有 `qualification-readiness/`。复算入口与冻结方案一致，使用新的输出路径。

## D5-R 暖目标预检

D5-R 执行候选 SHA 为 `42b83882a4a2045d3671c845479f08ef67ac76fd`，方案 SHA-256 为 `0bbda0693f6bffd9eff82c963ca9f06d41d17d83d3fd6f9009ada302852d4d9a`。执行端和独立审查端在完整波次排空、清 APC 并可见的原遥测窗口内，分别选取首个温度中位数在 64–66°C 的原规则合格平台。热试跑和测量仍使用 150 秒及 110 秒上限。D5 的 4/5/5/4、容量、六对热窗口、同档与跨档门槛全部继承。

完整 `UV_OFFLINE=1 make check` 为 924 项通过、3 项固定 Linux opt-in 跳过，预检清单 SHA-256 为 `ec88702f1eb5287ab84b36a06f18c77c0f219a756101a81e2ecf1fa94d5f4898`。专项子进程的 27 项检查覆盖温度边界、首个合格平台、遥测可见边界、超时、错序、伪造目标及独占授权。使用真实封存材料的子进程接受原始来源与五个叶规格，并拒绝八类改动。节点 CPU 从实际上传文件独立复核试跑与四轮规格，以及 47 个安装源码哈希。D5 在其原执行 SHA 上逐字节复算，十二次历史分配的资源账本与原件一致。

封存热数据的两条检查路径均拒绝旧试跑的 62°C 目标。将目标设为 64、65 或 66°C 的反事实复算均在历史 E1 第八个完整波次、70.764 秒通过，其余准入条件沿用原值。新试跑能否形成带内平台及准入优化效果仍待 GPU 实测。

本轮新增 GPU 分配为 0。D5-R 等待独立的 1 次串行单 RTX 3090、最多 20 分钟授权，预计 1,040 秒，作业内硬上限 1,050 秒。原 D5 保持 `diagnostic-incomplete`，原 C3 保持 `blocked`，资格 GPU 复验未启动。私有预检及封存复算位于 `runs/M5-paired-performance/d5-r-review/`。 [冻结规则、提交和复放入口](m5-design.md#d5-r-热目标修复)

## D5-R 热目标迁移与准入对照结果

作业 `140054` 在执行与冻结审查 SHA `42b83882a4a2045d3671c845479f08ef67ac76fd` 使用一次串行单 RTX 3090 分配，申请 20 分钟，实际 634 秒，Slurm 权威终态为 `COMPLETED/0:0`。独立 watcher 提交前就绪并正常退出。全队列查询、设备前中后、采样器、进程与 IPC 清理均通过，GPU 边界为 1→1 MiB。历史累计 13 次分配、260 申请分钟、5,941 秒实跑。

热试跑在第 10 个完整波次、87.016 秒冻结首个合格目标 64°C、SM 1,875 MHz、显存 9,501 MHz。E1–E4 分别在 72.057、80.660、49.498、62.314 秒完成热准入，预热平台中位温度为 65、66、63、65°C。四轮 240 条请求全部完成且按时取得首 token，实际配置逐轮核验。测量窗口中位温度为 65、66.5、65、66°C，SM 中位数均为 1,860 MHz，显存时钟众数均为 9,501 MHz。六对热窗口全部可比。按 E1→E2、E1→E3、E1→E4、E2→E3、E2→E4、E3→E4 顺序，测量窗口温差分别为 +1.5、0、+1.0、−1.5、−0.5、+1.0°C，SM 差均为 0 MHz，显存时钟众数均一致。热目标在本次作业迁移到全部四轮。

| 轮 | 活跃槽 | TTFT p95 | 末 15 条接纳前等待中位数 | 待接纳峰值 | 末终态请求/秒 | 末终态 token/秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 | 4 | 1,526.688 ms | 1,368.575 ms | 11 | 6.525 | 104.400 |
| E2 | 5 | 156.868 ms | 21.758 ms | 1 | 7.720 | 123.519 |
| E3 | 5 | 134.214 ms | 0.026 ms | 1 | 7.491 | 119.850 |
| E4 | 4 | 677.311 ms | 442.076 ms | 4 | 7.236 | 115.773 |

| 同档配对 | TTFT p95 差值 / 容限 | 请求及 token 吞吐变化 | 判定 |
| --- | ---: | ---: | --- |
| E1→E4，4 槽 | −849.377 / 76.334 ms | +10.893% | TTFT 和吞吐越界 |
| E2→E3，5 槽 | −22.654 / 7.843 ms | −2.971% | TTFT 越界 |

| 跨档配对，5 槽减 4 槽 | 末 15 条等待中位数差 | TTFT p95 差 | 请求及 token 吞吐变化 |
| --- | ---: | ---: | ---: |
| E1→E2 | −1,346.817 ms | −1,369.820 ms | +18.313% |
| E1→E3 | −1,368.549 ms | −1,392.474 ms | +14.798% |
| E4→E2 | −420.318 ms | −520.443 ms | +6.691% |
| E4→E3 | −442.050 ms | −543.097 ms | +3.521% |

两个 4 槽轮次都复现积压，四个跨档对照均满足尾部等待、TTFT 与吞吐方向和损失门槛。同档波动仍越界，因此冻结优化判定为 `unverified`，保留全部跨档观察。E1 p95 请求 `000054` 按同 ID 对照 E4，TTFT 减少 880.284 ms，其中原生入队至首次接纳减少 946.445 ms。此同 ID 分解与整轮 p95 差值分别计算。独立复算与冻结审查对逐请求时钟、配对、门槛和释放一致。

原 D5 保持 `diagnostic-incomplete`，原 C3 保持 `blocked`。原热状态、TTFT 和吞吐门槛保留。资格复验未启动，正式矩阵为 0/64。

私有原件在 `runs/M5-paired-performance/attempts/d5-r-01/`。主审查、独立对照、资源账本在 `runs/M5-paired-performance/d5-r-review/`。主审查 SHA-256 为 `bade7e99c3fb214f8545a1285df28a148f2062b1f6fa64d2d410aeeb55130882`，独立对照为 `4c3ad68e8763b2216871039acb8ff0f1a59ee736d9fc471c4aaa6b22f692449f`，资源账本为 `ccf7953a09d039e8e3d006e04d2818c30231d62a0349c29d262017735ac822dc`。复算在执行 SHA 上使用新的输出路径。

```bash
python scripts/check_m5_d2.py --attempt runs/M5-paired-performance/attempts/d5-r-01 --proposal configs/m5-d5r.json --output runs/M5-paired-performance/d5-r-review/replayed.json
python scripts/check_m5_resources.py runs/M5-paired-performance/attempts --history
```
