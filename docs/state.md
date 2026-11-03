# 技术状态与验证边界

## M5 部分校准

M5 从总站验收的 M4 交付 `f4d94d3` 建立独立工作树，增加固定负载、轻量原始采集、配对复算和释放门槛。校准 C1 与 C2 已通过，C3 因预先冻结的采集扰动门槛失败。D1 确认同档波动超出原门槛。新增一次 D2 完成首对 E/E，E3 未达到热准入，整组验证待补，资格复验仍受阻。C4–C6 待执行，正式预算等待校准完成后单独授权。当前指标与技术证据范围见 [M5 部分校准交接](research/m5-handoff.md) 和 [实验设计](research/m5-design.md)。

M5 离线修订 `b641d64` 已通过完整本地检查、子进程控制和历史复算。D2-R 在 `e4cfe66` 使用一次获批单卡分配，热试跑通过，E1 在 95 秒内未达到热准入，四个测量轮次均无计时数据。资源释放通过，测量期热状态及 E/E 稳定性仍未验证。原 C3 仍为 `blocked`，资格复验未启动。 [D2-R 交接](research/m5-handoff.md#d2-r-热准入未通过)

D2-R110 在 `9f8637d` 完成四轮共 240 条请求，三对测量窗口热状态可比，四轮均通过原 95 秒预热准入复算。E3/E4 的 TTFT p95 与吞吐同时越界，受控 E/E 仍不稳定。第九次分配已核验释放。原 C3 仍为 `blocked`，资格复验入口保持受阻。 [D2-R110 交接](research/m5-handoff.md#d2-r110-热可比性通过同档性能越界)

D3 在 `f4a4928` 完成 240 条带原生分段的请求并核验释放。三对热状态可比，三对均越过 TTFT 门槛，E1/E2 同时越过吞吐门槛。尾部同 ID 请求的主要差异落在原生入队至首次接纳。旧 E3/E4 同方向双越界形态未复现。原 C3 继续 `blocked`，资格复验未启动。 [D3 交接](research/m5-handoff.md#d3-同档积压复现入队至首次接纳为主要差异区间)

M5 后续离线修订 `dfc37b0` 已通过 841 项本地检查、子进程正负控与八份历史报告逐字节复算。新版资格入口复算同条件后继证据并绑定累计预算，正式各入口验证实际到达表，轻量采集推迟哈希和 runner 位置核验。D4 的 8/6/6/8 方案与输入已离线冻结，资格和正式矩阵继续受阻。 [离线修订交接](research/m5-handoff.md#d3-后的离线修订与-d4-方案)

D4 在 `ddc1c5e` 完成 8/6/6/8 请求/秒的四轮真实重放。六对测量窗口热可比，降速减轻该表的接纳前积压，6 请求/秒的同速率 TTFT 仍越过原门槛。第 11 次分配已核验释放，C3 继续 `blocked`，资格未启动。 [D4 交接](research/m5-handoff.md#d4-到达压力对照结果)

D5 在 `581c50b` 冻结活跃准入上限的单变量对照。固定 8 请求/秒、5 个 runner 槽位、2,048 token 批预算及 5,553 个实际 KV 块，测量按 4/5/5/4。完整本地检查与集群 CPU 输入预检通过，新增 GPU 分配尚未提交。 [D5 预检交接](research/m5-handoff.md#d5-准入上限对照预检)

D5 作业 `140052` 在 `581c50b` 通过五槽位启动和热试跑，E1 因 65–66°C 的稳定窗口偏离 62°C 目标而在正式计时前停止。四轮性能对照未验证，第 12 次分配已释放。资格七角色离线清单已生成，同条件前置证据仍缺，原 C3 保持 `blocked`。 [D5 结果](research/m5-handoff.md#d5-热准入停止与资格准备)

D5-R 在 `42b8388` 冻结热试跑 64–66°C 首个合格平台的选择规则。执行与独立审查、封存 D5 复算和集群 CPU 预检通过。新增 GPU 分配尚未提交，原 C3 保持 `blocked`。 [D5-R 预检](research/m5-handoff.md#d5-r-暖目标预检)

## M4 独立审查修订完成，待总站验收（2026-11-03）

M4 从 M3 交付12705a8开始，真实执行冻结为5525bda：native105叶/1200轮、CPU540组、GPU完整60例及同提交compat432/M1原12/full636均完成。原冻结导出因漏检case外排空反例而失败，原F永久保留FAILED。用户明确授权不追加GPU的独立审查修订后，分析冻结34fa4ea修复该三行检查，完整本地627通过/3原有跳过，全部49个真实子进程控制通过，双SHA数值白名单导出成功。执行、分析与文档交付版本不混同；完整成本、A–F证据及限制见[M4交接](research/m4-handoff.md)、[审查修订授权](research/m4-audit-revision.md)。

M4三次分配全部计入：首轮FAILED/1:0，两个正式角色COMPLETED/0:0；共39:15实际运行、60申请分钟，3/3用尽，全部终态与清理证据完整。生产策略未改，不支持由本轮推导TTFT、吞吐、自然流量公平性或加速。M3旧formal-02的UNVERIFIED_PURGED与原历史gate blocked保持。现停在总站待验收，不追加GPU、不合并PR、不进入M5。

截至 2026-11-03，M1/M2A/M2B 已通过总站验收；相关 PR 均未合并。M3 新批次在 `77844c447e7917264bb6f20fdb41e7056a7181b4` 完成真实GPU完整矩阵、同提交compat432/M1原入口12、完整审查、原30控件与新增完整正控/28负控、专用白名单导出及独立A–F审查，当前待总站验收。原四次历史及formal-02未知终态保持不变，原全历史门槛仍blocked；不能把新批次通过冒称历史缺口恢复。累计使用6/8次分配，余2次未申请。详见[M3新批次交接](research/m3-campaign-handoff.md)和[原历史交接](research/m3-handoff.md)。该段为M3历史状态；当前M4已单独获授权，未进入M5。

## 已验证能力

- 单张 RTX 3090 24 GiB 上，Python 3.12.16、vLLM 0.32.0、Qwen3-4B BF16 完成 7 个请求。重复的 2,230-token 提示命中 2,224 tokens，运行配置与 worker 均确认 APC 和原生 `AsyncScheduler` 开启，见[原生验证记录](../evidence/2026-11-01-rtx3090/validation.json)。
- KV cache 容量为 88,848 tokens（5,553 blocks × 16 tokens），约 12.20 GiB；0.5 秒采样的显存峰值为 20,802 MiB。测量定义及来源见[验证汇总](../evidence/2026-11-01-rtx3090/summary.json)与[来源清单](../evidence/2026-11-01-rtx3090/provenance.json)。
- 从版本化 checkout 复用独立环境、模型和缓存后再次通过原生 7 请求验证；引擎退出后 GPU 占用 1 MiB，作业已释放，见[重放验收记录](../evidence/2026-11-01-checkout-replay/workspace-verification.json)。

- M1 在真实 `AsyncScheduler` / Request / hash / KVCacheManager / BlockPool 上完成四组 × 三次重放：压力 B→C 的 C prefill 派发33，测试 C→B 为1；充足容量两序均为1。A自然完成释放后被B原生分配淘汰的因果 trace、独立审查与证据见 [M1记录](research/m1-native-bc-replay.md)。这只是合成 runner 驱动的 host-side 元数据和 scheduled/dispatched prefill tokens，没有真实 KV tensor 计算。

- M2A 在固定 vLLM 0.32.0 / `5df63a7` 上实际执行 24 组 × 3 次原生矩阵，被测代码为 `1091037dcc2a3a9c31d9692841faebff1f063808`。native/off 压力组 B+C prefill 派发均为82，prefix 自动保留 C 的32-token复用后为50；充足容量三模式均为50。full KV events、统计与 metrics 开启的探测无副作用、APC开关下的 off 等价、19种窗口子例、真实多步 chunk、自然抢占、取消、ID复用和 FIFO 排空均通过。完整来源、失败保留、独立审查及复现入口见 [M2A交接](research/m2a-handoff.md)。这些仍是合成 runner 驱动的 host-side 元数据验证，不代表实际 GPU 工作量或推理加速。

- M2B 最终 Linux 被测 `342f9082e05cd1bebee93a760dd02219eaabae34` 完成636个完整原生case、独立432个兼容case、87个正负子进程和12个M1回归case；每个命名叶子独立重建三次。guarded在500ms使冷B优先于新热H4，热流prompt派发为118（prefix86），如实增加32；不推导GPU性能。原始独审确认保护记录最终归零、原生资源分别排空；状态机、迁移、失败保留与复现见[M2B交接](research/m2b-handoff.md)。

## 历史 M3 收口记录

M3新批次技术闭环已完成并交总站待验收。保留所有失败：首次新冻结 `9dd9de4a9b71149f8102ad26eb8e9b3da48a8ded` 的组合审查错误地要求原M1替代声明为空，修复为精确版本化声明后在 `77844c447e7917264bb6f20fdb41e7056a7181b4` 全矩阵重跑。两次新分配分别09:16与09:36、COMPLETED/0:0、单卡20分钟以内、严格串行，独立监视保存提交前ready、终态、全队列和进程退出。原生产策略、模型、推理参数和历史记录未改；不自动合并，不开始M4/M5。

## 结果解释

早期 readiness 证据证明 BF16 推理、APC 复用、异步配置和工作区重放可用；其中16K是上下文上限，实际输入约2.2K tokens，CUDA Graph与torch.compile关闭。M2A另行证明了固定合成场景下scheduled/dispatched tokens的机制差异；两类证据均不支持调度优化的真实模型端到端性能收益结论。性能结论须按[实验规范](experiments.md)另行验证。

M3 新增真实模型闭环证据，但受控暂停积队与被动观察具有干预和开销，不用于推导自然TTFT、公平性或性能收益。派发区间、实际runner处理与可观测输入token数分开记录；它们不等于FLOPs。
