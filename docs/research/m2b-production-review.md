# M2B 生产实现独立审查

审查日期：2026-11-02（Asia/Hong_Kong）。审查者为独立源码调查/审查子任务，未编写本次生产实现或被审查单元测试。
范围是相对已验收 `a776cd3467ce1697f9dfd6bf8b858f1952a34de2` 的 `wait_guard.py`、`scheduler.py`、
`config.py` 与 `test_wait_guard.py`、`test_prefix_config.py`；设计阶段提交为 `d26920a`。
审查时实现尚未提交，因此以以下文件 SHA-256 绑定观察；后续修改必须另行复核。

| 文件 | 审查版本 SHA-256 |
| --- | --- |
| `src/vllm_prefix_scheduler/wait_guard.py` | `92dd24e34921703d07f2ebf5a85176bb1fb2e8fb87a4bf61523ee6f66ecda662` |
| `src/vllm_prefix_scheduler/scheduler.py` | `d65c89284e49ae4d7854b3f2e4cf45a4da9f7d2d3572698c9deacdffe8dda16d` |
| `src/vllm_prefix_scheduler/config.py` | `a24976b6728fab6352992d67ee45e5751c7326002b6e62bb454f254bc84a1608` |
| `tests/test_wait_guard.py` | `17abe9151056ba899248d1e1f75a4bb056ca658f70fc5077705f5882131eef74` |
| `tests/test_prefix_config.py` | `7ef19af53e6ff1c5f2f4a34ba5fc2392d3d61e441360f6f86418044b2e718bc4` |

## 结论与已排查风险

本轮静态审查和元数据/数值层独立验证未发现需要阻断实现提交的代码缺陷。
这不是 Linux 原生验收通过结论，也不替代独立 trace 审查；当前单元测试不构造真实 vLLM scheduler。

| 合同 | 代码观察与反证检查 |
| --- | --- |
| guarded 构造支持范围 | scheduler 106–128 同时启用既有构造前检查和构造后对象检查；guard 在检查成功后才创建。APC、async/V2、full-attention 等没有被 parser 放开绕过。 |
| off/prefix 保持 M2A | 这两模式不创建 `_wait_guard`；add/preempt/free/shutdown 直接父类调用；prefix 原窗口排序和事务恢复保留，不访问保护 clock。 |
| 入队与重复请求 | scheduler 183–203 先记录现有 native 对象，父类成功后核对 `requests[id] is request`；已有 ID 不登记。新 ID 父类失败 latch，重复/session 错误不刷新年龄；内部生命周期错误的 fault 保留。 |
| 时钟与数值 | `wait_threshold_ns` 对精确 int/float 作整数 ceil；没有 float 乘法、inf 或限幅。clock 严格要求 exact int、允许负 epoch/同 tick，回退与 provider 错误 latch；坏读数不登记、不增加序号。 |
| 原始次序 | `WaitRecord.sequence` 在成功登记后递增；protected key 使用 sequence，未 protected key 使用稳定负 prefix 分数。不能把已改序队列当原始入队次序。 |
| window=1 与共同 now | `_reorder_waiting` 在长度早退前验证所有候选记录；只有实际排序读一次 decision clock。未找到 off/prefix 或清理路径读取保护 clock。 |
| 最终派发边界 | schedule 162–169 等完整父类 schedule 成功返回，再仅处理最终正 token map 的实际对象。未覆盖 `_update_after_schedule`，避免 profiling context 退出失败导致提前清理。 |
| 身份与存活数量 | records 直接以实际 Request 对象为 key，持有强引用。固定 Request 无自定义 equality/hash，且候选要求 exact Request，因此旧对象删除不能命中新对象。没有字符串 ID 清理、裸 id 数字或历史 tombstone。 |
| preempt/free 异常 | 两 hook 各自 latch 原生异常，覆盖从 schedule 之外调用的路径。preempt 根据状态/计数是否实际转换清理；free 的 finally 只清理已终止对象；保留原生签名、参数传递与返回值。 |
| native 异步语义 | 未替换 `finish_requests`、输出消费、placeholder 更新或 `cache_blocks`；继承完整 AsyncScheduler 父链。未知/重复 ID 和一次性 generator 无新增消费。 |
| 关闭与引用 | shutdown finally 清表并 closed；后续 guarded add/schedule 在父类调用前拒绝。fault 仅存字符串，不存 exception/traceback；清理不依赖 active 状态。 |
| 异常事务 | pre-native probe/队头恢复位于原生 failure catch 之外，可按既有合同重试；clock/缺记录等保护一致性错误自行 latch。进入原生后不回滚原生状态。 |

固定原生接口/调用次序依据与 SHA-256 已记录于 [设计](m2b-wait-guard.md)。本轮复核特别关注：
Scheduler `add_request:2616–2647`、`schedule:1501–1556`、`_preempt_request:1571–1614`、
`finish_requests/_free_request:2649–2749`、`shutdown:2980–2990`，以及 AsyncScheduler 25–78。
实际对象 key 的成立条件是固定原生 Request；不能把该私有模块对任意自定义 equality/hash 对象的行为扩展成支持承诺。

## 实际独立验证

在 Mac 既有开发解释器执行：

```bash
.venv/bin/python -m pytest -q tests/test_wait_guard.py tests/test_prefix_config.py
```

结果为 **73 passed in 0.05s**，无 skip。本命令只证明本地元数据与配置层。

另以 Python `random.Random(20261102)` 产生 10,000 个 64-bit 模式，经 `struct.unpack('!d', ...)` 得到浮点数，
筛选 4,974 个非负有限值；独立使用 `fractions.Fraction(value) * 1_000_000` 验证：

```text
threshold >= exact_value
threshold == 0 or threshold - 1 < exact_value
```

全部通过。另核对整数 `0`、`1`、`10**400`、`10**10000` 的阈值恰为原值乘一百万，全部通过。
该探索性 oracle 未写入原生证据目录，不是冻结 native 矩阵的一部分；正式验收仍由既定 schema 2 采集。

测试已覆盖：浮点次正规数/最大有限值、0.1 二进制精确边界、巨大整数、负 epoch、同 tick 序号、旧对象清理新绑定、
错误 clock 类型/回退/负 elapsed、缺记录失败、clock 异常不被状态持有，以及 discard/close 后弱引用释放。

## 尚须由原生验收证明

以下是验收依赖，不是已通过项目；当前 73 项单元测试不能替代它们：

- 真正的 scheduler resolution/MRO、生效配置、构造前后拒绝及失败 shutdown。
- 普通新请求成功接纳后的登记、重复 ID/session 更新、窗口外等待、真实分配不足、实际 chunk 派发和自然抢占。
- window=1 的真实候选缺记录、同 tick 多 protected、已 prefix 改序队列、持续热流量冷请求保护及实际 token 代价。
- profiling context 退出、new add 部分插入、preempt 转换前后、free 与 native shutdown 的真实边界故障注入。
- str/list/generator/None/unknown/duplicate 的原生终止行为、旧代 pending 排空后新对象复用、重复生命周期后的存活表归零。
- stats/metrics/full KV events 同开只读 probe、placeholder/cache_blocks/stale/in-flight、原生队列/pending/引用/资源分别排空。
- 完整冻结矩阵、三次独立重建、原始事件独立重算和 tamper/缺多重复/错标/优化解释器失败封闭。

原生 shutdown 只关闭组件，不能以 guard 表清零代替 requests/pending/KV 排空。带存活请求的 shutdown 用例必须如实显示原生残留。
原生输出按字符串 ID 路由；旧 pending 未排空时复用同 ID 仍不支持，纯保护身份测试不证明原生并发代际安全。
本审查未运行 GPU、原生 replay、真实模型或 KV tensor，也未申请计算资源或改变生产文件。
