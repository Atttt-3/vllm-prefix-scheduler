# M2B：monotonic 等待保护与元数据生命周期

本设计于 2026-11-02（Asia/Hong_Kong）完成固定源码调查与独立反证审查，作为实现前合同。
本页记录设计和待验证行为，不代表实现、Linux 原生验收或总项目验收已经通过。
任务分工与逐项验收见 [M2B 任务包](m2b-task.md)；最终被测提交、实际命令及证据由交接记录绑定。

起点为已验收 M2A 交付 `a776cd3467ce1697f9dfd6bf8b858f1952a34de2`，其 Linux 原生被测版本为
`1091037dcc2a3a9c31d9692841faebff1f063808`。M2B 使用独立 `feat/m2b-wait-guard` 分支，
堆叠 PR 的 base 为 `feat/m2a-prefix-scheduler`。M1、M2A 及 M2B PR 均不自动合并。

## 固定源码与迁移依据

调查对象是 vLLM 0.32.0 / `5df63a7203cdb9328d824a27b682e77c7b9928be`，Apache-2.0，
版权归 vLLM contributors。本项目新增保护策略与生命周期包装；原生调度、异步输出消费及缓存实现保留。
本地固定源码快照与 M2A 冻结 SHA-256 核对相同；同时打开固定上游源码核查归属。
一次 shell 直接联网读取受沙箱限制，未将该失败冒充成功；已有固定快照与网页读取提供了实际调查材料。
以下行号是原始源码的一基行号，不使用网页文本转换后的行号。

| 固定源码 | SHA-256 |
| --- | --- |
| [`vllm/v1/core/sched/scheduler.py`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/sched/scheduler.py) | `d9a39870286c424cef2fc0730e24092ee3df67e848608e2278944846bab804ae` |
| [`vllm/v1/core/sched/async_scheduler.py`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/sched/async_scheduler.py) | `38d296e77e4d612e696241c67298eb1ca9b6a48b483cd33d35495bf31691032e` |
| [`vllm/v1/request.py`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/request.py) | `dd7c30bf8b0894dc123a739c299c0133566f8cc075b081e45905f1c29506e69d` |

| 位置 | 原生事实与设计后果 |
| --- | --- |
| Scheduler 2616–2647：`add_request` | 先按 ID 查现有对象；重复 ID 可能拒绝或更新已有 streaming session。首次插入先入队，再写 `requests`，再记指标/事件，不是事务。只能在父类成功返回后核对实际对象并登记。 |
| Request 84–241；Scheduler 1650–1691、3175–3207 | 普通新对象初始 WAITING、进度为零。grammar/session/remote 恢复不能当新请求；恢复到 WAITING 仍有对应特殊字段或抢占历史。 |
| Scheduler 2457–2491：入队与 session 停止处理 | `_enqueue_waiting_request` 也接收恢复对象，不能把该入口当作首次入队 hook。 |
| Scheduler 657–939、1255–1282、1343–1369 | running 先于 waiting；失败分配在最终准入之前退出。查找命中、分配尝试、RUNNING 状态均不能单独证明成功派发。 |
| Scheduler 1501–1556；AsyncScheduler 25–57 | 原生最后更新 computed/in-flight、chunk 和 placeholders，然后返回调度输出。`_update_after_schedule` 仍在 profiling context 内，其成功后 context 退出可能抛错；清理应在完整 `super().schedule()` 成功返回后。 |
| Scheduler 1571–1614：`_preempt_request` | 原生释放块、设置 PREEMPTED、清零 computed/placeholder、增加抢占计数并 prepend。释放异常可能发生在状态转换前；保护处理必须区分实际转换与失败尝试。 |
| Scheduler 2077–2103；AsyncScheduler 59–78 | 输出按字符串 ID 查当前对象并消费 in-flight/stale；stale 不再次扣 placeholder，RUNNING 输出继续 `cache_blocks`。保护层不替代这条路径。 |
| Scheduler 2224–2230、2649–2749 | 自然完成和外部 finish 汇入 `_free_request`。`finish_requests` 原生支持 str、iterable、None，先收集/移除队列，再逐个终止；保留这一实现。 |
| Scheduler 2980–2990：`shutdown` | 只关闭事件发布器与 connectors，不保证结束请求、消费 pending 或释放全部 KV。保护清表与原生排空必须分别验收。 |

固定源码中未发现受支持普通请求成功 add 后被资格检查排除、又自然变成 fresh 且没有记录的路径。
grammar 就绪后仍有 structured-output 对象；session 保留 streaming/resumable 字段；抢占计数不复位。
这个源码结论不替代运行时检查：guarded 遇到当前候选缺记录或错对象时仍必须失败封闭，不能延迟登记。

## 保护范围、排序与时钟

配置仍位于 `additional_config.prefix_scheduler`，默认 `mode=off`、`window=16`、`max_wait_ms=500`。
仅 guarded 创建并维护请求级保护状态；off/prefix 不读取保护时钟，不维护保护记录，保留 M2A 行为。
启用 guarded 同时执行 M2A 构造前支持检查与构造后原生对象检查，不能只放开配置 parser。
单 CUDA、有效 async/V2、FCFS、APC、单因果 full-attention group、块大小一致以及原有全局拒绝条件继续有效。

保护对象是**普通新请求从首次成功入队到首次返回的原生正 token 调度输出之前**的等待。
这里的派发是 host-side `SchedulerOutput.num_scheduled_tokens` 中的正 token 工作，不是首个采样 token、
模型 worker 已执行、真实 KV tensor 计算或用户收到首 token。首次 prefill chunk 已到达清理边界。

成功入队后记录实际 Request 强引用、首次入队纳秒值及严格递增序号。序号独立于时钟，相同 tick 仍可区分先后。
计时不从首次进入窗口开始；排序、分配失败、重复 step、进入或离开窗口不刷新记录。
每次 guarded schedule 都先取 M2A 连续 fresh 队头窗口，并校验候选记录及对象身份。
即使 `window=1` 或候选不足两个，也不能跳过身份检查；没有实际排序决定时不读取排序时钟。

一次实际排序只读一次共同 `now_ns`：

1. `now_ns - enqueue_ns >= threshold_ns` 的候选置前，按首次成功入队序号升序。
2. 其余候选按当前只读可复用前缀长度稳定降序；同分保留本次窗口顺序。
3. 原队列可能已被 prefix 规则改序，不能把其当前位置用作首次入队次序。

只保证当前连续候选窗口内的排序。窗口外、blocked/resumed/部分计算屏障、running 优先、原生资源不足和过载
仍可延长等待；不承诺全局 TTFT 上界或任意负载下无饥饿。原生再次查找与分配是最终准入依据。
评分沿用 M2A 只读 probe，不保存跨 step 分数、不 pin 块、不重做 hash 或缓存查找。
评分与排序先完成，再以既有 deque 事务替换头窗口；后缀不变，prepend 仍传反序。

生产时钟为私有 `time.monotonic_ns` 提供器；测试以私有接口或子类注入，不新增 HTTP 参数，不全局替换原生计时。
时钟值要求 `type(value) is int`，拒绝 bool、float、其他类型；任意正负整数 epoch 均可，零和同 tick 合法。
每个成功时钟读数不得小于该实例上一次成功读数，包括入队与排序之间；比较前检查，不能把负 elapsed 钳成零。
不读取 `Request.arrival_time`、wall clock 或客户端时间，也不把绝对 epoch 写入保护假设。

`max_wait_ms` 保留非负有限 int/float 输入，包括 `10**400`。转换采用 Python 整数运算：

```text
int ms:   threshold_ns = ms * 1_000_000
float ms: (numerator, denominator) = ms.as_integer_ratio()
          scaled = numerator * 1_000_000
          threshold_ns = (scaled + denominator - 1) // denominator
```

浮点值按其实际二进制数值解释，再向上取整至可观察的整数纳秒。因此 `0.1` ms 对应 `100001` ns；
这不是把十进制文本重新解释为有理数。零对应零；不经由 float 乘法，不溢出为 inf，不限幅。
独立审查器以精确有理数交叉相乘重算阈值，不依赖实现导出的 threshold/protected 标签。

## 完整状态机与失败合同

保护存储以实际 Request 对象为键，只保存仍待首次派发的对象及不可变记录；固定 Request 使用身份相等与哈希。
不得只保存字符串 ID 或可能复用的 `id(request)` 数字。序号作为代际标识单调递增，删除记录不回退序号。
正常清理只处理当前事件影响的对象，不每 step 扫描整个 waiting 或保护表。

| 事件/当前状态 | 父类调用与保护转换 | 时钟/序号 | 可继续性 |
| --- | --- | --- | --- |
| 新 ID、普通 fresh 对象，add 成功 | 父类成功后确认调用前无对象、当前映射就是传入对象，再登记 | 读一次入队 clock；成功登记才增加序号 | 可继续 |
| 新 ID、非普通或非 fresh 对象，add 成功 | 由原生管理；不登记、不晚登记 | 不读保护 clock、不增加序号 | 可继续原生路径 |
| 现有 ID，重复 add 拒绝或 session 更新 | 原样调用父类；现有记录不刷新，传入对象不建新代 | 不读保护 clock、不增加序号 | 原生错误原样传播；本事件不额外 fault，内部 preempt/free 已触发的 fault 保留 |
| 新 ID，父类 add 失败 | 不登记；可能存在原生部分入队状态，不回滚原生 | 不读入队 clock、不增加序号 | latch fault，重抛原异常；普通与非普通输入相同 |
| 新普通 add 已成功，但 clock 抛错、类型错或回退 | 不登记、不伪造入队时间；已接纳的原生对象保持可终止清理 | 不增加序号；无有效新读数 | latch fault，拒绝后续 guarded add/schedule |
| 等待、排序、分配失败、重复 step、进出窗口 | 原记录保持；失败查找/失败分配不是派发 | 不刷新入队值/序号；实际排序只读一次共同 now | 无异常时可继续 |
| 候选缺记录或错身份 | 不补登记、不当作未保护项 | 不为恢复身份读取 clock | latch fault，拒绝该次 guarded 调度 |
| 排序 clock 抛错、类型错或回退 | 队列尚未改序，不修正 elapsed | 不更新最后有效读数 | latch fault |
| pre-native probe/排序异常，或队头事务异常且成功恢复 | 原窗口及年龄保持；原生 schedule 尚未调用 | 已有有效读数可保留，不重置年龄 | 重抛；不因普通 probe/事务异常额外 fault，保护自身一致性错误仍 fault |
| 父类 schedule 失败 | 不将部分 computed、allocation 或 hook 更新视作成功派发；不回滚原生 | 不读清理 clock | latch fault，重抛原异常 |
| 父类 schedule 成功返回正 token 输出 | 仅按实际输出请求查当前对象，身份核对后删除对应记录；chunk 同样删除 | 不读清理 clock | 可继续 |
| 抢占成功，包括首次最终派发前的异常残留对象 | 原生设置 PREEMPTED/增加计数后身份清理；恢复不再登记 | 不读 clock、不增加序号 | 正常路径可继续 |
| 抢占父类异常 | latch fault；若已发生 PREEMPTED/抢占计数转换则清理，否则保留至后续终止/关闭 | 不读 clock | 重抛；不宣称原生恢复 |
| 自然终止、取消、批量/未知/重复 ID、None | 保留原生 finish；终止对象汇入 `_free_request`，finally 身份清理，未知/重复无新增操作 | 不读 clock | 正常路径可继续 |
| `_free_request` 父类异常 | latch fault；finally 只清理已终止的实际对象，不误删非终止对象或其他代际 | 不读 clock | 重抛；自有清理不等于 native 资源已释放 |
| finish 的一次性 generator 抛错 | 原生先物化 set，异常在队列/终止变更前；不包装或重复消费 | 记录不变 | 原异常传播；可重试 finish |
| finish 批量移除后部分释放失败 | 只清理实际触达的终止对象，其余记录不按输入 ID 批量删除 | 记录不刷新 | `_free_request` fault 生效；保留原生部分失败证据 |
| shutdown 成功或父类抛错 | finally 清空自有状态并置 closed；不替原生完成请求 | 不读 clock | 拒绝后续 guarded add/schedule；重复 shutdown 幂等清表 |

fault/closed 仅保存简单标量状态或固定原因代码，不保存异常对象、traceback、callback 历史或无限事件列表。
原异常继续传播；后续 guarded add/schedule 在父类调用前拒绝。finish/free/shutdown 清理路径仍可执行，
不能因为 fault 阻止终止清理。测试追踪与清理原因记录放在夹具/trace 层，不让生产状态随历史请求累积。

不覆盖 `_update_after_schedule` 或 `_update_request_with_output`。原生占位、cache_blocks、stale/in-flight 消费均保持。
`finish_requests` 不需要 wrapper：str、列表、一次性 generator、None、未知和重复 ID 原样传入父类。
取消的 set 迭代不保证 FIFO，审查不得把 cleanup 事件顺序误当请求顺序。

## 身份复用、异常与资源边界

正式支持旧代输出、finished 通知和资源排空后的同 ID 新对象复用；新对象获得新入队值和更大序号。
单独的保护模块测试证明旧对象清理不能删除同 ID 新绑定。该测试不得向原生 `_free_request` 或输出路由
注入危险旧对象回调来宣称并发代际安全：原生 `_free_blocks` 按 ID 删除 native map，输出也按 ID 查当前对象。
**旧代 pending 未排空时重名复用不受支持**；M2B 不重写输出路由。shutdown 后同一 scheduler 实例也不重开。

固定支持路径的普通请求在首次成功派发前不会自然被“本次新准入后又抢占”；running 处理早于新 waiting。
但 schedule 在准入后、成功返回前失败可能留下 RUNNING 对象，后续终止/抢占清理必须处理。
这类注入故障只证明异常边界，不代替真实自然抢占验收。

自然完成与显式取消仍需正常 FIFO 消费输出并检查原生 requests、waiting/running/kv_holding/deferred、
in-flight 集合、pending、活跃块引用、分配映射及 free queue。历史已取消 Python 对象的计数残留不能直接算活跃泄漏。
正常排空在 shutdown 前验证。带存活请求的 shutdown 用例应分别报告保护表清零与原生仍存活状态，
不能重新 schedule 已关闭实例来包装成“shutdown 排空”。释放失败保留失败证据，不以自有表为空宣称资源已释放。

## 契约迁移与验证门槛

M2A 的 guarded 未实现拒绝、原始矩阵、trace、失败记录及验收属于其原版本。
完整历史入口 `make native-m2a` / `make native-m2a-trace-check` 适用于 M2A 验收交付提交及其实际被测版本，
不能要求 M2B 启用 guarded 后继续通过旧全局拒绝用例。
M2B 最小更新当前配置测试，使用独立冻结矩阵与审查合同：guarded 受支持正例、APC-off/async/group 等真实负例，
并通过新的兼容回归入口保留 off/prefix 行为；不删除旧拒绝后留下覆盖空缺，不改写旧配置、trace 或验收。
兼容入口的实际命令由新 Make target 和交接记录固定，历史完整入口不会被静默重定义为新语义。

原生验收包含阈值前/等于/超过、零、精确有限浮点、超大整数、同 tick 序号、多 protected、已改序未准入队列、
窗口外年龄、重复资源失败、首 chunk、持续热请求越过机制、取消/完成/复用/异常/关闭及原生安全继承。
guarded 未到阈值与 prefix 决策相同；达到阈值可增加冷请求准入、降低缓存命中并增加派发工作，不能强迫保持 82→50。
同时报告 native/off/prefix 对照、guarded 保护前后准入/命中/prompt/decode/重复工作及越过次数。

冻结矩阵必须完整、非空、case 唯一，关键原生状态场景至少三次独立重建；禁止缺/多/重复/错标 case、重复不足、
无效配置、缺原生依赖、跳过或优化解释器通过。审查从原始 clock/入队序号/对象代际/窗口/最终派发/清理事件
重算保护与存活状态；summary 和 protected 标签只是可被核对的产物。
只读 probe 在 stats、metrics、full KV events 同时开启时保留无副作用验证；原生正例不能由 sleep 或手写列表代替。

Mac 仅轻量开发与完整 `make check`；固定 Linux 环境执行原生 host-side replay，不安装或覆盖共享环境/锁文件。
协调者独占单卡作业提交，串行、每次分配不超过 20 分钟，有有限 step 与超时，结束核对释放。
schema 2 私有新 runs 保存代码/环境/hash、冻结输入/clock/seed/case、原始 trace、逐项 acceptance、独立审查及失败/释放材料。
代码冻结后采集，修复后以新被测提交重建完整验收，不拼接版本。真实模型、KV tensor、GPU 性能、M3/M5 均不在本轮。
