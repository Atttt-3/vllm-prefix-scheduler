# M3 真实模型闭环：实现前合同

任务 `M3-real-model-integration`，2026-11-02。精确起点为 M2B 交付
`a1e57efc6a72811c53a8d259bdebcafd9b9c8c2e`；其 Linux 被测仍为
`342f9082e05cd1bebee93a760dd02219eaabae34`。后者至前者仅五个阶段/文档文件变化。
M2B 已由总站验收，本任务明确授权 M3；历史 worktree、锁、策略和历史证据不修改。
本设计不是 GPU 验收结论。所有 PR 保持未合并，停止点为总站待验收，不进入 M4/M5。

## 固定源码和执行入口

固定 vLLM 0.32.0 / `5df63a7203cdb9328d824a27b682e77c7b9928be`，Apache-2.0，
上游版权归 vLLM contributors。源码快照不含独立 Git，不能用父仓库 HEAD 代替上游身份；
旧 29 文件哈希加 `configs/m3-real-model.json` 的新增执行链哈希与实际安装逐项核对。

- `vllm/v1/engine/async_llm.py:270` 的 `from_engine_args` 创建异步客户端，
  `generate:698` 必须实际启动并消费。`input_processor.py:354–371` 默认随机化 ID；
  本诊断冻结原生 `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1` 便于精确关联，不修改策略。
- `vllm/v1/engine/core.py:161–182` 在 EngineCore 构造配置解析出的 scheduler。
  生产 smoke 以生产完整类名运行，不加载 scheduler/runner observer；来源、配置、固定构造源码
  与真实输出支持类选择，不能冒称 RPC 直接读取了该 scheduler 实例。
- 诊断 scheduler 子类的 `__init__` 直接记录 self 的 type/MRO/PID/生效配置/来源。
  `add_request` 只在父类成功返回且 requests 映射绑定同一对象之后记入队事件。
  客户端 task、generator 或发送成功均不是入队确认；确认等待有 10 秒上限。
- `async_llm.py:952,986,1144` 原生 pause keep/clear_cache=false、resume 和 reset API
  控制实验初态。A 的真实输出与终止通知先排空，重置缓存再预热；确认暂停后依次提交 B、C。
  不手改 waiting、不手写保护记录、不替换输出消费。
- `core.py:725–785` 是 batch queue 调度、非阻塞 execute/sample、appendleft/pop FIFO、
  `scheduler.update_from_output` 链。单 GPU 原生 UniProcExecutor 使 scheduler/worker/runner
  同 PID；观察层旁路关联原生 SchedulerOutput 对象，不向 dataclass 添加字段。
  非 uni 路径拒绝该诊断合同，不猜测跨进程对象身份。
- 活跃 `vllm/v1/worker/gpu/model_runner.py:1743–1748` 的零 token 通知可以不 forward。
  异步 execute 返回 None 后 sample，再由原生 future 消费 get_output，是合法路径。
  观察器只包原方法一次并保留返回值/异常，绝不提前消费、额外 sample 或 synchronize。

## 接口、归属与观察代价

主执行者拥有运行入口、来源检查、配置、启动脚本、Makefile 与文档；观察层执行者仅拥有
`m3_observer.py`、`m3_worker.py`、`m3_events.py` 及专用测试；独立契约执行者拥有
`m3_contract.py`、`check_m3_trace.py`、`check_m3_fail_closed.py` 与专用测试。
另一独立审查者只读差异/来源/原始 trace。仅主执行者提交 GPU 作业，严格串行。

每进程私有 JSONL：schema/event/pid/seq/monotonic_ns/clock_domain/engine_id/mode；
request_id=`case_id::role`，batch 以旁路序号和原始输出字段指纹关联。
客户端提交、首输出、完成独立；EngineCore 成功入队、评分/窗口/共同决策时钟、成功派发、
V2 runner execute/sample/get_output、原生 update 消费及结束分别记录。
时钟域含主机/boot 身份摘要；不同域不相减，保护只用同 EngineCore 的真实 monotonic 值。

观察副作用限定为有界字典、JSON 序列化和文件写入，存在诊断开销，不用于性能结论。
不额外调用评分/allocate/touch/cache stats/KV events；边界 RPC 读取 checkpoint，
不每 batch 全 KV 快照或阻塞 RPC。记录写入失败应使证据拒绝验收，不改变父方法异常含义。
完整资源扫描仅在初态、排空、shutdown边界：requests、原生队列、finished通知、
in-flight/stale、延迟释放、块映射、guard记录、非null引用、free queue，以及runner状态和旁路pending。
缓存 hash 保留但引用为零合法。shutdown清表与此前自然排空分别验收。

## 冻结矩阵、初态和预算

配置 `configs/m3-real-model.json` 固定文本、采样 seed=42/temperature=0/8 tokens、
16K/2048 batched/APC/async/chunked/eager/BF16/TP1/0.85。
生产 smoke 使用 baseline max_num_seqs=4；功能 profile 仅改 max_num_seqs=1，
实际读取 scheduler.max_num_active_reqs=1（max_num_active_seqs 配置可为 None）。window 仍为16。

20 个完整 case：native/off/prefix/guarded 各一个全新进程的 smoke；
native/off/prefix/guarded_unexpired/guarded_expired 各三次独立 reset+warmup+积队；
一个真实 guarded 客户端取消（先收到真实输出，再 abort 并继续消费）。
取消使用长输出上限；自然完成抢先则失败，不改标签。
各比较模式输入 token IDs、采样、缓存初态、容量一致；native/off greedy输出逐请求比较。
若 batching 引起差异，保留并调查，不能随意放宽门槛。

过期用例 B 确认实际入队后等待0.75秒，再提交较新 C；标签不构成证明。
独立审查以原始 enqueued_ns、sequence、一次共同 decision now 和500ms精确阈值复算：
B已过期且C未过期才有效。未过期guarded与prefix一致，prefix必须正命中且真实C→B。
每请求/场景60秒，入队/排空各10秒；条件失败保留，校准可调整，正式冻结后不改条件拼接。
暂停/人为等待是受控功能干预，不是自然排队、公平性或TTFT实验。

先一次限时校准，再按实际耗时拆正式批次；总上限四次、每次单卡≤20分钟，包含失败。
不嵌套分配、不在登录网关推理/安装、不覆盖CUDA_VISIBLE_DEVICES、不取消其他作业。
每次记录GPU前/中/进程退出后、子进程退出码、Slurm终态和有效活动队列查询。

## 验收与指标

schema2 collector 分开检查完整性、命令成功、行为验收；独立审查从原始事件复算20-case全集、
三次重复、请求/批次/FIFO闭环、保护和资源，不信summary标签。真实子进程正负控覆盖空结果、
缺模式/重复/错标、缺来源、断链、未完成、残留与优化解释器。fixture只算本地契约验证。

指标区分调度派发prompt/decode/重复区间、runner成功处理batch、物理forward有效/填充token。
第三层不能直接采到则写未测，不把prompt减cached或派发数称为GPU计算量/FLOPs/加速。
本专项不要求淘汰、容量矩阵、自然抢占压力、排序成本、TTFT或吞吐改善。
正式前冻结clean commit与配置；新增文档提交不能声称再跑过GPU。
