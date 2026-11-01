# M1 原生 B/C 缓存淘汰重放

任务：`M1-native-bc-replay`。研究问题：冷请求 B 的原生准入分配是否会淘汰已完成请求 A 的可复用前缀，使稍后 C 增加 prompt 派发工作；仅调整测试候选顺序能否保留该机会？

## 固定源码调查

- 阅读日期：2026-11-02，Asia/Hong_Kong；调查者：独立源码调查 agent。
- 上游：[vllm-project/vllm](https://github.com/vllm-project/vllm/tree/5df63a7203cdb9328d824a27b682e77c7b9928be)，vLLM 0.32.0，commit `5df63a7203cdb9328d824a27b682e77c7b9928be`。
- 取材：固定 commit 的 GitHub 源码 archive，逐文件阅读；没有以移动分支、搜索摘要或未合并 PR 代替固定源码。
- 许可证：Apache-2.0；上游源文件注明 Copyright contributors to the vLLM project。以下描述归属上游现有实现，本项目重放不复制缓存算法。
- 本节是源码事实和待由 trace 证明的断言；实际运行、环境来源和数值以本文后续验收记录为准。未调查其他版本或真实 GPU 执行的性能。

### 调用链及可观察状态

下列链接全部固定到同一 commit。行号对应下载源码。

| 层次 | 上游位置 | 已核查行为 / 对 trace 的要求 |
|---|---|---|
| 选择异步类 | [`SchedulerConfig.get_scheduler_cls`, L228–258](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/config/scheduler.py#L228-L258) | 未指定自定义类且 `async_scheduling=True` 时返回真正 `AsyncScheduler`。重放需记录对象类型和实际模块文件，不能仅记录配置标志。 |
| 原生请求与 hash | [`Request.__init__`, L61–237](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/request.py#L61-L237)；[`init_none_hash`, L131–173；`get_request_block_hasher`, L941–1003](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/kv_cache_utils.py#L941-L1003) | Request 初始化为 WAITING、进度和引用相关计数为零，调用原生 block hasher；hash 按完整块生成并链入前块 hash。记录 token IDs、hash 函数和 `PYTHONHASHSEED` / NONE_HASH seed。 |
| 入队 | [`Scheduler.add_request`, L2616–2649](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/sched/scheduler.py#L2616-L2649)；[`FCFSRequestQueue`, L86–127](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/sched/request_queue.py#L86-L127) | 普通新请求追加到 FCFS 队列，peek 读队头、pop 使用 popleft。`prepend_requests` 使用 extendleft，会反转输入顺序。控制组只能一次明确操作 B/C 候选，不改变到达字段。 |
| 原生 waiting scan | [`Scheduler.schedule`, L899–973、L1240–1381](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/sched/scheduler.py#L899-L973) | 先处理 running，再扫描 waiting；持有 KV 的 waiting 队列优先。对普通新请求先查命中，再分配，成功后才 pop waiting、加入 running。分配失败会 break，后续请求可能延后。队列快照须包含 waiting、kv_holding_waiting、running。 |
| 查找当前前缀 | [`_get_local_prefix_cache_hit`, L558–569](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/sched/scheduler.py#L558-L569) → [`KVCacheManager.get_computed_blocks`, L264–321](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/kv_cache_manager.py#L264-L321) → [`FullAttentionManager.find_longest_cache_hit`, L733–791](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/single_type_kv_cache_manager.py#L733-L791) | 普通 full-attention、hash block size = physical block size 时按完整块从首块找最长连续前缀。最大命中长度为 `request.num_tokens - 1`，最后 token 要重算 logits。观测包装只记录这次原生调用的返回值，不额外调用查找。 |
| 采用命中及分配新块 | [`KVCacheManager.allocate_slots`, L371–618](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/kv_cache_manager.py#L371-L618) → [`KVCacheCoordinator.allocate_new_computed_blocks`, L261–310](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/kv_cache_coordinator.py#L261-L310) → [`add_local_computed_blocks`, L266–316](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/single_type_kv_cache_manager.py#L266-L316) | 容量检查把零引用命中块也计入占用；成功后先 touch 命中块，再分配需要的新块。C 先成功准入时，命中块从 free queue 移出且引用计数增加，B 后续分配无法淘汰这些被引用的块。 |
| free queue 与淘汰 | [`BlockPool.__init__`, L159–206](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/block_pool.py#L159-L206)；[`get_new_blocks`, L666–700；`_maybe_evict_cached_block`, L729–750](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/block_pool.py#L666-L750) | 初始化把 block 0 从 free queue 取走作为 null block；可分配容量为总块数减一。分配从 free queue 头取块，调用真正淘汰方法移除旧 hash 元数据，然后把 ref_cnt 从 0 增为 1。需要在 B 的 allocate 调用上下文记录旧 hash、块号、返回布尔值和前后映射。 |
| 元数据缓存 | [`allocate_slots`, L607–616](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/kv_cache_manager.py#L607-L616)；[`SingleTypeKVCacheManager.cache_blocks`, L476–533](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/single_type_kv_cache_manager.py#L476-L533) | 分配路径已将不超过 request.num_tokens 的完整已知块登记原生缓存元数据，输出更新再补充新确定 token 的缓存。这是 host-side 元数据逻辑，不能表述为本重放计算了 KV tensors。 |
| 异步调度进度 | [`Scheduler._update_after_schedule`, L1616–1648](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/sched/scheduler.py#L1616-L1648)；[`AsyncScheduler._update_after_schedule`, L25–56](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/sched/async_scheduler.py#L25-L56) | schedule 末尾乐观增加 num_computed_tokens 和 num_in_flight_tokens。非末尾 prefill chunk 不加采样占位；末尾 prefill/decode 增加 num_output_placeholders。trace 需区分已派发进度和已消费输出的进度。 |
| FIFO 输出消费 | [`EngineCore.step_with_batch_queue`, L698–813](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/engine/core.py#L698-L813)；[`Scheduler.update_from_output`, L2026–2232](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/sched/scheduler.py#L2026-L2232)；[`AsyncScheduler._update_request_with_output`, L58–77](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/sched/async_scheduler.py#L58-L77) | 引擎优先填充 batch queue，按 FIFO 消费输出。输出更新先扣该 step 的 in_flight；新 token 更新停止状态，非 stale 输出扣 placeholder，再按 computed − placeholder 缓存。合成输出必须绑定生成它的 SchedulerOutput，不能读下一 step 已改变的 chunk 标志。 |
| 完成和释放 | [`Scheduler._free_request` / `_free_request_blocks`, L2714–2778](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/sched/scheduler.py#L2714-L2778)；[`SingleTypeKVCacheManager.free`, L581–589](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/single_type_kv_cache_manager.py#L581-L589)；[`BlockPool.free_blocks`, L774–807](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/block_pool.py#L774-L807) | 完成后移除请求并释放块引用；每请求按逆序释放。未缓存块插到 free queue 前部，带 hash 的零引用缓存块追加到尾部，仍保留映射，后续可被原生分配淘汰。A 初态必须同时证明 mapped、ref_cnt=0、在 free queue 中。 |

### 源码揭示的测量边界

`get_computed_blocks` 在 KV events 开启且请求 report mode 为 full 时会发出 `BlockStored`；因此额外“测一次命中”不是普遍无副作用的探针。本重放只能包装调度器本来调用的方法并透传返回值，或直接读取已有 block/hash/free-queue 元数据。统计在成功准入处调用 `record_prefix_cache_stats`（scheduler L1316–1319），也不能为了观测额外触发。

原生分配会登记 prompt 元数据，不能由这一事实推断真正执行过矩阵运算。合成 `ModelRunnerOutput` 仅驱动 host-side 生命周期。合法输出约束是：非末尾 prompt chunk 没有采样 token，末尾 prompt 或 decode step 采样一个固定 token（未启用投机解码），所有 pending outputs 按 FIFO 消费并在有限步数内排空。

对单 full-attention group、相同 physical/hash block size、无投机/多模态/连接器的夹具：长度 `k × block_size + 1` 允许命中前 k 个完整块并派发最后 1 token；长度恰为整块倍数则受 `num_tokens - 1` 上限约束，至少重派最后一个完整块。总块数还包含不可分配的 null block，不能把总块数直接当作可用块数。

调度区间应在每次实际派发时记录为 `[start, start + count)`，`start` 是此次采用前缀命中后的派发起点，不能简单使用 schedule 调用前的 0。与 `[0, prompt_length)` 的交集计为 **scheduled/dispatched prefill tokens**；与已派发 prompt 位置集合的重叠另计重复派发，包括抢占重算。区间超出 prompt 的部分计为 decode 派发。仅累加 `num_scheduled_tokens` 会混入 decode，只用 prompt − cached 会漏掉重算；二者均不足以直接表示全部 prefill 派发工作。

### 原生测试参照及迁移结论

上游 [`tests/v1/core/utils.py:create_scheduler/create_requests`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/tests/v1/core/utils.py#L56-L285) 使用原生 ModelConfig、SchedulerConfig、KVCacheConfig、FullAttentionSpec、StructuredOutputManager 和 block hasher；metadata-only cache 配置的 `kv_cache_tensors=[]`。`skip_tokenizer_init=True` 可避免 tokenizer 初始化，不需要替换命中、分配、缓存或输出更新算法。

上游 [`tests/v1/core/test_async_scheduler.py`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/tests/v1/core/test_async_scheduler.py#L20-L76) 已用 ModelRunnerOutput 驱动异步停止测试，L167–225 覆盖同 step 前缀去重。它们说明这种 host-side 夹具的接口用法，不能代替本任务要求的 B/C 压力反例、四组容量/顺序对照和三次重复。

本轮结论限定为原生机制重放。一次 C→B 候选顺序更换只用于因果控制，不是已实现的窗口策略、只读评分、等待保护或自定义 scheduler。真正的策略还需另行验证新鲜 WAITING 连续窗口、屏障、重排公平性和副作用约束；本轮不进入这些实现。

## 实现与冻结输入

实现入口为 `scripts/replay_bc.py` / `src/vllm_prefix_scheduler/native_replay.py`，没有自定义调度器子类。`configs/native-bc-replay.json` 固定完整 token IDs、到达记录、seed=42、15 个上游模块哈希及模型 config.json 哈希。模型配置读取既有本地 Qwen3-4B 元数据，`skip_tokenizer_init=True`，不创建 LLM、不加载权重。缓存几何是原生单层 `FullAttentionSpec` 的 metadata-only 夹具，`kv_cache_tensors=[]`；它不是模型真实 KV 容量。

| 固定项 | 值 |
|---|---|
| A / B / C prompt 长度 | 33 / 49 / 33 |
| A/C 共享部分 | 前 32 token IDs 完全相同，末 token 分别 200 / 201 |
| 到达与入队 | A=0；A 完成并排空后按 B=1、C=2 入队；时间是合成逻辑到达，不是性能时间 |
| block / hash block size | 16 / 16 |
| 总块容量 | 压力 5；充足 10；均包含不可分配 null block 0 |
| token budget / 活跃上限 / 执行槽数 | 64 / 2 / 2 |
| chunked prefill / reserve full ISL | true / true |
| watermark / long prefill threshold | 0 / 0，adaptive=false |
| max model length / max output tokens | 128 / 3，ignore_eos=true，合成输出 token=9001 |
| pending FIFO / 每阶段循环上限 | 2 / 64；warmup 和 B/C 分别有界 |
| 并行与特性 | TP=PP=DCP=PCP=1；APC=true；async=true；V2=true；无投机/LoRA/多模态/连接器 |
| 重复 | 四个独立重建条件 × 3 次 |

A 自然分配 3 块，前 2 块为完整可复用前缀。3 个输出 token 只派发其中前 2 个 decode 位置；最后生成 token 无需再次作为输入计算。所有请求都不会跨过下一块边界。未缓存尾块优先回 free queue；A 两个完整块保持 hash 映射、ref_cnt=0，并在 free queue 中，才进入 B/C 阶段。

所有 mock/stub：无。唯一合成执行结果为原生 `ModelRunnerOutput` 对象：对调度当时的非末尾 prefill chunk 填 `[]`，末尾 prefill/decode 填一个 9001。`ModelConfig`、`SchedulerConfig`、`VllmConfig`、`StructuredOutputManager`、`Request`、hash、AsyncScheduler、KVCacheManager、BlockPool 都是真实对象。

观测包装清单：manager 的 `get_computed_blocks/allocate_slots/cache_blocks/free`；pool 的 `get_new_blocks/_maybe_evict_cached_block/touch/free_blocks`；scheduler 的 `_preempt_request/_update_after_schedule`。每次原生调用只透传一次且原样返回；不提前消费 free_blocks 的迭代器。快照只读字段、hash map 和原生 free-list 遍历 helper，不额外 lookup、不 reset stats、不 take events。每个 case 结束后恢复实例方法。

在 `_update_after_schedule` 原方法调用前冻结 `[start,end)`，此时 start 已包含真实准入命中。prefill 是该区间与原 prompt 范围的交集，decode 是 prompt 之外的派发位置；重叠位置独立累计为 repeated prefill/decode。`computed_optimistic` 仅代表原生乐观进度。矩阵无抢占时 repeated=0，不能据此声称完成抢占回归。计数器的回滚/区间跨界由本地测试单独验证。

## 可复现入口

仅在现有 Linux 固定环境、获准计算节点上运行；`MODEL_CONFIG_DIR` 指向现有模型元数据目录，`NATIVE_PYTHON` 指向现有固定解释器。下列变量不包含公开的真实机器路径：

```bash
export PYTHONHASHSEED=42 VLLM_USE_V2_MODEL_RUNNER=1 HF_HUB_OFFLINE=1
make native-replay NATIVE_PYTHON="$NATIVE_PYTHON" \
  ARGS="--config configs/native-bc-replay.json --model-config '$MODEL_CONFIG_DIR' --output runs/m1-native-bc-replay/new-data"
make native-trace-check NATIVE_PYTHON="$NATIVE_PYTHON" \
  ARGS='--trace runs/m1-native-bc-replay/new-data/trace.jsonl'
make check
```

正式取证应以原有 `scripts/run_evidence.py --kind native --seed 42` 包住 native 命令，登记该配置、runtime/inputs/trace/summary 四项产物与 A–F 原始标准，再分别运行完整性、命令成功、人工 acceptance 和 require-accepted 门槛。独立 trace 审查入口仅依赖标准库；现有 native 环境未安装 pytest，不向共享环境补装依赖。本地 native pytest 明确 skip，不能满足原生验收。

## 已知失败与审查修复

首轮固定提交 `57d6590` 的 attempt-01 在原生 SchedulerConfig 初始化失败：未提供必填 `is_encoder_decoder`，未产生任何调度事件。此前独立静态 API 审查漏检此 InitVar；审查者已承认遗漏并重新系统核对所有配置必填字段。修复仅传入真实 `model.is_encoder_decoder`，未修改原生调度或缓存行为。完整失败、环境记录和空 trace 保留在私有 runs；summary 缺失由采集器如实列为 collection error，不给予验收。

独立审查另提出：显式冻结 reserve-full-ISL/长 prefill 阈值并记录生效配置；汇总计数不能代替因果验收。两项已落实为 `effective_config` 事件及独立 `tests/native/test_native_replay.py --trace` 检查。该检查直接读取原始事件，验证 allocation 嵌套中的 A 淘汰、C 原生查找、touch 保护、同初态、同到达、唯一改序、区间、FIFO 和释放。

## 实测结果与验收（2026-11-02）

起始 origin/main 为 `b353401aaa69a8938272d80aa48eee313f09e789`。成功被测项目提交为 `0288eb4e444c86d33b880b232e820bad4fb81a0b`；Linux x86_64、Python 3.12.16、vLLM 0.32.0。子进程记录真实解释器、类文件路径和关键模块 SHA-256，独立审查重新对固定上游源码字节计算，15 项全部匹配。模型 config 哈希也匹配冻结输入。原始 source-before/after 完全相同，65 个项目文件哈希与被测 Git commit 的字节一致。

下表每组重复 3 次均完全相同；去掉 case 标签后，整条 trace 也逐事件一致。单位是 **scheduled/dispatched tokens**。

| 条件 | C 原生首次命中 | B prefill | C prefill | B+C prefill | B+C decode |
|---|---:|---:|---:|---:|---:|
| 压力，B→C | 0 | 49 | 33 | 82 | 4 |
| 压力，测试 C→B | 32 | 49 | 1 | 50 | 4 |
| 充足，B→C | 32 | 49 | 1 | 50 | 4 |
| 充足，测试 C→B | 32 | 49 | 1 | 50 | 4 |

A 预热另计 33 prefill + 2 decode，不包含在 B+C 合计中。每个请求最终均输出 3 tokens；所有条件抢占次数、repeated prefill、repeated decode 都是 0。开启了 chunked prefill，但这组固定输入实际没有 chunk dispatch；它与抢占回归都仍未验证。

原始成功 trace 共 1,992 行、519 对严格嵌套的观测 call/return、144 个完整 FIFO 调度输出（含空 step）。以下行号对应私有 `attempt-02/artifacts/002-trace.jsonl`；公开[因果摘录](../../evidence/2026-11-02-m1-native-bc-replay/causal-excerpt.json)保留相同 `source_line` 和原始 trace 哈希。

- 压力 B→C：L61 的 A 缓存两块 mapped/ref0/free，free queue `[3,4,2,1]`。L66–77 的 B 原生分配获取四块，L73/75 淘汰 A 的 block 2/1；L79 的 C 原生查找命中 0。C 此次准入失败，直到 step10 的 L142 才派发 `[0,33)`；没有把失败分配当成工作量。
- 压力 C→B：L250 是唯一测试重排，L253 命中 32；L255–256 touch 将 A 两块引用升高并移出 free queue，L266 派发 `[32,33)`。C 完成后，B 仍于 step10 的 L323/325 淘汰 A。因此结论是保住 C 本次复用机会，不是永久保留 A。
- 充足 B→C / C→B：L452 / L584 命中 32，L461 / L607 派发 C 的末 token；整个条件没有 A 的对应淘汰。两组只增加总块数，其他生效配置一致。

### 原始标准逐项结论

| 标准 | 证据与观察 | 结论 |
|---|---|---|
| A 调用链 / 固定来源 | runtime + native_objects、15 项模块 SHA、真实调用嵌套；本页固定源码链接 | 通过 |
| B 四组独立初态 | inputs、effective_config、A_cached_and_evictable、enqueue、test_only_reorder；12 cases | 通过 |
| C 直接因果 / 负对照 | 上述原始 trace 行、公开因果摘录、数值结果；先淘汰再失去命中且增加32个 prompt 派发 | 通过 |
| D 完整异步生命周期 | 每个 output 关联所属 step 并 FIFO 消费；pending最大2最终0；所有请求长度停止，引用/占位/in-flight/stale归零，free queue恢复4/9块 | 通过 |
| E 派发工作计数 | 原始逐step区间和独立位置集合复算；prefill/decode/重复分开；无 tensor 或性能声称 | 通过 |
| F 重复与分层验证 | 四组各3次一致；完整 make check 为71 passed/1 native skipped；实际native CLI 12cases完成；缺依赖CLI退出1 | 通过 |

`attempt-02` 的 schema 2 完整性、`--require-command-success` 和人工逐项审查后的 `--require-accepted` 三个门槛分别通过。工具只检查记录契约；机制结论来自原始 trace 和独立源码/因果审查。独立审查者为 `source_audit` 与 `counterexample_review`，最终没有剩余阻断项。[审查与失败汇总](../../evidence/2026-11-02-m1-native-bc-replay/review.json)保留发现、修复与边界。

本地 `local-check` 的71个测试虽通过，但构建阶段获取 hatchling 被网络沙箱阻断，完整命令退出2；独立审查及时指出不能以测试通过代替 make check 通过。另建 `local-check-network` 后完整检查通过，wheel 和 sdist 均构建成功。两条记录分别保留，没有覆盖旧失败。原生环境不安装 pytest；真正原生入口和标准库 trace 审查都已在 Linux 执行。缺依赖负测试用同一 Linux 解释器的 `-S` 隐藏 site-packages，得到 `ModuleNotFoundError: vllm`、退出1且没有成功 summary，不改写共享环境。

唯一计算资源分配申请上限20分钟，实际5分50秒，作业 COMPLETED、ExitCode `0:0`，squeue 不再列出该作业，退出前 GPU memory 为1 MiB。没有启动模型或 GPU 性能实验。集群 `sacct` storage 禁用，因此释放证明采用 `scontrol`、squeue 与退出后显存快照；原始输出保留在私有材料中。

### 证据位置、交付与后续建议

私有材料统一位于新 worktree 的 `runs/m1-native-bc-replay/`：`attempt-01{,-data}`、`attempt-02{,-data}`、`trace-audit-02`、`fail-closed{,-data}`、`local-check`、`local-check-network`，以及 `resource-release.txt` 和 `gpu-after.csv`。原始 manifest、acceptance、配置、来源、日志、完整 trace、失败和汇总均保留。公开最小字段导出在 [`evidence/2026-11-02-m1-native-bc-replay/`](../../evidence/2026-11-02-m1-native-bc-replay/)，未调用 readiness 导出器、未覆盖历史证据，移除了实际解释器/内部机器/账号/路径，保留数值、定义、源哈希和原始行号。

专项分支 `feat/m1-native-bc-replay`；交付 commit 与未合并 PR 以交接消息和 Git HEAD 为准。成功被测代码、配置、测试与 Makefile 均来自 `0288eb4`，后续交付改动仅补技术说明和显式公开证据。原工作区始终保留 `feat/project-workflows`，其 `docs/workflows.md` 未提交修改前后 SHA-256 均为 `c3dca495872357a584ecc2464dc92dafa5e4467d16237f4c11dfdf9ab096bb82`。

后续设计应以“准入分配之前的复用机会”为观测对象，避免额外 `get_computed_blocks` 的条件性 event 副作用；保留区间计数、相同到达的容量负对照和 FIFO 排空验证。进入策略工作前仍需独立设计连续新鲜 WAITING 窗口、屏障、等待保护与原生回归。本轮只完成 M1 原生机制与派发基线，没有实现 M2、自定义调度器或性能优化。
