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
