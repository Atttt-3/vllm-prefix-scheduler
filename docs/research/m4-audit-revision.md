# M4 独立审查修订授权与来源边界

2026-11-03，用户在看到三行补丁、真实60case正控与八个边界反例后，明确授权：
“授权独立审查修订，不追加 GPU”。本记录前瞻修订[原冻结设计](m4-design.md)中分析程序必须与执行同一提交的约束；真实实验冻结不变。

## 已发生的失败不改写

实际执行提交为 `5525bda3eff2dad20f614eaca7718383c31448e4`。三个allocation全部结束，独立预算3/3、60申请分钟已耗尽。
原冻结导出在GPU第11个控制resource_residue停止：将GPU合并trace L1396的case_id为空的generation_drained快照中non_null_ref_sum从0改1，旧审查器仍返回0。
共有8条引擎boundary/before_shutdown排空记录未进入逐case检查；真实284条drain均无残留。该漏洞证明旧F失败封闭未通过，不代表真实发生泄漏。
原失败progress、派生输入、accepted=true的反例审查结果、stderr及导出失败全部保留；原公开导出未产生文件。
补充离线诊断中其余7项GPU和9项资源控制按预期完成，也没有将旧49项改称全部通过。

## 允许的最小修订

GPU事件按case筛选前，对所有m4_generation_drained复用既有audit_drained检查，只有三行语义新增。
未改负控选择、生产调度/等待策略、observer/fixture、模型参数、输入、采集器、原始数据或任何已执行提交。
独立模块候选在真实原件上完成60case正控及四模式×boundary/before_shutdown的8个单字段反例；每个反例均被拒绝。
候选合同SHA256为 `d44bfa03fcef0930f450dae0c222c0df3162e7a2dff215f2c9a1bfe6a4371c44`。
这只验证窄修复，不宣称涵盖所有可能篡改，也不替代完整控制和导出。

新审查版本必须单独提交。导出需显式提供完整analysis-commit，其对象必须是Git commit，不能隐式使用HEAD、缩写、tree或tag。
执行SHA固定为5525bda；actual runtime全部108份source哈希仍逐一对照该执行Git树，collector/raw/input/job与双角色资源绑定保持原样。
执行树、分析树和当前tracked执行目录的文件集合必须相同，禁止添加、删除或移动执行文件。分析树相对执行树只能变化以下四个路径：

- `src/vllm_prefix_scheduler/m4_contract.py`：全GPU排空记录检查。
- `scripts/export_m4.py`：显式双SHA绑定及公开来源说明。
- `tests/test_m4_contract.py`：八个边界反例回归。
- `tests/test_m4_export.py`：真实Git对象/目录/工作区/白名单负控。

其他producer、fixture、配置、原始回归入口及所有实际执行源码必须逐字节等于执行冻结。
当前每个执行目录文件须逐字节等于指定分析Git树；结束时再核目录和所有已绑定文件。默认省略analysis-commit仍保留原全字节严格绑定，不自动放宽。
四路径白名单不替代对三行语义修复及导出绑定差异的独立审查。

## 修订后的接受门槛

新审查版本重审全部native/CPU/GPU/旧回归原件与全部三次分配，再在新私有目录执行native17、CPU5、GPU18、resources9共49个真实子进程控制。
全部通过且最终hash稳定后，方可在此次明确授权的audit-only范围内导出接受结论；否则保持未验收。
summary与provenance同时记录tested_commit、analysis_commit、审查范围、旧F为FAILED，以及四路径旧/新source哈希。
旧失败保留由独立交接审查逐一复核私有483文件哈希；公开字段明确这属于独立核验，不是导出器自行认证。
不得把分析提交或最终文档交付提交声称为GPU被测提交。

不追加GPU，不改写旧历史，不合并PR，不进入M5。最终实测数值、审查SHA、控制结果与交付状态见[M4交接](m4-handoff.md)。
