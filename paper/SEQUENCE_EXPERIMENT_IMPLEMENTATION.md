# 附加序列实验实施说明

早期以 MS MARCO/BEIR 文本检索和 FLIP/TAPE/ProteinGym 蛋白质任务为核心的
计划已经停止。正式附加实验固定为：

1. GenomicBenchmarks 全部 8 个任务；
2. Long Range Arena 的 ListOps、Text、Retrieval、Pathfinder 四个任务。

UEA-30 已完成一次全任务探测，但其大量极小训练集与当前宽 RRLSSO 编码器不
匹配，因此从论文正式实验、主要指标与通用性结论中移除。相关 runner 和本地
结果保留，作为可复现的负结果与适用边界分析，不再占用正式实验预算。

统一的任务范围、最小自跑策略、公平引用规则和实施顺序见：

- `docs/auxiliary_experiments.md`
- `paper/EXPERIMENT_PLAN.md` 第 4、7、8 节

正式附加实验默认完整训练 RRLSSO，并在严格匹配公开协议时引用已发表基线；
每个套件最多补一个 MHA 管线锚点。主视觉实验继续承担完整的
MHA/LSSO/RRLSSO 受控比较。
