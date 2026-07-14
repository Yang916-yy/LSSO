# 附加序列实验实施说明（已重定向）

此前以 MS MARCO/BEIR 文本检索和 FLIP/TAPE/ProteinGym 蛋白质任务为核心的
附加实验计划已经停止，不再构成论文正式实验、代码开发优先级或 GPU 预算。
已有原型代码暂时保留，避免丢失可复用的数据管线和 masked sequence encoder。

新的正式附加实验固定为：

1. GenomicBenchmarks 全部 8 个任务；
2. Long Range Arena 的 ListOps、Text、Retrieval、Pathfinder 四个任务；
3. UEA-30 全部 30 个多变量时间序列分类任务。

统一的任务范围、最小自跑策略、公平引用规则和实施顺序见：

- `docs/auxiliary_experiments.md`
- `paper/EXPERIMENT_PLAN.md` 第 4、7、8 节

正式附加实验默认只完整训练 RRLSSO，并在严格匹配公开协议时直接引用已发表
基线；每个套件最多补一个 MHA 管线锚点。主视觉实验仍负责完整的
MHA/LSSO/RRLSSO 受控比较。
