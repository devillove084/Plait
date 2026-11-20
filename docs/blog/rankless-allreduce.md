# 没有 rank 0 的 AllReduce，和 4 张 3090 的排班

我在一台 8 卡 RTX PRO 5000 上跑完这组数，第一反应是觉得有点怪。Plait 的 device ring allreduce 比 NCCL 快，1 MiB 是 11.0 对 4.82 GB/s，16 MiB 是 16.3 对 12.72，64 MiB 是 15.7 对 13.60。

这组数只说明这一台机器上的 allreduce。它不说明训练会更快，也不说明 NCCL 不行。NCCL 的 ring 和 tree 很成熟。奇怪的地方在于，Plait 的设计起点跟 NCCL 不一样，跑出来却没有输。

我想把这个设计讲清楚。它没有固定 communicator，没有 rank 0，集合通信的控制信息和 tensor 的局部结果都走同一条有序日志。大 tensor 仍然走注册内存和单向 descriptor，日志不搬字节。

## 先讲清楚它是什么

传统 collective 先圈一组进程，给每个进程一个 rank，再在组里选 ring 或 tree。成员变了，rank 映射和通信计划要跟着改。节点重启再加入，通常要重新建 communicator。

Plait 里没有这个圈。每个节点有一个稳定的 128 位 `NodeId`。它不是数组下标，成员变化时也不改号。节点把命令提到无主日志，日志给出全序。命令可以是在声明一个 operator，也可以是在提交某个 contributor 的局部结果。

一个 operator 大概是这样。

```rust
OpSpec {
    id,
    kind: FuseKind::Sum,
    contributors: {A, B, C},
    tensor: TensorKey,
    epoch,
}
```

节点先执行 `DeclareOp`。contributor 再提交 `Contribute`，带上自己的局部值。最后一个 contributor 的命令执行完，每个副本都在本地把值合起来，得到同一份 tensor shard。没有单独的 allreduce 消息阶段，也没有“先通信再计算”的切换。

我第一次把 `Contribute` 放进日志时，担心日志会变成数据通道。后来把边界划在 partial 和 descriptor 上，大 tensor 不进日志，这个问题才算收住。

![Plait 的控制面与数据面](assets/plait-architecture.png)

另一件事是执行顺序。`ExecutionGraph` 维护在线的 SCC 凝聚图。新依赖边如果闭合环，就把环上的组件合并。执行时整个组件一起出队，冲突边不拆开。独立 operator 不会被一条无关的全局 barrier 拦住。

这些听起来像分布式系统课上的东西。难的是，成员变化、异构分片和融合操作最后都要落在同一套状态里。日志和 SCC 只是把这个状态撑起来。

## 4 张 3090 和 8 张 Pro 5000

异构集群里最容易犯的错，是让每张卡拿一样多的行。4 张 3090 和 8 张 Pro 5000 的算力、链路都不同，按卡数平均分，3090 会拿三分之一，尾延迟却落在它们身上。

Plait 现在让 `HeterogeneousLayout` 解一个 max-min 线性规划。

```text
minimize T
subject to  a_i * x_i + b_i <= T
            sum x_i = 1, x_i >= 0

a_i = 1 / compute_i + message_bytes / bandwidth_i
b_i = latency_i
```

只有一个耦合约束，不需要外部 solver。对 `T` 做参数搜索，得到每个 rank 的分数份额，再量化成整数权重，交给 `AsymmetricLayout::row_ranges` 切成连续行区间。

4+8 的参数按硬件填。

```text
4 × RTX 3090     PCIe, compute 1.0, 25 Gbps
8 × RTX PRO 5000 UB,   compute 4.0, 200 Gbps
```

可以直接跑仓库里的 demo。

```bash
cargo run -q -p plait-node -- hetero-plan
```

它会输出

```text
plait hetero-plan: ranks=12 message_mib=64 local_share=0.111 remote_share=0.889
plait hetero-plan: rows=8 local_rows=0 remote_rows=8 predicted_makespan_ms=27.822
plait hetero-plan: rows=64 local_rows=7 remote_rows=57 predicted_makespan_ms=27.822
plait hetero-plan: rows=256 local_rows=28 remote_rows=228 predicted_makespan_ms=27.822
plait hetero-plan: rows=64 equal_split_local=21 equal_split_remote=43
```

64 行的时候，平均分会给 4 张 3090 共 21 行，LP 只给 7 行。8 行的 demo 会把 local 量化成 0，这是整数分配在行数小于 rank 数时的边界，不能当成最终调度结果。调度器应该保留分数份额。

![4+8 异构行份额](assets/hetero-4plus8.png)

权重只改工作分配。投票仍然是一节点一票。把 quorum 权重一起改掉，会动到安全性。

## 为什么这组数字能赢

Plait 的 ring kernel 把 peer 指针、flag 和通道状态留在显存。每次 collective 只在最后同步一次，ring 步进没有每步 host barrier。NCCL 在这个 PCIe 拓扑上也走自己的 ring 或 tree，尺寸不同会换算法。

结构上的差别更直接。NCCL 的 communicator 仍是一个固定 rank 组。Plait 的成员和顺序是日志状态。4+8 的行分片、TP/EP/DP 的 contributor 集合、节点重新加入，都可以是一条命令，不需要重建 communicator。

NCCL 也能做拓扑感知和异构通信。它的抽象中心仍然是 rank 和 communicator。Plait 把中心换成了有序状态。

这套结构有代价。日志要处理冲突、依赖和恢复，控制面比 communicator 重。大 tensor 必须绕开日志，否则会把共识层压垮。上面的数字只覆盖单机 allreduce，不是完整训练吞吐。矩阵校验用 12 个逻辑 contributor 跑 `DP2×TP3×EP2`，8×8×6 和 64×64×64 的 checksum 是 103 和 197，和 NCCL 的 float allreduce 对得上。矩阵路径包含日志协议、12 次提案和 CPU f64 参考，不能拿来吹带宽。

## 设计有没有撞墙

比性能更让我不放心的是顺序和恢复。只要有一个状态组合能出现两个 replica 提交不同值，前面的性能都没有意义。

我们在 `specs/` 下留了有限 TLA 模型，用 TLC 跑。

```text
PlaitExecutionGraphSmall   3 节点调度器
PlaitExecutionGraphFive    5 节点调度器
PlaitOrderingSmall         ballot / preaccept / accept / commit / recovery
PlaitOrderingLive          两 instance recovery liveness
PlaitOrderingTiny          单实例 liveness
PlaitUnified               容量 / 计算 / 融合 / 提交 / 执行
PlaitUnifiedTiny           统一模型 liveness
```

检查的性质包括 `Agreement`、`CommittedStable`、`CommittedDepsStable`、`CapacityBound`、`FusionSound`、`CommitAfterFusion`、`DependencyClosure`。

ordering safety 和 liveness 一共跑过 1,986,053 个状态，其中 271,441 个不同状态，没有报错。统一设计的 TLC 跑过 738,227 个不同状态。模型里包含异构容量、贡献完整性、融合前置条件、dependency 顺序和最终执行。

这些结果只说明有限抽象下的不变量。Rust 代码和 TLA 还没有逐步 refinement。后来我们把严格 recovery 改成按成本选路，重算、snapshot replay、recovery ballot 三者取最小。这个选择只对确定性的 shard 成立，已经提交的命令仍然走共识。

## 参考过的东西

HeteCCL 把异构 GPU 集群的 collective 调度写成 MILP，用 DSL 描述硬件和传输约束，再合成 schedule。我的 4+8 线性规划比它小，但问题形状一样。

https://www.usenix.org/system/files/nsdi26-hei.pdf

ForestColl 讨论异构 network fabric 上的 throughput-optimal collective。它让我把 NVLink、PCIe、跨 NUMA 的 SYS 链路分开建模，不再把拓扑当成一层 ring。

https://www.usenix.org/system/files/nsdi26-zhao-liangyu.pdf

TCCL 专门做 PCIe GPU 集群的 collective 路径和路由优化。我们这台 8 卡机没有 NVLink，PCIe 瓶颈和它描述的很像。

https://dl.acm.org/doi/10.1145/3620666.3651362

AsymGroup 做异构硬件下 2D tensor parallelism 的非对称分组。HexiScale 走得更远，直接做非对称并行和非均匀分片。它们支撑了“快卡多拿行”的方向，也解释了为什么不能只按卡数平均分。

https://dtl.yonsei.ac.kr/docs/International_Journal/[2025][ACCESS]%20AsymGroup.pdf

https://arxiv.org/html/2409.01143v2

Blink 是早期把 collective 调度做成可组合算法的系统。NCCL 的 LL128 给了我另一条教训，128 字节 line 加 flag 在 NVLink 上合适，在 PCIe 上不一定。我们最后没有把 LL128 设成默认路径，PCIe 继续走 ring 加 Simple。

https://proceedings.mlsys.org/paper_files/paper/2020/file/cd3a9a55f7f3723133fa4a13628cdf03-Paper.pdf

华为 UB 的 multicast 工作也值得看。它把复制和归约往网络里推，和 Plait 把 partial 放进日志是两种不同的融合方向。

https://arxiv.org/pdf/2605.22428v1

## 还没做完

两机 GPU buffer collective 还缺真实 RNIC。容器里没有 `/dev/infiniband`，`gpu-rdma` 只能明确 skip。

LL128 在 PCIe 上仍然慢，代码保留为 opt-in，没有进默认路径。

TLA 还需要消息级模型和 Rust action trace 的逐步对拍。现在的结论是设计在这些有限状态里没撞墙，不是实现已经被证明。

Plait 还是研究原型。它现在最有价值的地方，是把 collective、异构调度和成员变化放进了同一套可排序状态。下一步要接的是 rejoin 的成本模型，以及 TLA action 和 Rust trace 的对拍。
