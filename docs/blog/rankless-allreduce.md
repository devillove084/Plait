# GPU 不排队，把 AllReduce 塞进共识日志

先说结果。8 张 RTX PRO 5000 在同一台机器上，PCIe 拓扑，没有 NVLink。Plait 的 device ring allreduce 和 NCCL 2.30.4 跑同一组尺寸，结果是这样。

| 消息大小 | Plait | NCCL |
|---:|---:|---:|
| 1 MiB | 11.0 GB/s | 4.82 GB/s |
| 16 MiB | 16.3 GB/s | 12.72 GB/s |
| 64 MiB | 15.7 GB/s | 13.60 GB/s |

这不是把 NCCL 包一层。Plait 里没有固定进程组，没有 rank 0，没有“所有 GPU 先到齐再开始”的全局 barrier。集合通信被拆成一条条可排序的命令，命令本身就是状态转移。

这篇文章讲三件事。这个结构怎么跑起来，4×3090 加 8×Pro 5000 这种不对等集群怎么分活，哪些地方还没有做完。

![Plait 的控制面与数据面](assets/plait-architecture.png)

## 一条命令如何变成一次 AllReduce

传统 collective 先约定一个 group，再按 rank 编号决定每个进程读写哪一段。进程组里通常有一个根，或者一条固定环。成员变化、节点重启、异构卡混插，都要改 group 和 rank 映射。

Plait 换了一个起点。每个节点有稳定的 128 位 `NodeId`，它从不等于数组下标，也不因为成员变化而改号。节点把一条命令提案到无主日志，日志给出全序。命令可以是声明一个 fused operator，也可以是提交某个 contributor 的局部结果。

一个 operator 长这样。

```rust
OpSpec {
    id,
    kind: FuseKind::Sum,
    contributors: {A, B, C},
    tensor: TensorKey,
    epoch,
}
```

节点先执行 `DeclareOp`。每个 contributor 再提交 `Contribute`，带上自己的局部值。最后一个 contributor 的命令执行完，所有副本都在本地把值合起来，得到同一份 tensor shard。这里没有单独的 allreduce 消息阶段，也没有“先通信再计算”的切换。

这套路径的快慢，取决于两件事。日志只承载控制信息和小的 partial，大 tensor 走注册内存和单向 descriptor。互相冲突的命令按依赖排序，互不冲突的 SCC 可以并行执行。

我在 `ExecutionGraph` 里把 ready frontier 做成在线 SCC 凝聚图。一个新依赖边如果闭合环，就把环上的组件合并。执行时整个组件一起出队，冲突边不拆开。这样做的直接结果是，独立 operator 不会被一条无关的全局 barrier 拦住。

## 4×3090 和 8×Pro 5000 怎么分

异构集群里最容易犯的错，是让每张卡拿一样多的行。4 张 3090 和 8 张 Pro 5000 算力、链路都不同，平均分会让慢的一组成为尾延迟。

Plait 现在的做法分两层。第一层由 `HeterogeneousLayout` 解一个 max-min 线性规划。

```text
minimize T
subject to  a_i * x_i + b_i <= T
            sum x_i = 1, x_i >= 0

a_i = 1 / compute_i + message_bytes / bandwidth_i
b_i = latency_i
```

这里只有一个耦合约束，不需要外部 LP solver。对 `T` 做参数搜索，算出每个 rank 的分数份额，再量化成整数权重，交给 `AsymmetricLayout::row_ranges` 切成连续行区间。

4+8 的实际参数按硬件填。

```text
4 × RTX 3090     PCIe, compute 1.0, 25 Gbps
8 × RTX PRO 5000 UB,   compute 4.0, 200 Gbps
```

解出来是 `local_share=0.111`，`remote_share=0.889`。64 行的矩阵里，4 张 3090 合计拿约 7 行，8 张 Pro 5000 拿其余 57 行。8 行的 demo 会出现整数量化把 local 抹成 0 的情况，这个边界我在文档里写清楚了，调度器应该优先看分数份额，不要只看行数。

![4+8 异构行份额](assets/hetero-4plus8.png)

权重只影响工作分配。投票仍然是一节点一票。把 quorum 权重也改掉，会动到安全性，不是调度优化。

## 和 NCCL 比，赢在哪里，没赢在哪里

上面那张表是同一台机器、同一批 GPU、同一组尺寸的 device ring allreduce。Plait 的 ring kernel 把 peer 指针、flag 和通道状态留在显存，每次 collective 只在最后同步一次。NCCL 在这个 PCIe 拓扑上也走自己的 ring 或 tree，尺寸不同会换算法。

Plait 在 1 MiB 到 64 MiB 的 allreduce 上领先，原因不神秘。小消息少一次 host 往返，大消息的 ring 步进没有每步 host barrier。更重要的差别在结构，NCCL 的 communicator 仍是一个固定 rank 组，Plait 的成员和顺序是日志状态。

这套结构对异构卡有直接好处。4+8 的行分片、TP/EP/DP 的 contributor 集合、节点重加入，都可以是一条命令，不需要重建 communicator。NCCL 也能做拓扑感知和异构通信，但它的抽象中心仍是 rank 和 communicator。

边界同样明确。上面的数字只覆盖单机 allreduce，不是完整训练吞吐。矩阵校验用 12 个逻辑 contributor 跑 `DP2×TP3×EP2`，8×8×6 和 64×64×64 的 checksum 都是 103 和 197，和 NCCL 的 float allreduce 对得上。但矩阵路径包含日志协议、12 次提案和 CPU f64 参考，不适合拿来吹带宽。

## 参考了哪些论文

这套东西不是凭空想出来的。下面几篇对我的设计影响最大。

HeteCCL 把异构 GPU 集群的 collective 调度写成 MILP，用 DSL 描述硬件和传输约束，再合成近似最优 schedule。我的 4+8 线性规划比它小得多，但问题形状一致。

https://www.usenix.org/system/files/nsdi26-hei.pdf

ForestColl 讨论异构 network fabric 上的 throughput-optimal collective。它提醒我，拓扑不是一层 ring，NVLink、PCIe、跨 NUMA 的 SYS 链路要分开建模。

https://www.usenix.org/system/files/nsdi26-zhao-liangyu.pdf

TCCL 专门做 PCIe GPU 集群的 collective 路径和路由优化。我们这台 8 卡机没有 NVLink，PCIe 的瓶颈和它描述的很像。

https://dl.acm.org/doi/10.1145/3620666.3651362

AsymGroup 做异构硬件下 2D tensor parallelism 的非对称分组。HexiScale 走得更远，直接做非对称并行和非均匀分片。这两个工作支撑了“快卡多拿行”的方向，也解释了为什么不能只按卡数平均分。

https://dtl.yonsei.ac.kr/docs/International_Journal/[2025][ACCESS]%20AsymGroup.pdf

https://arxiv.org/html/2409.01143v2

Blink 是早期把 collective 调度做成可组合算法的系统。NCCL 的 LL128 则给了我另一条教训，128 字节 line 加 flag 在 NVLink 上合适，在 PCIe 上不一定。我们最后没有把 LL128 设成默认路径，PCIe 继续走 ring 加 Simple。

https://proceedings.mlsys.org/paper_files/paper/2020/file/cd3a9a55f7f3723133fa4a13628cdf03-Paper.pdf

华为 UB 的 multicast 工作也值得看。它把“复制”和“归约”往网络里推，和 Plait 把 partial 放进日志是两种不同的融合方向。

https://arxiv.org/pdf/2605.22428v1

## TLA 证明到什么程度

我们在 `specs/` 下留了五组有限模型，统一用 `tlc` 跑。

```text
PlaitExecutionGraphSmall   3 节点调度器
PlaitExecutionGraphFive    5 节点调度器
PlaitOrderingSmall         ballot / preaccept / accept / commit / recovery
PlaitOrderingTiny          单实例 liveness
PlaitUnified               容量 / 计算 / 融合 / 提交 / 执行
PlaitUnifiedTiny           统一模型 liveness
```

检查的性质包括 `Agreement`、`CommittedStable`、`CommittedDepsStable`、`CapacityBound`、`FusionSound`、`CommitAfterFusion`、`DependencyClosure`。ordering safety 的 TLC 跑过 271,441 个不同状态，统一设计的 TLC 跑过 738,227 个不同状态，没有报错。

这些结果证明的是有限抽象下的不变量。它不证明 Rust 代码和 TLA 每一步等价，也不覆盖消息丢失、重复、重排和完整 recovery liveness。我们后来把“严格 recovery”改成按成本选路，重算、snapshot replay、recovery ballot 三者取最小。这个选择只对确定性的 shard 成立，已经提交的命令仍然走共识。

## 还没做完的

两机 GPU buffer collective 还缺真实 RNIC。容器里没有 `/dev/infiniband`，`gpu-rdma` 只能明确 skip。

LL128 在 PCIe 上仍然慢，代码保留为 opt-in，没有污染默认路径。

TLA 还需要消息级模型和 Rust action trace 的逐步对拍。现在的结论应该读成“设计在这些有限状态里没撞墙”，不是“实现已经被证明”。

Plait 还是研究原型。它的价值不在某一个 benchmark 数字，而在于把 collective、异构调度和成员变化放进同一套可排序状态里。下一步我会把 rejoin 的成本模型接到节点 join 路径，再把 TLA 的 action 和 Rust trace 对起来。
