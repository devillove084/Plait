# Plait 全量技术报告

版本基线：`7e5db6d`（2026-11-20）  
范围：从无 rank 0 的融合运行时，到 CUDA peer-memory collective、4+8 异构调度、TLA 模型，以及失败实验记录。

---

## 0. 一句话结论

Plait 把 collective 的控制信息、排序和局部贡献放进一条无主有序日志，把大 tensor 放进注册内存和 GPU peer 数据面。当前实现能在单机 8 卡 RTX PRO 5000 上跑出比 NCCL 更快的 allreduce，也完成了 4+8 异构卡的非对称行分片。它仍然是研究原型，不是生产训练系统。

---

## 1. 目标与边界

### 目标

1. 去掉固定 communicator 和 rank 0。
2. 用稳定 `NodeId` 和有序日志承载成员、依赖、贡献和提交。
3. 控制面与数据面分离：小 partial/descriptor 走日志，大 tensor 不走日志。
4. 让异构 GPU 按计算和链路能力分活，而不是按卡数平均分。
5. 用 TLA 检查调度器、ordering safety/liveness 和统一系统不变量。
6. 与 NCCL 做同机、同 buffer、同迭代次数的对照，不用 API 覆盖冒充性能。

### 边界

- 当前数据面以单机 PCIe/NODE/SYS 为主，跨机 GPUDirect RDMA 未完成。
- LL/LL128 已实现但保留为 opt-in，PCIe 默认走 ring + Simple。
- TLA 是有限抽象，不是 Rust 代码的 refinement proof。
- 下文的性能数字只对应文中写明的机器、构建和尺寸。

---

## 2. 架构总览

```mermaid
flowchart LR
  C[Client / local task] -->|propose command| L[(Plait ordering log)]
  L -->|ordered commands| R[Fused runtime]
  R -->|materialize shard| T[(Tensor state)]
  T -. bulk bytes .-> M[Registered memory / peer data plane]
  L --> S[SCC ready frontier]
  S --> R
```

### 2.1 控制面

- `NodeId`：稳定 128 位身份，不是数组下标。
- 无主日志：任意 voter 可提案；ballot 只用于恢复。
- 命令类型：`DeclareOp`、`Contribute`、`Fuse`、`Publish`、成员变更、设备注册、计算任务。
- 复制与快照：成员加入通过 staged learner + 快照，成员退出通过 staged remove。
- 控制传输：Tokio actor，TCP 帧；数据面另行处理。

### 2.2 融合运行时

`OpSpec` 描述一个 fused operator：

```rust
OpSpec {
    id,
    kind: FuseKind,
    contributors: BTreeSet<NodeId>,
    tensor: TensorKey,
    epoch,
}
```

`Runtime::apply` 是确定性的状态机。所有 contributor 的 `Contribute` 到齐后，`try_fuse` 在本地执行 `Sum`、`Max` 或 `Gather`，物化 tensor shard。没有单独的 allreduce 消息阶段。

### 2.3 执行调度

`ExecutionGraph` 维护在线 SCC 凝聚图：

- 新提交的 instance 形成一个 singleton component；
- 新依赖边如果闭合环，就把路径上的 component 合并；
- ready frontier 是没有任何未执行外部依赖的 component；
- 执行按 component 出队，冲突边不拆开。

`ready_batches_for(local)` 只在独立 SCC 之间按本地设备访问数、seq、instance id 重排，不改变冲突顺序或 quorum。

### 2.4 数据面

- CUDA IPC/VMM 分配与导入；
- peer 指针 + device flag 的 ring kernel；
- 专用 reduce-scatter/allgather/alltoall/broadcast kernel；
- 注册内存和单向 descriptor 为后续 RDMA 预留；
- `PeerGroup` 管理每卡 context、stream 和通道。

---

## 3. 代码地图

| 路径 | 作用 |
|---|---|
| `crates/plait-core` | NodeId、命令、配置、tensor、设备描述 |
| `crates/plait-consensus` | 无主 ordering、ballot、恢复、快照、SCC 执行 |
| `crates/plait-fabric` | `OpSpec`/`Contribute`/`Fuse` 的确定性运行时 |
| `crates/plait-collective` | CUDA peer-memory collective、IPC/VMM、拓扑、工具 |
| `crates/plait-parallel` | TP/PP/EP/DP/CP/SP 计划、DP/PP/MoE、异构 LP、rejoin |
| `crates/plait-node` | 节点服务、benchmark、matrix/hetero demo |
| `crates/plait-net` | TCP 控制传输 |
| `crates/plait-rdma` | ibverbs probe、GPUDirect 注册探测 |
| `crates/plait-gpu` | CUDA 执行器 |
| `crates/plait-device` | 设备发现、能力与调度策略 |
| `specs/` | TLA+ 模型与 TLC 配置 |

---

## 4. Collective 实现与结果

### 4.1 从 host ring 到 device ring

早期 ring 由 host 编排 peer copy，Nsight 里 host peer copy、分配和同步占主导：约 21 ms 对 3.4 ms 的 device-resident 路径。把 ring 状态和 flag 放进显存后，性能提升约两个数量级。

后续实现：

- `plait_ring_allreduce`：peer 指针、device flag、`float4` 向量化、channels；
- `plait_alltoall_push`：rank-owned output，避免目的端汇总；
- `plait_scatter_pull` + device allgather：broadcast；
- 持久 workspace：`DeviceRingWorkspace`、`AllToAllWorkspace`、`BroadcastPullWorkspace`、`Ll128Workspace`。

### 4.2 最新 allreduce 对照

机器：8× RTX PRO 5000，单机，PCIe NODE/SYS，无 NVLink；NCCL 2.30.4；CUDA 13.2。

| 大小 | Plait device ring | NCCL |
|---:|---:|---:|
| 1 MiB | 11.0 GB/s | 4.82 GB/s |
| 16 MiB | 16.3 GB/s | 12.72 GB/s |
| 64 MiB | 15.7 GB/s | 13.60 GB/s |

这是 2026-11-20 的新构建。更早的构建在同一台机器上曾测到 20.6 / 27.4 / 27.5 GB/s，说明当前机器的时钟和构建批次存在波动；报告以最新可复现实验为准。

### 4.3 六种 collective 历史结果

8 ranks，设备路径，较早期构建：

| 大小 | allreduce | broadcast | reduce | allgather | reduce-scatter | alltoall |
|---:|---:|---:|---:|---:|---:|---:|
| 1 MiB | 20.6 | 21.0 | 20.4 | 38.7 | 26.5 | 24.9 |
| 16 MiB | 27.4 | 40.7 | 27.6 | 76.5 | 41.8 | 43.5 |
| 64 MiB | 27.5 | 44.3 | 27.5 | 83.1 | 39.3 | 47.2 |

单位 GB/s，algbw。64 MiB 时 NCCL 的 broadcast/reduce-scatter/alltoall 约为 45.3/40.9/51.4，Plait 在这三项上略落后。1/16 MiB 的六项均高于同批 NCCL 对照。

### 4.4 最新专项

同一新构建：

| 操作 | 1 MiB | 16 MiB | 64 MiB |
|---|---:|---:|---:|
| reduce-scatter | 16.25 | 24.21 | 20.74 |
| allgather | 24.79 | 45.46 | 47.58 |

### 4.5 channels

`--channels 4`：

| 大小 | C=1 | C=2 | C=4 |
|---|---:|---:|---:|
| 1 MiB | 10.90 | 11.41 | 11.79 |
| 16 MiB | 16.16 | 15.92 | 15.55 |
| 64 MiB | 14.85 | 14.29 | 15.47 |

更多 channel 没有稳定收益。C=8 在 64 MiB 上出现 0.014 GB/s 的异常，已被记录为失败，不作为默认。

### 4.6 正确性对照

NCCL harness 把 `DP2×TP3×EP2` 的 12 个逻辑 contributor 映射到 8 个物理 rank，和 Plait fused sum 对 checksum：

```text
matrix 8x8x6   plait=103 nccl=103
matrix 64x64x64 plait=197 nccl=197
```

这证明 fused sum 语义一致，不证明吞吐。

---

## 5. 4+8 异构调度

### 5.1 LP 模型

```text
minimize T
subject to  a_i * x_i + b_i <= T
            sum_i x_i = 1, x_i >= 0

a_i = 1 / compute_i + message_bytes / bandwidth_i
b_i = latency_i
```

只有一个耦合约束，用参数搜索求最优 `T`。解出的分数份额再量化成整数 rank 权重，交给 `AsymmetricLayout::row_ranges`。

### 5.2 实际配置与结果

```text
4 × RTX 3090     PCIe, compute 1.0, 25 Gbps, latency 5 us
8 × RTX PRO 5000 UB,   compute 4.0, 200 Gbps, latency 2 us
```

```text
local_share  = 0.111
remote_share = 0.889
```

`plait-node hetero-plan` 输出：

```text
rows=8   local_rows=0  remote_rows=8
rows=64  local_rows=7  remote_rows=57
rows=256 local_rows=28 remote_rows=228
rows=64 equal_split_local=21 equal_split_remote=43
```

64 行时平均分给 local 21 行，LP 给 7 行。8 行时 local 被整数量化为 0，这是 `rows < ranks` 的边界，调度器应保留分数份额。

### 5.3 EPaxos 名称移除

代码和文档里的协议名称已统一改为 **Plait ordering**。保留的设计是：

- ballot；
- preaccept/accept/commit；
- fast/slow commit；
- recovery；
- dependency set 与 SCC 执行；
- 一节点一票，不按权重改 quorum。

### 5.4 重新加入

`RejoinCost` 在三种路径中取最小：

```text
StrictRecover
SnapshotReplay
Recompute
```

如果 shard 小且确定性强，重算可能比通信 + replay + recovery 便宜。这个策略只适用于确定性 shard；已提交命令仍然走共识。

---

## 6. LL / LL128 实验与失败记录

### 6.1 已实现

- `Protocol::{Simple, Ll, Ll128}`；
- LL device flag/ack；
- LL128 128-byte line、per-line flag、持久 workspace、warp-per-line；
- size/topology selection API；
- 2-rank LL128 正确性测试。

### 6.2 性能

| 版本 | 1 MiB | 16 MiB |
|---|---:|---:|
| 初始 LL128 | 0.049 | 0.054 |
| persistent workspace | 0.080 | 0.086 |
| 多 block line group | 0.631 | 1.317 |
| warp-per-line | 1.099 | 2.254 |
| 8 lines per warp | 1.250 | 2.188 |
| embedded flag，2 ranks | 0.118 | 未测 |

最终没有超过 NCCL 的 4.82/12.72。LL128 仍是 opt-in，不是默认。

### 6.3 失败点

1. 初始 per-line flag + 每行 block 串行等待，吞吐只有 0.05 GB/s。
2. 多 block 没有解决 line 级同步开销。
3. embedded flag 把 flag 放进 line 尾部，2-rank 通过，8-rank 出现：
   ```text
   LL128 mismatch index=2 actual=2.002 expected=3.004
   ```
4. 加 per-step recv buffer 后消除超时，但 8-rank 仍有贡献丢失。
5. 修正 allgather chunk 索引、step-tagged flag、清 data flag 后，仍无法稳定通过 8-rank。
6. vector/per-step 版本有一次 8-rank 超时，最终回滚。

结论：LL128 的 128-byte line 加 per-line flag 在 NVLink 上合理，在这台 PCIe 机器的现有实现里不划算。PCIe 默认选择 ring + Simple。

### 6.4 其他回滚实验

- Bruck allreduce：8.68 GB/s，回滚；
- broadcast chain/fanout/rounds：均慢于 pull-scatter + allgather；
- 多 stream alltoall：慢；
- `float4` push alltoall：慢；
- gather-sum allreduce：3.19/4.57/5.25 GB/s，未采用；
- resident two-shot：1 MiB 1.23 GB/s，16 MiB resident 复用需要重置，未通过；
- channels 2/4/8：没有稳定收益。

---

## 7. CUDA IPC、VMM、GPUDirect

### IPC

`crates/plait-collective/src/ipc.rs` 实现 `cuMemAlloc` + `cuIpcGetMemHandle` / `cuIpcOpenMemHandle`，`ipc-peer` 两个 OS 进程通过 TCP 交换 handle，跑两 rank allreduce。

```text
local same GPU: checksum=381 ok
pro5000 cross GPU: rank0 gpu0, rank1 gpu1, checksum=381 ok
```

本地跨 GPU 因 `CUDA_ERROR_PEER_ACCESS_UNSUPPORTED` 跳过，是预期行为。

### VMM

`VmmRegion` 实现 `cuMemCreate`、POSIX fd 导出/导入、`cuMemMap`、`cuMemSetAccess`。单进程 export/import 测试通过；跨进程 fd 传输（SCM_RIGHTS）未完成。

### GPUDirect

`RdmaDevice::probe_gpu_region` 用 ibverbs 注册 VMM region。当前 local 和 `pro5000` 都输出：

```text
plait gpu-rdma: skipped (no active ibverbs device)
```

两机 GPU buffer collective 因此没有真实验证。

---

## 8. TLA+ 模型与结果

### 8.1 模型

| 文件 | 内容 |
|---|---|
| `specs/PlaitExecutionGraph.tla` | 通用 SCC 调度器 |
| `PlaitExecutionGraphSmall.tla` | 3 节点实例 |
| `PlaitExecutionGraphFive.tla` | 5 节点实例 |
| `PlaitOrdering.tla` | ballot/preaccept/accept/commit/recovery/deps |
| `PlaitOrderingSmall.tla` | 2 instance safety + deps |
| `PlaitOrderingLive.cfg` | 2 instance recovery liveness |
| `PlaitOrderingTiny.tla` | 1 instance liveness |
| `PlaitUnified.tla` | 容量、计算、融合、提交、执行 |
| `PlaitUnifiedSmall.tla` | 3 rank/4 op 不变量 |
| `PlaitUnifiedTiny.tla` | 统一模型 liveness |

### 8.2 TLC 结果

```text
3-node scheduler        pass
5-node scheduler        pass
ordering safety         pass, 271,441 distinct states
ordering liveness       pass, 1,986,053 generated states
unified safety          pass, 738,227 distinct states
unified liveness        pass, 351 distinct states
```

检查的性质：

```text
Safety
ExecutedOnce
Agreement
CommittedStable
CommittedDepsStable
AssignedOnce
CapacityBound
ContributorsExact
FusionSound
CommitAfterFusion
ExecuteAfterCommit
DependencyClosure
```

### 8.3 建模过程中遇到的错误

1. TLC 配置里的元组/集合常量解析失败，改成 concrete wrapper module。
2. 终止状态被 TLC 报 deadlock，使用 `-deadlock`，并在模型中明确终止状态。
3. `preaccepted` 记录和 `0` 比较触发 record/non-record 运行时错误，引入 `NoneRecord`。
4. ballot 无界导致状态无限增长，加入 `MaxBallot`。
5. `SlowCommit` 的 `c` 作用域错误，把量词移到外层。
6. `Execute` 漏写 `committed` 的 UNCHANGED，TLC 报 successor 不完整。
7. 多 instance liveness 状态空间大，先做 tiny，再跑 2 instance，约 1 分 56 秒通过。

### 8.4 结论边界

这些是有限抽象结果。没有消息丢失/重复/重排的显式网络模型，也没有 Rust handler 到 TLA action 的逐步 refinement。

---

## 9. 复现命令

```bash
just fmt-check
just lint
just test
just nccl-ops
just gate-collectives
just matrix-asymmetric

cargo run -q -p plait-node -- hetero-plan

./scripts/check-tla.sh
```

TLC 需要 `/opt/tlaplus/tla2tools.jar` 和 `/usr/local/bin/tlc`。

---

## 10. 当前缺口

1. M5 两机 GPU buffer collective：缺 RNIC/rxe。
2. ordering 的消息丢失、重复、重排模型：未完成。
3. Rust ordering handler 的逐步 refinement trace：未完成。
4. 4+8 目前只加权 row 维度，TP/EP 维度仍统一。
5. 64 MiB 的 broadcast/reduce-scatter/alltoall 与 NCCL 还有 2% 到 8% 的差距。
6. 知乎博客图片上传接口返回 `status=init`，草稿图片仍为占位符。

---

## 11. 参考

- HeteCCL, NSDI'26: https://www.usenix.org/system/files/nsdi26-hei.pdf
- ForestColl, NSDI'26: https://www.usenix.org/system/files/nsdi26-zhao-liangyu.pdf
- TCCL, ASPLOS'24: https://dl.acm.org/doi/10.1145/3620666.3651362
- AsymGroup: https://dtl.yonsei.ac.kr/docs/International_Journal/[2025][ACCESS]%20AsymGroup.pdf
- HexiScale: https://arxiv.org/html/2409.01143v2
- Blink: https://proceedings.mlsys.org/paper_files/paper/2020/file/cd3a9a55f7f3723133fa4a13628cdf03-Paper.pdf
- UB multicast: https://arxiv.org/pdf/2605.22428v1
- NCCL: https://github.com/NVIDIA/nccl

---

## 12. 提交索引

```text
7e5db6d Rewrite the Plait blog around the actual test run
981c5ca Rewrite the Plait blog in a more natural engineering voice
a915ec5 Add a hetero planner demo and strengthen the design blog
9d870c6 Add the rankless AllReduce blog draft
805d031 Check liveness for the single-instance ordering model
9aec544 Model explicit dependency stability in Plait ordering
613ad4b Rename the consensus design to Plait ordering
42cb205 Rename ordering model and check unified design liveness
3282918 Weight four 3090s below eight Pro 5000s and add rejoin cost policy
9d7ce4d Trace fused-runtime events against the unified TLA actions
3b27564 Solve heterogeneous row shares with a max-min LP
9de6676 Model the unified Plait control-plane abstraction
6591f00 Model-check 3-node and 5-node execution schedulers
49c65ff Plan 4+8 heterogeneous row shares from compute and link costs
34fe99f Record scheduler execution actions as a conformance trace
5678485 Select ring and simple protocol on PCIe topologies
fe04cf1 Run LL128 one warp per 128-byte line
8af6ccb Reuse a persistent LL128 workspace across iterations
96af1f2 Parallelize LL128 line groups across multiple blocks
87cfed9 Add LL and LL128 protocol kernels and selection hooks
0258124 Add asymmetric 4+8 work partitioning and locality-biased SCC dispatch
```

---

## 13. 结论

Plait 的当前默认路径是：

```text
无主 ordering + fused runtime
        +
PCIe ring / Simple data plane
        +
4+8 max-min LP 行分片
        +
有限 TLA 验证
```

它不是 NCCL 的全面替代。它证明了一个方向：把 collective 的控制面和异构调度放进可排序状态，可以在不牺牲 allreduce 性能的前提下，换来成员变化和异构分片的自由度。剩下的工作是跨机 RDMA、消息级 ordering 模型和 Rust refinement trace。
