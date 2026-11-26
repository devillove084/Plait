# Plait TP transport and NCCL compatibility

The vLLM fork uses Plait's peer transport for tensor parallel groups when
`VLLM_USE_PLAIT=1`. Each worker publishes its own endpoint and contributes
to collectives with its local accelerator. CUDA, ROCm, Huawei CANN, and PPU
HGGC workers share the same wire format.

## Start a peer group

Start the vLLM workers with the same TP layout and the vendor's PyTorch
runtime on each worker. The fork uses Gloo to exchange endpoints and for
its existing CPU control operations. Set `VLLM_HOST_IP` to the address
that the other workers can reach when automatic address selection chooses
the wrong interface. The Rust layer binds a TCP port assigned by the OS.
`PLAIT_NCCL_TIMEOUT_MS` sets the TCP bootstrap and socket I/O timeouts and
defaults to 120 seconds.

Each worker binds before the Gloo all-gather. The ordered list of endpoints
and per-launch tickets forms the group manifest. Direct connections verify
its digest, the slots at both ends, and the wire lane. A restarted worker
has a new ticket. Slot numbers keep the vLLM tensor layout and determine the
pairing order. Every participant uses the same startup protocol.

The managed launcher sets `PLAIT_TP_GROUP_VERSION` from its committed launch
claim. Plait includes the group ID and epoch in the manifest digest. Peers
with different group versions fail the handshake. A managed replacement
and its predecessor can hold separate communicators at the same time.

## Execute a collective

For small all-reduces, each worker sends its complete contribution to each
peer and reduces the received values in slot order. A worker advances the
send and receive on each link independently under one deadline. The planner
uses this path up to 1 MiB of input for two workers. For larger groups it caps
the replicated traffic per worker at 64 KiB.

Larger all-reduces divide the tensor into contiguous shards and assign one
shard to each slot. In each round a worker sends its contribution to a
partner's shard and receives that partner's contribution to its own shard.
It reduces the received values in slot order and exchanges the completed
shard with its peers. Reduce-scatter ends after the first phase.
All-gather exchanges complete input shards in slot order. Broadcast and
reduce use the source or destination specified by that call.

Frame writes submit the header and payload together. Before consuming a
payload, each receiver checks the collective sequence, operation, dtype,
size, and stage. The point-to-point lane is separate.
A send stages the payload and enqueues it for that peer, so opposing sends
can be posted before either receive. The caller's stream is synchronized
before device input is read. Collective calls return after the host exchange
and output copy.

A failed TCP lane closes the affected communicator's sockets. Independent
communicators retain their own sockets, sequence counters, and send queues.
Each group holds two sockets per peer. This transport uses a fixed group
for its lifetime. Changing membership also requires a new vLLM tensor
layout and activation of a new application group.

## Compatibility API

The NCCL C ABI and Python `NCCLLibrary.ncclCommInitRank` keep the legacy
`(nranks, uniqueId, rank)` bootstrap. Its TCP star and local fast paths are
available to callers using that ABI. The Python `Group.join_root` and
`Group.join` helpers use the same star, with slot assignment at rendezvous.
The vLLM fork's `PlaitCommunicator` calls the peer bootstrap directly.

| Backend | Setting | Device adapter |
| --- | --- | --- |
| NVIDIA CUDA | `PLAIT_NCCL_BACKEND=cuda` (default) | CUDA driver copies and stream synchronization |
| AMD ROCm | `PLAIT_NCCL_BACKEND=rocm` | HIP runtime copies and stream synchronization |
| Huawei Ascend | `PLAIT_NCCL_BACKEND=huawei` | CANN runtime copies and stream synchronization |
| PPU | `PLAIT_NCCL_BACKEND=ppu` | HGGC runtime, with an adapter ABI for older deployments |

The peer host reduction supports FP32, FP16, BF16, and E4M3, and integer
sums. FP16 and BF16 accumulate in FP32 before the output is rounded.
Use an eager vLLM engine for this path.
