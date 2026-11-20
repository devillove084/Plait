# RDMA and registered memory

Plait separates **ordering** from **bulk movement**. Consensus commands carry
tensor identities, offsets, small partials, and `RegionDescriptor`s. Large
buffers stay in page-aligned registered memory.

## Why `rdma-io`

The backend uses the community [`rdma-io`](https://github.com/youyuanwu/rust-rdma-io)
crate (`0.0.2`, low-level synchronous surface via `default-features = false`).
It supplies safe RAII wrappers for devices, protection domains, completion
queues, memory regions, queue pairs, work requests, and completions.

Alternatives were evaluated before choosing it:

- `ibverbs` (jonhoo) is well maintained but exposes SEND/RECV only, not the
  one-sided `RDMA_WRITE`/`RDMA_READ` this data plane requires.
- `rdma` and `async-rdma` are attractive APIs but currently fail to resolve
  because their `hex-simd 0.5` dependency pulls yanked `simd-abstraction 0.5.x`.
- `rdma-core-sys` expects a musl build wrapper and toolchain this repository
  does not use.
- A project-local C shim was implemented first, then replaced by `rdma-io` once
  it was verified to build against the installed rdma-core. Maintaining verbs
  struct layouts and QP state transitions in Plait is unnecessary dependency.

`rdma-io-sys` is used directly only for the `ibv_qp_attr`/GID types needed to
choose a non-zero RoCE GID index and to set the RC route explicitly.

## Layers

| Layer | Responsibility |
| --- | --- |
| `plait-mem` | `AlignedBuffer`, size-class `BufferPool`, `RegionDescriptor`, `RdmaPlane` contract, validated loopback plane |
| `plait-rdma` | `rdma-io` RC queue pairs, aligned MR registration, one-sided read/write, completion polling |
| `plait-net` | framed TCP fallback with reused send/receive buffers |

`RegionDescriptor` carries:

```text
node, region, addr, offset, len, rkey, kind
```

`addr` is the base virtual address of the registered allocation; `offset`
selects a range. Loopback uses the offset; verbs adds it to `addr` to form the
remote virtual address. `rkey` authorizes the remote operation.

## Data path

1. `RdmaDevice::open_first` opens an active verbs device, allocates a PD,
   selects the active port and a usable GID index.
2. `create_qp` creates an RC QP and a completion queue, moves the QP to INIT,
   and returns the local endpoint (QPN, LID, GID, GID index).
3. Peers exchange endpoints out of band (today the caller supplies them; the
   two-QP self-test uses both endpoints in one process).
4. `register` pins a page-aligned `plait-mem::AlignedBuffer`; the resulting
   `Registration` exposes `lkey`, `rkey`, `addr`, and a transport-neutral
   `RegionDescriptor`.
5. `post_write`/`post_read` build `SendWr` with `WrOpcode::RdmaWrite` or
   `RdmaRead`, post it through `rdma-io`, and poll the CQ for the matching
   `wr_id`.

Large payloads never enter the consensus log. Small indices and partials stay
inline.

## Soft-RoCE setup

On a host with the `rdma_rxe` kernel module and `CAP_NET_ADMIN`:

```sh
sudo modprobe rdma_rxe
sudo rdma link add rxe0 type rxe netdev <interface>
rdma link show
ibv_devices
cargo run -q -p plait-node -- rdma-selftest
```

Expected self-test output on a working device:

```text
plait rdma: <device> loopback write/read passed (128 bytes)
```

Some distributions package the module separately (`rdma-core` /
`kmod-rdma`); the module must match the running kernel. Soft-iWARP (`siw`) is an
alternative software provider if the host supports it.

## This development container

The container cannot execute the hardware path:

- `/sys/class/infiniband` and `/dev/infiniband` are empty;
- the running 4.19 kernel has no `rdma_rxe.ko`;
- `CAP_NET_ADMIN` and `CAP_SYS_MODULE` are not in the effective/bounding set,
  so `rdma link add ... type rxe` and `modprobe rdma_rxe` cannot work.

The backend still compiles and links against the installed `libibverbs`
(`rdma-core` 55 headers/libraries locally; `rdma-core` 48 on the remote
`pro5000` container). On `pro5000`, `/sys/class/infiniband` and
`/dev/infiniband` are also empty; only a virtio NIC is exposed (no Mellanox/
Broadcom RDMA PCI function), the kernel has no `rdma_rxe` module, and the
container lacks `CAP_NET_ADMIN` and `CAP_SYS_MODULE`. The self-test therefore
exits successfully with a clear skip there too.

```text
plait rdma: skipped (no verbs device; configure Soft-RoCE or an RNIC)
```

To validate the hardware path, run the repository on a host with an RNIC or a
configured Soft-RoCE link. `Registration::descriptor` produces the same
`RegionDescriptor` consumed by the loopback plane, and
`post_write_descriptor`/`post_read_descriptor` accept it.

## Performance notes

- Memory registration uses page-aligned `plait-mem` buffers, so registration
  does not need a copy into a separate pinned pool.
- The buffer pool recycles aligned staging buffers per size class.
- TCP framing reuses send/receive buffers; the RDMA path never serializes bulk
  tensor bytes into the consensus log.
- The reference model runners use loopback memory for bulk Q and small inline
  consensus payloads. Wiring every runner to the verbs plane is a data-path
  optimization; the backend and descriptor bridge are in place.
- Completion-queue batching, multi-QP striping, and per-domain consensus groups
  remain future work for large clusters.
