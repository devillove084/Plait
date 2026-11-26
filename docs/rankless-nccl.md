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

### Assign different numbers of workers per host

For four TP workers across three hosts, start the same model on each host.
Use the control host's address and port for `MASTER_ADDR` and `MASTER_PORT`.

```sh
VLLM_USE_PLAIT=1 vllm serve "$MODEL" --enforce-eager \
  --distributed-executor-backend mp --nnodes 3 \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" --node-rank "$NODE_RANK" \
  --tensor-parallel-size 4 --plait-tp-layout 2 1 1
```

Set `NODE_RANK` to `0`, `1`, or `2` and append `--headless` on
hosts 1 and 2. Host 0 gets slots 0 and 1, host 1 gets slot 2, and host 2
gets slot 3. The host's visible device order determines the local workers'
device assignments. Each host registers its layout on the existing
vLLM rendezvous endpoint before creating workers. Conflicting layouts
fail startup. Workers that can share memory with the control worker
use shared-memory response queues. Other workers use socket queues.

The explicit layout applies to one TP group with DP, PP, and prefill
context parallel sizes of one. The counts describe workers per host.
The model's tensor-partitioning rules determine each worker's shard.
The placement is fixed for the lifetime of that vLLM process group.

### Start a committed group on several hosts

Run `scripts/serve-tiered.py` with `--host-launch-spec` to derive the
TP placement from a committed `GroupPlan`. Give each host the same
JSON document. The keys under `hosts` are the plan's host IDs.

```json
{
  "api_host": "00000000000000000000000000000001",
  "master_addr": "192.0.2.10",
  "master_port": 29500,
  "hosts": {
    "00000000000000000000000000000001": {
      "address": "192.0.2.10",
      "argv": ["/opt/runtime/bin/vllm"],
      "model_path": "/models/engine",
      "device_env": "CUDA_VISIBLE_DEVICES",
      "env": {"PLAIT_NCCL_BACKEND": "cuda"}
    },
    "00000000000000000000000000000002": {
      "address": "192.0.2.20",
      "argv": ["/opt/runtime/bin/vllm"],
      "model_path": "/models/engine",
      "device_env": "ASCEND_RT_VISIBLE_DEVICES",
      "env": {"PLAIT_NCCL_BACKEND": "huawei"}
    }
  }
}
```

The engine recipe can include host-specific `args` and `env`. Each
host address must be reachable by the control replicas and TP peers.
The script supplies the device mask, tensor parallel placement, and
peer group version from the launch claim.

1. Use `--print-launch-digest` with the common model, TP size, engine
   options, `--backend-port`, `--port-base`, `--advertise-http`, and
   `--host-launch-spec`. Put the digest in the plan's `launch_digest`.
2. Start the launcher on each planned host with those same options,
   the committed `--group-id` and `--group-epoch`, its own `--host-id`,
   and the control peer addresses. Use a vLLM fork with the hosted
   readiness hook. The API host serves HTTP. The other hosts run headless.
3. Activate the new version after all planned devices report `Ready`.
   The previous version begins draining. Inference responses identify
   the serving version in `X-Plait-Group-Version`.

Use a local `--verify-release` executable on each host to authorize
grant reuse after that host has closed its workers and transports.
Retirement requires every host's stop confirmation.

### Commit an automatic rollout or standby

Send one JSON line per operation to `plait-node serving`. Provide the same
control peers and reachable reply address used by the launchers.

```sh
plait-node serving --listen "$CONTROL_REPLY_ADDR" --peer "$VOTER_ID@$VOTER_ADDR" < requests.jsonl
```

`prepare_hosted` takes `id`, `group`, `expected_epoch`, `expected_active`,
`plan`, and `activation`. Use the latest known epoch as `expected_epoch`.
For an initial group, use epoch zero and a null `expected_active`. Group
and operation origin IDs are 16-byte arrays. A plan supplies the model,
role `Aggregated`, TP size, PP size of one, launch digest, and
device and host reservations. Represent the device and host maps as
arrays of key/value pairs. Launch each planned host as described above.

`replan_hosted` takes `id`, `forecast`, `candidates`, and `activation`.
Each candidate supplies `group`, `expected_epoch`, `plan`, `capacity`,
`ready_after_ms`, and `transition_loss_requests`. A capacity contains
`rps` and `valid_until_ms`, the Unix millisecond deadline for that
measurement. The planner checks device capacity, host memory, and current
grants, then scores the feasible candidates against the forecast's
demand, horizon, and minimum gain. Supply capacity measurements for the
candidate's topology. The control plane rechecks reservations and the
current epoch when it applies the selected plan. Retry with the same
operation ID and inputs to retrieve the original decision receipt.

| Activation | When the committed plan starts admitting requests |
| --- | --- |
| `OnReady` (default) | Activation waits for all worker grants to be ready and the HTTP endpoint to be published. The final report moves an active incumbent to `Draining` in the same operation. |
| `OnRevocation` | A ready standby is promoted when the named active epoch is revoked. If revocation arrives first, the last readiness report completes the switch. |

An explicit drain leaves a standby in `Ready`. A failed host's other
device grants are included in the same host-expiry transition. The model
and HTTP alias used by a replacement must match the incumbent. A prepared
standby keeps its grants while waiting, so account for those reservations
alongside the active group.

### Start groups from resident execution nodes

Use a version of `plait-node` with hosted recipes on the voters and
execution hosts. Install the launcher and each model profile on its host. The
profile supplies local executable paths, device visibility, weights and
backend settings. For example, one host can use this configuration.

```json
{
  "listen": "192.0.2.10:0",
  "host_memory": {"bytes": 68719476736, "pinned_bytes": 0},
  "lock_dir": "/var/lib/plait/execution",
  "devices": [{"ordinal": 0, "incarnation": 1, "device_bytes": 25769803776}],
  "launcher": ["/opt/engine/bin/python", "/opt/plait/scripts/serve-tiered.py"],
  "profiles": {
    "model-fp16": {
      "argv": ["/opt/engine/bin/vllm"],
      "model_path": "/models/model",
      "device_env": "CUDA_VISIBLE_DEVICES",
      "args": ["--dtype", "float16"],
      "env": {"PLAIT_NCCL_BACKEND": "cuda"}
    }
  }
}
```

```sh
plait-node run --compute-only --id "$HOST_ID" --seed "$VOTER_ADDR" \
  --hosted-execution /etc/plait/hosted.json
```

The node discovers control voters from seed IP addresses and registers
its devices and host memory. Repeat `--seed` for additional address
hints. Keep the host ID and lock directory stable across restarts.

Include a `recipe` in `prepare_hosted` or on a `replan_hosted` candidate.
The plan and recipe are committed together. Use 32 zero bytes for the
plan's `launch_digest` in this request. The CLI derives the digest from
the recipe. Set these fields on the recipe.

| Field | Meaning |
| --- | --- |
| `profile` | Installed profile name on each participating host. |
| `hosts`, `api_host` | Array of `{"host": node_id, "address": ip}` entries and the host ID serving HTTP. The hosts must match the plan. |
| `master_port`, `http_port`, `backend_port` | Distinct rendezvous, advertised HTTP, and private engine ports. |
| `api_model` | Model alias reported by the HTTP server and used by the ingress. |
| `max_model_len`, `gpu_memory_bps`, `prefix_caching` | Common engine settings. A value of 8000 sets GPU memory utilization to 80 percent. |

Each planned host claims its assignment before starting its launcher.
The launcher checks the grants and maps them to the local device mask.
The committed activation policy determines when the ingress switches.
If the supervisor exits, the launcher closes its worker process group.
A restarted supervisor uses the same local lock and revokes a claim
left without a live launcher. Use a profile's optional `verify_release`
executable to attest device release before its grants can be reused.

### Keep one HTTP endpoint across group versions

Point clients at `serve-aggregated.py`. Use the same API model alias
for every version of the group. The ingress follows the active version
through the control replicas.

```sh
python scripts/serve-aggregated.py --model "$SERVED_MODEL" --group-id "$GROUP_ID" \
  --control-peer "$VOTER_ID@$VOTER_ADDR" --control-listen "$INGRESS_REPLY_ADDR" \
  --host "$INGRESS_BIND_IP" --port 8192
```

Repeat `--control-peer` for each voter. The control reply address must be
reachable by the voters. Use the current `plait-node` binary for the
ordered group observation. The ingress serves `/v1/completions`,
`/v1/chat/completions`, `/v1/models`, and `/health`.

The ingress waits 500 ms between observations and expires its last
route after eight seconds without a fresh observation. The response header
identifies the group that first admitted the request. A request admitted by the old group
keeps its upstream connection while the old group drains. The ingress can
retry a connection that failed before sending request bytes or an explicit
Plait admission rejection. Once forwarding starts, a disconnect without
an admission rejection ends that attempt.

### Recover a greedy completion stream

Set `X-Plait-Stream-Recovery: replay-prefix` on `POST /v1/completions`
with `stream: true` and `temperature: 0`. Send one prompt and request one
completion. This mode supports text and token-array prompts and final usage.
Use plain greedy completions without echo, log probabilities, beam search,
continuous usage statistics, or an external KV transfer descriptor.

After an admitted stream breaks, the ingress waits for a newer active
epoch within the request timeout and submits the original completion
there. It checks the prompt token IDs and compares both the token IDs
and text already emitted, even when stream chunks have different boundaries.
It sends only the suffix after both comparisons pass. The client keeps
the same HTTP connection and completion ID. The ingress sends a single
successful `[DONE]` together with the final completion frames. A
prefix mismatch yields a `plait_recovery_error` event.

Recovery retains the visible prefix in the running ingress process and
rebuilds KV by recomputation on the replacement. The same connection
must remain open throughout recovery.

### Reuse prompt KV across TP layouts

Start a shared store on the engine network. It keeps completed pages in
memory, independent of either vLLM process group.

```sh
plait-node kv-store --listen 192.0.2.10:7952 --capacity-bytes 268435456
```

Pass the same model namespace, KV dtype, and logical block size to both
engines. Choose a namespace that identifies the same model weights,
position encoding, and compatible attention arithmetic. Use
`--enforce-eager` and `--disable-hybrid-kv-cache-manager` with the connector.
The two engines can use different TP sizes if each evenly partitions the
model's KV heads. This path supports dense full-attention FP16 and BF16 KV.

```sh
--kv-transfer-config '{
  "kv_connector": "PlaitKvConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "address": "192.0.2.10:7952",
    "namespace": "model-revision-fp16"
  }
}'
```

Set `kv_role` to `kv_consumer` on the destination, or `kv_both` for an engine
that reads and writes. Each producer writes its global head range for a
completed block. The store publishes a block after one write attempt has
covered all its heads. The destination looks up contiguous full prefix
blocks, then loads its own head range into the local attention layout before
the forward pass. The first unmatched token and subsequent tokens run through
the usual engine path.

A managed TP4 group on CUDA, ROCm, and PPU HGGC populated two 128-token
blocks. After activation, its TP2 replacement on CUDA and Ascend CANN
reported 256 external prefix hits. The replacement's output token IDs and
text matched its cold run, which reported zero external hits. The transfer
uses host staging over TCP. The cache starts empty after a store restart.

### Resume a stream from its last emitted token

Use `X-Plait-Stream-Recovery: resume-kv` on the same greedy completion
endpoint. Set `PlaitKvConnector` to produce on the source and consume on
the replacement. Use the V2 runner on the replacement. Set `kv_both` on
an engine that should publish additional pages after it takes over.
The request must carry a plain text or token-array prompt. Structured
outputs, prompt log probabilities, repetition detection, and thinking
budgets require their own continuation state.

After a disconnect, the ingress sends the original request and the token
IDs already delivered. The replacement restores those IDs as generation
history, initializes its incremental detokenizer from them, and looks up
KV using the combined prompt and generated prefix. It loads complete pages
into its own TP layout and computes any uncovered suffix of the known
sequence before sampling the next token. The scheduler can reuse a full
page that ends after the prompt. Prompt usage counts only prompt positions,
while the external cache metric counts all matched positions.

The replacement acknowledges the restored length in its first choice.
The ingress verifies that count, the prompt IDs, and the previously sent
token IDs and text before forwarding the suffix. The client keeps the
original completion ID, usage budget, and one terminal event. The ingress
and its client connection must survive the replacement.

In the four-backend fault test, the source emitted six tokens before one
of its TP4 workers was killed. The TP2 replacement reported 256 external
hits for a 254-token prompt. The client received all 16 output tokens on
the original SSE connection. Token IDs, text, and usage matched a cold
run on the replacement.

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
