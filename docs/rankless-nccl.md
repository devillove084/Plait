# Plait TP transport and NCCL compatibility

The vLLM fork uses Plait's peer transport for tensor parallel groups when
`VLLM_USE_PLAIT=1`. Each worker publishes its own endpoint and contributes
to collectives with its local accelerator. CUDA, ROCm, Huawei CANN, and PPU
HGGC workers share the same wire format.

## Start a peer group

### Join by device identity

Use the native [C ABI](../crates/plait-nccl/include/plait_peer.h) or Rust
`RanklessListener` for a caller that already owns a committed group version
and the corresponding worker grants. The caller supplies these grants and
distributes peer endpoint records through its control plane.

1. On every participant, call `plaitPeerListen` with the group ID and epoch,
   stable device ID, incarnation, fence, and a fresh per-listener ticket.
   The returned member record carries the selected endpoint and grant.
2. Give every participant the full member list. Call `plaitPeerConnect`
   with the list in any order and the local backend and device ordinal.
   Plait sorts by device ID and checks the group version, complete grant list,
   endpoints, and tickets in each handshake. Rust callers use
   `Comm::connect_rankless`.
3. Use the usual NCCL operations on the returned handle. Resolve a stable
   device ID to its current slot with `plaitPeerDeviceSlot` when an operation
   accepts a peer or root slot.

`plaitPeerCancel` closes the communicator's sockets and prevents new calls.
`plaitPeerFailed` reports failure recorded locally by cancellation or an
operation. Wait for active calls before destroying the handle or releasing its
buffers. Build a new communicator from the next group's grants and tickets
when the control plane changes membership. Independent communicators continue
using their own sockets and sequence state.

### Publish a manifest through the control plane

The native group API uses the same device fences and host memory ledger as
hosted execution. Its placement specifies a group ID, the expected epoch,
worker incarnations, and per-device and per-host resource demand. Any voter
can accept a proposal. To send newline-delimited requests, start a client with
`plait-node group --listen 192.0.2.10:0 --seed 192.0.2.11:7950`.
Rust callers can use `plait_core::group` and `plait_node::Client`.

1. Register capacity with `SetHostMemory` and `RegisterWorker`. Use the
   `prepare` request to reserve a version. Its `workers` and `hosts` fields
   are lists of `(identity, demand)` pairs. The version's view contains the
   granted fences and incarnations.
2. Each participant opens a listener and submits `PublishEndpoint` with its
   local grant and endpoint record. A published endpoint is immutable for the
   lifetime of the version. The version exposes a manifest after every grant
   has an endpoint. Clients can poll `view` and use `observe` to order a fresh
   read against group mutations.
3. Each participant joins using the observed manifest, then submits `Ready`
   with its grant and `plaitPeerManifestDigest` from the connected handle.
   The control plane accepts `Activate` after every member confirms the same
   manifest. Use `observe` without an explicit epoch to resolve the group
   for new work, and check that the returned phase is still `Active`.
4. `Drain` and `Revoke` remove the version from new admissions. After a worker
   finishes or cancels its calls and closes access to its device and buffers,
   submit `Release` with its exact grant. The device becomes eligible for a
   new fence immediately. The host's reservation remains until its last grant
   in that version is released. Later retries for a released grant preserve
   any successor reservation.

The `apply` request takes an `operation` containing an `id` and a serialized
`GroupAction`. Keep the operation ID when retrying an ambiguous result.
Voters with `run --state-dir` preserve applied group metadata through restart.
Callers use current metadata and device completion to decide when existing
handles can be reclaimed.

### Plan across heterogeneous resources

Use `AdvertiseWorker` with a `DeviceProfile`, incarnation, and allocatable
device bytes. The incarnation binds the profile to the registered device.
The `placement` request reads the group's versions, available device grants,
and host capacity and reservations from a replica.

Send a `plan` request with the group ID and a `workload` object. When the
workload sends bytes between workers, the planner reads the latest samples
from the hosts' probes. An explicit `links` array supplies the samples for
that request. Use `links: []` to evaluate with no link measurements.

| Field | Meaning |
| --- | --- |
| `widths` | Legal group sizes supplied by the caller. |
| `requirement` | Optional device capabilities, including the operation data type. The default data type is FP32. |
| `device_bytes` | Allocatable device memory required by each worker. |
| `per_worker_host`, `per_host` | Host memory charged per worker and once per participating host. Each has `bytes` and `pinned_bytes`. |
| `compute_work`, `bytes_per_peer` | Estimated work per operation and bytes sent per directed worker pair. A zero term excludes that cost. |
| `max_link_age_ms` | Freshness limit for the link samples. Each sample has `source`, `target`, `bytes_per_second`, `rtt_ns`, and Unix `sampled_at_ms`. |
| `operations`, `switch_ns`, `minimum_gain_ns` | Forecast horizon in operations, one-time replacement cost, and optional gain threshold. |

The planner filters unavailable devices and accounts for both versions' host
reservations during preparation. Its estimate uses the slowest compute rate
for equal work per device. The network estimate accounts for all worker pairs
sharing each directed host link, then takes the highest host ingress or egress
cost. A cross-host communication candidate needs fresh measurements in both
directions. The search retains up to 24 partial layouts per width. The network
estimate covers cross-host TCP.

The reply reports `prepare` with a placement and cost, `keep` with the active
version, `pending` with the version in preparation, `await_links` when the
incumbent cannot be compared, or `no_placement` when the search finds no
candidate. When both layouts are feasible, the estimated gain over the horizon
must exceed the replacement cost and threshold. Set the request's optional
`prepare` field to an operation ID to reserve the selected placement through
consensus. The transaction rechecks the latest and active epochs and the live
resource ledger. With automatic sampling, it also compares the probe registrations
used to score the chosen layout and the incumbent, when compared. A changed
registration returns `StaleProbe`. Read fresh samples and submit a new operation
ID to retry that plan. Retrying a submitted operation ID reads its applied receipt.
The new group follows the normal endpoint, readiness, and activation steps.

#### Replan with measured compute rates

Use a stable `calibration` ID for each workload you want to compare. Use the
same definition of an operation across participating backends, including the
operator, shapes, and precision requirements. Measure each device in those
units and send `PublishComputeRate` through `plait-node group`.

```json
{"request":"apply","operation":{"id":{"origin":[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"counter":1},"action":{"PublishComputeRate":{"calibration":[2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"device":{"node":[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"ordinal":0},"incarnation":1,"expected_revision":0,"operations_per_second":2000000000}}}}
```

Use the registered device identity and incarnation. The first publication
expects revision zero. Subsequent updates expect the revision returned in
the `ComputeRate` receipt. Sending the same value at the current revision
preserves that revision. Publish `null` to invalidate an unusable reading.
Registration of a new device incarnation clears that device's readings.
Updating the rate leaves existing device grants and communicator membership
intact.

Pass the same `calibration` to `placement` to see the inventory and rate
revisions from one replica snapshot. Pass it to `plan` with nonzero
`compute_work` to use those rates in the comparison. Candidates need valid
rates on every device. An otherwise eligible incumbent with a missing rate
causes `await_compute_rates`. A plan without `calibration` uses the advertised
compute scores.

When `prepare` is set, the reservation checks the exact rates for the chosen
devices and for the incumbent if it was scored. If a rate changes before
reservation, the command returns `StaleComputeRate` with the current
revision and holds no new resources. Read fresh rates and use a new operation
ID to rescore. Updates to other calibrations or to devices outside the
comparison leave the reservation eligible. Journal recovery restores applied
rate revisions along with the group metadata.

The caller supplies and invalidates measurements. Publish stable estimates
at the control-plane rate required by the planner. Upgrade the voting nodes
before enabling the measured-compute requests.

##### Collect rates during vLLM TP startup

With the matching vLLM fork and Plait wheel, add a `compute_calibration`
object to a TP group in the `VLLM_PLAIT_TP_GROUP_CONFIG` descriptor.

```json
{
  "compute_calibration": {"m": 64, "n": 1024, "k": 1024, "dtype": "float16"}
}
```

Each worker runs a local `torch.mm` on its assigned device after the
communicator warm-up. The timed batches include operator launch and device
synchronization. They use preallocated tensors and run without peer
communication. The rate counts two operations per scalar multiply-add and
uses the median of seven samples. The fork derives the calibration ID from
the benchmark version, data type, and shape. It logs the ID, applied
revision, and rate for use in a `plan` request.

The worker reads the revision before measuring and publishes against its
exact grant. A concurrent update, closed group, or invalid fence prevents
publication. The engine logs a skipped measurement and continues startup
when the control request or benchmark fails. The latest applied reading
remains available after device release while the same incarnation is
registered. Collect separate rates when the workload's precision or GEMM
shape changes. The startup benchmark supports FP16 and BF16 GEMM.

#### Run resident link probes

Add `--link-probe "$NODE_IP:0"` to `plait-node run` on each participating host.
Use the same stable `--id` as the host's worker profiles. The IP must be reachable
by the other probe hosts and by the planner. `:0` lets the OS select a port.
Upgrade the voters to a version that supports probe registration before
enabling this option. A nonvoting host can run a probe without an engine
configuration.

```sh
plait-node run --compute-only --id "$HOST_ID" \
  --seed "$SEED_IP:$SEED_PORT" --link-probe "$NODE_IP:0"
```

Each process publishes its bound endpoint with a new session ID. It discovers
targets through the replicated directory. A probe sends 1 MiB per target on a
ten-second interval and records throughput and RTT in the sending direction.
Durable voters recover the directory from their journals. Samples remain in
memory and are rebuilt after restart. A `links` request to `plait-node group` returns
the directory and the available samples. Sample timestamps use the probe
host's wall clock. Keep host clocks synchronized for `max_link_age_ms` checks.

The planner checks the source session when it requests samples and compares
the reported target session against the directory. When a probe is replaced,
samples for its old session are excluded. The ordered reservation checks the
registration revisions again in case they change after the planner reads them.

### Bootstrap from vLLM

Start the vLLM workers with the same TP layout and the vendor's PyTorch
runtime on each worker. The default peer bootstrap exchanges endpoints
over Gloo, which also carries the engine's CPU control operations. Set
`VLLM_HOST_IP` to the address the other workers can reach when automatic
address selection chooses the wrong interface. The Rust layer binds a TCP
port assigned by the OS.
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

### Join a committed TP group from vLLM

Reserve a native execution group first. Use a Plait Python package built with
the control-plane join and set `VLLM_USE_PLAIT=1` and
`VLLM_PLAIT_TP_GROUP_CONFIG` to the launch descriptor's path on every host.
Each host sets its own `host_id` and uses the same `seeds` and `tp_groups`.

```json
{
  "host_id": "00000000000000000000000000000002",
  "seeds": ["192.0.2.11:7950", "192.0.2.12:7950", "192.0.2.13:7950"],
  "tp_groups": [{
    "version": "00000000000000000000000000000003@1",
    "ranks": [0, 1],
    "devices": [
      ["00000000000000000000000000000002", 3],
      ["00000000000000000000000000000001", 1]
    ]
  }]
}
```

The corresponding elements in `ranks` and `devices` give each model shard
its stable device ID. The launcher sets the device mask so the local worker
runs on that device. The device ordinal in the descriptor identifies the
physical grant. The ordinal passed to the device runtime follows the process's
visibility mask. A deployment with multiple TP subgroups provides one entry per
TP subgroup.

Each worker independently resolves its grant, publishes its endpoint, and reads
the complete manifest through a voter. The peer handshake binds the manifest and
the descriptor's device order, so AllGather output and ReduceScatter shards
follow the engine's slots. Workers report `Ready` for the communicator after
connecting. The deployment controller can activate the version once transport
and engine readiness are both confirmed. Use a new epoch when a worker must
publish a new listener. The engine retains its CPU process groups and PP startup.

Rust clients can use `join_committed_group` with the `plait-nccl/control`
feature for the same sequence. C clients can publish and join through the
native group APIs and existing C ABI.

#### Verify four-vendor TP4 over HTTP

The integration test installed vLLM `0.31.0+plait.acea9b5ad` and the Plait
Python wheel in six Linux environments. Each vLLM wheel was built from the
same source revision. CUDA and ROCm builds reused the native extensions for
their installed PyTorch and runtime versions. CANN and HGGC used their vendor
plugins. The test checked the installed vLLM source files for the native TP
join and loaded the same Plait wheel on every host. The Plait wheel used the
dev build profile for this functional test.
[vLLM revision](https://github.com/devillove084/vllm/commit/acea9b5adb9919192b1b9ef1207c2f4868c64fcb)
[Plait build source](https://github.com/devillove084/Plait/commit/9bb383cf6c99ec314eab57d976593df85fd0cd06)

For Qwen3-0.6B in FP16, use one device each on CUDA, ROCm, CANN, and HGGC.
Start four hosts with `--distributed-executor-backend mp`, `--nnodes 4`,
`--tensor-parallel-size 4`, and `--plait-tp-layout 1 1 1 1`. Use
`VLLM_USE_V2_MODEL_RUNNER=1`, `--enforce-eager`, `--block-size 128`, and
`--disable-custom-all-reduce`. Give each host its own node rank. Run the API on
node rank 0 and append `--headless` on the other hosts. The small-model test set
`--max-model-len 96`, `--no-enable-prefix-caching`, and
`--kv-cache-memory-bytes 268435456` on each worker. It selected `TRITON_ATTN`
on CUDA, ROCm, and HGGC and used the Ascend plugin's attention backend on CANN.

On a PPU host using the vendor Triton runtime, make the SDK's `ppu-smi`
available to the vLLM worker before launch. Select the native compiler with
`TRITON_DEFAULT_BACKEND=ppu`. The vLLM worker still uses the vendor
PyTorch CUDA-compatible device interface. The Triton compiler targets
`ppu` for its attention kernels.

```sh
export PATH="$PPU_SDK/ppu-smi/bin:$PPU_SDK/bin:$PATH"
export TRITON_DEFAULT_BACKEND=ppu
```

Confirm one ready grant per worker and an HTTP 200 from `/health` on the
API host. Apply `Activate` to the native group version, then send the same
greedy completion request to the TP4 endpoint and a single-device baseline.
The test received HTTP 200 from both `/v1/completions` endpoints and compared
all eight returned token IDs.

#### Verify a planned change in TP width

The installed build also ran a TP2 successor on separate CUDA and ROCm
devices. Prepare it under the same group ID with a new epoch and a TP2 launch
descriptor. The TP4 incumbent continued to answer requests while TP2 loaded
the model. Both engines had their own HTTP endpoint.

Once TP2 was ready, the test opened a stream on TP4 and submitted `Activate`
after receiving its first token. The control plane marked TP4 `Draining` and
TP2 `Active`. The new endpoint returned HTTP 200 with eight token IDs matching
the single-device baseline. The old stream delivered all 48 tokens and the
final `[DONE]` frame. The deployment layer uses the active group to route new
requests and keeps an admitted stream on its original engine.

#### Route native groups through one HTTP endpoint

On each vLLM API host, add the Plait `scripts` directory to `PYTHONPATH` and
pass `--middleware serving_native.NativeGroupHeaders` to `vllm serve`.
The middleware reads the version from the unchanged
`VLLM_PLAIT_TP_GROUP_CONFIG` launch descriptor. This entry point supports an
API instance with one committed TP group. A request carrying a different
version receives HTTP 409 before model dispatch. Matching responses carry
the engine's group version, including `/health` and streaming responses.

Give each epoch its own HTTP origin and write the map to a JSON file.

```json
{
  "1": "http://192.0.2.21:8000",
  "2": "http://192.0.2.22:8000"
}
```

```sh
python scripts/serve-aggregated.py --model "$MODEL_ALIAS" --group-id "$GROUP_ID" \
  --native-routes "$ROUTES" --control-seed "$SEED" --control-listen "$LOCAL_LISTEN"
```

The ingress polls ordered `Observe` reads and opens each new request against
the latest observed `Active` epoch. Replace the route file atomically when
adding an endpoint. An observed active epoch without a usable origin pauses
new admissions until its binding is available. The ingress refreshes every
500 ms and expires its last route after eight seconds without a fresh read.
Retain the old engine until ingress observers have cut over and its admitted
requests have drained. Each admitted stream retains its upstream connection
and version across a route change.

The installed vLLM test used one client URL for a four-vendor TP4 and its
CUDA and ROCm TP2 successor. With the new binding withheld, `/health`
returned HTTP 503 after the ingress observed the switch. Publishing the
binding let the same URL serve new requests from TP2. Its eight token IDs
matched TP4, and the original TP4 stream returned all 24 tokens and `[DONE]`.

#### Cancel a committed communicator

Communicators created through the control-plane join keep their control
client for the lifetime of the handle. The client checks the local grant
every 250 ms, rotating through the configured voters. The compact lookup
checks the group epoch, device incarnation, and fence. With an older voter
the client uses the existing group version view.

Apply `Revoke` to stop a version's communication. When a communicator
observes the applied revocation or the release of its own grant, it shuts
down its collective and point-to-point sockets. Pending socket operations
return errors, and subsequent calls fail locally. Python callers can read
`CommHandle.peer_failed`. Other group versions retain their connections.
A missing reply or an unknown grant view prompts another query.

If vLLM raises during a model or sampler step, its worker closes the
affected Plait TP communicator and preserves the exception in the worker
log. Peers blocked in that group wake with a transport error. The worker
rejects subsequent model RPCs on the closed handle before execution. With
the matching Plait wheel, the watcher reports the execution failure under
the worker's exact grant. An eligible standby can then take over through the
same ordered revocation path.

Use `Drain` to stop admitting work at the deployment layer while admitted
requests finish. `Draining` preserves communication. `Release` still
requires the device owner to finish teardown, including any device work
already in progress. The observer closes sockets and leaves release to the
owner.

#### Arm a standby for communication group failover

Prepare a successor under the same group ID using available devices. Once its
communicators are `Ready` and the engine is healthy, send `SetStandby` through
the `apply` request. The action takes the incumbent `GroupVersion` in `active`
and the successor epoch in `standby`. The voters check that `active` is still
current and that the successor is the latest ready version. The `standby`
request takes the group ID and reads the armed version from a replica. Set
`standby` to `null` in `SetStandby` to disarm it.

A native control-plane communicator reports terminal transport or device
errors with `ReportFailure`. The background client retries the same exact-grant
operation for up to 15 seconds while its process is alive. On an applied report
for the incumbent, or an ordered host expiry, the control plane revokes the
affected version and promotes an eligible standby in the same operation. Host
expiry excludes a standby affected by that host. Explicit `Drain` and `Revoke`
disarm the incumbent. A failure in a standby or an older draining version stops
only that version.

Idle worker loss requires host liveness detection or a surviving participant
that observes the failure. Run voters that support these actions before arming
a standby. After promotion, a fresh `observe` resolves the new active version.
The deployment layer uses that result for new admissions and the engine owns
in-flight requests and KV. The revoked group keeps its device fences until its
owners confirm teardown and submit `Release`.

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

The counts describe workers per host and sum to `TP × PP`. Each pipeline
stage owns a consecutive set of TP workers in global worker order. The
model's tensor-partitioning rules determine the shards within a stage.
Use DP, PCP and DCP sizes of one with EP disabled. The layout is fixed for
the lifetime of that vLLM process group.

### Split model layers across hosts

For two stages using two devices each, start the same model with
`--tensor-parallel-size 2 --pipeline-parallel-size 2 --plait-tp-layout 2 2`
and the `mp` executor. The first host runs the first stage. Set
`VLLM_PP_LAYER_PARTITION` on both hosts to override the layer split. For a
28-layer model, `10,18` assigns ten layers to the first stage and eighteen
to the next. Each host must use the same worker counts, sizes and layer
partition. Set its device backend and mask locally.

Pipeline stages send device tensors through Plait's point-to-point links.
The receiver allocates them on its own device and reconstructs any TP
slices within its stage. Gloo carries tensor descriptors and CPU data.
The peer transport stages sends through host memory before the source
tensor can be reused. The receiving stage waits for its activation copy
before forwarding. This path requires eager execution.

### Start a committed group on several hosts

Run `scripts/serve-tiered.py` with `--host-launch-spec` to derive the
placement from a committed `GroupPlan`. Give each host the same
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
role `Aggregated`, TP and PP sizes, launch digest, and
device and host reservations. Represent the device and host maps as
arrays of key/value pairs. Reserve one device per worker, for a total of
`TP × PP`. When PP is greater than one, each host must run the Plait fork
with pipeline support. Launch each planned host as described above.

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

For managed groups with dedicated accelerator devices and Plait TCP, a
profile can use the built-in driver process check instead of an external
verifier. Set `release_driver` and the matching `PLAIT_NCCL_BACKEND` in
the engine profile.

```json
{
  "release_driver": { "driver": "ppu", "program": "/path/to/ppu-smi" },
  "env": { "PLAIT_NCCL_BACKEND": "ppu" }
}
```

| `driver` | `PLAIT_NCCL_BACKEND` | Local tool |
| --- | --- | --- |
| `nvidia` | `cuda` | `nvidia-smi` |
| `amd` | `rocm` | `amd-smi` |
| `amd_kfd` | `rocm` | `amd-smi` with Linux KFD sysfs |
| `ppu` | `ppu` | `ppu-smi` |
| `ascend` | `huawei` | `npu-smi` |

Use an absolute executable path and load its vendor libraries in the
resident environment. The driver tool and the engine's visibility mask
must use the same physical device numbering. The Ascend query addresses
device IDs whose chip index is zero. The launcher uses the built-in check
after the worker process group exits and the admission proxy drains.
It disables the Plait shared-memory and CUDA IPC paths for these groups.
Every local grant must return a clear process report. Busy devices,
unsupported reports, and timeouts leave the grants protected. Use
`verify_release` for profiles that rely on another release procedure,
such as device resets or transport-specific cleanup.

Use `amd_kfd` for a ROCm whole-device profile on a host where AMD SMI
returns the same process list for multiple cards. Use `CUDA_VISIBLE_DEVICES`
or `HIP_VISIBLE_DEVICES` as the profile's `device_env`. The launcher
captures the HIP-to-KFD mapping before starting vLLM. After drain, it
checks that the mapping still matches AMD SMI and kernel topology. KFD
queue IDs and per-device VRAM counters must be clear on the granted
devices. Work on another device can keep running. The verifier requires
a complete KFD process view and reconciles the PIDs reported by AMD SMI.
Partitioned identities, changed mappings and missing process data keep
the grants protected. The `amd` mode remains available on older hosts.

### Replan with measured host links

Set `probe_listen` in each hosted execution configuration to an address
reachable from the execution hosts and the planner. Use the same IP as
`listen`. Port zero lets the node choose a port. Each node registers
its probe endpoint with the control replicas, then samples its outgoing
links in sequence. A probe sends 1 MiB over TCP and waits for the peer's
acknowledgment. It also records the round-trip time of the opening
request. Samples are kept in the source node's memory.

Send `{"request":"hosted_links"}` to the serving CLI to read the current
endpoints and directed samples. A sample carries `bytes_per_second`,
`rtt_ns`, and `sampled_at_ms`. When a node restarts, its probe session
changes. The planner only uses samples that match the directory's
source and destination sessions.

To apply these measurements, add a `link_policy` to `replan_hosted`.
Supply `bytes_per_rank_pair`, the estimated bytes sent per request
between each directed pair of TP workers, and `max_sample_age_ms`.
For each pair of hosts, the planner multiplies the byte estimate by
the number of sending and receiving rank pairs. It sums traffic through
each host's send and receive port and uses the slowest measured path
through each port as its capacity. The smallest resulting rate caps
the candidate's supplied SLO capacity. RTT is available for diagnosis.

The response includes one `link_limits` entry per candidate. A
cross-host candidate needs fresh samples in both directions for every
host pair. Missing or expired evidence leaves that candidate unselected.
The measured cap describes the TCP traffic budget for the supplied
workload estimate. Application goodput still comes from the SLO
profiles.

### Generate placements from registered hosts

For automatic placement, send a `pool` in place of `candidates` on
`replan_hosted` and supply `link_policy`. Use
`{"request":"hosted_planning"}` to inspect the host inventory and committed
port reservations. Resident nodes advertise their reachable addresses,
installed profile names, and registered device incarnations. Each host
should install compatible weights and engine settings under the same
profile name. Local executable paths and backend settings may differ.
If a device registration changes, only that device loses eligibility
until the host advertises its current incarnation.

The example shows the `pool` field. Supply `id`, `forecast`, and
`link_policy` alongside it. In a running deployment, replace the example
capacity and Unix millisecond deadline with measurements for the intended
model and workload. List one entry for each supported TP size.

```json
{
  "group": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  "profile": "model-fp16",
  "api_model": "model",
  "max_model_len": 4096,
  "gpu_memory_bps": 8000,
  "prefix_caching": true,
  "ports": {"first": 25000, "count": 48},
  "sizes": [
    {
      "tensor_parallel": 2,
      "device_bytes": 8589934592,
      "host_base": {"bytes": 2147483648, "pinned_bytes": 0},
      "host_per_worker": {"bytes": 536870912, "pinned_bytes": 0},
      "capacity": {"rps": 4.0, "valid_until_ms": 0},
      "ready_after_ms": 15000,
      "transition_loss_requests": 0
    }
  ]
}
```

For each size, the planner combines free devices on at least two hosts.
It charges the base host memory once per host and the per-worker memory
for each assigned rank, including pinned memory. It retains a bounded
set of layouts ranked by the measured link ceiling, then applies the
forecast to score the retained candidates. Each size currently uses one
supplied SLO profile across its layouts. Run `replan_hosted` again with
a new operation ID after capacity or demand changes.

Keep the port range dedicated to hosted launches. Consecutive triplets
provide the rendezvous, public HTTP, and backend HTTP ports on the API
host. The control plane checks for overlapping recipe ports and verifies
the host inventory, device grants, and group versions when it applies
the plan. A draining or revoked group retains its ports until retirement.
`placements` reports generated layouts in the same order as the selection
scores. An empty list means there was no feasible layout in this search
using the current inventory and link samples.

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

### Hand off prompt KV between independent Prefill and Decode groups

Use the vLLM fork at `9122df3da` or later on `plait/v0.31-runtime`.
Start `plait-node kv-store`
at an address reachable from both groups. Use the same store, model
namespace, weights, KV dtype, and logical page size on both sides.
Give each `serve-tiered.py` launch the role `prefill` or `decode` and
these fetch and publish settings.

```sh
--plait-kv-namespace "$MODEL_NAMESPACE" --kv-host "$KV_HOST" --kv-port "$KV_PORT"
```

Use `--proxy-host` to bind the group proxy on an address reachable by
the router, and `--advertise-http` to publish its HTTP endpoint. The
Prefill endpoint's `prefill_kv` is the shared store address.
After the groups report Ready, activate and publish the versioned P/D
path. Start the router against the published directory.

```sh
python scripts/serve-disaggregated.py --model "$MODEL_ALIAS" --kv-transport plait \
  --control-peer "$VOTER_ID@$VOTER_ADDR" --control-listen "$ROUTER_REPLY_ADDR"
```

The router gives Prefill a ticket. Prefill returns the ticket, store
address, prompt digest, and number of published prompt tokens. The
digest binds the model namespace, page layout, token IDs and cache salt.
The router compares the ticket and the store advertised by the selected
Prefill group. Decode checks the digest and requires the complete pages
claimed in the receipt. It computes the remaining prompt tokens before
generating. A lost page fails the request under the managed Decode
configuration. Ordinary untagged requests retain the external cache
lookup and recomputation behavior.

This path uses the bounded, in-memory Plait KV store and supports the
eager, dense FP16 and BF16 attention layouts described below.

### Recover a P/D completion after Decode disconnects

With `serve-disaggregated.py --kv-transport plait`, send
`X-Plait-Stream-Recovery: replay-prefix` on `POST /v1/completions`. Use
`stream: true`, `temperature: 0`, and one token-addressable text completion.

If the admitted Decode disconnects, the ingress keeps the HTTP stream open
and excludes that Decode version. It uses an already live alternative or
waits for the route directory to publish one. The replacement attempt
obtains a fresh Prefill handoff, reads the prompt pages through the Plait
store, and generates from the start of the completion. The ingress checks
the token IDs and text already emitted before forwarding the suffix. It
keeps the original completion ID and emits one terminal marker on success.

The ingress releases each completed handoff lease after its Decode attempt.
When matching fails or the request deadline expires, an in-progress stream
receives a recovery error. Replay keeps state in the live ingress and uses
an available Prefill/Decode path. It regenerates the emitted prefix with
the replacement engine.

### Start and publish a managed Prefill and Decode deployment

Install the model profile on each resident execution host. The plan
supplies the KV connector and block size for both roles. Keep those
options out of the profile's `args`. Set the model namespace to an
identifier shared by profiles with compatible weights and KV data.

Send `prepare_hosted_pd` to the serving CLI. The request supplies these
fields.

| Field | Value |
| --- | --- |
| `id` | Operation identity. Retry an ambiguous response with the same ID and inputs. |
| `model` | API alias used by the router and all endpoint recipes. |
| `expected_revision` | Current route directory revision. Use zero for the initial publication. |
| `kv` | Shared `store` as an IPv4 address and port, `namespace`, and `block_size`. |
| `placements` | New groups with `group`, `expected_epoch`, `expected_active`, `plan`, and `recipe`. Use role `Prefill` or `Decode`. Reserve `TP × PP` devices for each group. The CLI computes the launch digest. A group may use one host or multiple hosts. |
| `paths` | Complete desired path list. Each entry names its `prefill` and `decode` group versions. Include any existing versions the deployment is keeping. |

The voters reserve all new groups in one operation. Each resident host
starts the groups assigned to its device grants. A voter proposes
publication after every new group is Ready. The publication operation
checks the staged revision, current host inventories, group readiness
and endpoints. It activates the new groups and publishes the complete
path list together. It drains versions removed from all paths. A
retained Prefill can serve an additional Decode without a restart.
To shrink capacity, submit a smaller path list with an empty
`placements` list.

Use `{"request":"hosted_pd_pending","model":"model"}` to inspect a staged
intent. The intent survives a voter restart. A failed candidate or
changed placement inventory cancels its proposal without publishing
the new paths. For an explicit cancellation, send `ResolveHostedPd`
with the returned `intent` and `cancel: true` through `execute`, using
a fresh operation ID. Claims remain reserved until the existing stop
and retirement protocol confirms release. This API takes explicit
placements.

### Plan Prefill and Decode placements together

Send `replan_hosted_pd` with `id`, the API alias `model`,
`expected_revision`, `forecast`, `link_policy`, and `pool`. Use the
checkpoint identity as `forecast.model` and supply a current capacity
entry for every group on an existing path. Leave the forecast's
`paths` and `link_bytes_per_second` empty. The planner derives them from
the registered layouts and current link samples.

The pool supplies the following fields.

| Field | Value |
| --- | --- |
| `kv` | Store address, namespace and block size used by the managed P/D deployment. The store address must match one registered host. |
| `kv_bytes_per_request` | Estimated bytes written by Prefill and read by Decode for the request class being planned. |
| `store_bytes_per_second`, `store_valid_until_ms` | Calibrated aggregate read and write budget and its Unix millisecond deadline. |
| `slots` | Each entry has a `role` of `Prefill` or `Decode` and a `pool` with the hosted pool fields described above. Give each slot a stable group ID and supported TP sizes. A P/D slot can use TP1 or local TP. The planner chooses which slots to populate. |
| `allow_scale_down` | With `true`, a layout can remove groups when it serves the forecast demand at the baseline rate with fewer devices. It must use already-active groups. |

Placement checks account for live grants, host and pinned memory,
ports and installed profiles. The search can retain an active group or
prepare its replacement alongside it. It keeps a bounded set of layouts
for each Prefill and Decode replica count.

The estimate follows the ingress's round-robin path allocation. The
planner charges Prefill-to-store writes, store-to-Decode reads, and
cross-host TP traffic to their directed ports. Workers contribute to
KV traffic in proportion to their share of the TP group. Paths sharing
a port consume the same budget. Capacity and startup estimates are
supplied per role and TP size. Re-run the command with refreshed inputs
to make another planning decision.

The response's `selection` reports the baseline and any preferred
layout. A selected layout is submitted through managed P/D staging.
The voters recheck route and resource versions when applying it. Retry
an ambiguous submission using the same operation ID and request.

### Drive P/D planning from ingress load

Start each managed P/D ingress with `--load-listen IP:PORT` on a private
address reachable from the execution node running the policy. Each
listener reports process-local arrival, completion, failure, and
in-flight counters for its model alias. Requests that pass the
ingress checks count towards arrivals while the route directory is
empty or unavailable. This permits initial placement under offered
load.

Add a `pd_scaling` list to the resident execution configuration. Each
policy includes the following fields.

| Field | Value |
| --- | --- |
| `model`, `checkpoint` | API alias and the serving plan's model identity. |
| `ingress` | Private load listener addresses for demand-based scaling. Omit for a repair-only policy. |
| `repair_on_failure` | When enabled, restore failed roles in a committed hosted P/D directory even when no ingress is running. Defaults to `false`. |
| `window_ms` | Interval between observations. |
| `stable_windows` | Consecutive complete windows that must select the same plan. |
| `cooldown_ms` | Delay after a deployment revision changes or a local attempt submits. |
| `horizon_ms`, `minimum_gain_requests` | Forecast window and improvement threshold. |
| `pool`, `link_policy` | The same calibrated pool and link evidence settings used by `replan_hosted_pd`. Enable `pool.allow_scale_down` to permit removing idle groups. |

The node adds arrival rates from all ingress processes. It includes the
current in-flight count over the forecast horizon. Missing or duplicate
counters, a process restart, or an excessive gap clears recent
decisions. A pending deployment blocks another automatic submission.
Any resident can propose a change. Route revision and reservation
checks arbitrate concurrent submissions.

Existing group capacities come from the configured slot that matches
the active role, launch profile, and TP size. Keep those measurements
and the shared store budget current. The loop supplies observed demand
to the planner and follows the normal staged rollout. Model capacity
and startup time remain calibration inputs.

### Restore a failed hosted P/D group

Set `repair_on_failure` on a `pd_scaling` policy. Upgrade the voting
nodes before enabling the policy. Each `window_ms` tick first checks for
revoked or retired groups referenced by the committed directory. The
policy can omit `ingress` when only recovery is needed. A failed slot
uses the matching group ID and role in `pool.slots`. Each host advertises
the profile names it can launch, with local settings for its own engine
and devices. A replacement may use a different TP size from the failed
group if the checkpoint and KV contract match.

A repair preserves the paths and healthy group versions in the committed
directory. The node stages a replacement for every failed endpoint in
one operation. Healthy paths stay available during preparation. If no
complete layout fits current device grants, host memory, ports and
directed link evidence, the policy retries on a later tick. The normal
readiness and revision checks still govern publication. Failed groups
hold their old device grants until the regular release procedure finishes.
To remove a slot permanently, publish a hosted directory without it.

### Change a role on an occupied device

Submit `prepare_hosted_pd_transition` when one rollout needs to move traffic
away from a group and reuse its devices for another role or TP layout. Send
`id`, `model`, `expected_revision`, and the same `kv` contract used by
`prepare_hosted_pd`. Send both `bridge` and `target`, each containing a full
`paths` list and the `placements` it needs at that step. `release` lists the
old group versions excluded by the bridge. For a group ID reused after
retirement, set the target placement's `expected_epoch` to the last epoch and
its `expected_active` to `null`. Upgrade the voting nodes before submitting
the new operation.

For example, keep the current Prefill, prepare a replacement Decode on the
new host, and make that pair the bridge. The target can keep the bridge path
and add a path from a new Prefill on the old Decode's devices. Plait checks
both layouts before launching the replacement. It protects the target's
device IDs and launch ports, and holds the host memory shortfall above the
retiring groups' reservations. As those groups retire, their memory credit
moves into the target hold. The target receives a new grant only after
all the listed release records have committed. Reusing a group ID or port
offers the target under a new epoch. The group proxy rejects an admission
for the old epoch.

Configure `verify_release` on the retiring groups' resident launch profiles.
The verifier must attest that the devices and transport are safe to reuse.
Without a release record, an exited process can leave the rollout serving
through the bridge and waiting for release.

Query `hosted_pd_transition` with the model alias for the most recent
transition's operation ID, route revision, phase and any staging error.
`PreparingBridge` leaves the old directory live. `WaitingForRelease` and
`PreparingTarget` serve the bridge directory. `Published` exposes the target
directory. `Cancelled` leaves the last published directory in effect and
clears the deferred holds. A voter restarted from its WAL can continue the
same intent. Automatic P/D policies wait for an active transition to finish.

The two layouts are supplied by the caller. Assess the bridge's capacity
for the expected traffic during drain and target warm-up. Hosted vLLM
completes existing requests on the old process; the target serves newly
admitted work after cutover.

### Let the planner choose a two-stage transition

Add `"reconfigure": { "drain_ms": 12000 }` to the hosted P/D pool to
enable two-stage planning. `drain_ms` is an estimate from bridge publication
to release of its displaced groups. The control plane still waits for the
actual release records. The `ready_after_ms` in each size profile estimates
group startup. Slot IDs may have one Prefill and one Decode profile. A
published layout selects at most one role and version for each ID.

Each resident advertises the engine profiles that have a configured,
executable `verify_release`. Planning can reclaim a group's resources only
when every participating host advertises that profile. Submission checks
the current directory revision and release capability again before staging.

The planner estimates the bridge on currently available devices. It then
checks target layouts against a projected ledger that includes the bridge's
reservations and credits only the displaced groups' memory, devices and
ports. Bridge capacity must cover the incumbent rate during the estimated
drain and target startup. The score counts requests completed over the
remaining window in both layouts and subtracts both transition costs.
Two-stage changes compete with ordinary changes in the same scoring pass.

`replan_hosted_pd` reports the bridge paths, new placements, release
versions and estimated target readiness when it selects a transition.
`pd_scaling` submits the same plan after the full intent remains stable
across the configured observation windows. Keep the incumbent and target
profiles calibrated for this model and load.

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

#### Align different Prefill and Decode pipelines

Upgrade the store to a `plait-node` build that accepts `PLAITKV3` before
starting a multi-stage Prefill or Decode group. Use the matching vLLM fork
on the group's hosts. Choose TP, PP and `VLLM_PP_LAYER_PARTITION` for each
role independently. Within one group, the hosts agree on the layer split
before loading the model. The two roles use the same weights, KV dtype,
block size and store namespace.

The engine gives each stage the same global attention layer order. A
worker sends one range request per page for its local layers and heads.
The store publishes the page when one write attempt covers every layer
and head. Until then the page is a miss, including at the sender's
handoff check. The receiver requests its own ranges before executing
the first unmatched token. Single-stage engines continue to use
`PLAITKV2` whole-head requests against the same page keys and bytes.

### Resume a stream from its last emitted token

Use `X-Plait-Stream-Recovery: resume-kv` on the same completion endpoint.
Set `PlaitKvConnector` to produce on the source and consume on
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

On the unified ingress (`serve-disaggregated.py --kv-transport plait`), the
same header also opts in to planned handoff. When the active directory stops
routing to the current generator and a replacement path accepts work, the
ingress cuts after a complete emitted SSE frame. It closes the source attempt
before starting the replacement. A changed Prefill with the same available
Decode version leaves the generation attempt in place.

Set `stream_checkpoints: true` on the resident Decode engine profile to
publish pages for these requests. A destination advertises token resume when
its launch uses a V2 Plait KV consumer. A split destination obtains a fresh
Prefill receipt and checks every promised prompt page. Complete pages that
include generated tokens can extend the hit, and the engine computes any
uncovered known tokens before continuing. A greedy request can use prefix
replay on a destination without token resume. The request stays on the original
ingress connection in either case.

#### Resume a completion with an explicit seed

Set `seeded_stream_resume: true` on both resident Decode profiles and use
the unified ingress. Use a positive `temperature` and an explicit signed
64-bit `seed` other than `-1`. The supported engine uses the standard
V2 FP32 sampler without speculative decoding, watermarking, or custom logits
processors. Every host in a group must use the same profile setting. The
launcher verifies the engine's sampler capability before declaring it ready.

The engine derives sampling randomness from the request seed and absolute
token position. The destination keeps the seed, sampling parameters, and
committed token history. Its first choice confirms the seed and checkpoint
length. The ingress verifies both before publishing the next token. During
a planned change, the source keeps generating until an accepting destination
advertises the seeded sampler capability. Different backends and layouts can
compute different logits, so matching the seed and history gives the same
random stream with the destination's logits.

In the four-backend fault test, the source emitted six tokens before one
of its TP4 workers was killed. The TP2 replacement reported 256 external
hits for a 254-token prompt. The client received all 16 output tokens on
the original SSE connection. Token IDs, text, and usage matched a cold
run on the replacement.

#### Share checkpoints between aggregate and split routes

Use one `kv` contract in `prepare_hosted_serving` and set
`aggregate_stream_checkpoints: true` on each new aggregate placement that
should participate. For automatic placement, set the same flag under
`pd_scaling.serving_modes`. The committed binding fixes the store, namespace,
and block size for the group's lifetime. Use the same model checkpoint and
KV dtype in the installed engine profiles. Set `stream_checkpoints: true`
on Decode profiles to export generated pages from split routes.

An aggregate consumes and publishes shared pages for requests using
`X-Plait-Stream-Recovery: resume-kv`. Its ordinary requests keep their
local KV path. Enable `seeded_stream_resume` across the generating profiles
to carry a positive-temperature request with an explicit seed. During a
switch into split serving, the ingress still obtains the target Prefill's
KV receipt. During a switch into aggregation, the engine looks up the
original prompt and committed output history directly. Both can import
complete pages from the old layout and compute the remaining known tokens.
#### Budget checkpoint traffic during a rollout

Add `checkpoint_traffic` to `pd_scaling.pool`. Set
`write_bytes_per_stream` and `read_bytes_per_stream` from measurements of
checkpoint traffic. For split serving, exclude the usual prompt handoff
bytes already in `kv_bytes_per_request`. Include prompt pages in the
aggregate generator's checkpoint measurements when it stores or reads them.
Use the upper budget across the layouts in the pool.

`minimum_fraction_bps` sets the floor for the fraction of requests that
use `resume-kv`, in basis points. The load listener reports the opted-in
arrivals and their in-flight count. An older listener without these counters
is budgeted as if all its arrivals and in-flight requests opted in.
The planner combines those estimates with the normal TP, PP, and split
handoff traffic on each directional port and at the store.

`handoff_window_ms` is the calibrated interval over which the in-flight
requests are expected to migrate. When a generating group is replaced, the
planner reserves old-generation writes, target imports, and any new split
prefill handoff during that interval. It reserves enough for concentration
on one target because resumed requests can converge on the first available
group. A candidate must sustain the incumbent's modeled service rate after
those reserves.

#### Limit optional checkpoint transfers

Run the upgraded KV store with `--checkpoint-bytes-per-second` and
`--checkpoint-burst-bytes`. Set `checkpoint_admission: true` on the
participating Decode and aggregate engine profiles. The launcher checks
the actual engine capability. Multi-host groups agree on the setting
before starting workers.

The store combines optional checkpoint reads and writes into one byte
budget. Use a burst at least as large as a full logical page if the
store should admit optional imports. A writer requests credit before it
copies or sends the payload. A reader reserves the full page for 30
seconds. All TP and PP shards draw from that reservation, and in-flight
transfers occupy credit until their socket operation finishes. Required
P/D prompt handoffs use their existing path. When an optional prefix
reservation is declined, the target computes the uncovered known tokens.

Run `plait-node kv-store --stats HOST:PORT` for completed payload bytes,
refusals, expired read reservations, and bytes currently held. Use the
change in the byte counters over a sampling window to measure the store's
optional transfer rate.

#### Limit checkpoint traffic by peer and direction

Pass `--checkpoint-peer-limits PATH` to the store with a JSON map of
worker source IPs. Each entry has a `read` budget for traffic from the store
to that peer and a `write` budget for traffic from that peer to the store.
Use `bytes_per_second` and `burst_bytes` in each direction.

```json
{
  "192.0.2.10": {
    "read": {"bytes_per_second": 16777216, "burst_bytes": 8388608},
    "write": {"bytes_per_second": 4194304, "burst_bytes": 2097152}
  }
}
```

Set `checkpoint_peer_budget: true` together with `checkpoint_admission`
on participating Decode and aggregate profiles. Workers use their
configured host IP for the KV connection. The scheduler gathers their
actual page sizes and reserves the complete import atomically across the
store and all peer read budgets. A shard can consume only the reservation
for its observed TCP source IP. For writes, the store checks the sender's
write budget and the shared store budget before accepting payload.

With a peer map, optional traffic from an unlisted address and optional
reads without a complete peer manifest are declined. The target computes
the uncovered prefix when the read reservation cannot be admitted.
Required P/D prompt I/O uses the original protocol. The store's
`--stats` response reports per-peer successful bytes, refusals, and
pending bytes in each direction.

#### Share a peer budget across KV stores

Set `--checkpoint-peer-budget-plan PATH` on each store when several stores
use the same worker's network connection. The plan's `peers` list declares
total budgets. Each member of `stores` names a concrete listening
`endpoint` and its own `peers` list. Each peer entry has an `address`
and a `limit` with the same read and write fields as the per-store map.
For example, a shared 8 MiB/s read budget with a 2 MiB burst can allocate
5 and 3 MiB/s with a 1 MiB burst to each of two stores. Add the read rates,
read bursts, write rates, and write bursts separately. All four totals must
fit the declared peer budgets.

```json
{
  "peers": [
    {"address": "192.0.2.10", "limit": {
      "read": {"bytes_per_second": 8388608, "burst_bytes": 2097152},
      "write": {"bytes_per_second": 6291456, "burst_bytes": 2097152}
    }}
  ],
  "stores": [
    {"endpoint": "192.0.2.11:7952", "peers": [
      {"address": "192.0.2.10", "limit": {
        "read": {"bytes_per_second": 5242880, "burst_bytes": 1048576},
        "write": {"bytes_per_second": 2097152, "burst_bytes": 1048576}
      }}
    ]},
    {"endpoint": "192.0.2.12:7952", "peers": [
      {"address": "192.0.2.10", "limit": {
        "read": {"bytes_per_second": 3145728, "burst_bytes": 1048576},
        "write": {"bytes_per_second": 4194304, "burst_bytes": 1048576}
      }}
    ]}
  ]
}
```

A store installs only the share for its `--listen` endpoint. An empty
`peers` list on a member gives that store no optional peer capacity. A
direction with both fields set to zero can be disabled independently.
Store buckets start empty on each restart and refill at their assigned
rate, so restarting a member never supplies another initial burst.

With all stores listening, run
`plait-node kv-store --audit-checkpoint-peer-budget-plan PATH`. It
verifies the installed plan ID and actual read and write shares at every
endpoint. A successful audit prints the common ID and member list.
With static configuration, retire the old allocation before starting stores
with a different plan.

#### Replace peer shares online

Start each store with `--checkpoint-peer-budget-plan OLD.json` and
`--checkpoint-budget-control DIR` to enable online changes. Use a durable,
private directory per endpoint. The first launch installs generation 0.
Later launches recover the installed plan and any pending change from the
same directory. The local Unix socket accepts changes from the store owner.
Activation reads peer status from the plan's TCP endpoints on the trusted
store network. Keep the same endpoints throughout a change.

Run the following on each store host, with `0` replaced by the currently
installed generation.

```shell
plait-node kv-store --prepare-checkpoint-peer-budget-plan DIR 0 OLD.json NEW.json
plait-node kv-store --activate-checkpoint-peer-budget-plan DIR 0 OLD.json NEW.json
```

1. Prepare all members. Each store closes new optional reservations and
   persists the target. Already granted reads and writes can finish. The
   normal P/D prompt handoff and the stored KV stay available.
2. Use `plait-node kv-store --stats ENDPOINT` to see the target in
   `peer_budget_epoch.target_plan_id` and outstanding work in `pending_bytes`.
   Each activation checks every old member. A member must report zero pending
   bytes for the same target or show that it already activated that successor.
   Retry if a peer is unavailable or still draining.
3. Audit `NEW.json` after activation. The result includes the common
   `generation`. The new buckets start empty and refill at the new rates.
   Use the generation to interpret per-peer counters, which reset when
   a new plan is installed. Store-level counters continue across changes.

A prepare can revise a draining target. If any member has already activated,
prepare the remaining members for that same successor. A retry for the
installed successor returns its epoch. A write failure closes optional
admission and withdraws the local proof. Repair the disk issue and restart
with the same control directory to recover the last durable state.

## Execute a collective

For small all-reduces, each worker sends its complete contribution to each
peer and polls all peer sockets together under one collective deadline.
A slow link leaves the other sends and receives free to progress. The worker
reduces the complete inputs in slot order. The planner uses this path up to
1 MiB of input for two workers. For larger groups it caps the replicated
traffic per worker at 64 KiB.

Larger all-reduces divide the tensor into contiguous shards and assign one
shard to each slot. Each worker concurrently sends contributions to the peers
that own those shards and receives contributions to its own shard. It reduces
the received values in slot order, then starts a second concurrent exchange
of the reduced shard. Each phase has its own collective deadline. Reduce-scatter
returns after the first phase. All-gather receives each peer's shard directly
into its slot's output range. Broadcast and reduce use the source or destination
specified by that call.

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
