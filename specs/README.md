# Plait formal verification

TLC explores the finite configurations in this directory. TLAPS checks the
symbolic statements in the proof modules below. The models cover ordered
control state, communication and kernel contracts, and the boundaries
between them.

## Reproduced results

The release-line check ran these configurations from Plait source `207ac56`
with TLC 2.19. Each run exhausted its queue with zero violations.

| Configuration | Distinct states | Checked behavior |
| --- | ---: | --- |
| [Native host expiry](PlaitNativeHostExpiry.cfg) | 126 | Expiry uses the current heartbeat and affected-version set. It blocks placement while retaining fences. |
| [Confirmed host quarantine](PlaitHostQuarantine.cfg) | 8,268 | Exact fault evidence revokes affected groups, excludes a same-host standby, and retains the drain after grant release. |
| [Mixed axes, TP2 × PP2](PlaitMixedPeerAxes.cfg) | 1,925 | Both axes connect before readiness and close after the worker observes the end of its grant and runs cancellation. |
| [Mixed axes, TP2 × PP3](PlaitMixedPeerAxesOdd.cfg) | 48,073 | The same predicates hold with three pipeline stages. Each worker's pending operations share a staging allowance. |
| [Peer failover](PlaitPeerFailover.cfg) | 18 | Exact grant reports, ready standby eligibility, and one active version. |
| [Kernel primitives](PlaitKernelPrimitivesCheck.cfg), [W4A4](PlaitW4A4Check.cfg), [W4A8/W4A16](PlaitW4OperatorsCheck.cfg) | 2 each | Constant-domain arithmetic checks and kernel indexing guards. |
| [Async reduction](PlaitAsyncReduction.cfg) | 3,600 | Matching events and deterministic folding gate result delivery. |
| [Async reduction with a stalled producer](PlaitAsyncReductionStalled.cfg) | 1,937 | An independent operation eventually completes under the configuration's fairness conditions. |
| [Ordered publication](PlaitOrderedPublication.cfg) | 70,968 | Replicas and durable recovery preserve the applied prefix. Retries return stable receipts. |

Removing either `CheckPulse` or `CheckGroups` from the host-expiry guard
produces a counterexample to `ExpiryUsesCurrentEvidence`. One trace commits a
stale expiry after a renewal. The other commits expiry after the live
version set changes.

## Symbolic proof results

TLAPS `e4d66f4` reran both proof modules in strict mode against release
source `207ac56` and discharged all 164 obligations.

| TLAPS proof | Obligations discharged | Statement |
| --- | ---: | --- |
| [Kernel and publication contracts](PlaitContractsProof.tla) | 128 | Zero-point correction over finite groups, guarded packed addressing, and the event and copy conditions for publication. |
| [Mixed-axis geometry](PlaitMixedPeerAxesProof.tla) | 36 | Logical slots map into a positive rectangular TP × PP layout. Two distinct workers cannot share both coordinates. |

## Reproduce

Run these commands from the repository root with TLC 2.19 and `tlc` on
`PATH`. The runner checks the configured invariants and temporal
properties. The `-deadlock` switch disables TLC's implicit deadlock check
for models with intended terminal states.

```sh
./scripts/check-tla.sh
```

Run a single configuration or reproduce a stale-expiry counterexample:

```sh
tlc -deadlock -config specs/PlaitNativeHostExpiry.cfg specs/PlaitNativeHostExpiry.tla
(
  cfg=$(mktemp)
  trap 'rm -f "$cfg"' EXIT
  sed 's/CheckPulse = TRUE/CheckPulse = FALSE/' specs/PlaitNativeHostExpiry.cfg > "$cfg"
  tlc -deadlock -config "$cfg" specs/PlaitNativeHostExpiry.tla
)
```

Run proofs with TLAPS 1.6.0 pre-release, commit `e4d66f4`. The fresh cache
forces proof checking. [TLAPS release](https://github.com/tlaplus/tlapm/releases/tag/1.6.0-pre)

```sh
(
  set -e
  cache=$(mktemp -d)
  trap 'rm -rf "$cache"' EXIT
  tlapm --strict --threads 2 --cache-dir "$cache" specs/PlaitContractsProof.tla
  tlapm --strict --threads 2 --cache-dir "$cache" specs/PlaitMixedPeerAxesProof.tla
)
```

## Model boundaries

The device contracts assume stream ordering and host visibility after a
successful wait and copy. Ordered publication assumes a common order for
commands that affect the same group. Every TLC configuration fixes finite
input and fault domains. End-to-end refinement from the Rust and vendor
implementations and floating-point equivalence across vendors remain open.

## Model catalog

| TLA+ module | Properties checked with TLC |
| --- | --- |
| [Kernel primitives](PlaitKernelPrimitives.tla) | Signed decoding, packed coordinate mapping, and exact integer dot products with accumulator bounds. |
| [W4A4](PlaitW4A4.tla) and [W4A8 / W4A16](PlaitW4Operators.tla) | INT32 zero-point correction and tail reads for W4A4, byte-wise subtraction for W4A8, and packed indexing and dequantization before rounding for W4A16. |
| [Caller-owned W4 stream](PlaitW4CallerStream.tla) | Quantization precedes GEMM, queued work retains access to its scratch range, and a non-finite activation affects only its output row. TLC checked 7,263 distinct states. |
| [CANN caller-owned W4 stream](PlaitW4CallerStreamCANN.tla) | FIFO execution with aligned Cube scratch and a scalar branch. Partial enqueue retains scratch until completion of the submitted stages is observed. TLC checked 7,420 distinct states. |
| [Asynchronous reduction](PlaitAsyncReduction.tla) | W4A4 device completion, stale-event rejection, and deterministic folding. An independent operation completes even while a producer or pull in another operation stalls. |
| [Ordered publication](PlaitOrderedPublication.tla) | Completion and revocation in the agreed command order, receipt identity on retry, and recovery from a replica's durable command prefix. |
| [Peer collectives](PlaitPeerCollective.tla) | Independent link progress within scatter and gather, compatibility with paired rounds, and completion of a separate communicator after participant loss. |
| [FP16 host fold order](PlaitHostFoldOrder.tla) | Each block folds slots in order before narrowing. A nonfinite input discards provisional blocks and selects the scalar path. TLC checked 68 states. Rust bit-pattern tests cover the arithmetic. |
| [Peer AllReduce pricing](PlaitPeerAllReducePricing.tla) | Separate ingress and egress budgets, one RTT charge per exchange, exact shard traffic, and rejection of a quote with a missing direction. TLC checked 32 states across six representative cases. |
| [Peer AllReduce budget](PlaitPeerAllReduceBudget.tla) | A new version verifies the egress budget and slot layout before joining. A failed candidate leaves the existing epoch usable. Default peers remain wire-compatible. TLC checked 4,472 distinct states. |
| [Native TP recipe budget](PlaitNativeTpRecipeBudget.tla) | A recipe binds width and egress budget. Fresh link samples can change the next choice while the incumbent keeps its prior binding. Activation requires a matching, warmed replacement. TLC checked 46,608 distinct states. |
| [Hosted AllReduce profile](PlaitHostedAllReduceProfile.tla) | Each candidate host advertises the committed profile. Startup resolves the exact profile and publication requires the peers to agree on its budget. TLC checked 352 distinct states. |
| [Hosted budget selection](PlaitHostedBudgetSelection.tla) | A candidate requires fresh directed samples and support from every host. The incumbent is quoted with its committed profile as the link conditions change. TLC checked 2,484 distinct states. |
| [Rankless peer join](PlaitRanklessPeerJoin.tla) | Socket handshakes bind the group version and grant view. Cancellation closes the participant's links. |
| [Logical slot layout](PlaitRanklessLayout.tla) | A participant reports ready after connecting to all members with a common slot layout. |
| [Mixed TP × PP peer axes](PlaitMixedPeerAxes.tla) | Both axes connect before readiness and share one staging allowance. After an exact grant ends, workers that observe the end and run cancellation close both axes. TLC checked 1,925 states for TP2 × PP2 and 48,073 for TP2 × PP3. |
| [Peer revocation](PlaitPeerRevocation.tla) | Observing that an exact grant has ended closes that communicator and wakes a pending call. A draining group can keep its links. |
| [Peer failover](PlaitPeerFailover.tla) | Exact-grant failure reports, readiness and host-loss eligibility for standby activation, one active version, and grant retention until release. |
| [Host quarantine](PlaitHostQuarantine.tla) | A confirmed fault revokes every affected group and excludes a same-host standby. Grants remain held until worker exit and release. The drain survives release and excludes new placements on the host. TLC checked 8,268 states. |
| [Native host expiry](PlaitNativeHostExpiry.tla) | Expiry matches the last committed heartbeat and current affected versions. It removes the host from placement and revokes its groups while retaining their grants. TLC checked 126 distinct states. |
| [Device-scoped shared-host faults](PlaitSharedHostFault.tla) | A device fault revokes its group version. A separate group on the same host keeps its communicator and grants. The model also checks standby activation, asynchronous cancellation, and per-device release after stop. TLC checked 36,895 states. |
| [Unavailable worker](PlaitWorkerUnavailable.tla) | An exact report withdraws one device and revokes the group. Release preserves the device drain, healthy devices can join a successor, and a stale report cannot drain a device held by the successor. TLC checked 113 states. |
| [PCI device watch](PlaitPciWatch.tla) | Two successful absence observations qualify the idle report. A race with allocation or a newer incarnation rejects the report. TLC checked 27 states. |
| [PCI watch reload](PlaitPciReload.tla) | A new binding requires fresh absence observations. The watch retains binding history through removal, and a delayed report applies only to its registered incarnation. TLC checked 76 states. |
| [Worker readmission](PlaitWorkerReadmission.tla) | The ordered transition matches the drain and old incarnation, checks for grants and deferred placements, resets measurements, and preserves the fence counter. TLC checked 102 states. |
| [Peer collective requests](PlaitPeerRequests.tla) | Blocking and ticketed operations share the wire order. A dropped ticket keeps its sequence, and a cancelled communicator denies delivery of a buffered result. TLC checked 695,556 states. |
| [Peer request admission](PlaitPeerAdmission.tla) | Queue and byte limits are checked before input staging. Capacity rejection leaves the stream and sequence untouched and returns any acquired slot. TLC checked 186 states. |
| [Peer staging budgets](PlaitPeerStaging.tla) | Asynchronous inputs and results retain their local byte reservations through exchange, abandonment, and delivery. TLC checked 19,584 states. |
| [Peer scatter/gather pipeline](PlaitPeerScatterGatherPipeline.tla) | Each block requires contributions from all slots. Scatter frames precede gather frames in both directions. TLC checked 582 states for two pipelined workers, 275 for a mixed pair, and 18,270 for three workers with one peer using two-phase exchange. |
| [Staging and host reservations](PlaitPeerStagingGrants.tla) | Co-located communicators divide the pageable part of their epoch's host reservation across device grants. TLC checked 75 states. |
| [Sharded prompt scoring](https://github.com/devillove084/vllm/blob/plait/v0.31-runtime/tools/plait/serving/specs/PlaitShardedPromptScores.tla) | Exact-integer candidate coverage and rank sums across 2,187 score vectors. TLC checked 1,024 states for agreement on fallback rows and collective order. |
| [Prompt scoring plan](https://github.com/devillove084/vllm/blob/plait/v0.31-runtime/tools/plait/serving/specs/PlaitPromptPlan.tla) | Every TP slot selects the same scoring path before issuing a collective. Sharded scoring requires matching vocabulary geometry. Overlap requires group-wide opt-in, and dense overlap requires a submitter on every slot. TLC checked 1,321 distinct states. |
| [Pipeline KV cache budgets](https://github.com/devillove084/vllm/blob/ece8393afb3174dc195b3aaa3acfd8c1362219b3/tools/plait/serving/specs/PlaitPipelineCacheBudget.tla) | Hosts agree on the per-stage vector before workers start. TP workers use the budget of their global pipeline stage. TLC checked 64 distinct states. |
| [Native serving handoff through a temporary group](https://github.com/devillove084/vllm/blob/9ad438a44299bb9bf56b26b2eb42cab27beae23c/tools/plait/serving/specs/PlaitNativeBridgeHandoff.tla) | The incoming version acquires shared devices after release. A rejected request can retry once on a different active route. An admitted stream stays on its engine. TLC checked 4,729 distinct states. |
| [Automatic device reuse](https://github.com/devillove084/vllm/blob/008461a1f5bd1aee6d73e9f9e32d1c629e8e545c/tools/plait/serving/specs/PlaitAutomaticDeviceReuse.tla) | A feasible preview and sufficient bridge capacity gate the first switch. The target waits for the source grants and takes traffic when ready. An active bridge can continue preparation after a controller restart. TLC checked 152 states. |
| [Measured capacity for device reuse](https://github.com/devillove084/vllm/blob/be30b12127020f523a8051ef0ebde50c613f7404/tools/plait/serving/specs/PlaitMeasuredDeviceReuse.tla) | In measured mode, both handoffs require a current, unexpired profile from timed trials. A warmup wave waits for the first token from each stream before injecting a prompt. A warmup change cancels the pending benchmark and preserves completed profiles. TLC checked 3,628 distinct states and found counterexamples for all four warmup rules. |
| [Stable demand before bridge return](https://github.com/devillove084/vllm/blob/c68befa4dc452b46fed0c684605302f3fe01b5af/tools/plait/serving/specs/PlaitBridgeReturnDemand.tla) | Return requires consecutive qualifying windows for the same target epoch and capacity profile. TLC checked 643 distinct states and found counterexamples when allowing one window, keeping a count across a miss, or combining different profile identities. |
| [Native launch failure](https://github.com/devillove084/vllm/blob/plait/v0.31-runtime/tools/plait/serving/specs/PlaitNativeLaunchFailure.tla) | Launchers recheck the version under the local claim. A failed candidate preserves its incumbent, and each grant stays held until the worker has stopped or the executable has been rejected. The failed launch keeps its claim until the release receipt. TLC checked 101 distinct states. |
| [Failed launch recovery](https://github.com/devillove084/vllm/blob/plait/v0.31-runtime/tools/plait/serving/specs/PlaitFailedLaunchRecovery.tla) | The launcher records a rejected exec before reporting it. After a crash, it retries with the original operation IDs. A reused device keeps its successor grant, and the journal stays until every release receipt has arrived. TLC checked 71 distinct states. |
| [Native engine stop](https://github.com/devillove084/vllm/blob/plait/v0.31-runtime/tools/plait/serving/specs/PlaitNativeEngineStop.tla) | Only the revoked process group is stopped. Grants remain held until its descendants exit, and the launcher preserves release IDs across restart. TLC checked 34 distinct states. |
| [Engine ownership across launcher restart](https://github.com/devillove084/vllm/blob/plait/v0.31-runtime/tools/plait/serving/specs/PlaitEngineOwnerRecovery.tla) | The local engine parent retains the launch lock, records ownership before exec, accepts the current instance's stop request, and records exit before grant return. TLC checked 41 distinct states. |
| [Orphaned engine recovery](https://github.com/devillove084/vllm/blob/plait/v0.31-runtime/tools/plait/serving/specs/PlaitOrphanCgroupRecovery.tla) | Recovery stops the exact revoked cgroup subtree, preserves independent groups, and releases grants after recording that the subtree is empty. TLC checked 32 distinct states. |
| [Sharded prompt overlap](https://github.com/devillove084/vllm/blob/plait/v0.31-runtime/tools/plait/serving/specs/PlaitShardedPromptOverlap.tla) | Workers may mix blocking and asynchronous calls while preserving collective order and retaining at most two projections. TLC checked that unfinished requests can take another step across 362 two-worker and 22,742 four-worker states. |
| [Replicated group manifest](PlaitGroupManifest.tla) | Activation follows complete endpoint and handshake evidence. Retrying an old release preserves any successor grant. |
| [Native group placement](PlaitGroupPlacement.tla) | A planned reservation checks the version, incumbent, and resource availability against concurrent changes. |
| [Asymmetric device selection](PlaitAsymmetricDeviceChoice.tla) | For each local width, slot-ordered selection matches the exhaustive minimum compute cost among assignments that fit each slot's device memory. TLC checked 864 states. Fastest-first selection and ignoring slot memory each produce a counterexample. |
| [Native probe sessions](PlaitProbePlacement.tla) | A planned reservation checks both endpoint revisions if a probe restarts after measurement. |
| [Native compute rates](PlaitComputeRatePlacement.tla) | A measured reservation checks the selected devices and the scored incumbent. A worker can publish a rate while its exact grant is open. |
| [Resident group replanning](PlaitGroupReplan.tla) | A planner submits after consecutive matching recommendations. Competing planners can reserve only one successor for the observed epoch. |
| [Replanning link measurements](PlaitReplanLinkWindows.tla) | Each confirmation advances every priced direction under the same probe registrations. TLC checked 275 states and found counterexamples for reused samples, partial updates, and mixed probe sessions. |
| [Committed group recipes](PlaitGroupRecipe.tla) | Reservation binds the recipe and grants together. A launch uses the recipe recorded for its grant epoch. |
| [Active group sets](PlaitActiveGroupSet.tla) | An ordered switch checks every expected epoch and applies the complete change. Routing uses one observation and requires every group in a P/D path. Existing work keeps its draining grants. |
| [Group set preparation](PlaitGroupPrepareSet.tla) | Two placements share a staging ledger on one host. A stale guard or failed reservation leaves both versions and the ledger unchanged. A retry returns the same receipt. TLC checked 50,128 distinct states. |
| [Failed candidate sets](PlaitAbortFailedSet.tla) | An abort uses the exact preparation receipt and requires a failed member and a batch that has never activated. Each grant remains held until its owner releases it. TLC checked 223,616 distinct states. |
| [Native serving mode policy](https://github.com/devillove084/vllm/blob/plait/v0.31-runtime/tools/plait/serving/specs/PlaitServingModePolicy.tla) | A switch requires repeated windows, a viable target, fresh profiles before send, and a matching active set at commit. A retry returns the original receipt. |
| [On-demand serving preparation](https://github.com/devillove084/vllm/blob/plait/v0.31-runtime/tools/plait/serving/specs/PlaitServingProvision.tla) | Stable demand can prepare a complete candidate set. Launch follows a grant. Switch dispatch checks the exact measured versions and each endpoint before the runtime compares the active set. TLC checked 101,760 distinct states. |
| [Serving calibration](https://github.com/devillove084/vllm/blob/plait/v0.31-runtime/tools/plait/serving/specs/PlaitServingCalibration.tla) | The private benchmark pins a complete candidate and leaves production counters unchanged. Changes to the observed active set, bindings, or configuration invalidate a running job. An accepted measurement can become the next incumbent baseline. TLC checked 1,211,296 distinct states. |
| [Calibration restore](https://github.com/devillove084/vllm/blob/e60c3bd0ca0c6633b643040a77a550ebf078253e/tools/plait/serving/specs/PlaitServingProfileRestore.tla) | An atomic file update preserves the more recent measurement for the same identity. A restart revalidates the route, device grants, and lifetime. Changing the warmup count preserves the profile identity. TLC checked 445,604 distinct states. |
| [Native serving cutover](PlaitNativeServingCutover.tla) | The deployment publishes a ready engine's origin before activation. An admitted stream keeps its version as ingress replicas observe the switch. |
| [Native serving retirement](PlaitNativeServingRetire.tla) | Planned retirement closes admission, waits for the engine to drain, and releases each host's grants after its process group exits. |
| [Hosted rollout and failover](PlaitHostLaunch.tla) | Full readiness before activation, ordered standby promotion on incumbent revocation, and exclusion of groups affected by the same host loss. |
| [Hosted assignment](PlaitHostedAssignment.tla) | An observed assignment requires an ordered claim before launch. The local lock remains held through supervisor loss until the launcher exits. |
| [KFD device release](PlaitKfdRelease.tla) | A complete report for the captured device checks its queues and VRAM. Activity on another device leaves release enabled. |
| [Hosted link evidence](PlaitHostedLinkEvidence.tla) | Fresh directional samples match both probe sessions. Placement looks up the target role and TP width, then the shared width and configured default. TLC checked 682,534 states with [strict profiles](PlaitHostedLinkEvidence.cfg) and 1,174,918 with the [fallback enabled](PlaitHostedLinkEvidenceFallback.cfg). |
| [Hosted pool reservations](PlaitHostedPool.tla) | A placement is checked against current inventory when prepared. Competing placements cannot reserve the same devices or API ports, including while a revoked group awaits retirement. |
| [Hosted P/D publication](PlaitHostedPd.tla) | Both roles cut over in one publication after their participants report readiness. Candidate failure or a changed directory revision prevents a stale cutover. |
| [Hosted Decode width switch](PlaitHostedPdDecodeSwitch.tla) | A ready Decode replaces its predecessor while Prefill keeps its grants. An admitted stream keeps its Decode epoch and retirement requires both old stop reports. TLC checked 3,929 distinct states. |
| [Hosted serving modes](PlaitHostedServingModes.tla) | An aggregate, a split path, or both can be published in one revision after all target groups are ready. Removed groups drain admitted requests before retiring. |
| [Hosted checkpoint contract](PlaitHostedCheckpointContract.tla) | A published route agrees with the KV contract of each participating group. A retained group version keeps its launch binding as the directory changes serving shapes. |
| [Hosted checkpoint budget](PlaitHostedCheckpointBudget.tla) | A replacement preserves the incumbent rate after steady traffic and in-flight checkpoint reservations are charged to shared ports and storage. |
| [Checkpoint admission](https://github.com/devillove084/vllm/blob/plait/v0.31-runtime/tools/plait/kv-store/specs/PlaitCheckpointAdmission.tla) | Optional writes and reserved reads share a burst. Refill accounts for bytes already granted to unfinished transfers. |
| [Checkpoint peers](https://github.com/devillove084/vllm/blob/plait/v0.31-runtime/tools/plait/kv-store/specs/PlaitCheckpointPeers.tla) | A complete page reservation fits every peer and the store. Each peer claims only its own reserved bytes. |
| [Shared checkpoint peer budget](https://github.com/devillove084/vllm/blob/plait/v0.31-runtime/tools/plait/kv-store/specs/PlaitCheckpointPeerPlan.tla) | Independent store shares stay inside one directional budget over any sampled interval, including a store restart. |
| [Checkpoint budget replacement](https://github.com/devillove084/vllm/blob/plait/v0.31-runtime/tools/plait/kv-store/specs/PlaitCheckpointBudgetSwitch.tla) | Each generation installs one plan. Older generations stop admitting optional transfers and drain their grants before the next generation opens, including during conflicting proposals and restarts. |
| [Hosted mode selection](PlaitHostedModeSelection.tla) | Mixed routes fit the combined port and KV-store budgets. A switch preserves the incumbent SLO rate and has positive benefit after startup and drain. |
| [Execution maintenance](PlaitExecutionMaintenance.tla) | A new grant avoids a withdrawn target, a separate device keeps its group, and confirmation waits for both grants and deferred holds to release. |
| [Hosted P/D budgeting](PlaitHostedPdBudget.tla) | Equal path shares consume the same Prefill and directional port budgets. The model checks combined TP and KV demand on shared ports. |
| [Hosted pipeline placement](PlaitHostedPipelinePlacement.tla) | An API-first grant order assigns workers to the intended stage. A two-stage plan fits device and host memory, and the sum of TP, PP, and KV traffic fits the shared port. |
| [Hosted P/D scaling](PlaitHostedPdScale.tla) | A proposed change uses repeated observations from the current ingress session and directory revision. Publication starts a new cooldown. |
| [Hosted P/D role transition](PlaitHostedPdTransition.tla) | The replacement route becomes live before the old group retires. The target acquires the retired device only after release and becomes visible only after readiness. |
| [Hosted P/D reconfiguration](PlaitHostedPdReconfigure.tla) | A bridge preserves the incumbent rate. The projected gain pays for the transition. Commit checks the directory and declared release capability. |
| [Hosted P/D recovery](PlaitHostedPdRecovery.tla) | A spare can restore a failed slot without ingress traffic. Cutover preserves healthy epochs, checks readiness and directory revision, and requires proof to release old grants. Before publication, the model checks that a transition remains enabled. |
| [Hosted pipeline rollout](PlaitHostedPipeline.tla) | Both pipeline lanes deliver activations for the admitted epoch. The next version waits for all workers, and the old version retires after its requests drain. |
| [Serving ingress](https://github.com/devillove084/vllm/blob/plait/v0.31-runtime/tools/plait/serving/specs/PlaitServingIngress.tla) | One admission per ordinary forwarded request, monotonic route refresh, version checks on native HTTP bindings, and output pinned to the admitted epoch. |
| [Completion replay and KV resume](https://github.com/devillove084/vllm/blob/plait/v0.31-runtime/tools/plait/serving/specs/PlaitCompletionReplay.tla) | Visible positions are preserved. Each KV resume sends the latest checkpoint and requires a matching acknowledgment and prefix before publishing a suffix. |
| [Completion prefix resume](https://github.com/devillove084/vllm/blob/plait/v0.31-runtime/tools/plait/serving/specs/PlaitCompletionPrefixResume.tla) | A V2 engine rebuilds context from the latest visible history. The ingress verifies its acknowledgment and both output prefixes before publishing a suffix. TLC checked 15,868 states. |
| [Planned stream handoff](https://github.com/devillove084/vllm/blob/plait/v0.31-runtime/tools/plait/serving/specs/PlaitPlannedStreamHandoff.tla) | A cut uses the last visible position and waits for a ready replacement. A seeded request requires a compatible sampler and a seed acknowledgment before the target publishes. |
| [KV publication](https://github.com/devillove084/vllm/blob/plait/v0.31-runtime/tools/plait/kv-store/specs/PlaitKvPublication.tla) | All heads must come from one write attempt before a page becomes readable. A consumer reads its requested head range, and a complete page stays immutable. |
| [Layered KV](https://github.com/devillove084/vllm/blob/plait/v0.31-runtime/tools/plait/kv-store/specs/PlaitLayeredKv.tla) | A page publishes after one write attempt covers every layer and head. A consumer can read it with a different stage partition or a whole-head request. |
| [Prefill to Decode handoff](https://github.com/devillove084/vllm/blob/plait/v0.31-runtime/tools/plait/serving/specs/PlaitKvHandoff.tla) | The receipt matches the request context. Decode emits output after loading the required pages, including when the shared store can reset between admission and load. |
| [P/D stream replay](https://github.com/devillove084/vllm/blob/plait/v0.31-runtime/tools/plait/serving/specs/PlaitPdStreamReplay.tla) | A failed Decode version is excluded. Each attempt releases its own handoff, and suffix output follows a checked prefix. The ingress can wait for a route within the original deadline. TLC checked 12,597 states. |

The ordered publication configuration explores 70,968 distinct states.
