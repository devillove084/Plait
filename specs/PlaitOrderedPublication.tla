---- MODULE PlaitOrderedPublication ----
\* One managed W4 admission in the incumbent epoch, with a prepared successor.
\* Same-group commands share a conflict key. The sequence abstracts their durable
\* execution order, supplied by consensus. Each replica installs an arbitrary
\* prefix; a restart reconstructs the serving state from its durable prefix.
\* The device and W4A4 models supply the two K fragments. The submitter folds
\* their published values in order and sends their grants to the control plane.
\* The completion receipt authorizes that output and contains no tensor bytes.
EXTENDS PlaitDeviceCompletion, PlaitW4A4, FiniteSets

CONSTANT Replicas
VARIABLES history, durable, online, state, sawGrant, data, delivered

vars == <<history, durable, online, state, sawGrant, data, delivered>>

Old == 1
Work == 1
Shards == 1..2
CompletionIds == {11, 12}
Grant(device, fence) == [epoch |-> Old, device |-> device, incarnation |-> 1, fence |-> fence]
OldGrants == <<Grant(1, 1), Grant(2, 1)>>
StaleGrants == <<Grant(1, 1), Grant(2, 0)>>

Stamp(k, fence) == CompletionStamp(k, Old, Work, k, fence)
ActCodes(k) == <<Work, 16 - k>>
WeightCodes(k) == <<10 + k, 2 + Work>>
ZeroPoint(k) == 2 * k + 1
KernelValue(k) == NativeDot(ActCodes(k), WeightCodes(k), ZeroPoint(k))
ReferenceValue(k) == LogicalDot(ActCodes(k), WeightCodes(k), ZeroPoint(k))
ReferencePrefix(count) ==
    (IF count >= 1 THEN ReferenceValue(1) ELSE 0) + (IF count >= 2 THEN ReferenceValue(2) ELSE 0)

Completion(id, grants) == [kind |-> "Complete", id |-> id, version |-> Old, work |-> Work, grants |-> grants]
Change(kind) == [kind |-> kind, id |-> 0, version |-> Old, work |-> 0, grants |-> <<>>]

Activate == Change("ActivateSuccessor")
Revoke == Change("Revoke")
Expire == Change("Expire")
Completions == {Completion(11, OldGrants), Completion(11, StaleGrants), Completion(12, OldGrants)}
Commands == Completions \cup {Activate, Revoke, Expire}

Receipt(command, outcome, index) == [command |-> command, outcome |-> outcome, index |-> index]
NoReceipt == Receipt(Change("None"), "NotApplied", 0)
InitialState == [phase |-> "Active", pending |-> TRUE, successor |-> FALSE,
                 receipts |-> [id \in CompletionIds |-> NoReceipt]]
LocalResult(receipt, value) == [receipt |-> receipt, value |-> value]
NoDelivery == LocalResult(NoReceipt, 0)

Init ==
    /\ history = <<>>
    /\ durable = [r \in Replicas |-> 0]
    /\ online = Replicas
    /\ state = [r \in Replicas |-> InitialState]
    /\ sawGrant = FALSE
    /\ data = [jobs |-> [k \in Shards |-> EmptyDeviceJob],
               checkedAt |-> [k \in Shards |-> 0], published |-> {}, count |-> 0, sum |-> 0]
    /\ delivered = NoDelivery

AllowsOldGrants(s) == s.phase \in {"Active", "Draining"}
Admitted(s) == AllowsOldGrants(s) /\ s.pending

GrantsAreCurrent(grants) ==
    /\ grants # <<>>
    /\ \A i \in DOMAIN grants : grants[i] \in {OldGrants[1], OldGrants[2]}

CompletingOutcome(s, command) ==
    IF ~AllowsOldGrants(s) \/ ~GrantsAreCurrent(command.grants) THEN "GrantMismatch"
    ELSE IF ~s.pending THEN "NotAdmitted"
    ELSE IF s.phase = "Draining" THEN "AcceptedRevoked"
    ELSE "AcceptedActive"

Accepted(receipt) == receipt.outcome \in {"AcceptedActive", "AcceptedRevoked"}

ApplyCompletion(s, command, index) ==
    IF s.receipts[command.id] # NoReceipt
    THEN s
    ELSE LET outcome == CompletingOutcome(s, command)
             first == Receipt(command, outcome, index)
         IN [s EXCEPT !.phase = IF outcome = "AcceptedRevoked" THEN "Revoked" ELSE s.phase,
                      !.pending = s.pending /\ ~Accepted(first),
                      !.receipts[command.id] = first]

Apply(s, command, index) ==
    CASE command = Activate ->
           [s EXCEPT !.phase = IF s.phase = "Revoked" \/ ~s.pending THEN "Revoked" ELSE "Draining",
                     !.successor = TRUE]
      [] command \in {Revoke, Expire} ->
           [s EXCEPT !.phase = "Revoked", !.pending = FALSE]
      [] OTHER -> ApplyCompletion(s, command, index)

RECURSIVE Replay(_)
Replay(through) ==
    IF through = 0 THEN InitialState
    ELSE Apply(Replay(through - 1), history[through], through)

Agreed == Replay(Len(history))
OutputReady == data.count = Cardinality(Shards)

\* A proposal acknowledgement or an old group lookup carries no receipt.
ReadGrant(r) ==
    /\ r \in online /\ Admitted(state[r])
    /\ ~sawGrant
    /\ sawGrant' = TRUE
    /\ UNCHANGED <<history, durable, online, state, data, delivered>>

QueueShard(k) ==
    /\ sawGrant /\ Admitted(Agreed) /\ data.jobs[k].phase = "Idle"
    /\ data' = [data EXCEPT !.jobs[k] = QueueDevice(Stamp(k, 1))]
    /\ UNCHANGED <<history, durable, online, state, sawGrant, delivered>>

WriteShard(k) ==
    /\ CanWriteDevice(data.jobs[k])
    /\ data' = [data EXCEPT !.jobs[k] = WriteDevice(@, KernelValue(k))]
    /\ UNCHANGED <<history, durable, online, state, sawGrant, delivered>>

SignalShard(k, stamp) ==
    /\ CanSignalDevice(data.jobs[k], stamp)
    /\ data' = [data EXCEPT !.jobs[k] = SignalDevice(@, stamp)]
    /\ UNCHANGED <<history, durable, online, state, sawGrant, delivered>>

CopyShard(k) ==
    /\ CanCopyDevice(data.jobs[k], Stamp(k, 1))
    /\ data' = [data EXCEPT !.jobs[k] = CopyDevice(@)]
    /\ UNCHANGED <<history, durable, online, state, sawGrant, delivered>>

\* CheckW4 observes the agreed prefix, then the worker caches its result
\* locally. Revocation can execute between the check and the local store.
CheckShard(k) ==
    /\ Admitted(Agreed) /\ data.checkedAt[k] = 0
    /\ CanPublishDevice(data.jobs[k], Stamp(k, 1))
    /\ data' = [data EXCEPT !.checkedAt[k] = Len(history) + 1]
    /\ UNCHANGED <<history, durable, online, state, sawGrant, delivered>>

PublishShard(k) ==
    /\ data.checkedAt[k] # 0 /\ k \notin data.published
    /\ data' = [data EXCEPT !.published = @ \cup {k}]
    /\ UNCHANGED <<history, durable, online, state, sawGrant, delivered>>

PullAndFold ==
    /\ LET k == data.count + 1
       IN /\ k \in data.published
          /\ data' = [data EXCEPT !.count = k, !.sum = @ + data.jobs[k].snapshot]
    /\ UNCHANGED <<history, durable, online, state, sawGrant, delivered>>

CanAppend(command) ==
    CASE command = Activate -> ~Agreed.successor
      [] command = Revoke -> AllowsOldGrants(Agreed)
      [] command = Expire -> Agreed.phase = "Draining" /\ Agreed.pending
      [] OTHER -> OutputReady

\* Exact retries are observable through the original receipt. Repeating the
\* same command in the sequence would stutter at the serving state boundary.
AppendCommitted(command) ==
    /\ online # {}
    /\ command \notin {history[i] : i \in DOMAIN history}
    /\ CanAppend(command)
    /\ history' = Append(history, command)
    /\ UNCHANGED <<durable, online, state, sawGrant, data, delivered>>

Install(r) ==
    /\ r \in online /\ durable[r] < Len(history)
    /\ LET index == durable[r] + 1
       IN /\ durable' = [durable EXCEPT ![r] = index]
          /\ state' = [state EXCEPT ![r] = Apply(@, history[index], index)]
    /\ UNCHANGED <<history, online, sawGrant, data, delivered>>

Crash(r) ==
    /\ r \in online
    /\ online' = online \ {r}
    /\ state' = [state EXCEPT ![r] = InitialState]
    /\ UNCHANGED <<history, durable, sawGrant, data, delivered>>

Restart(r) ==
    /\ r \notin online
    /\ online' = online \cup {r}
    /\ state' = [state EXCEPT ![r] = Replay(durable[r])]
    /\ UNCHANGED <<history, durable, sawGrant, data, delivered>>

Reply(s, command) ==
    LET receipt == s.receipts[command.id]
    IN IF receipt = NoReceipt \/ receipt.command = command THEN receipt
       ELSE Receipt(command, "OperationIdReused", 0)

Deliver(r, command) ==
    /\ r \in online /\ OutputReady /\ delivered = NoDelivery
    /\ LET receipt == Reply(state[r], command)
       IN /\ Accepted(receipt)
          /\ delivered' = LocalResult(receipt, data.sum)
    /\ UNCHANGED <<history, durable, online, state, sawGrant, data>>

Next ==
    \/ \E command \in Commands : AppendCommitted(command)
    \/ \E r \in Replicas : ReadGrant(r) \/ Install(r) \/ Crash(r) \/ Restart(r)
    \/ \E k \in Shards :
           \/ QueueShard(k) \/ WriteShard(k) \/ CopyShard(k) \/ CheckShard(k) \/ PublishShard(k)
           \/ \E fence \in {0, 1} : SignalShard(k, Stamp(k, fence))
    \/ PullAndFold
    \/ \E r \in Replicas, command \in Completions : Deliver(r, command)

Spec == Init /\ [][Next]_vars

ReplicaPrefix ==
    /\ \A r \in Replicas : durable[r] \in 0..Len(history)
    /\ \A r \in online : state[r] = Replay(durable[r])

MaterializedPrefix ==
    /\ data.count \in 0..Cardinality(Shards)
    /\ 1..data.count \subseteq data.published
    /\ data.published \subseteq {k \in Shards : CanPublishDevice(data.jobs[k], Stamp(k, 1))}
    /\ \A k \in Shards :
           /\ DeviceJobSound(data.jobs[k], Stamp(k, 1), ReferenceValue(k))
           /\ ~ENABLED SignalShard(k, Stamp(k, 0))
    /\ data.sum = ReferencePrefix(data.count)

CheckedBeforeClose ==
    \A k \in Shards :
        LET checked == data.checkedAt[k]
        IN /\ k \in data.published => checked # 0
           /\ checked # 0 =>
               /\ checked \in 1..(Len(history) + 1)
               /\ Admitted(Replay(checked - 1))

\* A returned receipt is identical to the first receipt and to any later copy.
\* Different arguments under the same identity yield a reuse error.
StableReceipt ==
    \A i \in DOMAIN history, id \in CompletionIds :
        LET receipt == Replay(i).receipts[id]
        IN receipt # NoReceipt =>
           /\ receipt.index \in 1..i
           /\ receipt.command = history[receipt.index]
           /\ receipt = Agreed.receipts[id]
           /\ \A r \in online :
                 \A command \in Completions :
                     command.id = id /\ durable[r] >= receipt.index =>
                         IF command = receipt.command THEN Reply(state[r], command) = receipt
                         ELSE Reply(state[r], command).outcome = "OperationIdReused"

AuthorizedBeforeClose ==
    \A id \in CompletionIds :
        LET receipt == Agreed.receipts[id]
        IN Accepted(receipt) =>
           /\ receipt.command = Completion(id, OldGrants)
           /\ \A i \in 1..(receipt.index - 1) : history[i] \notin {Revoke, Expire}
           /\ Admitted(Replay(receipt.index - 1))

LiveCompletionAccepted ==
    \A i \in DOMAIN history :
        LET command == history[i]
            before == Replay(i - 1)
        IN command \in Completions /\ command.grants = OldGrants /\ Admitted(before) /\
           before.receipts[command.id] = NoReceipt =>
              LET receipt == Replay(i).receipts[command.id]
              IN /\ receipt.index = i
                 /\ receipt.outcome = IF before.phase = "Active" THEN "AcceptedActive" ELSE "AcceptedRevoked"

SingleAuthorization ==
    Cardinality({id \in CompletionIds : Accepted(Agreed.receipts[id])}) <= 1

DeliveredFromReceipt ==
    delivered # NoDelivery =>
        /\ OutputReady /\ Accepted(delivered.receipt)
        /\ delivered.receipt = Agreed.receipts[delivered.receipt.command.id]
        /\ delivered.receipt.command.grants = OldGrants
        /\ delivered.value = ReferencePrefix(Cardinality(Shards))

DeliveryRemainsAvailable ==
    \A r \in online, command \in Completions :
        LET receipt == Agreed.receipts[command.id]
        IN OutputReady /\ delivered = NoDelivery /\ Accepted(receipt) /\ receipt.command = command /\
           receipt.index <= durable[r] => ENABLED Deliver(r, command)
====
