---- MODULE PlaitAsyncReduction ----
\* Two admitted operations overlap across two epochs or use the same two workers
\* in one epoch. Each operation uses two immutable K shards.
\* Publish and Pull model dispatch_and_reduce, Reduce models KReduction,
\* and Complete models the ordered commit authorizing delivery after reduction.
EXTENDS Naturals, FiniteSets

CONSTANTS StopOldShard, HoldOldPull, AllowRevoke, AllowExpire, AllowFailure, SharedEpoch
VARIABLES active, revoked, admitted, pending, published, pulled, prefix, committed, delivered, failed

Ops == {1, 2}
Shards == {1, 2}
Epochs == {1, 2}
Old == 1
New == 2
vars == <<active, revoked, admitted, pending, published, pulled, prefix, committed, delivered, failed>>
EmptyReceipt == [count |-> 0, authorized |-> FALSE]

Init ==
    /\ active = Old
    /\ revoked = FALSE
    /\ admitted = [o \in Ops |-> 0]
    /\ pending = {}
    /\ published = [o \in Ops |-> {}]
    /\ pulled = [o \in Ops |-> {}]
    /\ prefix = [o \in Ops |-> 0]
    /\ committed = [o \in Ops |-> EmptyReceipt]
    /\ delivered = committed
    /\ failed = FALSE

LiveGrant(o) == o \in pending /\ (admitted[o] # Old \/ ~revoked)

Admit(o) ==
    /\ active = (IF SharedEpoch THEN Old ELSE o)
    /\ admitted[o] = 0
    /\ (HoldOldPull /\ o = New) => 2 \in published[Old]
    /\ admitted' = [admitted EXCEPT ![o] = active]
    /\ pending' = pending \cup {o}
    /\ UNCHANGED <<active, revoked, published, pulled, prefix, committed, delivered, failed>>

Activate ==
    /\ ~SharedEpoch /\ active = Old
    /\ admitted[Old] = Old
    /\ active' = New
    /\ UNCHANGED <<revoked, admitted, pending, published, pulled, prefix, committed, delivered, failed>>

RevokeOld ==
    /\ AllowRevoke /\ active = New /\ ~revoked
    /\ revoked' = TRUE
    /\ UNCHANGED <<active, admitted, pending, published, pulled, prefix, committed, delivered, failed>>

\* Exercise the race after the data plane already has the full reduction.
ExpireOld ==
    /\ AllowExpire /\ active = New /\ Old \in pending
    /\ prefix[Old] = Cardinality(Shards)
    /\ pending' = pending \ {Old}
    /\ UNCHANGED <<active, revoked, admitted, published, pulled, prefix, committed, delivered, failed>>

\* After a terminal error on one shard, the caller's report and the ordered
\* admission close can advance independently of other shard results.
ReportOldFailure ==
    /\ AllowFailure /\ Old \in pending /\ ~failed
    /\ failed' = TRUE
    /\ UNCHANGED <<active, revoked, admitted, pending, published, pulled, prefix, committed, delivered>>

CloseFailedAdmission ==
    /\ failed /\ Old \in pending
    /\ pending' = pending \ {Old}
    /\ UNCHANGED <<active, revoked, admitted, published, pulled, prefix, committed, delivered, failed>>

Collecting(o) == ~(o = Old /\ failed)

\* A producer has already published the first operation's shard. Only that
\* transfer is stalled, so the same producer can serve another operation.
PullHeld(o, k) == HoldOldPull /\ o = Old /\ k = 2

Publish(o, k) ==
    /\ LiveGrant(o)
    /\ ~(StopOldShard /\ o = Old /\ k = 2)
    /\ k \notin published[o]
    /\ published' = [published EXCEPT ![o] = @ \cup {k}]
    /\ UNCHANGED <<active, revoked, admitted, pending, pulled, prefix, committed, delivered, failed>>

Pull(o, source, epoch, k) ==
    /\ Collecting(o)
    /\ o = source /\ epoch = admitted[o] /\ epoch # 0
    /\ k \in published[source] \ pulled[o]
    /\ ~PullHeld(o, k)
    /\ pulled' = [pulled EXCEPT ![o] = @ \cup {k}]
    /\ UNCHANGED <<active, revoked, admitted, pending, published, prefix, committed, delivered, failed>>

Reduce(o) ==
    /\ Collecting(o)
    /\ prefix[o] + 1 \in pulled[o]
    /\ prefix' = [prefix EXCEPT ![o] = @ + 1]
    /\ UNCHANGED <<active, revoked, admitted, pending, published, pulled, committed, delivered, failed>>

Complete(o) ==
    /\ Collecting(o)
    /\ LiveGrant(o)
    /\ prefix[o] = Cardinality(Shards)
    /\ committed[o] = EmptyReceipt
    /\ pending' = pending \ {o}
    /\ committed' = [committed EXCEPT ![o] = [count |-> prefix[o], authorized |-> LiveGrant(o)]]
    /\ UNCHANGED <<active, revoked, admitted, published, pulled, prefix, delivered, failed>>

Deliver(o) ==
    /\ committed[o] # EmptyReceipt
    /\ delivered[o].count = 0
    /\ delivered' = [delivered EXCEPT ![o] = committed[o]]
    /\ UNCHANGED <<active, revoked, admitted, pending, published, pulled, prefix, committed, failed>>

Next ==
    \/ Activate
    \/ RevokeOld
    \/ ExpireOld
    \/ ReportOldFailure
    \/ CloseFailedAdmission
    \/ \E o \in Ops : Admit(o) \/ Reduce(o) \/ Complete(o) \/ Deliver(o)
    \/ \E o \in Ops, k \in Shards : Publish(o, k)
    \/ \E o, source \in Ops, epoch \in Epochs, k \in Shards : Pull(o, source, epoch, k)

\* Every productive transition advances a finite operation lifecycle.
\* Fair progress permits a permanently stalled or revoked old contributor.
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

PrefixOfReceived ==
    \A o \in Ops : pulled[o] \subseteq published[o] /\ 1..prefix[o] \subseteq pulled[o]

CompleteAndAuthorized ==
    \A o \in Ops :
        /\ committed[o] # EmptyReceipt =>
            committed[o].count = Cardinality(Shards) /\ committed[o].authorized /\ o \notin pending
        /\ delivered[o] # EmptyReceipt => delivered[o] = committed[o]

ForeignPayloadRejected ==
    \A o, source \in Ops, epoch \in Epochs, k \in Shards :
        (source # o \/ epoch # admitted[o]) => ~ENABLED Pull(o, source, epoch, k)

EarlyPullEnabled ==
    \A o \in Ops :
        \A k \in published[o] \ pulled[o] :
            Collecting(o) /\ ~PullHeld(o, k) => ENABLED Pull(o, o, admitted[o], k)

IncrementalFoldEnabled ==
    \A o \in Ops : Collecting(o) /\ prefix[o] + 1 \in pulled[o] => ENABLED Reduce(o)

SharedWorkerCanPublish ==
    \A o \in Ops, k \in Shards :
        SharedEpoch /\ LiveGrant(o) /\ k \notin published[o] => ENABLED Publish(o, k)

StalledOldNeverDelivers == StopOldShard => delivered[Old].count = 0
HeldPullNeverDelivers ==
    HoldOldPull => /\ 2 \notin pulled[Old]
                   /\ delivered[Old] = EmptyReceipt
                   /\ delivered[New] # EmptyReceipt =>
                       Old \in pending /\ admitted[Old] = admitted[New] /\ 2 \in published[Old]
FailureReportEnabled == AllowFailure /\ Old \in pending /\ ~failed => ENABLED ReportOldFailure
FailedOldCloses == <> (failed /\ Old \notin pending)
NewCompletes == <> (delivered[New].count # 0)
OldDrains ==
    (~StopOldShard /\ ~HoldOldPull /\ ~AllowRevoke /\ ~AllowExpire /\ ~AllowFailure)
        => <> (delivered[Old].count # 0)
====
