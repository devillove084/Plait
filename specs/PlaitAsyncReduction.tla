---- MODULE PlaitAsyncReduction ----
\* Two admitted operations overlap across two epochs. Each operation uses
\* two immutable K shards; its consumers need only this operation's plan.
\* Publish and Pull model dispatch_and_reduce, Reduce models KReduction,
\* and Deliver models the ordered grant check after reduction.
EXTENDS Naturals, FiniteSets

CONSTANTS StopOldShard, AllowRevoke
VARIABLES active, revoked, admitted, published, pulled, prefix, delivered

Ops == {1, 2}
Shards == {1, 2}
Epochs == {1, 2}
Old == 1
New == 2
vars == <<active, revoked, admitted, published, pulled, prefix, delivered>>

Init ==
    /\ active = Old
    /\ revoked = FALSE
    /\ admitted = [o \in Ops |-> 0]
    /\ published = [o \in Ops |-> {}]
    /\ pulled = [o \in Ops |-> {}]
    /\ prefix = [o \in Ops |-> 0]
    /\ delivered = [o \in Ops |-> [count |-> 0, authorized |-> FALSE]]

LiveGrant(o) == admitted[o] # 0 /\ (admitted[o] # Old \/ ~revoked)

Admit(o) ==
    /\ o = active
    /\ admitted[o] = 0
    /\ admitted' = [admitted EXCEPT ![o] = active]
    /\ UNCHANGED <<active, revoked, published, pulled, prefix, delivered>>

Activate ==
    /\ active = Old
    /\ admitted[Old] = Old
    /\ active' = New
    /\ UNCHANGED <<revoked, admitted, published, pulled, prefix, delivered>>

RevokeOld ==
    /\ AllowRevoke /\ active = New /\ ~revoked
    /\ revoked' = TRUE
    /\ UNCHANGED <<active, admitted, published, pulled, prefix, delivered>>

Publish(o, k) ==
    /\ LiveGrant(o)
    /\ ~(StopOldShard /\ o = Old /\ k = 2)
    /\ k \notin published[o]
    /\ published' = [published EXCEPT ![o] = @ \cup {k}]
    /\ UNCHANGED <<active, revoked, admitted, pulled, prefix, delivered>>

Pull(o, source, epoch, k) ==
    /\ o = source /\ epoch = admitted[o] /\ epoch # 0
    /\ k \in published[source] \ pulled[o]
    /\ pulled' = [pulled EXCEPT ![o] = @ \cup {k}]
    /\ UNCHANGED <<active, revoked, admitted, published, prefix, delivered>>

Reduce(o) ==
    /\ prefix[o] + 1 \in pulled[o]
    /\ prefix' = [prefix EXCEPT ![o] = @ + 1]
    /\ UNCHANGED <<active, revoked, admitted, published, pulled, delivered>>

Deliver(o) ==
    /\ LiveGrant(o)
    /\ prefix[o] = Cardinality(Shards)
    /\ delivered[o].count = 0
    /\ delivered' = [delivered EXCEPT ![o] = [count |-> prefix[o], authorized |-> LiveGrant(o)]]
    /\ UNCHANGED <<active, revoked, admitted, published, pulled, prefix>>

Next ==
    \/ Activate
    \/ RevokeOld
    \/ \E o \in Ops : Admit(o) \/ Reduce(o) \/ Deliver(o)
    \/ \E o \in Ops, k \in Shards : Publish(o, k)
    \/ \E o, source \in Ops, epoch \in Epochs, k \in Shards : Pull(o, source, epoch, k)

\* All enabled productive transitions change a finite, monotone state.
\* Fair progress permits a permanently stalled or revoked old contributor.
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

PrefixOfReceived ==
    \A o \in Ops : pulled[o] \subseteq published[o] /\ 1..prefix[o] \subseteq pulled[o]

CompleteAndAuthorized ==
    \A o \in Ops :
        delivered[o].count # 0 => delivered[o].count = Cardinality(Shards) /\ delivered[o].authorized

ForeignPayloadRejected ==
    \A o, source \in Ops, epoch \in Epochs, k \in Shards :
        (source # o \/ epoch # admitted[o]) => ~ENABLED Pull(o, source, epoch, k)

EarlyPullEnabled ==
    \A o \in Ops :
        \A k \in published[o] \ pulled[o] : ENABLED Pull(o, o, admitted[o], k)

IncrementalFoldEnabled ==
    \A o \in Ops : prefix[o] + 1 \in pulled[o] => ENABLED Reduce(o)

StalledOldNeverDelivers == StopOldShard => delivered[Old].count = 0
NewCompletes == <> (delivered[New].count # 0)
OldDrains == (~StopOldShard /\ ~AllowRevoke) => <> (delivered[Old].count # 0)
====
