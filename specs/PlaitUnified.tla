---- MODULE PlaitUnified ----
\* Unified TLA+ abstraction of the Plait control plane.
\*
\* Mapping to Rust:
\*   Nodes       <-> PeerGroup ranks / RankWeight
\*   Ops         <-> OpSpec / fused operator
\*   Contributors<-> OpSpec.contributors
\*   Kind        <-> FuseKind (Sum, Max, Gather)
\*   Deps        <-> EPaxos committed dependency edges
\*   Assign      <-> HeterogeneousLayout row assignment / capacity
\*   Contribute  <-> local compute result
\*   Fuse        <-> Runtime::try_fuse
\*   Commit      <-> EPaxos commit
\*   Execute     <-> ExecutionGraph::mark_executed
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS Nodes, Ops, Contributors, Kind, Deps, Capacity
VARIABLES assigned, contributed, fused, committed, executed

vars == <<assigned, contributed, fused, committed, executed>>

Init ==
    /\ assigned = [o \in Ops |-> 0]
    /\ contributed = [o \in Ops |-> {}]
    /\ fused = {}
    /\ committed = {}
    /\ executed = {}

Work(r) == Cardinality({o \in Ops : assigned[o] = r})

Assign(o, r) ==
    /\ assigned[o] = 0
    /\ Work(r) < Capacity[r]
    /\ assigned' = [assigned EXCEPT ![o] = r]
    /\ UNCHANGED <<contributed, fused, committed, executed>>

Contribute(o, r) ==
    /\ r \in Contributors[o]
    /\ r \notin contributed[o]
    /\ o \notin fused
    /\ contributed' = [contributed EXCEPT ![o] = contributed[o] \cup {r}]
    /\ UNCHANGED <<assigned, fused, committed, executed>>

Fuse(o) ==
    /\ Contributors[o] \subseteq contributed[o]
    /\ o \notin fused
    /\ fused' = fused \cup {o}
    /\ UNCHANGED <<assigned, contributed, committed, executed>>

Commit(o) ==
    /\ o \in fused
    /\ Deps[o] \subseteq committed
    /\ o \notin committed
    /\ committed' = committed \cup {o}
    /\ UNCHANGED <<assigned, contributed, fused, executed>>

Execute(o) ==
    /\ o \in committed
    /\ o \notin executed
    /\ executed' = executed \cup {o}
    /\ UNCHANGED <<assigned, contributed, fused, committed>>

Next ==
    \/ \E o \in Ops, r \in Nodes : Assign(o, r)
    \/ \E o \in Ops, r \in Nodes : Contribute(o, r)
    \/ \E o \in Ops : Fuse(o)
    \/ \E o \in Ops : Commit(o)
    \/ \E o \in Ops : Execute(o)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

AssignedOnce ==
    \A o \in Ops : assigned[o] = 0 \/ assigned[o] \in Nodes

CapacityBound ==
    \A r \in Nodes : Work(r) <= Capacity[r]

ContributorsExact ==
    \A o \in Ops : \A r \in Nodes : r \in contributed[o] => r \in Contributors[o]

FusionSound ==
    \A o \in Ops : o \in fused => Contributors[o] \subseteq contributed[o]

CommitAfterFusion == committed \subseteq fused
ExecuteAfterCommit == executed \subseteq committed

DependencyClosure ==
    \A o \in committed : Deps[o] \subseteq committed

Liveness == \A o \in Ops : <> (o \in executed)

THEOREM Spec => []AssignedOnce
THEOREM Spec => []FusionSound
THEOREM Spec => []CommitAfterFusion
====
