---- MODULE PlaitUnifiedSmall ----
EXTENDS Naturals, FiniteSets
Nodes == {1, 2, 3}
Ops == {1, 2, 3, 4}
Contributors == [o \in Ops |-> IF o <= 2 THEN {1, 2} ELSE {1, 2, 3}]
Kind == [o \in Ops |-> IF o <= 2 THEN "Sum" ELSE "Max"]
Deps == [o \in Ops |-> IF o <= 2 THEN {} ELSE {1}]
Capacity == [r \in Nodes |-> IF r <= 2 THEN 2 ELSE 1]

VARIABLE assigned, contributed, fused, committed, executed

I == INSTANCE PlaitUnified

Spec == I!Spec
AssignedOnce == I!AssignedOnce
CapacityBound == I!CapacityBound
ContributorsExact == I!ContributorsExact
FusionSound == I!FusionSound
CommitAfterFusion == I!CommitAfterFusion
ExecuteAfterCommit == I!ExecuteAfterCommit
DependencyClosure == I!DependencyClosure
Liveness == I!Liveness
====
