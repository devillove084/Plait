---- MODULE PlaitUnifiedTiny ----
EXTENDS Naturals, FiniteSets
Nodes == {1, 2}
Ops == {1, 2}
Contributors == [o \in Ops |-> Nodes]
Kind == [o \in Ops |-> "Sum"]
Deps == [o \in Ops |-> IF o = 1 THEN {} ELSE {1}]
Capacity == [r \in Nodes |-> 2]

VARIABLE assigned, contributed, fused, committed, executed

I == INSTANCE PlaitUnified

Spec == I!Spec
AssignedOnce == I!AssignedOnce
CapacityBound == I!CapacityBound
FusionSound == I!FusionSound
CommitAfterFusion == I!CommitAfterFusion
ExecuteAfterCommit == I!ExecuteAfterCommit
DependencyClosure == I!DependencyClosure
Liveness == I!Liveness
====
