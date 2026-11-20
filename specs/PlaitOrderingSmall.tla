---- MODULE PlaitOrderingSmall ----
EXTENDS Naturals, FiniteSets
Replicas == {1, 2, 3}
Instances == {1, 2}
Commands == {1, 2}
MaxBallot == 2
Deps == {<<1, 2>>, <<2, 1>>}

VARIABLE proposed, preaccepted, accepted, committed, committed_deps, ballot

I == INSTANCE PlaitOrdering

Spec == I!Spec
Agreement == I!Agreement
CommittedStable == I!CommittedStable
CommittedDepsStable == I!CommittedDepsStable
Live == I!Live
====
