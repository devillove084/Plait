---- MODULE PlaitHostedCheckpointBudget ----
EXTENDS Integers

\* The target may receive all resumed streams on a single route. Steady
\* arrivals and old checkpoints share the same store and directional ports.
Targets == {"aggregate", "split"}
NoTarget == "none"
VARIABLES port, store, live, checkpointShare, sourceWrites, demand, baseline,
          target, chosenRate
vars == <<port, store, live, checkpointShare, sourceWrites, demand, baseline,
          target, chosenRate>>

Init ==
    /\ port \in 4..10
    /\ store \in 4..10
    /\ live \in 0..3
    /\ checkpointShare \in 0..1
    /\ sourceWrites \in BOOLEAN
    /\ demand \in 1..4
    /\ baseline \in 0..demand
    /\ target = NoTarget
    /\ chosenRate = 0

SplitCost(shape) == IF shape = "split" THEN 2 ELSE 0
PortSteady(shape, rate) == (1 + checkpointShare + SplitCost(shape)) * rate
StoreSteady(shape, rate) == (2 * checkpointShare + SplitCost(shape)) * rate
PortReserved(shape) == (2 + SplitCost(shape)) * live
StoreReserved(shape) == (2 + SplitCost(shape) + IF sourceWrites THEN 1 ELSE 0) * live

FitsPort(shape, rate) == PortSteady(shape, rate) + PortReserved(shape) <= port
FitsStore(shape, rate) == StoreSteady(shape, rate) + StoreReserved(shape) <= store

Choose(shape, rate) ==
    /\ target = NoTarget
    /\ rate >= baseline
    /\ FitsPort(shape, rate)
    /\ FitsStore(shape, rate)
    /\ target' = shape
    /\ chosenRate' = rate
    /\ UNCHANGED <<port, store, live, checkpointShare, sourceWrites, demand, baseline>>

Next == \E shape \in Targets, rate \in 0..demand : Choose(shape, rate)
Spec == Init /\ [][Next]_vars

SharedPort == target # NoTarget => FitsPort(target, chosenRate)
SharedStore == target # NoTarget => FitsStore(target, chosenRate)
MaintainsRate == target # NoTarget => chosenRate >= baseline
====
