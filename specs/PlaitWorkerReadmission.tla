---- MODULE PlaitWorkerReadmission ----
EXTENDS Naturals

Incarnations == 1..3
DrainIds == 1..2
Fences == 1..3
None == 0

CONSTANTS CheckDrain, CheckGeneration, CheckQuiescence, ResetRates, PreserveFence
VARIABLES incarnation, drain, lastDrain, owner, deferred, rate, fence, lastGrant, evidence
vars == <<incarnation, drain, lastDrain, owner, deferred, rate, fence, lastGrant, evidence>>

Init ==
    /\ incarnation = 1
    /\ drain = None
    /\ lastDrain = None
    /\ owner = None
    /\ deferred \in BOOLEAN
    /\ rate = None
    /\ fence = None
    /\ lastGrant = None
    /\ evidence = [exactDrain |-> TRUE, newer |-> TRUE, idle |-> TRUE]

Reserve ==
    /\ drain = None /\ owner = None /\ ~deferred
    /\ fence + 1 \in Fences
    /\ owner' = incarnation
    /\ fence' = fence + 1
    /\ lastGrant' = fence'
    /\ UNCHANGED <<incarnation, drain, lastDrain, deferred, rate, evidence>>

Withdraw ==
    /\ drain = None
    /\ lastDrain < 2
    /\ drain' = lastDrain + 1
    /\ lastDrain' = drain'
    /\ UNCHANGED <<incarnation, owner, deferred, rate, fence, lastGrant, evidence>>

Release ==
    /\ owner # None
    /\ owner' = None
    /\ UNCHANGED <<incarnation, drain, lastDrain, deferred, rate, fence, lastGrant, evidence>>

ResolveDeferred ==
    /\ deferred
    /\ deferred' = FALSE
    /\ UNCHANGED <<incarnation, drain, lastDrain, owner, rate, fence, lastGrant, evidence>>

Calibrate ==
    /\ rate = None
    /\ rate' = incarnation
    /\ UNCHANGED <<incarnation, drain, lastDrain, owner, deferred, fence, lastGrant, evidence>>

Readmit(observedDrain, observedIncarnation, replacement) ==
    /\ drain # None
    /\ IF CheckDrain THEN observedDrain = drain ELSE TRUE
    /\ IF CheckGeneration THEN
           /\ observedIncarnation = incarnation
           /\ replacement > observedIncarnation
       ELSE TRUE
    /\ IF CheckQuiescence THEN owner = None /\ ~deferred ELSE TRUE
    /\ evidence' = [exactDrain |-> observedDrain = drain,
                    newer |-> observedIncarnation = incarnation /\ replacement > incarnation,
                    idle |-> owner = None /\ ~deferred]
    /\ incarnation' = replacement
    /\ drain' = None
    /\ rate' = IF ResetRates THEN None ELSE rate
    /\ fence' = IF PreserveFence THEN fence ELSE None
    /\ UNCHANGED <<lastDrain, owner, deferred, lastGrant>>

Next ==
    \/ Reserve \/ Withdraw \/ Release \/ ResolveDeferred \/ Calibrate
    \/ \E d \in DrainIds, g \in Incarnations, new \in Incarnations : Readmit(d, g, new)

Spec == Init /\ [][Next]_vars
ExactDrain == evidence.exactDrain
NewIncarnation == evidence.newer
Quiescent == evidence.idle /\ (owner = None \/ owner = incarnation)
CurrentRates == rate = None \/ rate = incarnation
MonotonicFence == fence >= lastGrant
====
