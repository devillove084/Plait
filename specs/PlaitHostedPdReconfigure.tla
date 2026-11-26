---- MODULE PlaitHostedPdReconfigure ----
EXTENDS Integers

Base == 4
Horizon == 10
Empty == [revision |-> 0, bridge |-> 0, target |-> 0, elapsed |-> 0]
VARIABLES revision, releaseKnown, forecast, applied
vars == <<revision, releaseKnown, forecast, applied>>

Init ==
    /\ revision = 0 /\ releaseKnown = TRUE
    /\ forecast = Empty /\ applied = Empty

Gain(bridge, target, elapsed) == (bridge - Base) * elapsed + (target - Base) * (Horizon - elapsed)

Forecast(bridge, target, elapsed) ==
    /\ forecast = Empty /\ applied = Empty /\ releaseKnown
    /\ bridge >= Base
    /\ target >= Base /\ elapsed < Horizon /\ Gain(bridge, target, elapsed) > 0
    /\ forecast' = [revision |-> revision, bridge |-> bridge, target |-> target, elapsed |-> elapsed]
    /\ UNCHANGED <<revision, releaseKnown, applied>>

Withdraw ==
    /\ releaseKnown /\ releaseKnown' = FALSE
    /\ UNCHANGED <<revision, forecast, applied>>

ReplaceDirectory ==
    /\ revision = 0 /\ revision' = 1
    /\ UNCHANGED <<releaseKnown, forecast, applied>>

Apply ==
    /\ applied = Empty /\ forecast # Empty
    /\ releaseKnown /\ forecast.revision = revision
    /\ applied' = [plan |-> forecast, revision |-> revision, releaseKnown |-> releaseKnown]
    /\ UNCHANGED <<revision, releaseKnown, forecast>>

Next == Withdraw \/ ReplaceDirectory \/ Apply \/
    \E bridge \in 3..6, target \in 3..8, elapsed \in {1, 5, 10} : Forecast(bridge, target, elapsed)

Spec == Init /\ [][Next]_vars

BridgeCapacity == applied # Empty => applied.plan.bridge >= Base
PaysBack == applied # Empty =>
    /\ applied.plan.target >= Base /\ applied.plan.elapsed < Horizon
    /\ Gain(applied.plan.bridge, applied.plan.target, applied.plan.elapsed) > 0
CurrentReleaseAtCommit == applied # Empty =>
    applied.plan.revision = applied.revision /\ applied.releaseKnown
====
