---- MODULE PlaitHostedModeSelection ----
EXTENDS Integers

\* One aggregate route, one split route, or both under equal path shares.
\* Bandwidth costs are bytes per unit of load. The incumbent continues
\* serving during startup and removed workers still occupy slots while draining.
Modes == {"aggregate", "split", "mixed"}
Aggregate(mode) == IF mode = "split" THEN 0 ELSE 1
Split(mode) == IF mode = "aggregate" THEN 0 ELSE 1
Routes(mode) == Aggregate(mode) + Split(mode)
Devices(mode) == Aggregate(mode) + 2 * Split(mode)
Horizon == 4
Start == 1
Cost == 1
ChangeLoss == 1

VARIABLES incumbent, demand, wire, store, drain, picked, rate
vars == <<incumbent, demand, wire, store, drain, picked, rate>>

Init ==
    /\ incumbent \in Modes
    /\ demand \in 1..6
    /\ wire \in 3..7
    /\ store \in 3..7
    /\ drain \in 0..2
    /\ picked = "none"
    /\ rate = 0

AggregateBytes(mode, load) == 2 * Aggregate(mode) * load
SplitBytes(mode, load) == Split(mode) * load
StoreBytes(mode, load) == 2 * Split(mode) * load
Fits(mode, load) ==
    /\ Aggregate(mode) * load <= 3 * Routes(mode)
    /\ Split(mode) * load <= 6 * Routes(mode)
    /\ AggregateBytes(mode, load) + SplitBytes(mode, load) <= wire * Routes(mode)
    /\ StoreBytes(mode, load) <= store * Routes(mode)

IncumbentRate == CHOOSE b \in 0..demand :
    Fits(incumbent, b) /\ \A other \in (b+1)..demand : ~Fits(incumbent, other)

Added(mode) ==
    2 * (IF Split(mode) > Split(incumbent) THEN 1 ELSE 0)
      + (IF Aggregate(mode) > Aggregate(incumbent) THEN 1 ELSE 0)
Removed(mode) ==
    2 * (IF Split(mode) < Split(incumbent) THEN 1 ELSE 0)
      + (IF Aggregate(mode) < Aggregate(incumbent) THEN 1 ELSE 0)
Gain(mode, load, retirementDelay) ==
    LET ready == IF Added(mode) > 0 THEN Start ELSE 0
        reclaim == Removed(mode) * (Horizon - ready - retirementDelay)
        reserve == Added(mode) * Horizon
    IN (load - IncumbentRate) * (Horizon - ready)
       + Cost * (reclaim - reserve) - ChangeLoss

Select(mode, load) ==
    /\ picked = "none"
    /\ mode # incumbent
    /\ load >= IncumbentRate
    /\ Fits(mode, load)
    /\ Gain(mode, load, drain) > 0
    /\ picked' = mode
    /\ rate' = load
    /\ UNCHANGED <<incumbent, demand, wire, store, drain>>

Next == \E mode \in Modes, load \in 0..demand : Select(mode, load)
Spec == Init /\ [][Next]_vars

SharedBudgets == picked = "none" \/ (
    /\ AggregateBytes(picked, rate) + SplitBytes(picked, rate) <= wire * Routes(picked)
    /\ StoreBytes(picked, rate) <= store * Routes(picked))
NoCapacityRegression == picked = "none" \/ rate >= IncumbentRate
ProfitableAfterDrain == picked = "none" \/ Gain(picked, rate, drain) > 0
====
