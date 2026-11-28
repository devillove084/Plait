---- MODULE PlaitAsymmetricDeviceChoice ----
\* One host contributes a contiguous, device-ID ordered slice of a recipe.
\* For a fixed host slice, the collective charge depends on the host and
\* slot range. Check minimum compute cost among device assignments that
\* fit each slot's reservation in effective device memory.
EXTENDS Naturals, Sequences

CONSTANTS UseOrderedSearch, CheckSlotMemory

Slots == 0..3
DeviceCount == 5
Infeasible == 10000
RateFixtures ==
    {<<1, 100, 90, 40, 30>>, <<0, 5, 1, 100, 90>>, <<20, 20, 15, 30, 30>>}
WorkFixtures ==
    {<<1, 100, 200>>, <<0, 60, 10>>, <<90, 15, 0>>, <<12, 12, 12>>}
MemoryFixtures ==
    {<<24, 24, 24, 24, 24>>, <<8, 24, 16, 24, 8>>, <<24, 8, 8, 8, 8>>}
DemandFixtures == {<<8, 24, 16>>, <<24, 8, 24>>}

VARIABLES rates, work, memory, demand, mode, scanned, best
vars == <<rates, work, memory, demand, mode, scanned, best>>

Min(left, right) == IF left <= right THEN left ELSE right
DeviceCost(slot, device) ==
    IF work[slot] = 0 THEN 0
    ELSE IF rates[device] = 0 THEN Infeasible
    ELSE (work[slot] + rates[device] - 1) \div rates[device]
FittedCost(slot, device) ==
    IF memory[device] >= demand[slot] THEN DeviceCost(slot, device) ELSE Infeasible
Combine(left, right) ==
    IF mode = "parallel" THEN IF left >= right THEN left ELSE right
    ELSE Min(Infeasible, left + right)

Ordered(count, prefix) ==
    {placement \in [1..count -> 1..prefix] :
        \A slot \in 1..(count - 1) : placement[slot] < placement[slot + 1]}
Members(placement) == {placement[slot] : slot \in DOMAIN placement}

RECURSIVE Cost(_, _)
Cost(placement, count) ==
    IF count = 0 THEN 0
    ELSE Combine(Cost(placement, count - 1), FittedCost(count, placement[count]))
Minimum(values) == CHOOSE x \in values : \A y \in values : x <= y

\* Reference enumerates every valid subset in canonical ID order.
Exhaustive(count, prefix) ==
    Minimum({Cost(placement, count) : placement \in Ordered(count, prefix)}
             \cup {Infeasible})

\* Counterexample variant commits to the fastest devices before assigning slots.
Ahead(left, right) ==
    \/ rates[left] > rates[right]
    \/ rates[left] = rates[right] /\ left < right
FastestOnly(count, prefix) ==
    IF prefix < count THEN Infeasible
    ELSE LET placement == CHOOSE placement \in Ordered(count, prefix) :
                 \A left \in Members(placement), right \in (1..prefix) \ Members(placement) :
                     Ahead(left, right)
         IN Cost(placement, count)

Init ==
    /\ rates \in RateFixtures /\ work \in WorkFixtures
    /\ memory \in MemoryFixtures /\ demand \in DemandFixtures
    /\ mode \in {"parallel", "sequential"}
    /\ scanned = 0
    /\ best = [count \in Slots |-> IF count = 0 THEN 0 ELSE Infeasible]

ConsiderDevice ==
    /\ scanned < DeviceCount
    /\ scanned' = scanned + 1
    /\ best' = [count \in Slots |->
           IF count = 0 THEN 0
           ELSE IF UseOrderedSearch
                THEN LET nextCost == IF CheckSlotMemory THEN FittedCost(count, scanned + 1)
                                     ELSE DeviceCost(count, scanned + 1)
                     IN Min(best[count], Combine(best[count - 1], nextCost))
                ELSE FastestOnly(count, scanned + 1)]
    /\ UNCHANGED <<rates, work, memory, demand, mode>>

Spec == Init /\ [][ConsiderDevice]_vars
MinimumForEachWidth == \A count \in Slots : best[count] = Exhaustive(count, scanned)
====
