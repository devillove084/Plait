---- MODULE PlaitHostedPdBudget ----
EXTENDS Naturals

\* The router sends the same fraction of requests to each path. Costs stand
\* for the summed TP and KV bytes on a shared directional port.
Paths == 1..2

VARIABLES port, prefill, decode, costs, perPath
vars == <<port, prefill, decode, costs, perPath>>

Init ==
    /\ port \in 2..8
    /\ prefill \in 2..8
    /\ decode \in [Paths -> 1..4]
    /\ costs \in [Paths -> 1..3]
    /\ perPath = 0

Select(rate) ==
    /\ perPath = 0
    /\ 2 * rate <= prefill
    /\ \A path \in Paths : rate <= decode[path]
    /\ rate * (costs[1] + costs[2]) <= port
    /\ perPath' = rate
    /\ UNCHANGED <<port, prefill, decode, costs>>

Next == \E rate \in 1..4 : Select(rate)
Spec == Init /\ [][Next]_vars

CombinedPortBound == perPath * (costs[1] + costs[2]) <= port
SharedPrefillBound == 2 * perPath <= prefill
DecodeBound == \A path \in Paths : perPath <= decode[path]
====
