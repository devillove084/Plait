---- MODULE PlaitHostedPipelinePlacement ----
EXTENDS Naturals, Sequences

\* A two-stage TP2 Prefill plan. Each trial layout is an ordered projection
\* of grants. Hosts are visited API-first, then in identity order, and
\* ordinals on a host always increase. Ordinals below are stage-independent.
Hosts == 1..2
Slots == 1..4
Devices == 1..5
Host(d) == IF d <= 2 THEN 1 ELSE 2
Hbm(d) == IF d \in {1, 3} THEN 8 ELSE 16
Stage(i) == IF i <= 2 THEN 1 ELSE 2
WorkerMemory(i) == IF Stage(i) = 1 THEN 2 ELSE 3
RankHbm(i) == IF Stage(i) = 1 THEN 8 ELSE 16
Layers(i) == IF Stage(i) = 1 THEN 1 ELSE 2
Layouts == {<<1, 2, 4, 5>>, <<1, 3, 4, 5>>, <<3, 4, 1, 2>>,
             <<3, 4, 2, 5>>, <<1, 2, 3, 4>>, <<3, 5, 2, 4>>}

VARIABLES api, memory, nic, fresh, selected, rate
vars == <<api, memory, nic, fresh, selected, rate>>

Ordered(layout) ==
    /\ Host(layout[1]) = api
    /\ \A i, j \in Slots :
        i < j => IF Host(layout[i]) = Host(layout[j])
                 THEN layout[i] < layout[j]
                 ELSE Host(layout[i]) = api
WithinHbm(layout) == \A i \in Slots : Hbm(layout[i]) >= RankHbm(i)
MemoryDemand(layout, host) ==
    IF \E i \in Slots : Host(layout[i]) = host
    THEN 1 + LET m(i) == IF Host(layout[i]) = host THEN WorkerMemory(i) ELSE 0
             IN m(1) + m(2) + m(3) + m(4)
    ELSE 0

\* The modeled outbound interface is shared by TP, forward activations,
\* last-stage feedback and layer-proportional writes to a remote KV store.
Traffic(layout, host) ==
    LET crosses(i,j) == Host(layout[i]) = host /\ Host(layout[j]) /= host
        edge(i,j,cost) == IF crosses(i,j) THEN cost ELSE 0
        kv(i) == IF Host(layout[i]) = host THEN Layers(i) ELSE 0
    IN [
      tp |-> edge(1,2,1) + edge(2,1,1) + edge(3,4,1) + edge(4,3,1),
      pp |-> edge(1,3,3) + edge(2,4,3) + edge(3,1,1) + edge(4,2,1),
      kv |-> kv(1) + kv(2) + kv(3) + kv(4)]
HasLink(layout, host) ==
    LET cost == Traffic(layout, host)
    IN cost.tp + cost.pp = 0 \/ host \in fresh

Init ==
    /\ api \in Hosts
    /\ memory \in [Hosts -> {5, 7, 9}]
    /\ nic \in [Hosts -> {4, 8, 12, 16, 24}]
    /\ fresh \in SUBSET Hosts
    /\ selected = <<>>
    /\ rate = 0

Admit(layout, r) ==
    /\ selected = <<>>
    /\ Ordered(layout)
    /\ WithinHbm(layout)
    /\ \A host \in Hosts :
        /\ MemoryDemand(layout, host) <= memory[host]
        /\ HasLink(layout, host)
        /\ LET cost == Traffic(layout, host)
           IN r * (cost.tp + cost.pp + cost.kv) <= nic[host]
    /\ selected' = layout
    /\ rate' = r
    /\ UNCHANGED <<api, memory, nic, fresh>>

Next == \E layout \in Layouts, r \in 1..3 : Admit(layout, r)
Spec == Init /\ [][Next]_vars

StageMatchesGrants ==
    selected = <<>> \/ (Ordered(selected) /\ WithinHbm(selected))
HostBudget ==
    selected = <<>> \/ \A host \in Hosts : MemoryDemand(selected, host) <= memory[host]
SharedLinkBudget ==
    selected = <<>> \/ \A host \in Hosts :
        /\ HasLink(selected, host)
        /\ LET cost == Traffic(selected, host)
           IN rate * (cost.tp + cost.pp + cost.kv) <= nic[host]
====
