---- MODULE PlaitPeerAllReducePricing ----
\* The planner's estimate for fixed-width PeerGroup AllReduce. An exchange
\* phase runs peer directions concurrently. In each direction, host-pair
\* serialization times sharing a NIC add; the slowest direction across all
\* hosts and the largest directed RTT are charged once for the phase.
\* Rates are bytes per model tick. Calls contain whole, fixed-width elements.
\* This abstracts frame bytes (the peers still exchange headers), kernel/host
\* work, queueing, shared fabric paths, sampled rates and wider FP8 gather output.
EXTENDS Naturals, Sequences

Hosts == 1..3
HostPairs == {<<h, q>> : h \in Hosts, q \in Hosts}
Rates == << <<0, 1024, 2048>>, <<4096, 0, 2048>>, <<1024, 512, 0>> >>
Rtt == <<<<0, 2, 11>>, <<3, 0, 6>>, <<4, 5, 0>>>>
Call(count, bytes, elementBytes) == [count |-> count, bytes |-> bytes, elementBytes |-> elementBytes]
Phase(stage, call) == [stage |-> stage, bytes |-> call.bytes, elem |-> call.elementBytes]
Charge(stage, ticks) == [stage |-> stage, ticks |-> ticks]

TwoCalls == <<Call(2, 1048576, 4), Call(3, 1048580, 4)>>
ThreeCalls == <<Call(1, 32768, 4), Call(1, 32772, 4)>>
FourCalls == <<Call(1, 21844, 4), Call(2, 21848, 4)>>
Scenario(name, layout, hidden, calls) ==
    [name |-> name, layout |-> layout, hidden |-> hidden, calls |-> calls]
Two == Scenario("Two", <<1, 2>>, {}, TwoCalls)
Three == Scenario("Three", <<1, 2, 3>>, {}, ThreeCalls)
Four == Scenario("Four", <<1, 1, 2, 3>>, {}, FourCalls)
Local == Scenario("Local", <<1, 1, 1>>, HostPairs, ThreeCalls)
Solo == Scenario("Solo", <<1>>, HostPairs, <<Call(1, 1048580, 4)>>)
Missing == Scenario("Missing", <<1, 2, 3>>, {<<3, 2>>}, <<Call(1, 32772, 4)>>)
Scenarios == {Two, Three, Four, Local, Solo, Missing}

Sum(xs) ==
    LET prefix[n \in 0..Len(xs)] == IF n = 0 THEN 0 ELSE prefix[n - 1] + xs[n]
    IN prefix[Len(xs)]
Largest(values) == CHOOSE value \in values : \A v \in values : v <= value
CeilDiv(n, d) == (n + d - 1) \div d

IsReplicated(width, bytes) ==
    IF width = 2 THEN bytes <= 1024 * 1024 ELSE bytes * (width - 1) <= 64 * 1024

Phases(width, call) ==
    IF IsReplicated(width, call.bytes) THEN <<Phase("Replicate", call)>>
    ELSE <<Phase("Scatter", call), Phase("Gather", call)>>

RECURSIVE Repeat(_, _), OrderedCalls(_, _)
Repeat(phases, count) == IF count = 0 THEN <<>> ELSE phases \o Repeat(phases, count - 1)
OrderedCalls(width, calls) ==
    IF Len(calls) = 0 THEN <<>>
    ELSE Repeat(Phases(width, Head(calls)), Head(calls).count) \o OrderedCalls(width, Tail(calls))

\* The runtime gives one extra whole element to each low-numbered shard.
Shard(width, phase, slot) ==
    LET elements == phase.bytes \div phase.elem
    IN (elements \div width + (IF slot <= elements % width THEN 1 ELSE 0)) * phase.elem

PeerBytes(width, phase, from, to) ==
    CASE from = to -> 0
      [] phase.stage = "Replicate" -> phase.bytes
      [] phase.stage = "Scatter" -> Shard(width, phase, to)
      [] OTHER -> Shard(width, phase, from)

HostBytes(layout, phase, from, to) ==
    IF from = to THEN 0
    ELSE Sum([i \in DOMAIN layout |->
         IF layout[i] = from THEN Sum([j \in DOMAIN layout |->
              IF layout[j] = to THEN PeerBytes(Len(layout), phase, i, j) ELSE 0]) ELSE 0])

PhaseCost(c, phase) ==
    LET assigned == {c.layout[i] : i \in DOMAIN c.layout}
        needed == {pair \in HostPairs : pair[1] # pair[2] /\ pair[1] \in assigned /\ pair[2] \in assigned}
        traffic == [h, q \in Hosts |-> HostBytes(c.layout, phase, h, q)]
    IN IF needed \cap c.hidden # {} THEN [usable |-> FALSE, ticks |-> 0]
       ELSE LET serialize == [h, q \in Hosts |->
                    IF traffic[h, q] = 0 THEN 0 ELSE CeilDiv(traffic[h, q], Rates[h][q])]
                outgoing == {Sum([q \in Hosts |-> serialize[h, q]]) : h \in Hosts}
                incoming == {Sum([q \in Hosts |-> serialize[q, h]]) : h \in Hosts}
                latency == Largest({0} \cup {Rtt[p[1]][p[2]] : p \in needed})
            IN [usable |-> TRUE, ticks |-> Largest(outgoing \cup incoming) + latency]

VARIABLES scenario, charged, status
vars == <<scenario, charged, status>>

Init ==
    /\ scenario \in Scenarios
    /\ charged = <<>> /\ status = "Pricing"

Next ==
    /\ status = "Pricing"
    /\ LET work == OrderedCalls(Len(scenario.layout), scenario.calls)
           index == Len(charged) + 1
       IN IF index > Len(work) THEN /\ status' = "Done" /\ UNCHANGED charged
          ELSE LET phase == work[index]
                   cost == PhaseCost(scenario, phase)
               IN IF cost.usable THEN
                      /\ charged' = Append(charged, Charge(phase.stage, cost.ticks))
                      /\ UNCHANGED status
                  ELSE /\ status' = "Unusable" /\ UNCHANGED charged
    /\ UNCHANGED scenario

Spec == Init /\ [][Next]_vars

\* Independent fixture totals include asymmetric rates, shared NIC fanout,
\* odd shard sizes, cutoff boundaries and multiple calls. The RTT of the
\* slowest path need not belong to the most loaded direction.
ExpectedTwo == <<Charge("Replicate", 1027), Charge("Replicate", 1027),
    Charge("Scatter", 515), Charge("Gather", 516), Charge("Scatter", 515), Charge("Gather", 516),
    Charge("Scatter", 515), Charge("Gather", 516)>>
ExpectedThree == <<Charge("Replicate", 107), Charge("Scatter", 44), Charge("Gather", 44)>>
ExpectedFour == <<Charge("Replicate", 97), Charge("Scatter", 33), Charge("Gather", 33),
    Charge("Scatter", 33), Charge("Gather", 33)>>
ExpectedLocal == <<Charge("Replicate", 0), Charge("Scatter", 0), Charge("Gather", 0)>>
Expected(c) == CASE c = Two -> ExpectedTwo [] c = Three -> ExpectedThree [] c = Four -> ExpectedFour
                 [] c = Local -> ExpectedLocal [] c = Solo -> <<Charge("Replicate", 0)>> [] OTHER -> <<>>

PriceMatchesOrderedPhases == charged = SubSeq(Expected(scenario), 1, Len(charged))

PricingCanAdvance == status = "Pricing" => ENABLED Next

TotalsAndFeasibility ==
    /\ status = "Done" =>
        /\ scenario # Missing /\ charged = Expected(scenario)
        /\ Sum([i \in DOMAIN charged |-> charged[i].ticks]) =
            CASE scenario = Two -> 5147 [] scenario = Three -> 195 [] scenario = Four -> 229 [] OTHER -> 0
    /\ status = "Unusable" => scenario = Missing /\ charged = <<>>

UnevenShardsUseCorrectDirection ==
    LET call == Four.calls[2]
        scatter == Phase("Scatter", call)
        gather == Phase("Gather", call)
    IN /\ [slot \in 1..4 |-> Shard(4, scatter, slot)] = <<5464, 5464, 5460, 5460>>
       /\ HostBytes(Four.layout, scatter, 1, 2) = 10920
       /\ HostBytes(Four.layout, gather, 1, 2) = 10928
       /\ HostBytes(Four.layout, scatter, 2, 1) = 10928
       /\ HostBytes(Four.layout, gather, 2, 1) = 10920

EvidenceOrZeroCrossHostTraffic ==
    /\ ~PhaseCost(Missing, Phase("Scatter", Missing.calls[1])).usable
    /\ PhaseCost(Local, Phase("Scatter", Local.calls[2])) = [usable |-> TRUE, ticks |-> 0]
====
