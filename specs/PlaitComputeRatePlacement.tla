---- MODULE PlaitComputeRatePlacement ----
EXTENDS Naturals

CONSTANTS CheckIncumbent, CheckCandidate
Incumbent == "incumbent"
Candidate == "candidate"
Devices == {Incumbent, Candidate}
Speeds == 0..3

VARIABLES rates, plan, reserved, stale
vars == <<rates, plan, reserved, stale>>

Rate(incarnation, revision, speed) ==
    [incarnation |-> incarnation, revision |-> revision, speed |-> speed]

NoPlan == [d \in Devices |-> Rate(0, 0, 0)]

Init ==
    /\ rates = [d \in Devices |-> Rate(1, 1, IF d = Incumbent THEN 1 ELSE 2)]
    /\ plan = NoPlan
    /\ reserved = FALSE
    /\ stale = FALSE

Publish(device, incarnation, revision, speed) ==
    /\ rates[device].incarnation = incarnation
    /\ rates[device].revision = revision
    /\ revision < 3
    /\ rates[device].speed # speed
    /\ rates' = [rates EXCEPT ![device] = Rate(incarnation, revision + 1, speed)]
    /\ UNCHANGED <<plan, reserved, stale>>

ReplaceDevice ==
    /\ ~reserved
    /\ rates[Candidate].incarnation = 1
    /\ rates' = [rates EXCEPT ![Candidate] = Rate(2, 0, 0)]
    /\ UNCHANGED <<plan, reserved, stale>>

Score ==
    /\ ~reserved
    /\ rates[Incumbent].speed > 0
    /\ rates[Candidate].speed > rates[Incumbent].speed
    /\ plan' = rates
    /\ UNCHANGED <<rates, reserved, stale>>

Reserve ==
    /\ ~reserved
    /\ plan # NoPlan
    /\ CheckIncumbent => plan[Incumbent] = rates[Incumbent]
    /\ CheckCandidate => plan[Candidate] = rates[Candidate]
    /\ reserved' = TRUE
    /\ stale' = (plan # rates)
    /\ UNCHANGED <<rates, plan>>

Next ==
    \/ Score \/ Reserve \/ ReplaceDevice
    \/ \E d \in Devices, i \in 1..2, r \in 0..2, s \in Speeds : Publish(d, i, r, s)

Spec == Init /\ [][Next]_vars
CurrentAtReservation == ~stale
====
