---- MODULE PlaitComputeRatePlacement ----
EXTENDS Naturals

CONSTANTS CheckIncumbent, CheckCandidate, CheckGrant
Incumbent == "incumbent"
Candidate == "candidate"
Devices == {Incumbent, Candidate}
Speeds == 0..3

VARIABLES rates, plan, reserved, stale
VARIABLES fence, open, stalePublisher
vars == <<rates, plan, reserved, stale, fence, open, stalePublisher>>

Rate(incarnation, revision, speed) ==
    [incarnation |-> incarnation, revision |-> revision, speed |-> speed]

NoPlan == [d \in Devices |-> Rate(0, 0, 0)]

Init ==
    /\ rates = [d \in Devices |-> Rate(1, 1, IF d = Incumbent THEN 1 ELSE 2)]
    /\ plan = NoPlan
    /\ reserved = FALSE
    /\ stale = FALSE
    /\ fence = 1
    /\ open = TRUE
    /\ stalePublisher = FALSE

UpdateRate(device, incarnation, revision, speed) ==
    /\ rates[device].incarnation = incarnation
    /\ rates[device].revision = revision
    /\ revision < 3
    /\ rates[device].speed # speed
    /\ rates' = [rates EXCEPT ![device] = Rate(incarnation, revision + 1, speed)]
    /\ UNCHANGED <<plan, reserved, stale>>

Publish(device, incarnation, revision, speed) ==
    /\ UpdateRate(device, incarnation, revision, speed)
    /\ UNCHANGED <<fence, open, stalePublisher>>

PublishFromWorker(claim, revision, speed) ==
    /\ CheckGrant => (open /\ claim = fence)
    /\ UpdateRate(Incumbent, 1, revision, speed)
    /\ stalePublisher' = (stalePublisher \/ ~open \/ claim # fence)
    /\ UNCHANGED <<fence, open>>

Quiesce ==
    /\ open
    /\ open' = FALSE
    /\ UNCHANGED <<rates, plan, reserved, stale, fence, stalePublisher>>

Reassign ==
    /\ ~open /\ fence = 1
    /\ open' = TRUE /\ fence' = 2
    /\ UNCHANGED <<rates, plan, reserved, stale, stalePublisher>>

ReplaceDevice ==
    /\ ~reserved
    /\ rates[Candidate].incarnation = 1
    /\ rates' = [rates EXCEPT ![Candidate] = Rate(2, 0, 0)]
    /\ UNCHANGED <<plan, reserved, stale, fence, open, stalePublisher>>

Score ==
    /\ ~reserved
    /\ rates[Incumbent].speed > 0
    /\ rates[Candidate].speed > rates[Incumbent].speed
    /\ plan' = rates
    /\ UNCHANGED <<rates, reserved, stale, fence, open, stalePublisher>>

Reserve ==
    /\ ~reserved
    /\ plan # NoPlan
    /\ CheckIncumbent => plan[Incumbent] = rates[Incumbent]
    /\ CheckCandidate => plan[Candidate] = rates[Candidate]
    /\ reserved' = TRUE
    /\ stale' = (plan # rates)
    /\ UNCHANGED <<rates, plan, fence, open, stalePublisher>>

Next ==
    \/ Score \/ Reserve \/ ReplaceDevice \/ Quiesce \/ Reassign
    \/ \E d \in Devices, i \in 1..2, r \in 0..2, s \in Speeds : Publish(d, i, r, s)
    \/ \E f \in 1..fence, r \in 0..2, s \in 1..3 : PublishFromWorker(f, r, s)

Spec == Init /\ [][Next]_vars
CurrentAtReservation == ~stale
CurrentGrantAtPublication == ~stalePublisher
====
