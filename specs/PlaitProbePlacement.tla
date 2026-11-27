---- MODULE PlaitProbePlacement ----
EXTENDS Naturals

CONSTANTS CheckSource, CheckTarget
Source == "source"
Target == "target"
Other == "other"
Hosts == {Source, Target, Other}
NoEvidence == [source |-> 0, target |-> 0]
VARIABLES revision, samples, plan, reserved, stale
vars == <<revision, samples, plan, reserved, stale>>

Evidence == [source |-> revision[Source], target |-> revision[Target]]
Current == plan = Evidence

Init ==
    /\ revision = [h \in Hosts |-> 1]
    /\ samples = NoEvidence /\ plan = NoEvidence
    /\ reserved = FALSE /\ stale = FALSE

Restart(h) ==
    /\ revision[h] < 3
    /\ revision' = [revision EXCEPT ![h] = @ + 1]
    /\ UNCHANGED <<samples, plan, reserved, stale>>

Measure ==
    /\ samples' = Evidence
    /\ UNCHANGED <<revision, plan, reserved, stale>>

Read ==
    /\ ~reserved /\ samples = Evidence
    /\ plan' = samples
    /\ UNCHANGED <<revision, samples, reserved, stale>>

Reserve ==
    /\ ~reserved /\ plan # NoEvidence
    /\ CheckSource => plan.source = revision[Source]
    /\ CheckTarget => plan.target = revision[Target]
    /\ reserved' = TRUE /\ stale' = ~Current
    /\ UNCHANGED <<revision, samples, plan>>

Next == Measure \/ Read \/ Reserve \/ \E h \in Hosts : Restart(h)
Spec == Init /\ [][Next]_vars
CurrentAtReservation == ~stale
====
