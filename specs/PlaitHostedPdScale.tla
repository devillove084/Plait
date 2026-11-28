---- MODULE PlaitHostedPdScale ----
EXTENDS Naturals, Sequences

Plans == {"add", "remove"}
Empty == [revision |-> 0, session |-> 0, observations |-> <<>>]
VARIABLES revision, session, cooling, observations, submitted
vars == <<revision, session, cooling, observations, submitted>>

Init ==
    /\ revision = 0 /\ session = 0 /\ cooling = 0
    /\ observations = <<>> /\ submitted = Empty

Measure(plan) ==
    /\ cooling = 0
    /\ LET observation == [revision |-> revision, session |-> session, plan |-> plan]
       IN observations' =
           IF Len(observations) = 0 THEN <<observation>>
           ELSE IF observations[Len(observations)].plan = plan
                THEN <<observations[Len(observations)], observation>>
                ELSE <<observation>>
    /\ UNCHANGED <<revision, session, cooling, submitted>>

Miss ==
    /\ observations' = <<>>
    /\ UNCHANGED <<revision, session, cooling, submitted>>

RestartIngress ==
    /\ session = 0 /\ session' = 1
    /\ observations' = <<>> \* discard the old process's counters
    /\ UNCHANGED <<revision, cooling, submitted>>

Publish ==
    /\ revision = 0 /\ revision' = 1 /\ cooling' = 2
    /\ observations' = <<>>
    /\ UNCHANGED <<session, submitted>>

Tick ==
    /\ cooling > 0 /\ cooling' = cooling - 1
    /\ UNCHANGED <<revision, session, observations, submitted>>

Submit ==
    /\ submitted = Empty /\ cooling = 0
    /\ Len(observations) = 2
    /\ submitted' = [revision |-> revision, session |-> session, observations |-> observations]
    /\ UNCHANGED <<revision, session, cooling, observations>>

Next == Miss \/ RestartIngress \/ Publish \/ Tick \/ Submit \/ \E plan \in Plans : Measure(plan)
Spec == Init /\ [][Next]_vars

CurrentObservationWindows == submitted # Empty =>
    /\ Len(submitted.observations) = 2
    /\ submitted.observations[1].plan = submitted.observations[2].plan
    /\ \A obs \in 1..2 :
           /\ submitted.observations[obs].revision = submitted.revision
           /\ submitted.observations[obs].session = submitted.session
====
