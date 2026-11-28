---- MODULE PlaitPciWatch ----
EXTENDS Naturals

CONSTANTS ConfirmAbsence, GuardIdle, GuardIncarnation
VARIABLES missing, incarnation, held, drained, queued, confirmed
vars == <<missing, incarnation, held, drained, queued, confirmed>>

NoReport == [inc |-> 0, stable |-> FALSE]
Init ==
    /\ missing = 0 /\ incarnation = 1 /\ held = FALSE /\ drained = FALSE
    /\ queued = NoReport /\ confirmed = FALSE

ReadAbsent ==
    /\ missing < 2
    /\ missing' = missing + 1
    /\ UNCHANGED <<incarnation, held, drained, queued, confirmed>>

ReadPresentOrUnknown ==
    /\ missing > 0
    /\ missing' = 0
    /\ UNCHANGED <<incarnation, held, drained, queued, confirmed>>

QueueIdle ==
    /\ queued = NoReport
    /\ ~held /\ ~drained /\ incarnation = 1
    /\ ConfirmAbsence => missing = 2
    /\ queued' = [inc |-> incarnation, stable |-> missing = 2]
    /\ UNCHANGED <<missing, incarnation, held, drained, confirmed>>

Acquire ==
    /\ ~held /\ ~drained
    /\ held' = TRUE
    /\ UNCHANGED <<missing, incarnation, drained, queued, confirmed>>

Release ==
    /\ held
    /\ held' = FALSE
    /\ UNCHANGED <<missing, incarnation, drained, queued, confirmed>>

RegisterReplacement ==
    /\ incarnation = 1 /\ ~held /\ ~drained
    /\ incarnation' = 2
    /\ UNCHANGED <<missing, held, drained, queued, confirmed>>

ApplyIdle ==
    /\ queued # NoReport
    /\ LET accept == (~GuardIdle \/ ~held)
                      /\ (~GuardIncarnation \/ queued.inc = incarnation)
       IN /\ drained' = (drained \/ accept)
          /\ confirmed' = IF accept THEN queued.stable ELSE confirmed
    /\ queued' = NoReport
    /\ UNCHANGED <<missing, incarnation, held>>

Next ==
    ReadAbsent \/ ReadPresentOrUnknown \/ QueueIdle \/ Acquire \/ Release
    \/ RegisterReplacement \/ ApplyIdle

Spec == Init /\ [][Next]_vars

ConfirmedReads == drained => confirmed
IdleIsolation == drained => ~held
IncarnationIsolation == drained => incarnation = 1

====
