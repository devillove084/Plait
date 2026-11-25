---- MODULE PlaitEpochAdmission ----
\* The old epoch is saturated and none of its requests can finish.
\* The submitter holds both its host and its admitted epoch slot.
EXTENDS Naturals

CONSTANT HostBudget, EpochBudget
VARIABLES active, oldSlots, newSlot, delivered

Old == 1
New == 2
vars == <<active, oldSlots, newSlot, delivered>>
HostSlots == oldSlots + (IF newSlot THEN 1 ELSE 0)

Init ==
    /\ active = Old
    /\ oldSlots = 0
    /\ newSlot = FALSE
    /\ delivered = FALSE

AdmitOld ==
    /\ active = Old
    /\ HostSlots < HostBudget
    /\ oldSlots < EpochBudget
    /\ oldSlots' = oldSlots + 1
    /\ UNCHANGED <<active, newSlot, delivered>>

ActivateNew ==
    /\ active = Old
    /\ oldSlots = EpochBudget
    /\ active' = New
    /\ UNCHANGED <<oldSlots, newSlot, delivered>>

AdmitNew ==
    /\ active = New
    /\ HostSlots < HostBudget
    /\ ~newSlot /\ ~delivered
    /\ newSlot' = TRUE
    /\ UNCHANGED <<active, oldSlots, delivered>>

DeliverNew ==
    /\ newSlot
    /\ newSlot' = FALSE
    /\ delivered' = TRUE
    /\ UNCHANGED <<active, oldSlots>>

Next == AdmitOld \/ ActivateNew \/ AdmitNew \/ DeliverNew
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

Bounded == HostSlots <= HostBudget /\ oldSlots <= EpochBudget
HeadroomForNew == active = New /\ ~newSlot /\ ~delivered => ENABLED AdmitNew
NewCompletes == <> delivered
====
