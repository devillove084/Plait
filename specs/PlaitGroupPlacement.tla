---- MODULE PlaitGroupPlacement ----
EXTENDS Naturals

CONSTANTS CheckEpoch, CheckActive, CheckSpare
Planners == {"a", "b"}
NoView == [latest |-> 0, active |-> 0]
VARIABLES latest, active, pending, spare, views, staleEpoch, staleActive, doubleGrant
vars == <<latest, active, pending, spare, views, staleEpoch, staleActive, doubleGrant>>

Init ==
    /\ latest = 1 /\ active = 1
    /\ pending = FALSE /\ spare = "Free"
    /\ views = [p \in Planners |-> NoView]
    /\ staleEpoch = FALSE /\ staleActive = FALSE /\ doubleGrant = FALSE

Read(p) ==
    /\ ~pending
    /\ spare = "Free"
    /\ views' = [views EXCEPT ![p] = [latest |-> latest, active |-> active]]
    /\ UNCHANGED <<latest, active, pending, spare, staleEpoch, staleActive, doubleGrant>>

Revoke ==
    /\ active # 0
    /\ active' = 0
    /\ UNCHANGED <<latest, pending, spare, views, staleEpoch, staleActive, doubleGrant>>

HoldSpare ==
    /\ spare = "Free"
    /\ spare' = "OtherGroup"
    /\ UNCHANGED <<latest, active, pending, views, staleEpoch, staleActive, doubleGrant>>

ReleaseSpare ==
    /\ spare = "OtherGroup"
    /\ spare' = "Free"
    /\ UNCHANGED <<latest, active, pending, views, staleEpoch, staleActive, doubleGrant>>

Prepare(p) ==
    /\ ~pending /\ latest < 3
    /\ views[p] # NoView
    /\ CheckEpoch => views[p].latest = latest
    /\ CheckActive => views[p].active = active
    /\ CheckSpare => spare = "Free"
    /\ latest' = latest + 1 /\ pending' = TRUE /\ spare' = "Candidate"
    /\ staleEpoch' = (staleEpoch \/ views[p].latest # latest)
    /\ staleActive' = (staleActive \/ views[p].active # active)
    /\ doubleGrant' = (doubleGrant \/ spare # "Free")
    /\ UNCHANGED <<active, views>>

Cancel ==
    /\ pending
    /\ pending' = FALSE /\ spare' = "Free"
    /\ UNCHANGED <<latest, active, views, staleEpoch, staleActive, doubleGrant>>

Next ==
    \/ Revoke \/ HoldSpare \/ ReleaseSpare \/ Cancel
    \/ \E p \in Planners : Read(p) \/ Prepare(p)

Spec == Init /\ [][Next]_vars
CurrentEpoch == ~staleEpoch
CurrentIncumbent == ~staleActive
ExclusiveReservation == ~doubleGrant
====
