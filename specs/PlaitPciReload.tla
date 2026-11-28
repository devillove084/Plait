---- MODULE PlaitPciReload ----
EXTENDS Naturals

Incarnations == 1..2
Addresses == 1..2
NoReport == [inc |-> 0, fresh |-> TRUE]

CONSTANTS ResetEvidence, CheckBinding, CheckReportIncarnation
VARIABLES binding, enabled, missing, freshReads, validBinding, registered, queued, withdrawn
vars == <<binding, enabled, missing, freshReads, validBinding, registered, queued, withdrawn>>

Init ==
    /\ binding = [inc |-> 1, bdf |-> 1]
    /\ enabled = TRUE
    /\ missing = 0 /\ freshReads = 0
    /\ validBinding = TRUE
    /\ registered = 1
    /\ queued = NoReport
    /\ withdrawn = 0

Load(inc, bdf) ==
    LET permitted == inc > binding.inc \/ (inc = binding.inc /\ bdf = binding.bdf)
        changed == ~enabled \/ inc # binding.inc \/ bdf # binding.bdf
    IN /\ changed
       /\ (~CheckBinding \/ permitted)
       /\ validBinding' = permitted
       /\ binding' = [inc |-> inc, bdf |-> bdf]
       /\ enabled' = TRUE
       /\ freshReads' = 0
       /\ missing' = IF ResetEvidence THEN 0 ELSE missing
       /\ UNCHANGED <<registered, queued, withdrawn>>

Disable ==
    /\ enabled
    /\ enabled' = FALSE
    /\ missing' = 0 /\ freshReads' = 0
    /\ UNCHANGED <<binding, validBinding, registered, queued, withdrawn>>

Absence ==
    /\ enabled
    /\ missing' = IF missing < 2 THEN missing + 1 ELSE missing
    /\ freshReads' = IF freshReads < 2 THEN freshReads + 1 ELSE freshReads
    /\ UNCHANGED <<binding, enabled, validBinding, registered, queued, withdrawn>>

PresenceOrReadError ==
    /\ (missing > 0 \/ freshReads > 0)
    /\ missing' = 0 /\ freshReads' = 0
    /\ UNCHANGED <<binding, enabled, validBinding, registered, queued, withdrawn>>

Submit ==
    /\ enabled /\ missing = 2
    /\ binding.inc = registered
    /\ withdrawn = 0 /\ queued = NoReport
    /\ queued' = [inc |-> binding.inc, fresh |-> freshReads = 2]
    /\ UNCHANGED <<binding, enabled, missing, freshReads, validBinding, registered, withdrawn>>

Apply ==
    /\ queued # NoReport
    /\ withdrawn' = IF ~CheckReportIncarnation \/ queued.inc = registered
                    THEN queued.inc ELSE withdrawn
    /\ queued' = NoReport
    /\ UNCHANGED <<binding, enabled, missing, freshReads, validBinding, registered>>

PublishSuccessor ==
    /\ registered = 1
    /\ registered' = 2
    /\ withdrawn' = 0
    /\ UNCHANGED <<binding, enabled, missing, freshReads, validBinding, queued>>

Next ==
    \/ \E inc \in Incarnations, bdf \in Addresses : Load(inc, bdf)
    \/ Disable \/ Absence \/ PresenceOrReadError \/ Submit \/ Apply \/ PublishSuccessor

Spec == Init /\ [][Next]_vars
BindingAdvances == validBinding
FreshEvidence == queued.fresh
OnlyCurrentIncarnationWithdrawn == withdrawn = 0 \/ withdrawn = registered
====
