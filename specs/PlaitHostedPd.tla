---- MODULE PlaitHostedPd ----
EXTENDS Naturals

\* Two target roles, with two independently ready hosts each.
Roles == {"prefill", "decode"}
Participants == {"p0", "p1", "d0", "d1"}

VARIABLES revision, staged, ready, failed, published, baseRevision, oldActive, newActive
vars == <<revision, staged, ready, failed, published, baseRevision, oldActive, newActive>>

Init ==
    /\ revision = 1 /\ staged = 0
    /\ ready = {} /\ failed = FALSE
    /\ published = FALSE /\ baseRevision = 0
    /\ oldActive = Roles /\ newActive = {}

Prepare ==
    /\ staged = 0 /\ ~published
    /\ staged' = revision /\ ready' = {} /\ failed' = FALSE
    /\ UNCHANGED <<revision, published, baseRevision, oldActive, newActive>>

Ready(participant) ==
    /\ staged # 0 /\ ~failed
    /\ ready' = ready \cup {participant}
    /\ UNCHANGED <<revision, staged, failed, published, baseRevision, oldActive, newActive>>

Fail ==
    /\ staged # 0 /\ ~failed
    /\ failed' = TRUE
    /\ UNCHANGED <<revision, staged, ready, published, baseRevision, oldActive, newActive>>

ChangeDirectory ==
    /\ ~published /\ revision = 1
    /\ revision' = 2
    /\ UNCHANGED <<staged, ready, failed, published, baseRevision, oldActive, newActive>>

Cancel ==
    /\ staged # 0 /\ (failed \/ staged # revision)
    /\ staged' = 0
    /\ UNCHANGED <<revision, ready, failed, published, baseRevision, oldActive, newActive>>

Publish ==
    /\ staged # 0 /\ ~failed
    /\ staged = revision
    /\ ready = Participants
    /\ published' = TRUE /\ baseRevision' = staged
    /\ revision' = revision + 1 /\ staged' = 0
    /\ oldActive' = {} /\ newActive' = Roles
    /\ UNCHANGED <<ready, failed>>

Next == Prepare \/ Fail \/ ChangeDirectory \/ Cancel \/ Publish
        \/ \E participant \in Participants : Ready(participant)

Spec == Init /\ [][Next]_vars

AtomicCutover ==
    IF published
    THEN oldActive = {} /\ newActive = Roles
    ELSE oldActive = Roles /\ newActive = {}

AllParticipantsAtCutover == published => ready = Participants /\ ~failed

CurrentRevisionAtCutover == published => revision = baseRevision + 1
====
