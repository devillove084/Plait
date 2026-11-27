---- MODULE PlaitNativeServingRetire ----
EXTENDS Naturals

CONSTANTS CheckAdmission, CheckDrain, CheckExit
Clients == {"west", "east"}
Hosts == {"a", "b"}

VARIABLES phase, gate, requests, paused, exited, reaped, released, reused
vars == <<phase, gate, requests, paused, exited, reaped, released, reused>>

Init ==
    /\ phase = "active" /\ gate = TRUE /\ requests = {}
    /\ paused = FALSE /\ exited = {} /\ reaped = {} /\ released = {}
    /\ reused = {}

(* A stale ingress can keep trying the old version after activation. *)
Admit(c) ==
    /\ phase \in {"active", "draining"} /\ c \notin requests
    /\ CheckAdmission => gate
    /\ requests' = requests \cup {c}
    /\ UNCHANGED <<phase, gate, paused, exited, reaped, released, reused>>

Finish(c) ==
    /\ c \in requests /\ requests' = requests \ {c}
    /\ UNCHANGED <<phase, gate, paused, exited, reaped, released, reused>>

ActivateSuccessor ==
    /\ phase = "active" /\ phase' = "draining"
    /\ UNCHANGED <<gate, requests, paused, exited, reaped, released, reused>>

CloseAdmission ==
    /\ phase = "draining" /\ gate /\ gate' = FALSE
    /\ UNCHANGED <<phase, requests, paused, exited, reaped, released, reused>>

PauseEngine ==
    /\ ~gate /\ requests = {} /\ ~paused
    /\ paused' = TRUE
    /\ UNCHANGED <<phase, gate, requests, exited, reaped, released, reused>>

Revoke ==
    /\ phase = "draining" /\ (CheckDrain => paused)
    /\ phase' = "revoked"
    /\ UNCHANGED <<gate, requests, paused, exited, reaped, released, reused>>

LeaderExit(h) ==
    /\ phase = "revoked" /\ h \notin exited
    /\ exited' = exited \cup {h}
    /\ UNCHANGED <<phase, gate, requests, paused, reaped, released, reused>>

ReapGroup(h) ==
    /\ h \in exited \ reaped /\ reaped' = reaped \cup {h}
    /\ UNCHANGED <<phase, gate, requests, paused, exited, released, reused>>

Release(h) ==
    /\ phase = "revoked" /\ h \notin released
    /\ CheckExit => h \in reaped
    /\ released' = released \cup {h}
    /\ phase' = IF released' = Hosts THEN "retired" ELSE phase
    /\ UNCHANGED <<gate, requests, paused, exited, reaped, reused>>

(* Retries for released grants leave state unchanged, even after reuse. *)
Reuse(h) ==
    /\ h \in released \ reused /\ reused' = reused \cup {h}
    /\ UNCHANGED <<phase, gate, requests, paused, exited, reaped, released>>

Next ==
    \/ ActivateSuccessor \/ CloseAdmission \/ PauseEngine \/ Revoke
    \/ \E c \in Clients : Admit(c) \/ Finish(c)
    \/ \E h \in Hosts : LeaderExit(h) \/ ReapGroup(h) \/ Release(h) \/ Reuse(h)

Spec == Init /\ [][Next]_vars
PausedHasNoRequests == paused => ~gate /\ requests = {}
RevokedAfterDrain == phase \in {"revoked", "retired"} => paused
ReleaseAfterGroupExit == released \subseteq reaped
RetiredAfterAllReleases == phase = "retired" => released = Hosts
====
