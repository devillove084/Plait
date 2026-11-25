---- MODULE PlaitResidentReclaim ----
\* Two workers hold an old native group. The unhealthy worker loses its
\* response pipe while the healthy worker can shut down normally.
EXTENDS Naturals

CONSTANT CanVerifyFault
VARIABLES phase, timedOut, exited, verified, stopped, leases, rebound, hostReserved

Healthy == 1
Fault == 2
Devices == {Healthy, Fault}
OldGrant == [group |-> "old", fence |-> 1]
Free == [group |-> "none", fence |-> 1]
NewGrant == [group |-> "new", fence |-> 2]
vars == <<phase, timedOut, exited, verified, stopped, leases, rebound, hostReserved>>

Init ==
    /\ phase = "active"
    /\ timedOut = FALSE
    /\ exited = {}
    /\ verified = {}
    /\ stopped = {}
    /\ leases = [d \in Devices |-> OldGrant]
    /\ rebound = FALSE
    /\ hostReserved = TRUE

Timeout ==
    /\ ~timedOut
    /\ timedOut' = TRUE
    /\ UNCHANGED <<phase, exited, verified, stopped, leases, rebound, hostReserved>>

Revoke ==
    /\ timedOut /\ phase = "active"
    /\ phase' = "revoked"
    /\ UNCHANGED <<timedOut, exited, verified, stopped, leases, rebound, hostReserved>>

Exit(d) ==
    /\ phase = "revoked" /\ d \notin exited
    /\ exited' = exited \cup {d}
    /\ UNCHANGED <<phase, timedOut, verified, stopped, leases, rebound, hostReserved>>

Verify(d) ==
    /\ d \in exited \ verified
    /\ d = Healthy \/ CanVerifyFault
    /\ verified' = verified \cup {d}
    /\ UNCHANGED <<phase, timedOut, exited, stopped, leases, rebound, hostReserved>>

Reclaim(d) ==
    /\ phase = "revoked" /\ d \in verified \ stopped
    /\ leases[d] = OldGrant
    /\ stopped' = stopped \cup {d}
    /\ leases' = [leases EXCEPT ![d] = Free]
    /\ phase' = IF stopped' = Devices THEN "retired" ELSE "revoked"
    /\ hostReserved' = (phase' # "retired")
    /\ UNCHANGED <<timedOut, exited, verified, rebound>>

AcquireReplacement ==
    /\ leases[Healthy] = Free /\ ~rebound
    /\ leases' = [leases EXCEPT ![Healthy] = NewGrant]
    /\ rebound' = TRUE
    /\ UNCHANGED <<phase, timedOut, exited, verified, stopped, hostReserved>>

Next ==
    \/ Timeout \/ Revoke \/ AcquireReplacement
    \/ \E d \in Devices : Exit(d) \/ Verify(d) \/ Reclaim(d)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

NoUnverifiedReuse ==
    /\ stopped \subseteq verified
    /\ \A d \in Devices : d \notin stopped => leases[d] = OldGrant

ReboundLeasePreserved == rebound => leases[Healthy] = NewGrant
SharedBudgetKept == hostReserved = (stopped # Devices)
HealthyRejoins == <> rebound
VerifiedGroupRetires == CanVerifyFault => <> (phase = "retired")
====
