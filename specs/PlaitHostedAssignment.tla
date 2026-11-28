---- MODULE PlaitHostedAssignment ----
EXTENDS Naturals

\* A durable assignment for one host. A later supervisor can read a stale
\* assignment, but the claim and stop transitions apply to the current state.
Sessions == 1..2
None == 0

VARIABLES phase, seen, parent, lock, owner, launcher
vars == <<phase, seen, parent, lock, owner, launcher>>

Init ==
    /\ phase = "Absent" /\ seen = {}
    /\ parent = None /\ lock = None /\ owner = None /\ launcher = None

Prepare ==
    /\ phase = "Absent" /\ phase' = "Preparing"
    /\ UNCHANGED <<seen, parent, lock, owner, launcher>>

Discover(s) ==
    /\ phase # "Absent" /\ s \notin seen
    /\ seen' = seen \cup {s}
    /\ UNCHANGED <<phase, parent, lock, owner, launcher>>

StartSupervisor(s) ==
    /\ lock = None /\ parent' = s /\ lock' = s
    /\ UNCHANGED <<phase, seen, owner, launcher>>

Claim(s) ==
    /\ parent = s /\ s \in seen /\ phase = "Preparing"
    /\ owner \in {None, s} /\ owner' = s
    /\ UNCHANGED <<phase, seen, parent, lock, launcher>>

Launch(s) ==
    /\ parent = s /\ owner = s /\ launcher = None
    /\ phase \in {"Preparing", "Revoked"} /\ launcher' = s
    /\ UNCHANGED <<phase, seen, parent, lock, owner>>

Ready(s) ==
    /\ launcher = s /\ parent = s /\ phase = "Preparing"
    /\ phase' = "Active"
    /\ UNCHANGED <<seen, parent, lock, owner, launcher>>

LoseSupervisor ==
    /\ parent # None /\ parent' = None
    /\ lock' = launcher
    /\ UNCHANGED <<phase, seen, owner, launcher>>

StopLauncher ==
    /\ launcher # None /\ launcher' = None /\ lock' = parent
    /\ UNCHANGED <<phase, seen, parent, owner>>

Revoke ==
    /\ phase \in {"Preparing", "Active"} /\ phase' = "Revoked"
    /\ UNCHANGED <<seen, parent, lock, owner, launcher>>

Retire ==
    /\ phase = "Revoked" /\ launcher = None /\ phase' = "Retired"
    /\ UNCHANGED <<seen, parent, lock, owner, launcher>>

Next == Prepare \/ LoseSupervisor \/ StopLauncher \/ Revoke \/ Retire
    \/ \E s \in Sessions : Discover(s) \/ StartSupervisor(s) \/ Claim(s) \/ Launch(s) \/ Ready(s)
Spec == Init /\ [][Next]_vars

ClaimBeforeLaunch == launcher # None => owner = launcher
LiveLauncherHoldsLock == launcher # None => lock = launcher
NoConcurrentSupervisor == launcher # None => parent \in {None, launcher}
RetiredHasNoLauncher == phase = "Retired" => launcher = None
====
