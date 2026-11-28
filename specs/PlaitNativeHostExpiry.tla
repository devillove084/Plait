---- MODULE PlaitNativeHostExpiry ----
EXTENDS Naturals

CONSTANTS CheckPulse, CheckGroups

Versions == 1..2
Pulses == 1..3
VARIABLES pulse, online, live, held, snapshot
vars == <<pulse, online, live, held, snapshot>>

Init ==
    /\ pulse = 1 /\ online = TRUE
    /\ live = {1} /\ held = live
    /\ snapshot = [pulse |-> 0, groups |-> {}]

Renew ==
    /\ pulse < 3
    /\ pulse' = pulse + 1 /\ online' = TRUE
    /\ UNCHANGED <<live, held, snapshot>>

Prepare(version) ==
    /\ online /\ version \notin held
    /\ live' = live \cup {version}
    /\ held' = held \cup {version}
    /\ UNCHANGED <<pulse, online, snapshot>>

Revoke(version) ==
    /\ version \in live
    /\ live' = live \ {version}
    /\ UNCHANGED <<pulse, online, held, snapshot>>

\* The voter has observed this pulse for the timeout period. The model
\* abstracts elapsed time and admits races while the proposal is ordered.
ObserveTimeout ==
    /\ online
    /\ snapshot' = [pulse |-> pulse, groups |-> live]
    /\ UNCHANGED <<pulse, online, live, held>>

Expire ==
    /\ online /\ snapshot.pulse \in Pulses
    /\ CheckPulse => snapshot.pulse = pulse
    /\ CheckGroups => snapshot.groups = live
    /\ online' = FALSE
    /\ live' = live \ snapshot.groups
    /\ UNCHANGED <<pulse, held, snapshot>>

Next ==
    \/ Renew \/ ObserveTimeout \/ Expire
    \/ \E version \in Versions : Prepare(version) \/ Revoke(version)

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ pulse \in Pulses /\ online \in BOOLEAN
    /\ live \subseteq held /\ held \subseteq Versions
    /\ snapshot \in [pulse : 0..3, groups : SUBSET Versions]

OfflineHostHasNoLiveVersions == ~online => live = {}

ExpiryUsesCurrentEvidence ==
    [][(online /\ ~online') => (snapshot.pulse = pulse /\ snapshot.groups = live)]_vars

ExpiryRetainsFences ==
    [][(online /\ ~online') => held' = held]_vars
====
