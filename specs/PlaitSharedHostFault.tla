---- MODULE PlaitSharedHostFault ----
EXTENDS FiniteSets

CONSTANTS ScopeDeviceFault, ScopeObservation

Incumbent == "current"
Standby == "standby"
Independent == "independent"
Groups == {Incumbent, Standby, Independent}
Hosts == {"first", "second"}
Devices == {"old-first", "old-second", "spare", "other-first", "other-second"}

Members(group) ==
    CASE group = Incumbent -> {"old-first", "old-second"}
      [] group = Standby -> {"spare"}
      [] group = Independent -> {"other-first", "other-second"}

Host(device) ==
    IF device \in {"old-first", "spare", "other-first"} THEN "first" ELSE "second"

GroupsOn(host) == {group \in Groups : \E device \in Members(group) : Host(device) = host}

VARIABLES phase, authorized, observed, closed, pending, held, stopped
vars == <<phase, authorized, observed, closed, pending, held, stopped>>

Init ==
    /\ phase = [group \in Groups |-> IF group = Standby THEN "Ready" ELSE "Active"]
    /\ authorized = {} /\ observed = {} /\ closed = {}
    /\ pending = {}
    /\ held = Devices /\ stopped = {}

HasEnded(group) == phase[group] \in {"Revoked", "Retired"}

End(victims, permitted) ==
    LET activate ==
        /\ Incumbent \in victims /\ phase[Incumbent] = "Active"
        /\ phase[Standby] = "Ready" /\ Standby \notin victims
    IN
    /\ phase' = [group \in Groups |->
        IF HasEnded(group) THEN phase[group]
        ELSE IF group \in victims THEN "Revoked"
        ELSE IF group = Standby /\ activate THEN "Active"
        ELSE phase[group]]
    /\ authorized' = authorized \cup permitted
    /\ UNCHANGED <<observed, closed, pending, held, stopped>>

ReportDevice(group, device) ==
    /\ device \in Members(group) \cap held
    /\ ~HasEnded(group)
    /\ End(IF ScopeDeviceFault THEN {group} ELSE GroupsOn(Host(device)), {group})

ReportHost(host) == End(GroupsOn(host), GroupsOn(host))

Poll(group) ==
    /\ HasEnded(group)
    /\ observed' = observed \cup {group}
    /\ UNCHANGED <<phase, authorized, closed, pending, held, stopped>>

Close(local, proof) ==
    /\ proof \in observed
    /\ ScopeObservation => local = proof
    /\ closed' = closed \cup {local}
    /\ pending' = pending \ {local}
    /\ UNCHANGED <<phase, authorized, observed, held, stopped>>

Submit(group) ==
    /\ phase[group] = "Active" /\ group \notin closed
    /\ pending' = pending \cup {group}
    /\ UNCHANGED <<phase, authorized, observed, closed, held, stopped>>

Complete(group) ==
    /\ group \in pending
    /\ pending' = pending \ {group}
    /\ UNCHANGED <<phase, authorized, observed, closed, held, stopped>>

Stop(group, device) ==
    /\ HasEnded(group) /\ device \in Members(group) \cap held
    /\ stopped' = stopped \cup {device}
    /\ UNCHANGED <<phase, authorized, observed, closed, pending, held>>

Release(group, device) ==
    /\ phase[group] = "Revoked" /\ device \in Members(group) \cap held \cap stopped
    /\ held' = held \ {device}
    /\ phase' = IF Members(group) \cap held' = {}
                THEN [phase EXCEPT ![group] = "Retired"] ELSE phase
    /\ UNCHANGED <<authorized, observed, closed, pending, stopped>>

Next ==
    \/ \E host \in Hosts : ReportHost(host)
    \/ \E group \in Groups :
         \/ Poll(group) \/ Submit(group) \/ Complete(group)
         \/ \E proof \in Groups : Close(group, proof)
         \/ \E device \in Devices :
              ReportDevice(group, device) \/ Stop(group, device) \/ Release(group, device)

Spec == Init /\ [][Next]_vars

RevocationIsScoped == {group \in Groups : HasEnded(group)} \subseteq authorized
OnlyRevokedCommunicatorsClose == closed \subseteq {group \in Groups : HasEnded(group)}
ClosedCallsWake == closed \cap pending = {}
StandbyPromotionScoped ==
    phase[Standby] = "Active" => HasEnded(Incumbent) /\ Standby \notin authorized
ReleasedDevicesStopped == Devices \ held \subseteq stopped
IndependentRemainsAvailable ==
    Independent \notin authorized =>
        /\ phase[Independent] = "Active"
        /\ Independent \notin closed
        /\ Members(Independent) \subseteq held
====
