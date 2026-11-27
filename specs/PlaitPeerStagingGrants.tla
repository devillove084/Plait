---- MODULE PlaitPeerStagingGrants ----
EXTENDS Naturals, FiniteSets

CONSTANT DivideByLocalWorkers

Versions == 1..2
Devices == {"a", "b", "c"}
Hosts == {"left", "right"}
Host(device) == IF device = "c" THEN "right" ELSE "left"
Assigned(version) == IF version = 1 THEN Devices ELSE {"a", "c"}
Local(version, host) == {device \in Assigned(version) : Host(device) = host}
Pageable(version, host) == IF host = "left" /\ version = 1 THEN 5 ELSE 3
Grants == Versions \X Devices

VARIABLES prepared, held, draining, opened
vars == <<prepared, held, draining, opened>>

Init ==
    /\ prepared = {} /\ draining = {} /\ opened = {}
    /\ held = [version \in Versions |-> {}]

Prepare(version) ==
    /\ version \notin prepared
    /\ version = 2 => 1 \in prepared
    /\ \A other \in prepared : held[other] \cap Assigned(version) = {}
    /\ prepared' = prepared \cup {version}
    /\ held' = [held EXCEPT ![version] = Assigned(version)]
    /\ UNCHANGED <<draining, opened>>

Open(grant) ==
    /\ grant[2] \in held[grant[1]] /\ grant[1] \notin draining
    /\ opened' = opened \cup {grant}
    /\ UNCHANGED <<prepared, held, draining>>

Drain(version) ==
    /\ version \in prepared \ draining
    /\ draining' = draining \cup {version}
    /\ UNCHANGED <<prepared, held, opened>>

Close(grant) ==
    /\ grant \in opened
    /\ opened' = opened \ {grant}
    /\ UNCHANGED <<prepared, held, draining>>

Release(grant) ==
    /\ grant[1] \in draining /\ grant[2] \in held[grant[1]]
    /\ grant \notin opened
    /\ held' = [held EXCEPT ![grant[1]] = @ \ {grant[2]}]
    /\ UNCHANGED <<prepared, draining, opened>>

Next ==
    \/ \E version \in Versions : Prepare(version) \/ Drain(version)
    \/ \E grant \in Grants : Open(grant) \/ Close(grant) \/ Release(grant)

Spec == Init /\ [][Next]_vars

Ceiling(version, device, host) ==
    IF <<version, device>> \in opened /\ Host(device) = host
    THEN Pageable(version, host) \div
         (IF DivideByLocalWorkers THEN Cardinality(Local(version, host)) ELSE 1)
    ELSE 0

Sum(version, host) == Ceiling(version, "a", host) + Ceiling(version, "b", host)
                     + Ceiling(version, "c", host)
Reserved(version, host) == IF held[version] \cap Local(version, host) # {}
                           THEN Pageable(version, host) ELSE 0

EpochFitsReservation == \A version \in Versions, host \in Hosts :
    Sum(version, host) <= Reserved(version, host)

CommunicatorHasGrant == \A grant \in opened : grant[2] \in held[grant[1]]
====
