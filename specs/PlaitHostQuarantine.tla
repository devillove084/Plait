---- MODULE PlaitHostQuarantine ----
EXTENDS Naturals, FiniteSets, Sequences

CONSTANTS CheckEvidence, RevokeHostGroups, ExcludeLocalStandby, CheckHostDrain

Incumbent == "mixed"
Standby == "standby"
Neighbor == "same host"
Independent == "other host"
Groups == {Incumbent, Standby, Neighbor, Independent}

Failed == 0
Survivor == 2
ReportIds == 1..2
Members(group) ==
    CASE group = Incumbent -> 0..3
      [] group = Standby -> {4}
      [] group = Neighbor -> {5}
      [] group = Independent -> {6}

Devices == UNION {Members(group) : group \in Groups}

VARIABLES colocated, phase, armed, held, stopped, drain, receipts, checked, placed
vars == <<colocated, phase, armed, held, stopped, drain, receipts, checked, placed>>

BadHost == {0, 1, 5} \cup (IF colocated THEN {4} ELSE {})
OnBadHost == {group \in Groups : Members(group) \cap BadHost # {}}
Live == {"Preparing", "Ready", "Active", "Draining"}

Init ==
    /\ colocated \in BOOLEAN
    /\ phase = [group \in Groups |-> IF group = Standby THEN "Preparing" ELSE "Active"]
    /\ armed = FALSE
    /\ held = Devices /\ stopped = {}
    /\ drain = 0 /\ receipts = <<>> /\ checked = {}
    /\ placed = {}

ReadyStandby ==
    /\ phase[Standby] = "Preparing"
    /\ phase' = [phase EXCEPT ![Standby] = "Ready"]
    /\ UNCHANGED <<colocated, armed, held, stopped, drain, receipts, checked, placed>>

Arm ==
    /\ phase[Incumbent] = "Active" /\ phase[Standby] = "Ready" /\ ~armed
    /\ armed' = TRUE
    /\ UNCHANGED <<colocated, phase, held, stopped, drain, receipts, checked, placed>>

CanPromote == phase[Incumbent] = "Active" /\ armed /\ phase[Standby] = "Ready"

\* A transport failure may end the version before confirmed device-loss
\* evidence arrives. It leaves the worker grant held until explicit release.
TransportLoss ==
    /\ phase[Incumbent] = "Active"
    /\ phase' = [phase EXCEPT ![Incumbent] = "Revoked",
         ![Standby] = IF CanPromote THEN "Active" ELSE @]
    /\ armed' = FALSE
    /\ UNCHANGED <<colocated, held, stopped, drain, receipts, checked, placed>>

\* The represented device has an original grant and a current inventory
\* incarnation. The report can carry stale evidence in either mode.
MatchesEvidence(kind, current) ==
    /\ current
    /\ CASE kind = "grant" -> Failed \in held
         [] kind = "idle" -> Failed \notin held /\ Failed \notin placed

Quarantine(id, kind, current) ==
    LET valid == MatchesEvidence(kind, current)
        victims == IF RevokeHostGroups THEN OnBadHost ELSE {Incumbent}
        promote == CanPromote /\ (~ExcludeLocalStandby \/ Standby \notin victims)
    IN
    /\ id \notin {receipts[i] : i \in DOMAIN receipts}
    /\ CheckEvidence => valid
    /\ drain' = IF drain = 0 THEN id ELSE drain
    /\ receipts' = Append(receipts, id)
    /\ checked' = IF valid THEN checked \cup {id} ELSE checked
    /\ phase' = [group \in Groups |->
         IF group = Standby /\ promote THEN "Active"
         ELSE IF group \in victims /\ phase[group] \in Live THEN "Revoked"
         ELSE phase[group]]
    /\ armed' = FALSE
    /\ UNCHANGED <<colocated, held, stopped, placed>>

\* A stop notification is the owner's proof that the process exited.
OwnerStops(group, device) ==
    /\ phase[group] = "Revoked"
    /\ device \in (Members(group) \cap held) \ stopped
    /\ stopped' = stopped \cup {device}
    /\ UNCHANGED <<colocated, phase, armed, held, drain, receipts, checked, placed>>

Release(group, device) ==
    /\ phase[group] = "Revoked"
    /\ device \in Members(group) \cap held \cap stopped
    /\ held' = held \ {device}
    /\ phase' = IF Members(group) \cap held' = {}
                THEN [phase EXCEPT ![group] = "Retired"] ELSE phase
    /\ UNCHANGED <<colocated, armed, stopped, drain, receipts, checked, placed>>

\* Probe a fresh placement after a confirmed fault. It may reuse a released
\* survivor, and it must still exclude a released device from the drained host.
Place(device) ==
    /\ drain # 0 /\ device \notin held \cup placed
    /\ CheckHostDrain => device \notin BadHost
    /\ placed' = placed \cup {device}
    /\ UNCHANGED <<colocated, phase, armed, held, stopped, drain, receipts, checked>>

Next ==
    \/ ReadyStandby \/ Arm \/ TransportLoss
    \/ \E id \in ReportIds, kind \in {"grant", "idle"}, current \in BOOLEAN :
        Quarantine(id, kind, current)
    \/ \E group \in Groups :
        \E device \in Members(group) : OwnerStops(group, device) \/ Release(group, device)
    \/ \E device \in {Failed, Survivor} : Place(device)

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ colocated \in BOOLEAN /\ armed \in BOOLEAN
    /\ phase \in [Groups -> Live \cup {"Revoked", "Retired"}]
    /\ held \subseteq Devices /\ stopped \subseteq Devices
    /\ drain \in {0} \cup ReportIds
    /\ receipts \in UNION {[1..size -> ReportIds] : size \in 0..2}
    /\ checked \subseteq ReportIds /\ placed \subseteq Devices

ExactEvidence == {receipts[i] : i \in DOMAIN receipts} = checked

StableDrain ==
    IF Len(receipts) = 0 THEN drain = 0 ELSE drain = receipts[1]

AffectedGroupsEnded ==
    drain # 0 => \A group \in OnBadHost : phase[group] \notin Live

OnlyOneActive ==
    phase[Incumbent] = "Active" => phase[Standby] # "Active"

UnrelatedGroupUnaffected ==
    /\ phase[Independent] = "Active"
    /\ Members(Independent) \subseteq held

ReleaseAfterStop == Devices \ held \subseteq stopped

NewPlacementUsesHealthyReleasedDevices ==
    /\ placed \cap held = {}
    /\ placed \cap BadHost = {}

StandbyWasReadyAndArmed ==
    [][(phase[Standby] # "Active" /\ phase'[Standby] = "Active") => CanPromote]_vars
====
