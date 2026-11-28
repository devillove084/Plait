---- MODULE PlaitWorkerUnavailable ----
EXTENDS FiniteSets

CONSTANTS CheckGrant, CheckDrain, ScopeDevice
VARIABLES phase, held, stopped, drains, reports, successor
vars == <<phase, held, stopped, drains, reports, successor>>

Failed == "failed"
Survivor == "survivor"
Sibling == "same-host-sibling"
Workers == {Failed, Survivor}
Devices == Workers \cup {Sibling}
ReportIds == {1, 2}

Init ==
    /\ phase = "Active"
    /\ held = Workers /\ stopped = {}
    /\ drains = [device \in Devices |-> 0]
    /\ reports = {}
    /\ successor = {}

TransportFailure ==
    /\ phase = "Active"
    /\ phase' = "Revoked"
    /\ UNCHANGED <<held, stopped, drains, reports, successor>>

ReportUnavailable(id) ==
    /\ id \notin reports
    /\ CheckGrant => Failed \in held
    /\ phase' = IF phase = "Active" THEN "Revoked" ELSE phase
    /\ drains' = [device \in Devices |->
         IF drains[device] = 0 /\ (device = Failed \/ ~ScopeDevice)
         THEN id ELSE drains[device]]
    /\ reports' = reports \cup {id}
    /\ UNCHANGED <<held, stopped, successor>>

Stop(device) ==
    /\ phase = "Revoked"
    /\ device \in held \ stopped
    /\ stopped' = stopped \cup {device}
    /\ UNCHANGED <<phase, held, drains, reports, successor>>

Release(device) ==
    /\ device \in held \cap stopped
    /\ held' = held \ {device}
    /\ phase' = IF held' = {} THEN "Retired" ELSE phase
    /\ UNCHANGED <<stopped, drains, reports, successor>>

Prepare(device) ==
    /\ device \in Workers \ (held \cup successor)
    /\ CheckDrain => drains[device] = 0
    /\ successor' = successor \cup {device}
    /\ UNCHANGED <<phase, held, stopped, drains, reports>>

Resume(id) ==
    /\ drains[Failed] = id
    /\ drains' = [drains EXCEPT ![Failed] = 0]
    /\ UNCHANGED <<phase, held, stopped, reports, successor>>

Next ==
    \/ TransportFailure
    \/ \E id \in ReportIds : ReportUnavailable(id) \/ Resume(id)
    \/ \E device \in Workers : Stop(device) \/ Release(device) \/ Prepare(device)

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ phase \in {"Active", "Revoked", "Retired"}
    /\ held \subseteq Workers /\ stopped \subseteq Workers
    /\ drains \in [Devices -> {0} \cup ReportIds]
    /\ reports \subseteq ReportIds /\ successor \subseteq Workers

RevokedBeforeRelease == phase = "Active" => held = Workers
ReleasedAfterStop == Workers \ held \subseteq stopped
OnlyFailedDeviceDrains == \A device \in Devices \ {Failed} : drains[device] = 0
SuccessorIsAvailable == \A device \in successor : drains[device] = 0

====
