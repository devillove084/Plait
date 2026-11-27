---- MODULE PlaitHostedPdDecodeSwitch ----
EXTENDS Naturals

\* One planned Decode replacement with a retained Prefill. Admission binds
\* a directory snapshot. Release evidence is produced by each old launcher
\* after its workers stop. A grace period expiry aborts unfinished streams.
CONSTANTS BindStream, RequireReady, RequireRequestDrain, RequireAllStops

Requests == {1, 2, 3}
Limit == 2
OldApi == "old-api"
OldFollower == "old-follower"
OldDevices == {OldApi, OldFollower}
PrefillDevices == {"prefill-a", "prefill-b"}
NewDevice == "new-decode"
Devices == OldDevices \cup PrefillDevices \cup {NewDevice}
Absent == [epoch |-> 0, tokens |-> 0, producer |-> 0, terminal |-> "Open"]

VARIABLES directory, staged, acknowledged, oldPhase, requests, owners, stopped
vars == <<directory, staged, acknowledged, oldPhase, requests, owners, stopped>>

Init ==
    /\ directory = 1
    /\ staged = "Absent"
    /\ acknowledged = FALSE
    /\ oldPhase = "Active"
    /\ requests = [r \in Requests |-> Absent]
    /\ owners = [d \in Devices |-> IF d \in OldDevices THEN "Old"
                                  ELSE IF d \in PrefillDevices THEN "Prefill" ELSE "Free"]
    /\ stopped = {}

Prepare ==
    /\ staged = "Absent"
    /\ owners[NewDevice] = "Free"
    /\ staged' = "Preparing"
    /\ owners' = [owners EXCEPT ![NewDevice] = "New"]
    /\ UNCHANGED <<directory, acknowledged, oldPhase, requests, stopped>>

Ready ==
    /\ staged = "Preparing"
    /\ staged' = "Ready"
    /\ acknowledged' = TRUE
    /\ UNCHANGED <<directory, oldPhase, requests, owners, stopped>>

Cancel ==
    /\ staged \in {"Preparing", "Ready"}
    /\ staged' = "Cancelled"
    /\ acknowledged' = FALSE
    /\ UNCHANGED <<directory, oldPhase, requests, owners, stopped>>

Publish ==
    /\ staged \in {"Preparing", "Ready"}
    /\ RequireReady => staged = "Ready"
    /\ directory' = 2
    /\ staged' = "Active"
    /\ oldPhase' = "Draining"
    /\ UNCHANGED <<acknowledged, requests, owners, stopped>>

Admit(r) ==
    /\ requests[r] = Absent
    /\ requests' = [requests EXCEPT ![r].epoch = directory]
    /\ UNCHANGED <<directory, staged, acknowledged, oldPhase, owners, stopped>>

OldInFlight(r) == requests[r].epoch = 1 /\ requests[r].terminal = "Open"
Emit(r) ==
    LET epoch == IF BindStream THEN requests[r].epoch ELSE directory
    IN /\ requests[r].epoch # 0
       /\ requests[r].terminal = "Open"
       /\ requests[r].tokens < Limit
       /\ IF epoch = 1 THEN oldPhase \in {"Active", "Draining"} /\ stopped = {}
                       ELSE staged = "Active"
       /\ requests' = [requests EXCEPT ![r].tokens = @ + 1, ![r].producer = epoch]
       /\ UNCHANGED <<directory, staged, acknowledged, oldPhase, owners, stopped>>

Finish(r) ==
    /\ requests[r].terminal = "Open"
    /\ requests[r].tokens = Limit
    /\ requests' = [requests EXCEPT ![r].terminal = "Done"]
    /\ UNCHANGED <<directory, staged, acknowledged, oldPhase, owners, stopped>>

ExpireGrace ==
    /\ oldPhase = "Draining"
    /\ \E r \in Requests : OldInFlight(r)
    /\ requests' = [r \in Requests |->
        IF OldInFlight(r) THEN [requests[r] EXCEPT !.terminal = "Aborted"] ELSE requests[r]]
    /\ UNCHANGED <<directory, staged, acknowledged, oldPhase, owners, stopped>>

ConfirmStop(device) ==
    /\ oldPhase = "Draining"
    /\ device \notin stopped
    /\ IF device = OldApi
          THEN RequireRequestDrain => (\A r \in Requests : ~OldInFlight(r))
          ELSE OldApi \in stopped
    /\ stopped' = stopped \cup {device}
    /\ UNCHANGED <<directory, staged, acknowledged, oldPhase, requests, owners>>

Retire ==
    /\ oldPhase = "Draining"
    /\ RequireAllStops => stopped = OldDevices
    /\ oldPhase' = "Retired"
    /\ owners' = [d \in Devices |-> IF d \in OldDevices THEN "Free" ELSE owners[d]]
    /\ UNCHANGED <<directory, staged, acknowledged, requests, stopped>>

Next ==
    \/ Prepare \/ Ready \/ Cancel \/ Publish \/ ExpireGrace \/ Retire
    \/ \E r \in Requests : Admit(r) \/ Emit(r) \/ Finish(r)
    \/ \E device \in OldDevices : ConfirmStop(device)

Spec == Init /\ [][Next]_vars
\* Finished finite executions are legal even when no transition is enabled.

PublishedOnlyWhenReady == directory = 2 => staged = "Active" /\ acknowledged

PreservesAdmission ==
    \A r \in Requests :
        requests[r].producer # 0 => requests[r].producer = requests[r].epoch

NoStopsBeforeDrain ==
    stopped # {} => \A r \in Requests : ~OldInFlight(r)

RetiredAfterEveryStop == oldPhase = "Retired" => stopped = OldDevices

RetainsUnrelatedGrants ==
    /\ \A d \in PrefillDevices : owners[d] = "Prefill"
    /\ staged # "Absent" => owners[NewDevice] = "New"
    /\ oldPhase # "Retired" => \A d \in OldDevices : owners[d] = "Old"

CompleteTerminal ==
    \A r \in Requests : requests[r].terminal = "Done" => requests[r].tokens = Limit
====
