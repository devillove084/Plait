---- MODULE PlaitServingIngress ----
EXTENDS Naturals

CONSTANT Requests
VARIABLES active, route, oldOpen, requests

Epochs == 1..2
vars == <<active, route, oldOpen, requests>>
GateOpen(e) == IF e = 1 THEN oldOpen ELSE active = e

Init ==
    /\ active = 1 /\ route = 1 /\ oldOpen = TRUE
    /\ requests = [r \in Requests |->
        [phase |-> "Pending", target |-> 0, admissions |-> 0,
         admittedOn |-> 0, outputOn |-> {}, chunks |-> 0]]

Activate ==
    /\ active = 1 /\ active' = 2
    /\ UNCHANGED <<route, oldOpen, requests>>

\* The ingress can continue to see the previous epoch during a refresh.
Observe(epoch) ==
    /\ epoch <= active /\ epoch >= route
    /\ route' = epoch
    /\ UNCHANGED <<active, oldOpen, requests>>

CloseOld ==
    /\ active = 2 /\ oldOpen
    /\ oldOpen' = FALSE
    /\ UNCHANGED <<active, route, requests>>

Connect(r) ==
    /\ requests[r].phase = "Pending"
    /\ requests' = [requests EXCEPT ![r].phase = "Connecting", ![r].target = route]
    /\ UNCHANGED <<active, route, oldOpen>>

Send(r) ==
    /\ requests[r].phase = "Connecting"
    /\ requests' = [requests EXCEPT ![r].phase = "Sent"]
    /\ UNCHANGED <<active, route, oldOpen>>

Retry(r) ==
    /\ (requests[r].phase = "Connecting"
        \/ (requests[r].phase = "Sent" /\ ~GateOpen(requests[r].target)))
    /\ requests' = [requests EXCEPT ![r].phase = "Pending", ![r].target = 0]
    /\ UNCHANGED <<active, route, oldOpen>>

Admit(r) ==
    /\ requests[r].phase = "Sent" /\ GateOpen(requests[r].target)
    /\ requests' = [requests EXCEPT ![r].phase = "Streaming",
        ![r].admittedOn = requests[r].target, ![r].admissions = @ + 1]
    /\ UNCHANGED <<active, route, oldOpen>>

AmbiguousFailure(r) ==
    /\ requests[r].phase \in {"Sent", "Streaming"}
    /\ requests' = [requests EXCEPT ![r].phase = "Failed"]
    /\ UNCHANGED <<active, route, oldOpen>>

Forward(r) ==
    /\ requests[r].phase = "Streaming"
    /\ requests' = [requests EXCEPT ![r].chunks = @ + 1,
        ![r].outputOn = @ \cup {requests[r].target},
        ![r].phase = IF requests[r].chunks = 1 THEN "Done" ELSE "Streaming"]
    /\ UNCHANGED <<active, route, oldOpen>>

Next == Activate \/ CloseOld \/ (\E e \in Epochs : Observe(e))
    \/ (\E r \in Requests :
        Connect(r) \/ Send(r) \/ Retry(r) \/ Admit(r) \/ AmbiguousFailure(r) \/ Forward(r))

Spec == Init /\ [][Next]_vars

SingleAdmission == \A r \in Requests : requests[r].admissions <= 1
PinnedOutput == \A r \in Requests : requests[r].outputOn \subseteq {requests[r].admittedOn}
NoRouteRegression == [][route' >= route]_vars
OldStreamCanFinish == \A r \in Requests :
    active = 2 /\ ~oldOpen /\ requests[r].phase = "Streaming" /\ requests[r].admittedOn = 1
    => ENABLED Forward(r)
====
