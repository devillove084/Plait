---- MODULE PlaitPeerStaging ----
EXTENDS Naturals

CONSTANT ReleaseOnAbandon

Groups == {"tight", "roomy"}
Requests == Groups \X (1..2)
Limit(group) == IF group = "tight" THEN 2 ELSE 3
Weight(request) == request[2]
HasPayload == {"staging", "queued", "running", "buffered", "uploading"}

VARIABLES phase, credits, cancelled, abandoned
vars == <<phase, credits, cancelled, abandoned>>

Used(group) == (IF <<group, 1>> \in credits THEN 1 ELSE 0)
               + (IF <<group, 2>> \in credits THEN 2 ELSE 0)

Init ==
    /\ phase = [request \in Requests |-> "new"]
    /\ credits = {} /\ cancelled = {} /\ abandoned = {}

Reserve(request) ==
    /\ phase[request] = "new" /\ request[1] \notin cancelled
    /\ Used(request[1]) + Weight(request) <= Limit(request[1])
    /\ phase' = [phase EXCEPT ![request] = "staging"]
    /\ credits' = credits \cup {request}
    /\ UNCHANGED <<cancelled, abandoned>>

Move(request, from, to) ==
    /\ phase[request] = from
    /\ phase' = [phase EXCEPT ![request] = to]
    /\ UNCHANGED <<credits, cancelled, abandoned>>

Retire(request) ==
    /\ phase' = [phase EXCEPT ![request] = "closed"]
    /\ credits' = credits \ {request}
    /\ UNCHANGED <<cancelled, abandoned>>

FinishWire(request) ==
    /\ phase[request] = "running"
    /\ IF request \in abandoned
          THEN Retire(request)
          ELSE Move(request, "running", "buffered")

Abandon(request) ==
    /\ phase[request] \in {"queued", "running"} /\ request \notin abandoned
    /\ abandoned' = abandoned \cup {request}
    /\ credits' = IF ReleaseOnAbandon THEN credits \ {request} ELSE credits
    /\ UNCHANGED <<phase, cancelled>>

Cancel(group) ==
    /\ group \notin cancelled
    /\ cancelled' = cancelled \cup {group}
    /\ UNCHANGED <<phase, credits, abandoned>>

Next ==
    \/ \E group \in Groups : Cancel(group)
    \/ \E request \in Requests :
        \/ Reserve(request)
        \/ /\ request[1] \notin cancelled
           /\ \/ Move(request, "staging", "queued")
              \/ /\ \A other \in Requests : other[1] = request[1] => phase[other] # "running"
                 /\ Move(request, "queued", "running")
              \/ Move(request, "buffered", "uploading")
        \/ FinishWire(request)
        \/ Abandon(request)
        \/ /\ \/ phase[request] \in {"staging", "buffered", "uploading"}
              \/ request[1] \in cancelled /\ phase[request] \in {"queued", "running"}
           /\ Retire(request)

Spec == Init /\ [][Next]_vars

WithinLocalLimits == \A group \in Groups : Used(group) <= Limit(group)
PayloadHasReservation == \A request \in Requests : phase[request] \in HasPayload => request \in credits
ReservationHasOwner == \A request \in credits : phase[request] \in HasPayload
====
