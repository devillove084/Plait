---- MODULE PlaitPeerRequests ----
EXTENDS Naturals

CONSTANTS EnforceTurn, CheckBeforeDelivery
Groups == {"first", "second"}
Seqs == 1..3
Requests == [group : Groups, seq : Seqs]
Modes == {"blocking", "ticket"}
Request(group, seq) == [group |-> group, seq |-> seq]

VARIABLES submitted, completed, running, mode, cancelled, results, abandoned,
          delivered, outOfOrder, deliveredAfterCancel
vars == <<submitted, completed, running, mode, cancelled, results, abandoned,
          delivered, outOfOrder, deliveredAfterCancel>>

Init ==
    /\ submitted = [group \in Groups |-> 0]
    /\ completed = [group \in Groups |-> 0]
    /\ running = [group \in Groups |-> 0]
    /\ mode = [request \in Requests |-> "unsubmitted"]
    /\ cancelled = {} /\ results = {} /\ abandoned = {} /\ delivered = {}
    /\ outOfOrder = FALSE /\ deliveredAfterCancel = FALSE

Admit(group, kind) ==
    /\ group \notin cancelled /\ submitted[group] < 3
    /\ submitted' = [submitted EXCEPT ![group] = @ + 1]
    /\ mode' = [mode EXCEPT ![Request(group, submitted'[group])] = kind]
    /\ UNCHANGED <<completed, running, cancelled, results, abandoned,
                   delivered, outOfOrder, deliveredAfterCancel>>

EnterWire(group, seq) ==
    /\ group \notin cancelled /\ running[group] = 0
    /\ completed[group] < seq /\ seq <= submitted[group]
    /\ EnforceTurn => seq = completed[group] + 1
    /\ running' = [running EXCEPT ![group] = seq]
    /\ outOfOrder' = (outOfOrder \/ seq # completed[group] + 1)
    /\ UNCHANGED <<submitted, completed, mode, cancelled, results, abandoned,
                   delivered, deliveredAfterCancel>>

FinishWire(group) ==
    /\ group \notin cancelled /\ running[group] # 0
    /\ results' = results \cup {Request(group, running[group])}
    /\ completed' = [completed EXCEPT ![group] = running[group]]
    /\ running' = [running EXCEPT ![group] = 0]
    /\ UNCHANGED <<submitted, mode, cancelled, abandoned, delivered,
                   outOfOrder, deliveredAfterCancel>>

Cancel(group) ==
    /\ group \notin cancelled
    /\ cancelled' = cancelled \cup {group}
    /\ running' = [running EXCEPT ![group] = 0]
    /\ UNCHANGED <<submitted, completed, mode, results, abandoned, delivered,
                   outOfOrder, deliveredAfterCancel>>

Abandon(request) ==
    /\ mode[request] = "ticket" /\ request \notin delivered
    /\ abandoned' = abandoned \cup {request}
    /\ UNCHANGED <<submitted, completed, running, mode, cancelled, results,
                   delivered, outOfOrder, deliveredAfterCancel>>

Deliver(request) ==
    /\ request \in results \ (abandoned \cup delivered)
    /\ CheckBeforeDelivery => request.group \notin cancelled
    /\ delivered' = delivered \cup {request}
    /\ deliveredAfterCancel' = (deliveredAfterCancel \/ request.group \in cancelled)
    /\ UNCHANGED <<submitted, completed, running, mode, cancelled, results,
                   abandoned, outOfOrder>>

Next ==
    \/ \E group \in Groups :
        \/ FinishWire(group) \/ Cancel(group)
        \/ \E kind \in Modes : Admit(group, kind)
        \/ \E seq \in Seqs : EnterWire(group, seq)
    \/ \E request \in Requests : Abandon(request) \/ Deliver(request)

Spec == Init /\ [][Next]_vars

WireOrder == ~outOfOrder
NoStaleDelivery == ~deliveredAfterCancel
DeliveredResultsExist == delivered \subseteq results
AbandonedResultsStayPrivate == abandoned \cap delivered = {}
CancelledLeavesWire == \A group \in cancelled : running[group] = 0
====
