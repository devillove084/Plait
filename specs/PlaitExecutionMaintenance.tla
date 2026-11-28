---- MODULE PlaitExecutionMaintenance ----
EXTENDS Naturals

\* Two single-device groups share a host. A maintenance target covers
\* either one device or the whole host. Devices 3 and 4 are off-host.
Devices == 1..4
Groups == 1..2
Targets == {{1}, {1, 2}}
NoToken == 0

VARIABLES target, token, serial, live, held, deferred, staged, ready, confirmed
vars == <<target, token, serial, live, held, deferred, staged, ready, confirmed>>

Init ==
    /\ target \in Targets
    /\ token = NoToken
    /\ serial = 0
    /\ live = [g \in Groups |-> g]
    /\ held = {1, 2}
    /\ deferred \in SUBSET target
    /\ staged = 0
    /\ ready = FALSE
    /\ confirmed = FALSE

Begin ==
    /\ token = NoToken
    /\ serial < 2
    /\ serial' = serial + 1
    /\ token' = serial'
    /\ UNCHANGED <<target, live, held, deferred, staged, ready, confirmed>>

Resume(expected) ==
    /\ token # NoToken
    /\ expected = token
    /\ token' = NoToken
    /\ confirmed' = FALSE
    /\ UNCHANGED <<target, serial, live, held, deferred, staged, ready>>

Stage(d, observed) ==
    /\ staged = 0
    /\ observed # NoToken
    /\ observed = token
    /\ d \notin target \cup held \cup deferred
    /\ \E g \in Groups : live[g] \in target
    /\ staged' = d
    /\ held' = held \cup {d}
    /\ UNCHANGED <<target, token, serial, live, deferred, ready, confirmed>>

Report ==
    /\ staged # 0
    /\ ~ready
    /\ ready' = TRUE
    /\ UNCHANGED <<target, token, serial, live, held, deferred, staged, confirmed>>

Publish(g) ==
    /\ ready
    /\ staged # 0
    /\ live[g] \in target
    /\ live' = [live EXCEPT ![g] = staged]
    /\ staged' = 0
    /\ ready' = FALSE
    /\ UNCHANGED <<target, token, serial, held, deferred, confirmed>>

Retire(d) ==
    /\ d \in held
    /\ d # staged
    /\ \A g \in Groups : live[g] # d
    /\ held' = held \ {d}
    /\ UNCHANGED <<target, token, serial, live, deferred, staged, ready, confirmed>>

DropDeferred(d) ==
    /\ d \in deferred
    /\ deferred' = deferred \ {d}
    /\ UNCHANGED <<target, token, serial, live, held, staged, ready, confirmed>>

Confirm(expected) ==
    /\ token # NoToken
    /\ expected = token
    /\ (held \cup deferred) \cap target = {}
    /\ confirmed' = TRUE
    /\ UNCHANGED <<target, token, serial, live, held, deferred, staged, ready>>

Next ==
    \/ Begin
    \/ \E expected \in 1..2 : Resume(expected) \/ Confirm(expected)
    \/ \E d \in Devices, observed \in 1..2 : Stage(d, observed)
    \/ Report
    \/ \E g \in Groups : Publish(g)
    \/ \E d \in Devices : Retire(d) \/ DropDeferred(d)

Spec == Init /\ [][Next]_vars
ConfirmedHasNoHolds ==
    confirmed => token # NoToken /\ (held \cup deferred) \cap target = {}
LiveOwnsGrant == \A g \in Groups : live[g] \in held
OtherDeviceKeepsItsGroup == target = {1} => live[2] = 2
OrderedNewGrant ==
    [][(staged' # 0 /\ staged' # staged) => (token # NoToken /\ staged' \notin target)]_vars
====
