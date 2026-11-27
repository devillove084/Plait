---- MODULE PlaitRanklessPeerJoin ----
EXTENDS Naturals, FiniteSets

\* Two group versions reuse the same devices and endpoint descriptors. The
\* grants in an observed manifest may be stale for any remote device.
CONSTANTS BindVersion, BindGrants
Epochs == 1..2
Devices == 1..3
Listeners == Epochs \X Devices
VARIABLES views, links, cancelled
vars == <<views, links, cancelled>>

\* Sorting stable IDs assigns each device the same temporary slot in every
\* permutation of this membership.
Slot(device) == Cardinality({other \in Devices : other < device})
Grants(stale) == [device \in Devices |-> IF device = stale THEN 0 ELSE 1]
NoView == [epoch |-> 0, grants |-> Grants(0)]

Init ==
    /\ views = [listener \in Listeners |-> NoView]
    /\ links = {}
    /\ cancelled = {}

ReadManifest(listener, stale) ==
    /\ views[listener] = NoView
    /\ listener \notin cancelled
    /\ stale # listener[2]
    /\ views' = [views EXCEPT ![listener] = [epoch |-> listener[1], grants |-> Grants(stale)]]
    /\ UNCHANGED <<links, cancelled>>

Hello(left, right) ==
    /\ Slot(left[2]) < Slot(right[2])
    /\ views[left] # NoView /\ views[right] # NoView
    /\ left \notin cancelled /\ right \notin cancelled
    /\ BindVersion => views[left].epoch = views[right].epoch
    /\ BindGrants => views[left].grants = views[right].grants
    /\ links' = links \cup {<<left, right>>}
    /\ UNCHANGED <<views, cancelled>>

Abort(listener) ==
    /\ listener \notin cancelled
    /\ cancelled' = cancelled \cup {listener}
    /\ links' = {link \in links : listener # link[1] /\ listener # link[2]}
    /\ UNCHANGED views

Next ==
    \/ \E listener \in Listeners :
           Abort(listener) \/ \E stale \in Devices \cup {0} : ReadManifest(listener, stale)
    \/ \E left, right \in Listeners : Hello(left, right)

Spec == Init /\ [][Next]_vars

ScopedLinks == \A link \in links : link[1][1] = link[2][1]

SameGrants == \A link \in links : views[link[1]].grants = views[link[2]].grants

CancelledLinksClosed == \A link \in links :
    /\ link[1] \notin cancelled
    /\ link[2] \notin cancelled
====
