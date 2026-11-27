---- MODULE PlaitRanklessLayout ----
EXTENDS Naturals

CONSTANT CheckLayout
Devices == 1..3
Layouts == {<<1, 2, 3>>, <<1, 3, 2>>}
NoView == <<>>
VARIABLES views, links, ready
vars == <<views, links, ready>>

Init ==
    /\ views = [device \in Devices |-> NoView]
    /\ links = {} /\ ready = {}

Read(device, order) ==
    /\ views[device] = NoView
    /\ views' = [views EXCEPT ![device] = order]
    /\ UNCHANGED <<links, ready>>

Connect(a, b) ==
    /\ a < b
    /\ views[a] # NoView /\ views[b] # NoView
    /\ CheckLayout => views[a] = views[b]
    /\ links' = links \cup {{a, b}}
    /\ UNCHANGED <<views, ready>>

Ready(a) ==
    /\ \A b \in Devices \ {a} : {a, b} \in links
    /\ ready' = ready \cup {a}
    /\ UNCHANGED <<views, links>>

Next ==
    \/ \E a \in Devices : Ready(a) \/ \E order \in Layouts : Read(a, order)
    \/ \E a, b \in Devices : Connect(a, b)

Spec == Init /\ [][Next]_vars
PairedLayouts == \A link \in links : \A a, b \in link : views[a] = views[b]
ReadyUsesOneLayout == \A a \in ready : \A b \in Devices : views[a] = views[b]
====
