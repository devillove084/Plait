---- MODULE PlaitHostedPool ----
EXTENDS Naturals

\* Two group proposals can read the same free devices and API port range.
\* Capability 0 means offline. Values 1 and 2 stand for distinct inventories.
Groups == {1, 2}
Hosts == {"api", "peer"}
Slots == {1, 2}
PortTriples == {1, 2}
Capabilities == [Hosts -> 0..2]
Placements == [slot : Slots, port : PortTriples, inventory : Capabilities]
Offline == [h \in Hosts |-> 0]
NoPlacement == [slot |-> 0, port |-> 0, inventory |-> Offline]
Empty == [placement |-> NoPlacement, phase |-> "empty", checkedInventory |-> Offline]

VARIABLES inventory, drafts, grants
vars == <<inventory, drafts, grants>>

Init ==
    /\ inventory = [h \in Hosts |-> 1]
    /\ drafts = [g \in Groups |-> NoPlacement]
    /\ grants = [g \in Groups |-> Empty]

Matches(p) ==
    \A h \in Hosts : inventory[h] # 0 /\ inventory[h] = p.inventory[h]

\* Each abstract slot represents one device on each host. The port triplet
\* belongs to the common API host. Revoked grants still reserve both.
Available(p) ==
    \A g \in Groups : grants[g] # Empty =>
        /\ grants[g].placement.slot # p.slot
        /\ grants[g].placement.port # p.port

Advertise(h, capability) ==
    /\ inventory' = [inventory EXCEPT ![h] = capability]
    /\ UNCHANGED <<drafts, grants>>

Plan(g, p) ==
    /\ grants[g] = Empty
    /\ drafts[g] = NoPlacement
    /\ Matches(p) /\ Available(p)
    /\ drafts' = [drafts EXCEPT ![g] = p]
    /\ UNCHANGED <<inventory, grants>>

Discard(g) ==
    /\ drafts[g] # NoPlacement
    /\ drafts' = [drafts EXCEPT ![g] = NoPlacement]
    /\ UNCHANGED <<inventory, grants>>

Prepare(g) ==
    /\ grants[g] = Empty /\ drafts[g] # NoPlacement
    /\ Matches(drafts[g])
    /\ Available(drafts[g])
    /\ grants' = [grants EXCEPT ![g] =
        [placement |-> drafts[g], phase |-> "prepared",
         checkedInventory |-> inventory]]
    /\ drafts' = [drafts EXCEPT ![g] = NoPlacement]
    /\ UNCHANGED inventory

Revoke(g) ==
    /\ grants[g] # Empty /\ grants[g].phase = "prepared"
    /\ grants' = [grants EXCEPT ![g].phase = "revoked"]
    /\ UNCHANGED <<inventory, drafts>>

Retire(g) ==
    /\ grants[g] # Empty /\ grants[g].phase = "revoked"
    /\ grants' = [grants EXCEPT ![g] = Empty]
    /\ UNCHANGED <<inventory, drafts>>

Next ==
    \/ \E h \in Hosts, capability \in 0..2 : Advertise(h, capability)
    \/ \E g \in Groups, p \in Placements : Plan(g, p)
    \/ \E g \in Groups : Discard(g) \/ Prepare(g) \/ Revoke(g) \/ Retire(g)

Spec == Init /\ [][Next]_vars

DisjointReservations ==
    \A g, h \in Groups :
        g # h /\ grants[g] # Empty /\ grants[h] # Empty =>
            /\ grants[g].placement.slot # grants[h].placement.slot
            /\ grants[g].placement.port # grants[h].placement.port

InventoryAtPreparation ==
    \A g \in Groups : grants[g] # Empty =>
        \A h \in Hosts :
            /\ grants[g].checkedInventory[h] # 0
            /\ grants[g].checkedInventory[h] = grants[g].placement.inventory[h]
====
