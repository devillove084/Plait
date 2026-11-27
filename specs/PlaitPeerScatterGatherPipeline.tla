---- MODULE PlaitPeerScatterGatherPipeline ----
EXTENDS Naturals

CONSTANTS Legacy, PreserveFrameOrder

Slots == {0, 1}
Peer(slot) == 1 - slot
Chunks(slot) == IF slot = 0 THEN 2 ELSE 3

VARIABLES sentScatter, readScatter, reduced, sentGather, readGather
vars == <<sentScatter, readScatter, reduced, sentGather, readGather>>

Init ==
    /\ sentScatter = [slot \in Slots |-> 0]
    /\ readScatter = [slot \in Slots |-> 0]
    /\ reduced = [slot \in Slots |-> 0]
    /\ sentGather = [slot \in Slots |-> 0]
    /\ readGather = [slot \in Slots |-> 0]

ScatterDone(slot) ==
    /\ sentScatter[slot] = Chunks(Peer(slot))
    /\ readScatter[slot] = Chunks(slot)

SendScatter(slot) ==
    /\ sentScatter[slot] < Chunks(Peer(slot))
    /\ sentScatter' = [sentScatter EXCEPT ![slot] = @ + 1]
    /\ UNCHANGED <<readScatter, reduced, sentGather, readGather>>

ReadScatter(slot) ==
    /\ readScatter[slot] < sentScatter[Peer(slot)]
    /\ readScatter' = [readScatter EXCEPT ![slot] = @ + 1]
    /\ UNCHANGED <<sentScatter, reduced, sentGather, readGather>>

Fold(slot) ==
    /\ reduced[slot] < readScatter[slot]
    /\ slot \in Legacy => ScatterDone(slot)
    /\ reduced' = [reduced EXCEPT ![slot] = @ + 1]
    /\ UNCHANGED <<sentScatter, readScatter, sentGather, readGather>>

SendGather(slot) ==
    /\ sentGather[slot] < reduced[slot]
    /\ slot \in Legacy => ScatterDone(slot)
    /\ PreserveFrameOrder => sentScatter[slot] = Chunks(Peer(slot))
    /\ sentGather' = [sentGather EXCEPT ![slot] = @ + 1]
    /\ UNCHANGED <<sentScatter, readScatter, reduced, readGather>>

ReadGather(slot) ==
    /\ readGather[slot] < sentGather[Peer(slot)]
    /\ readScatter[slot] = Chunks(slot)
    /\ slot \in Legacy => ScatterDone(slot)
    /\ readGather' = [readGather EXCEPT ![slot] = @ + 1]
    /\ UNCHANGED <<sentScatter, readScatter, reduced, sentGather>>

Next == \E slot \in Slots : SendScatter(slot) \/ ReadScatter(slot) \/ Fold(slot)
                           \/ SendGather(slot) \/ ReadGather(slot)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

FrameOrder == \A slot \in Slots :
    /\ sentGather[slot] > 0 => sentScatter[slot] = Chunks(Peer(slot))
    /\ readGather[slot] > 0 => readScatter[slot] = Chunks(slot)

ReadyPrefix == \A slot \in Slots :
    /\ sentGather[slot] <= reduced[slot]
    /\ reduced[slot] <= readScatter[slot]

Progress == <>(\A slot \in Slots : readGather[slot] = Chunks(Peer(slot)))
====
