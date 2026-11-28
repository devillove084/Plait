---- MODULE PlaitPeerScatterGatherPipeline ----
\* The sender and receiver track each direction of the TCP stream separately.
EXTENDS Naturals

CONSTANTS World, Legacy, PreserveFrameOrder, RequireAllSources

Slots == 0..(World - 1)
Links == {link \in Slots \X Slots : link[1] # link[2]}
Peers(slot) == Slots \ {slot}
Chunks(slot) == IF slot = 0 THEN 2 ELSE IF World = 2 THEN 3 ELSE 1

VARIABLES sentScatter, readScatter, reduced, sentGather, readGather
vars == <<sentScatter, readScatter, reduced, sentGather, readGather>>

Init ==
    /\ sentScatter = [link \in Links |-> 0]
    /\ readScatter = [link \in Links |-> 0]
    /\ reduced = [slot \in Slots |-> 0]
    /\ sentGather = [link \in Links |-> 0]
    /\ readGather = [link \in Links |-> 0]

ScatterDone(slot) ==
    \A peer \in Peers(slot) :
        /\ sentScatter[<<slot, peer>>] = Chunks(peer)
        /\ readScatter[<<peer, slot>>] = Chunks(slot)

SendScatter(link) ==
    /\ sentScatter[link] < Chunks(link[2])
    /\ sentScatter' = [sentScatter EXCEPT ![link] = @ + 1]
    /\ UNCHANGED <<readScatter, reduced, sentGather, readGather>>

ReadScatter(link) ==
    /\ readScatter[link] < sentScatter[link]
    /\ readScatter' = [readScatter EXCEPT ![link] = @ + 1]
    /\ UNCHANGED <<sentScatter, reduced, sentGather, readGather>>

Fold(slot) ==
    /\ reduced[slot] < Chunks(slot)
    /\ slot \in Legacy => ScatterDone(slot)
    /\ RequireAllSources =>
        \A peer \in Peers(slot) : reduced[slot] < readScatter[<<peer, slot>>]
    /\ reduced' = [reduced EXCEPT ![slot] = @ + 1]
    /\ UNCHANGED <<sentScatter, readScatter, sentGather, readGather>>

SendGather(link) ==
    /\ sentGather[link] < reduced[link[1]]
    /\ link[1] \in Legacy => ScatterDone(link[1]) /\ reduced[link[1]] = Chunks(link[1])
    /\ PreserveFrameOrder => sentScatter[link] = Chunks(link[2])
    /\ sentGather' = [sentGather EXCEPT ![link] = @ + 1]
    /\ UNCHANGED <<sentScatter, readScatter, reduced, readGather>>

ReadGather(link) ==
    /\ readGather[link] < sentGather[link]
    /\ readScatter[link] = Chunks(link[2])
    /\ link[2] \in Legacy => ScatterDone(link[2])
    /\ readGather' = [readGather EXCEPT ![link] = @ + 1]
    /\ UNCHANGED <<sentScatter, readScatter, reduced, sentGather>>

Next ==
    \/ \E link \in Links :
           SendScatter(link) \/ ReadScatter(link) \/ SendGather(link) \/ ReadGather(link)
    \/ \E slot \in Slots : Fold(slot)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

FrameOrder == \A link \in Links :
    /\ sentGather[link] > 0 => sentScatter[link] = Chunks(link[2])
    /\ readGather[link] > 0 => readScatter[link] = Chunks(link[2])

ReadyPrefix == \A link \in Links :
    /\ sentGather[link] <= reduced[link[1]]
    /\ reduced[link[2]] <= readScatter[link]

Progress == <>(\A link \in Links : readGather[link] = Chunks(link[1]))
====
