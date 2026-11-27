---- MODULE PlaitPeerCollective ----
\* Sharded phases follow paired rounds. Replicated inputs can arrive in any order.
\* The scopes share participants but use separate connections and failure state.
\* Tensor chunks contain one integer each to model ownership and materialization.
EXTENDS Naturals

CONSTANTS World, LoseSlotZero, ReplicatedGroups
VARIABLES local, lost, aborted

Groups == {1, 2}
Slots == 0..(World - 1)
First == 1
Independent == 2
Pivot == IF World % 2 = 0 THEN World - 1 ELSE World
vars == <<local, lost, aborted>>

Partner(slot, round) ==
    IF slot = Pivot THEN round
    ELSE IF slot = round THEN Pivot
    ELSE (2 * round + Pivot - slot) % Pivot

PairSchedule == \A slot \in Slots :
    /\ {Partner(slot, r) : r \in 0..(Pivot - 1)} \cap Slots = Slots \ {slot}
    /\ \A r \in 0..(Pivot - 1) :
           LET other == Partner(slot, r)
           IN other \in Slots => Partner(other, r) = slot
    /\ \A r, s \in 0..(Pivot - 1) : r # s => Partner(slot, r) # Partner(slot, s)

Input(group, source, shard) == 100 * group + 10 * source + shard
Expected(group, shard) ==
    World * (100 * group + shard) + 5 * World * (World - 1)

Initial(group, slot) == [phase |-> IF group \in ReplicatedGroups THEN "Replicated" ELSE "Scatter", round |-> 0,
    seen |-> {slot}, sum |-> Input(group, slot, slot),
    gathered |-> [shard \in Slots |->
        IF group \in ReplicatedGroups THEN Input(group, slot, shard) ELSE 0]]

Init ==
    /\ local = [g \in Groups |-> [slot \in Slots |-> Initial(g, slot)]]
    /\ lost = FALSE
    /\ aborted = {}

Alive(g, slot) == ~(lost /\ g = First /\ slot = 0)
Running(g, slot) == g \notin aborted /\ Alive(g, slot)
InRound(state) == state.phase # "Done" /\ state.round < Pivot

Receive(state, peer, payload) ==
    CASE state.phase = "Scatter" ->
        [state EXCEPT !.seen = @ \cup {peer}, !.sum = @ + payload, !.round = @ + 1]
      [] state.phase = "Replicated" ->
        [state EXCEPT !.seen = @ \cup {peer}, !.round = @ + 1,
            !.gathered = [shard \in Slots |-> state.gathered[shard] + payload[shard]]]
      [] OTHER ->
        [state EXCEPT !.seen = @ \cup {peer}, !.gathered[peer] = payload, !.round = @ + 1]

Payload(g, sender, receiver) ==
    CASE local[g][sender].phase = "Scatter" -> Input(g, sender, receiver)
      [] local[g][sender].phase = "Replicated" -> [shard \in Slots |-> Input(g, sender, shard)]
      [] OTHER -> local[g][sender].sum

Exchange(g, a, b) ==
    /\ a < b
    /\ Running(g, a) /\ Running(g, b)
    /\ LET left == local[g][a]
           right == local[g][b]
       IN /\ left.phase = right.phase /\ left.phase # "Done"
          /\ IF left.phase = "Replicated"
                THEN b \notin left.seen /\ a \notin right.seen
                ELSE /\ InRound(left) /\ left.round = right.round
                     /\ b = Partner(a, left.round)
          /\ local' = [local EXCEPT
                ![g][a] = Receive(@, b, Payload(g, b, a)),
                ![g][b] = Receive(@, a, Payload(g, a, b))]
    /\ UNCHANGED <<lost, aborted>>

Idle(g, slot) ==
    /\ Running(g, slot) /\ InRound(local[g][slot])
    /\ local[g][slot].phase # "Replicated"
    /\ Partner(slot, local[g][slot].round) \notin Slots
    /\ local' = [local EXCEPT ![g][slot].round = @ + 1]
    /\ UNCHANGED <<lost, aborted>>

BeginGather(g, slot) ==
    /\ Running(g, slot)
    /\ local[g][slot].phase = "Scatter" /\ local[g][slot].round = Pivot
    /\ local' = [local EXCEPT ![g][slot].phase = "Gather", ![g][slot].round = 0,
            ![g][slot].seen = {slot}, ![g][slot].gathered[slot] = local[g][slot].sum]
    /\ UNCHANGED <<lost, aborted>>

Finish(g, slot) ==
    /\ Running(g, slot)
    /\ LET state == local[g][slot]
       IN \/ state.phase = "Gather" /\ state.round = Pivot
          \/ state.phase = "Replicated" /\ state.seen = Slots
    /\ local' = [local EXCEPT ![g][slot].phase = "Done"]
    /\ UNCHANGED <<lost, aborted>>

Lose ==
    /\ LoseSlotZero /\ ~lost
    /\ lost' = TRUE
    /\ UNCHANGED <<local, aborted>>

Timeout(g, slot) ==
    /\ Running(g, slot)
    /\ LET state == local[g][slot]
       IN IF state.phase = "Replicated"
             THEN \E peer \in Slots \ state.seen : ~Alive(g, peer)
             ELSE /\ InRound(state)
                  /\ LET peer == Partner(slot, state.round)
                     IN peer \in Slots /\ ~Alive(g, peer)
    /\ aborted' = aborted \cup {g}
    /\ UNCHANGED <<local, lost>>

Next ==
    \/ Lose
    \/ \E g \in Groups, slot \in Slots :
           Idle(g, slot) \/ BeginGather(g, slot) \/ Finish(g, slot) \/ Timeout(g, slot)
    \/ \E g \in Groups, a, b \in Slots : Exchange(g, a, b)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

MaterializedShards ==
    \A g \in Groups, slot \in Slots :
        LET state == local[g][slot]
        IN /\ g \notin ReplicatedGroups /\ state.phase # "Scatter" => state.sum = Expected(g, slot)
           /\ g \notin ReplicatedGroups /\ state.phase # "Scatter" =>
               \A shard \in state.seen : state.gathered[shard] = Expected(g, shard)
           /\ state.phase = "Done" =>
               /\ state.seen = Slots
               /\ \A shard \in Slots : state.gathered[shard] = Expected(g, shard)

IsolatedFailure == aborted \subseteq {First}
AllDone(g) == \A slot \in Slots : local[g][slot].phase = "Done"
IndependentProgress == <>AllDone(Independent)
HealthyProgress == ~LoseSlotZero => <>AllDone(First)
====
