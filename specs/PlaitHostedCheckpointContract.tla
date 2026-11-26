---- MODULE PlaitHostedCheckpointContract ----
EXTENDS FiniteSets

\* A group version carries its launch binding with it between aggregate,
\* split and mixed directories. "plain" is the legacy aggregate launch.
A == "aggregate"
P == "prefill"
D == "decode"
Split == {P, D}
Groups == {A, P, D}
Shapes == {{A}, Split, Groups}
Stores == {"store1", "store2"}
Contracts == Stores \cup {"plain"}

VARIABLES live, directory, bindings, pending, announced, ready
vars == <<live, directory, bindings, pending, announced, ready>>

Init ==
    /\ live = {}
    /\ directory = "plain"
    /\ bindings = [g \in Groups |-> "absent"]
    /\ pending = {}
    /\ announced = "plain"
    /\ ready = {}

Compatible(g, contract, advertised) ==
    \/ g = A /\ contract = "plain"
    \/ contract \in Stores /\ contract = advertised

Stage(target, contract, proposed) ==
    /\ pending = {}
    /\ target \in Shapes
    /\ (contract \in Stores) = (\E g \in target : proposed[g] \in Stores)
    /\ \A g \in target :
           /\ Compatible(g, proposed[g], contract)
           /\ bindings[g] # "absent" => g \in live /\ proposed[g] = bindings[g]
    /\ \A g \in Groups \ target : proposed[g] = bindings[g]
    /\ pending' = target
    /\ announced' = contract
    /\ bindings' = proposed
    /\ ready' = target \cap live
    /\ UNCHANGED <<live, directory>>

Report(g) ==
    /\ g \in pending \ ready
    /\ ready' = ready \cup {g}
    /\ UNCHANGED <<live, directory, bindings, pending, announced>>

Publish ==
    /\ pending # {}
    /\ ready = pending
    /\ live' = pending
    /\ directory' = announced
    /\ pending' = {}
    /\ announced' = "plain"
    /\ ready' = {}
    /\ UNCHANGED bindings

Next == \/ \E target \in Shapes, contract \in Contracts,
              proposed \in [Groups -> Contracts \cup {"absent"}] :
            Stage(target, contract, proposed)
        \/ \E g \in Groups : Report(g)
        \/ Publish

Spec == Init /\ [][Next]_vars
PublishedContract == \A g \in live : Compatible(g, bindings[g], directory)
ImmutableBinding == [][\A g \in Groups : bindings[g] # "absent" =>
    bindings'[g] = bindings[g]]_vars
PublishAfterReady == [][(live' # live) => live' = ready]_vars
====
