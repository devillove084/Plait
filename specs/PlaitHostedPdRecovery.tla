---- MODULE PlaitHostedPdRecovery ----
EXTENDS Naturals

Groups == {"p", "d1", "d2"}
Paths == {<<"p", "d1">>, <<"p", "d2">>}

VARIABLES revision, lost, spares, releaseProof, grants, stage, epochs, published
vars == <<revision, lost, spares, releaseProof, grants, stage, epochs, published>>

Idle == [base |-> 0, replace |-> {}, ready |-> {}]
Unpublished == [base |-> 0, ready |-> {}]

Init ==
    /\ revision = 1 /\ stage = Idle /\ published = Unpublished
    /\ lost = {} /\ spares = {} /\ releaseProof = {} /\ grants = Groups
    /\ epochs = [g \in Groups |-> 1]

Fail(g) ==
    /\ published = Unpublished /\ g \notin lost
    /\ lost' = lost \cup {g}
    /\ UNCHANGED <<revision, spares, releaseProof, grants, stage, epochs, published>>

RegisterSpare(g) ==
    /\ g \notin spares
    /\ spares' = spares \cup {g}
    /\ UNCHANGED <<revision, lost, releaseProof, grants, stage, epochs, published>>

VerifyOldRelease(g) ==
    /\ g \in lost \ releaseProof
    /\ releaseProof' = releaseProof \cup {g} /\ grants' = grants \ {g}
    /\ UNCHANGED <<revision, lost, spares, stage, epochs, published>>

\* The inputs are placement and committed faults. Ingress may be absent.
Prepare ==
    /\ published = Unpublished /\ stage = Idle
    /\ lost # {} /\ lost \subseteq spares
    /\ stage' = [base |-> revision, replace |-> lost, ready |-> {}]
    /\ UNCHANGED <<revision, lost, spares, releaseProof, grants, epochs, published>>

Ready(g) ==
    /\ g \in stage.replace \ stage.ready
    /\ stage' = [stage EXCEPT !.ready = @ \cup {g}]
    /\ UNCHANGED <<revision, lost, spares, releaseProof, grants, epochs, published>>

Supersede ==
    /\ published = Unpublished /\ revision = 1
    /\ revision' = 2
    /\ UNCHANGED <<lost, spares, releaseProof, grants, stage, epochs, published>>

Cancel ==
    /\ stage # Idle /\ (stage.base # revision \/ stage.replace # lost)
    /\ stage' = Idle
    /\ UNCHANGED <<revision, lost, spares, releaseProof, grants, epochs, published>>

Publish ==
    /\ stage # Idle /\ stage.base = revision
    /\ stage.ready = lost /\ stage.replace = lost
    /\ epochs' = [g \in Groups |-> IF g \in stage.replace THEN 2 ELSE epochs[g]]
    /\ published' = [base |-> stage.base, ready |-> stage.ready]
    /\ revision' = revision + 1 /\ stage' = Idle
    /\ UNCHANGED <<lost, spares, releaseProof, grants>>

Next == Prepare \/ Supersede \/ Cancel \/ Publish
        \/ \E g \in Groups : Fail(g) \/ RegisterSpare(g) \/ VerifyOldRelease(g) \/ Ready(g)

Spec == Init /\ [][Next]_vars

Routable(g) == g \notin lost \/ epochs[g] = 2
LivePaths == {path \in Paths : Routable(path[1]) /\ Routable(path[2])}

HealthyVersionsPreserved == \A g \in Groups \ lost : epochs[g] = 1
OldReleaseRequiresProof == releaseProof \subseteq lost /\ grants = Groups \ releaseProof

Cutover ==
    IF published = Unpublished
    THEN \A g \in Groups : epochs[g] = 1
    ELSE /\ revision = published.base + 1 /\ published.ready = lost
         /\ \A g \in lost : epochs[g] = 2 /\ g \in spares
         /\ LivePaths = Paths
====
