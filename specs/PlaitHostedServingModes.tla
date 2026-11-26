---- MODULE PlaitHostedServingModes ----
EXTENDS Naturals

\* A is the incumbent aggregate. P and D are the split pair. B is a
\* replacement aggregate that can be placed while the split pair is live.
\* A directory can publish the aggregate and the complete split path together.
Agg == {"A", "B"}
Split == {"P", "D"}
Groups == Agg \cup Split
Shapes == {{"A"}, {"B"}, Split, {"A"} \cup Split, {"B"} \cup Split}
Requests == 1..2
NoRoute == {}
Phases == {"free", "preparing", "ready", "active", "draining", "failed", "retired"}

VARIABLES rev, live, epoch, next, phase, work
vars == <<rev, live, epoch, next, phase, work>>

Init ==
    /\ rev = 1
    /\ live = {"A"}
    /\ epoch = 0
    /\ next = NoRoute
    /\ phase = [g \in Groups |-> IF g = "A" THEN "active" ELSE "free"]
    /\ work = [r \in Requests |-> NoRoute]

Stage(target) ==
    /\ next = NoRoute
    /\ target \in Shapes \ {live}
    /\ \A g \in target \ live : phase[g] = "free"
    /\ next' = target
    /\ epoch' = rev
    /\ phase' = [g \in Groups |-> IF g \in target \ live THEN "preparing" ELSE phase[g]]
    /\ UNCHANGED <<rev, live, work>>

Report(g, outcome) ==
    /\ g \in next \ live
    /\ phase[g] = "preparing"
    /\ phase' = [phase EXCEPT ![g] = outcome]
    /\ UNCHANGED <<rev, live, epoch, next, work>>

Publish ==
    /\ next # NoRoute
    /\ epoch = rev
    /\ \A g \in next \ live : phase[g] = "ready"
    /\ rev' = rev + 1
    /\ live' = next
    /\ phase' = [g \in Groups |->
         IF g \in next THEN "active"
         ELSE IF g \in live THEN "draining"
         ELSE phase[g]]
    /\ next' = NoRoute
    /\ epoch' = 0
    /\ UNCHANGED work

Cancel ==
    /\ next # NoRoute
    /\ phase' = [g \in Groups |-> IF g \in next \ live THEN "free" ELSE phase[g]]
    /\ next' = NoRoute
    /\ epoch' = 0
    /\ UNCHANGED <<rev, live, work>>

Admit(r, route) ==
    /\ work[r] = NoRoute
    /\ route \subseteq live
    /\ route \in {{"A"}, {"B"}, Split}
    /\ \A g \in route : phase[g] = "active"
    /\ work' = [work EXCEPT ![r] = route]
    /\ UNCHANGED <<rev, live, epoch, next, phase>>

Finish(r) ==
    /\ work[r] # NoRoute
    /\ work' = [work EXCEPT ![r] = NoRoute]
    /\ UNCHANGED <<rev, live, epoch, next, phase>>

Retire(g) ==
    /\ phase[g] = "draining"
    /\ \A r \in Requests : g \notin work[r]
    /\ phase' = [phase EXCEPT ![g] = "retired"]
    /\ UNCHANGED <<rev, live, epoch, next, work>>

Next == \/ \E target \in Shapes : Stage(target)
        \/ \E g \in Groups, outcome \in {"ready", "failed"} : Report(g, outcome)
        \/ Publish
        \/ Cancel
        \/ \E r \in Requests, route \in SUBSET Groups : Admit(r, route)
        \/ \E r \in Requests : Finish(r)
        \/ \E g \in Groups : Retire(g)

Spec == Init /\ [][Next]_vars
TypeOK == /\ live \in Shapes
          /\ next \in Shapes \cup {NoRoute}
          /\ phase \in [Groups -> Phases]
          /\ work \in [Requests -> SUBSET Groups]
OnlyCommittedRoutes == {g \in Groups : phase[g] = "active"} = live
HeldGroupsRemainLive == \A r \in Requests : \A g \in work[r] :
    phase[g] \in {"active", "draining"}
PublicationWasReady == [][(rev' > rev) => (\A g \in live' \ live : phase[g] = "ready")]_vars
====
