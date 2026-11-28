---- MODULE PlaitHostedAllReduceProfile ----
EXTENDS Naturals

\* The qualified profile stores the option symbolically as part of its name.
\* Scope is one hosted role moving from the default recipe to a candidate.
CONSTANT ResolveByName
Hosts == {"A", "B"}
Default == <<>>
Options == {Default, <<0>>, <<393216>>}
NoLaunch == <<0, 0>>
NoCandidate == [epoch |-> 0, profile |-> <<>>]
Profile(option) ==
    IF option = Default THEN <<"model">> ELSE <<"model", "all-reduce", option[1]>>

VARIABLES inventory, candidate, launch, active, stopped
vars == <<inventory, candidate, launch, active, stopped>>

Init ==
    /\ inventory \in [Hosts -> SUBSET {Profile(o) : o \in Options}]
    /\ candidate = NoCandidate
    /\ launch = [host \in Hosts |-> NoLaunch]
    /\ active = [epoch |-> 1, profile |-> Profile(Default)]
    /\ stopped = FALSE

Prepare(option) ==
    /\ candidate = NoCandidate /\ ~stopped /\ active.epoch = 1
    /\ \A host \in Hosts : Profile(option) \in inventory[host]
    /\ candidate' = [epoch |-> 2, profile |-> Profile(option)]
    /\ UNCHANGED <<inventory, launch, active, stopped>>

Start(host, option) ==
    /\ candidate # NoCandidate /\ ~stopped /\ active.epoch = 1
    /\ launch[host] = NoLaunch
    /\ Profile(option) \in inventory[host]
    /\ ResolveByName => Profile(option) = candidate.profile
    /\ launch' = [launch EXCEPT ![host] = option]
    /\ UNCHANGED <<inventory, candidate, active, stopped>>

\* Both peers must finish the startup handshake before the group may serve.
Publish ==
    /\ candidate # NoCandidate /\ ~stopped /\ active.epoch = 1
    /\ \A host \in Hosts : launch[host] # NoLaunch
    /\ \A a, b \in Hosts : launch[a] = launch[b]
    /\ active' = candidate
    /\ UNCHANGED <<inventory, candidate, launch, stopped>>

Abort ==
    /\ candidate # NoCandidate /\ ~stopped /\ active.epoch = 1
    /\ stopped' = TRUE
    /\ launch' = [host \in Hosts |-> NoLaunch]
    /\ UNCHANGED <<inventory, active, candidate>>

Next ==
    \/ \E o \in Options : Prepare(o)
    \/ \E h \in Hosts, o \in Options : Start(h, o)
    \/ Publish
    \/ Abort
Spec == Init /\ [][Next]_vars

QualifiedOptionsAreDistinct ==
    \A left, right \in Options : (Profile(left) = Profile(right)) = (left = right)

PreparedHostsAdvertiseProfile ==
    candidate # NoCandidate => \A h \in Hosts : candidate.profile \in inventory[h]

PublishedBudgetMatchesRecipe ==
    active.epoch = 2 =>
        /\ ~stopped
        /\ \A h \in Hosts :
            /\ launch[h] # NoLaunch
            /\ Profile(launch[h]) = active.profile

AbortPreservesIncumbent == stopped => active.epoch = 1 /\ active.profile = Profile(Default)
====
