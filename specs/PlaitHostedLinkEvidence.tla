---- MODULE PlaitHostedLinkEvidence ----
EXTENDS Naturals

CONSTANTS EnforceTpRttBudget, UseTargetTpWidth, HasDefaultTpProfile

\* A planner reads a directory snapshot and one sample in each direction.
\* Published samples can outlive a receiver or sender session.
Hosts == {"a", "b"}
Other(h) == CHOOSE peer \in Hosts : peer # h
MaxAge == 2
TpWidths == {2, 4, 8}
NoProfile == [rounds |-> 0, wait |-> 0, bytes |-> 0]
DefaultProfile == [rounds |-> 1, wait |-> 2, bytes |-> 3]
ProfileFor(width) ==
    CASE width = 2 -> [rounds |-> 1, wait |-> 3, bytes |-> 2]
      [] width = 4 -> [rounds |-> 2, wait |-> 4, bytes |-> 5]
      [] OTHER -> IF HasDefaultTpProfile THEN DefaultProfile ELSE NoProfile
\* A multihost search prefix can already have two ranks even for a wider target.
SearchProfile(width) == ProfileFor(IF UseTargetTpWidth THEN width ELSE 2)

SampleRtts == {1, 2, 3}
NoSample == [time |-> 0, source |-> 0, target |-> 0, rtt |-> 0]
NoDecision == [time |-> 0, directory |-> [h \in Hosts |-> 0],
               used |-> [h \in Hosts |-> NoSample],
               width |-> 0, profile |-> NoProfile]

VARIABLES clock, sessions, samples, decision
vars == <<clock, sessions, samples, decision>>

Init ==
    /\ clock = 0
    /\ sessions = [h \in Hosts |-> 1]
    /\ samples = [h \in Hosts |-> NoSample]
    /\ decision = NoDecision

Tick ==
    /\ clock < 3 /\ clock' = clock + 1
    /\ UNCHANGED <<sessions, samples, decision>>

Restart(h) ==
    /\ sessions[h] < 2
    /\ sessions' = [sessions EXCEPT ![h] = @ + 1]
    /\ UNCHANGED <<clock, samples, decision>>

Measure(h, rtt) ==
    /\ samples' = [samples EXCEPT ![h] =
         [time |-> clock, source |-> sessions[h], target |-> sessions[Other(h)], rtt |-> rtt]]
    /\ UNCHANGED <<clock, sessions, decision>>

Valid(h) ==
    /\ samples[h].source = sessions[h] /\ samples[h].target = sessions[Other(h)]
    /\ clock - samples[h].time < MaxAge

WithinRttBudget(observed, profile) ==
    \A h \in Hosts : observed[h].rtt * profile.rounds <= profile.wait

Select(width) ==
    LET profile == SearchProfile(width)
    IN /\ decision = NoDecision
       /\ profile # NoProfile
       /\ \A h \in Hosts : Valid(h)
       /\ EnforceTpRttBudget => WithinRttBudget(samples, profile)
       /\ decision' = [time |-> clock, directory |-> sessions, used |-> samples,
                        width |-> width, profile |-> profile]
       /\ UNCHANGED <<clock, sessions, samples>>

Next == \/ Tick
        \/ \E width \in TpWidths : Select(width)
        \/ \E h \in Hosts : Restart(h) \/ \E rtt \in SampleRtts : Measure(h, rtt)
Spec == Init /\ [][Next]_vars

FreshDirections == decision # NoDecision =>
    \A h \in Hosts :
       /\ decision.used[h].time <= decision.time
       /\ decision.time - decision.used[h].time < MaxAge

MatchingSessions == decision # NoDecision =>
    \A h \in Hosts :
       /\ decision.used[h].source = decision.directory[h]
       /\ decision.used[h].target = decision.directory[Other(h)]

UsesTargetTpProfile == decision # NoDecision =>
    /\ ProfileFor(decision.width) # NoProfile
    /\ decision.profile = ProfileFor(decision.width)

BoundedTpWait == decision # NoDecision =>
    WithinRttBudget(decision.used, ProfileFor(decision.width))
====
