---- MODULE PlaitHostedLinkEvidence ----
EXTENDS Naturals

CONSTANT EnforceTpRttBudget

\* A planner reads a directory snapshot and one sample in each direction.
\* Published samples can outlive a receiver or sender session.
Hosts == {"a", "b"}
Other(h) == CHOOSE peer \in Hosts : peer # h
MaxAge == 2
SerialRounds == 2
TpWaitBudget == 4
SampleRtts == {1, 2, 3}
NoSample == [time |-> 0, source |-> 0, target |-> 0, rtt |-> 0]
NoDecision == [time |-> 0, directory |-> [h \in Hosts |-> 0],
               used |-> [h \in Hosts |-> NoSample]]

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

WithinRttBudget(observed) ==
    \A h \in Hosts : observed[h].rtt * SerialRounds <= TpWaitBudget

Select ==
    /\ decision = NoDecision
    /\ \A h \in Hosts : Valid(h)
    /\ EnforceTpRttBudget => WithinRttBudget(samples)
    /\ decision' = [time |-> clock, directory |-> sessions, used |-> samples]
    /\ UNCHANGED <<clock, sessions, samples>>

Next == Tick \/ Select \/ \E h \in Hosts : Restart(h) \/ \E rtt \in SampleRtts : Measure(h, rtt)
Spec == Init /\ [][Next]_vars

FreshDirections == decision # NoDecision =>
    \A h \in Hosts :
       /\ decision.used[h].time <= decision.time
       /\ decision.time - decision.used[h].time < MaxAge

MatchingSessions == decision # NoDecision =>
    \A h \in Hosts :
       /\ decision.used[h].source = decision.directory[h]
       /\ decision.used[h].target = decision.directory[Other(h)]

BoundedTpWait == decision # NoDecision => WithinRttBudget(decision.used)
====
