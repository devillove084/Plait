---- MODULE PlaitHostedBudgetSelection ----
EXTENDS Naturals

\* One networked Prefill TP slot has two measured profiles. Decode is local.
\* The two directed links share a bounded freshness window.
CONSTANT UseCommittedQuote
Default == "default"
Capped == "capped"
Options == {Default, Capped}
Hosts == {"left", "right"}
Rtts == {50, 100}
Directions == {"outbound", "inbound"}
Traffic(option) == IF option = Default THEN 125 ELSE 250
Rounds(option) == IF option = Default THEN 2 ELSE 1
Fits(option, links) == \A d \in Directions : links.rtt[d] * Rounds(option) <= 150
Fresh(links, at) == at >= links.at /\ at - links.at < 2

Binding(epoch, option) == [epoch |-> epoch, option |-> option]
Empty == Binding(0, Default)
Observation(at, rtt) == [at |-> at, rtt |-> rtt]
InitialLinks == Observation(0, [d \in Directions |-> 50])
NoCandidate == [binding |-> Empty, decision |-> 0, observation |-> InitialLinks]
NoQuote == [binding |-> Empty, bytes |-> 0]

VARIABLES installed, now, sample, active, candidate, quote
vars == <<installed, now, sample, active, candidate, quote>>

Choices ==
    {o \in Options : /\ \A host \in Hosts : o \in installed[host]
                     /\ Fits(o, sample)}
Best == {o \in Choices : \A other \in Choices : Traffic(o) <= Traffic(other)}

Init ==
    /\ installed \in [Hosts -> {{Default}, Options}]
    /\ now = 0 /\ sample = InitialLinks
    /\ active = Binding(1, Default)
    /\ candidate = NoCandidate /\ quote = NoQuote

Refresh(rtt) ==
    /\ sample' = Observation(now, rtt)
    /\ UNCHANGED <<installed, now, active, candidate, quote>>

Tick ==
    /\ now < 3 /\ now' = now + 1
    /\ UNCHANGED <<installed, sample, active, candidate, quote>>

PriceIncumbent ==
    /\ Fresh(sample, now)
    /\ LET option == IF UseCommittedQuote \/ Best = {} THEN active.option
                     ELSE CHOOSE o \in Best : TRUE
       IN quote' = [binding |-> active, bytes |-> Traffic(option)]
    /\ UNCHANGED <<installed, now, sample, active, candidate>>

Prepare(option) ==
    /\ active.epoch = 1 /\ candidate = NoCandidate /\ Fresh(sample, now)
    /\ option \in Best
    /\ candidate' = [binding |-> Binding(2, option), decision |-> now,
                     observation |-> sample]
    /\ UNCHANGED <<installed, now, sample, active, quote>>

Activate ==
    /\ candidate # NoCandidate
    /\ active' = candidate.binding /\ candidate' = NoCandidate
    /\ quote' = NoQuote
    /\ UNCHANGED <<installed, now, sample>>

Next ==
    \/ \E rtt \in [Directions -> Rtts] : Refresh(rtt)
    \/ \E option \in Options : Prepare(option)
    \/ Tick \/ PriceIncumbent \/ Activate

Spec == Init /\ [][Next]_vars

CandidateUsesCurrentEvidence ==
    candidate # NoCandidate =>
        /\ Fresh(candidate.observation, candidate.decision)
        /\ Fits(candidate.binding.option, candidate.observation)
        /\ \A host \in Hosts : candidate.binding.option \in installed[host]

IncumbentChargedForCommittedProfile ==
    quote # NoQuote =>
        /\ quote.binding = active
        /\ quote.bytes = Traffic(active.option)
====
