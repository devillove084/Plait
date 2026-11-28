---- MODULE PlaitReplanLinkWindows ----
\* One proposal needs the two directed paths between its host endpoints.
\* PlaitGroupReplan checks proposal identity and reservation arbitration.
EXTENDS Naturals

CONSTANTS RequireNewSample, RequireEveryDirection, MatchProbeSession

Paths == 1..2
Empty == [session |-> 0, samples |-> [path \in Paths |-> 0]]
NoProof == [first |-> Empty, last |-> Empty]

VARIABLES session, samples, arrived, seen, prepared
vars == <<session, samples, arrived, seen, prepared>>

Current == [session |-> session, samples |-> samples]
Advanced == {path \in Paths : samples[path] > seen.samples[path]}

Init ==
    /\ session = 1
    /\ samples = Empty.samples /\ arrived = {}
    /\ seen = Empty /\ prepared = NoProof

NewSample(path) ==
    /\ samples[path] < 3
    /\ samples' = [samples EXCEPT ![path] = @ + 1]
    /\ arrived' = arrived \cup {path}
    /\ UNCHANGED <<session, seen, prepared>>

ReplaceProbe ==
    /\ session = 1
    /\ session' = 2 /\ arrived' = {}
    /\ UNCHANGED <<samples, seen, prepared>>

Poll ==
    /\ prepared = NoProof
    /\ IF arrived # Paths
       THEN /\ seen' = Empty
            /\ UNCHANGED prepared
       ELSE IF seen = Empty \/ (MatchProbeSession /\ seen.session # session)
            THEN /\ seen' = Current
                 /\ UNCHANGED prepared
            ELSE /\ UNCHANGED seen
                 /\ prepared' =
                        IF ~RequireNewSample
                           \/ (RequireEveryDirection /\ Advanced = Paths)
                           \/ (~RequireEveryDirection /\ Advanced # {})
                        THEN [first |-> seen, last |-> Current]
                        ELSE prepared
    /\ UNCHANGED <<session, samples, arrived>>

Next == (\E path \in Paths : NewSample(path)) \/ ReplaceProbe \/ Poll
Spec == Init /\ [][Next]_vars

SameProbe ==
    prepared # NoProof => prepared.first.session = prepared.last.session
DistinctDirections ==
    prepared # NoProof =>
        \A path \in Paths : prepared.last.samples[path] > prepared.first.samples[path]
====
