---- MODULE PlaitCompletionReplay ----
EXTENDS Naturals, Sequences

\* A value represents an output token and its corresponding text.
Original == << <<1, "a">>, <<2, "b">>, <<3, "c">> >>
Candidates == {Original,
    << <<1, "a">>, <<2, "b">>, <<4, "d">> >>,
    << <<1, "x">>, <<2, "b">>, <<3, "c">> >>,
    << <<9, "a">>, <<2, "b">>, <<3, "c">> >>,
    << <<1, "a">> >>}

VARIABLES active, attempt, output, cursor, visible, phase
vars == <<active, attempt, output, cursor, visible, phase>>

Init ==
    /\ active = 1 /\ attempt = 1 /\ output = Original
    /\ cursor = 0 /\ visible = <<>> /\ phase = "Reading"

Activate ==
    /\ active < 3 /\ active' = active + 1
    /\ UNCHANGED <<attempt, output, cursor, visible, phase>>

Interrupt ==
    /\ phase = "Reading" /\ phase' = "Waiting"
    /\ UNCHANGED <<active, attempt, output, cursor, visible>>

Replay(candidate) ==
    /\ phase = "Waiting" /\ active > attempt
    /\ attempt' = active /\ output' = candidate
    /\ cursor' = 0 /\ phase' = "Reading"
    /\ UNCHANGED <<active, visible>>

Match ==
    /\ phase = "Reading"
    /\ cursor < Len(visible) /\ cursor < Len(output)
    /\ output[cursor + 1] = visible[cursor + 1].value
    /\ cursor' = cursor + 1
    /\ UNCHANGED <<active, attempt, output, visible, phase>>

Publish ==
    /\ phase = "Reading"
    /\ cursor = Len(visible) /\ cursor < Len(output)
    /\ visible' = Append(visible, [position |-> cursor + 1, value |-> output[cursor + 1]])
    /\ cursor' = cursor + 1
    /\ UNCHANGED <<active, attempt, output, phase>>

Reject ==
    /\ phase = "Reading" /\ cursor < Len(visible)
    /\ (IF cursor = Len(output) THEN TRUE
        ELSE output[cursor + 1] # visible[cursor + 1].value)
    /\ phase' = "Failed"
    /\ UNCHANGED <<active, attempt, output, cursor, visible>>

Finish ==
    /\ phase = "Reading" /\ cursor = Len(output) /\ cursor = Len(visible)
    /\ phase' = "Done"
    /\ UNCHANGED <<active, attempt, output, cursor, visible>>

Next == Activate \/ Interrupt \/ (\E c \in Candidates : Replay(c))
    \/ Match \/ Publish \/ Reject \/ Finish
Spec == Init /\ [][Next]_vars

PositionsOnce == \A i \in DOMAIN visible : visible[i].position = i
CheckedPrefix ==
    phase \in {"Reading", "Done"} =>
        /\ cursor <= Len(visible)
        /\ \A i \in 1..cursor : visible[i].value = output[i]
TerminalComplete == phase = "Done" => Len(visible) = Len(output)
PrefixNeverChanges == [][
    Len(visible') >= Len(visible) /\ \A i \in DOMAIN visible : visible'[i] = visible[i]
]_vars
====
