---- MODULE PlaitPlannedStreamHandoff ----
EXTENDS Naturals

\* The source can compute ahead. It remains live while the replacement warms.
\* The ingress transfers the last visible position and grants one connection
\* authority to publish the suffix. A capability bit selects resume or replay.
Limit == 4
VARIABLES routed, ready, resume, produced, pages, visible, checkpoint, imported, phase
vars == <<routed, ready, resume, produced, pages, visible, checkpoint, imported, phase>>

Init ==
    /\ routed = TRUE
    /\ ready = FALSE
    /\ resume \in BOOLEAN
    /\ produced = 0
    /\ pages = 0
    /\ visible = 0
    /\ checkpoint = 0
    /\ imported = 0
    /\ phase = "source"

Warm ==
    /\ ~ready
    /\ ready' = TRUE
    /\ UNCHANGED <<routed, resume, produced, pages, visible, checkpoint, imported, phase>>

RemoveSource ==
    /\ routed
    /\ routed' = FALSE
    /\ UNCHANGED <<ready, resume, produced, pages, visible, checkpoint, imported, phase>>

Compute ==
    /\ phase \in {"source", "cut", "target"}
    /\ produced < Limit
    /\ produced' = produced + 1
    /\ UNCHANGED <<routed, ready, resume, pages, visible, checkpoint, imported, phase>>

Store ==
    /\ pages < produced
    /\ pages' = pages + 1
    /\ UNCHANGED <<routed, ready, resume, produced, visible, checkpoint, imported, phase>>

EmitSource ==
    /\ phase = "source"
    /\ visible < produced
    /\ visible' = visible + 1
    /\ UNCHANGED <<routed, ready, resume, produced, pages, checkpoint, imported, phase>>

Cut ==
    /\ phase = "source"
    /\ 0 < visible /\ visible < Limit
    /\ ~routed
    /\ ready
    /\ checkpoint' = visible
    /\ phase' = "cut"
    /\ UNCHANGED <<routed, ready, resume, produced, pages, visible, imported>>

Verify(ack, matches, hit) ==
    /\ phase = "cut"
    /\ hit <= pages /\ hit <= checkpoint
    /\ imported' = IF resume THEN hit ELSE 0
    /\ phase' = IF matches /\ (~resume \/ checkpoint = ack) THEN "target" ELSE "failed"
    /\ UNCHANGED <<routed, ready, resume, produced, pages, visible, checkpoint>>

EmitTarget ==
    /\ phase = "target"
    /\ visible >= checkpoint /\ visible < Limit
    /\ visible' = visible + 1
    /\ UNCHANGED <<routed, ready, resume, produced, pages, checkpoint, imported, phase>>

Finish ==
    /\ phase \in {"source", "target"}
    /\ visible = Limit
    /\ phase' = "done"
    /\ UNCHANGED <<routed, ready, resume, produced, pages, visible, checkpoint, imported>>

Next == Warm \/ RemoveSource \/ Compute \/ Store \/ EmitSource \/ Cut
    \/ (\E ack \in 0..Limit, matches \in BOOLEAN, hit \in 0..Limit : Verify(ack, matches, hit))
    \/ EmitTarget \/ Finish
Spec == Init /\ [][Next]_vars

PreservesCut == checkpoint <= visible
ImportWithinVisible == imported <= checkpoint /\ imported <= pages
TerminalOnceComplete == phase = "done" => visible = Limit
CutAtVisible == [][(phase = "source" /\ phase' = "cut") =>
    (ready /\ ~routed /\ checkpoint' = visible)]_vars
OneOutputOwner == [][(visible' > visible) =>
    (visible' = visible + 1 /\ phase \in {"source", "target"})]_vars
====
