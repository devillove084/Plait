---- MODULE PlaitMixedPeerAxes ----
EXTENDS Naturals, FiniteSets

\* One committed mixed epoch. A link bundles its collective and direct
\* TCP lanes. A manifest, logical order, and TP degree bind the handshake.
\* The link records retain the bootstrap history after a handle closes.
CONSTANTS BindLayout, RequireBothAxes, MatchEndedGrant, ShareStaging
CONSTANTS TensorParallel, PipelineParallel
ASSUME /\ TensorParallel \in Nat \ {0, 1}
       /\ PipelineParallel \in Nat \ {0, 1}

Workers == 0..(TensorParallel * PipelineParallel - 1)
Axes == {"TP", "PP"}
Handles == Workers \X Axes
Peers == {pair \in Workers \X Workers : pair[1] < pair[2]}

Stage(slot, width) == slot \div width
TensorSlot(slot, width) == slot % width

Axis(pair) ==
    IF Stage(pair[1], TensorParallel) = Stage(pair[2], TensorParallel) THEN "TP"
    ELSE IF TensorSlot(pair[1], TensorParallel) = TensorSlot(pair[2], TensorParallel)
         THEN "PP" ELSE "none"

LinksOn(worker, axis) ==
    {pair \in Peers : Axis(pair) = axis /\ worker \in {pair[1], pair[2]}}

Tags == {"current", "older epoch", "different grant",
         "different slot order", "different TP degree"}

VARIABLES links, ready, phase, observed, closed, staged
vars == <<links, ready, phase, observed, closed, staged>>

Init ==
    /\ links = [pair \in Peers |-> "absent"]
    /\ ready = {}
    /\ phase = "Preparing"
    /\ observed = {}
    /\ closed = {}
    /\ staged = {}

Connect(pair, tag) ==
    /\ phase = "Preparing" /\ links[pair] = "absent"
    /\ Axis(pair) \in Axes
    /\ BindLayout => tag = "current"
    /\ links' = [links EXCEPT ![pair] = tag]
    /\ UNCHANGED <<ready, phase, observed, closed, staged>>

Linked(worker, axis) ==
    \A pair \in LinksOn(worker, axis) : links[pair] # "absent"

ReportReady(worker) ==
    /\ phase = "Preparing" /\ worker \notin ready
    /\ Linked(worker, "TP")
    /\ RequireBothAxes => Linked(worker, "PP")
    /\ ready' = ready \cup {worker}
    /\ phase' = IF ready' = Workers THEN "Ready" ELSE phase
    /\ UNCHANGED <<links, observed, closed, staged>>

Activate ==
    /\ phase = "Ready"
    /\ phase' = "Active"
    /\ UNCHANGED <<links, ready, observed, closed, staged>>

Revoke ==
    /\ phase = "Active"
    /\ phase' = "Revoked"
    /\ UNCHANGED <<links, ready, observed, closed, staged>>

\* A voter may answer before its current-grant view has caught up. An ended
\* predecessor's grant supplies no evidence about the local communicator.
ObserveEnded(worker, exact) ==
    /\ worker \notin observed
    /\ exact => phase = "Revoked"
    /\ MatchEndedGrant => exact
    /\ observed' = observed \cup {worker}
    /\ UNCHANGED <<links, ready, phase, closed, staged>>

Cancel(handle) ==
    /\ handle[1] \in observed /\ handle \notin closed
    /\ closed' = closed \cup {handle}
    /\ UNCHANGED <<links, ready, phase, observed, staged>>

\* Exercise a single worker's shared byte allowance with one unit per request.
\* Work already staged can retain its bytes after local cancellation.
Reserve(axis) ==
    /\ phase \in {"Active", "Revoked"}
    /\ <<0, axis>> \notin closed /\ axis \notin staged
    /\ ShareStaging => staged = {}
    /\ staged' = staged \cup {axis}
    /\ UNCHANGED <<links, ready, phase, observed, closed>>

Return(axis) ==
    /\ axis \in staged
    /\ staged' = staged \ {axis}
    /\ UNCHANGED <<links, ready, phase, observed, closed>>

Next ==
    \/ Activate \/ Revoke
    \/ \E pair \in Peers, tag \in Tags : Connect(pair, tag)
    \/ \E worker \in Workers :
        \/ ReportReady(worker)
        \/ \E exact \in BOOLEAN : ObserveEnded(worker, exact)
    \/ \E handle \in Handles : Cancel(handle)
    \/ \E axis \in Axes : Reserve(axis) \/ Return(axis)

Spec ==
    /\ Init /\ [][Next]_vars
    /\ \A worker \in Workers : WF_vars(ObserveEnded(worker, TRUE))
    /\ \A handle \in Handles : WF_vars(Cancel(handle))

TypeOK ==
    /\ links \in [Peers -> Tags \cup {"absent"}]
    /\ ready \subseteq Workers
    /\ phase \in {"Preparing", "Ready", "Active", "Revoked"}
    /\ observed \subseteq Workers /\ closed \subseteq Handles
    /\ staged \subseteq Axes

ScopedConnections ==
    \A pair \in Peers : links[pair] # "absent" =>
        /\ Axis(pair) \in Axes
        /\ links[pair] = "current"

ReadyAfterBothAxes ==
    \A worker \in ready : \A axis \in Axes : Linked(worker, axis)

PublishedAfterAllReady == phase # "Preparing" => ready = Workers

ExactGrantCancellation ==
    \A handle \in closed : phase = "Revoked" /\ handle[1] \in observed

StagingCeiling == Cardinality(staged) <= 1

EndedGrantClosesBothAxes == (phase = "Revoked") ~> (closed = Handles)
====
