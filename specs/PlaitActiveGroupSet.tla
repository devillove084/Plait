---- MODULE PlaitActiveGroupSet ----
\* Three peer groups share an ordering domain for the actions that touch them.
\* The model starts after preparation of the aggregate and prefill candidates.
\* The decode candidate can become ready independently. Standbys are disabled.
\* An observation authorizes selecting a layout at that point in the order.
\* A request can race with a later fault. The model admits only attempts that
\* reach the required active epochs, and models their drain separately.
EXTENDS Naturals

Aggregate == 1
Prefill == 2
Decode == 3
Groups == {Aggregate, Prefill, Decode}
None == 0
A1 == <<Aggregate, 1>>
A2 == <<Aggregate, 2>>
P1 == <<Prefill, 1>>
D1 == <<Decode, 1>>
Versions == {A1, A2, P1, D1}
Latest == <<2, 1, 1>>

InitialAggregate == <<1, None, None>>
NewAggregate == <<2, None, None>>
Split == <<None, 1, 1>>
PrefillOnly == <<None, 1, None>>
DecodeOnly == <<None, None, 1>>
Layouts == {InitialAggregate, NewAggregate, Split}
Empty == <<None, None, None>>

Change(before, after) == [expected |-> before, target |-> after]
SplitTraffic == Change(InitialAggregate, Split)
ReplaceAggregate == Change(InitialAggregate, NewAggregate)
JoinTraffic == Change(Split, NewAggregate)
GuardSplit == Change(Split, Split)
TryOldAggregate == Change(Split, InitialAggregate)
RepairPrefillOnly == Change(PrefillOnly, NewAggregate)
RepairDecodeOnly == Change(DecodeOnly, NewAggregate)
Commands == {SplitTraffic, ReplaceAggregate, JoinTraffic, GuardSplit, TryOldAggregate,
             RepairPrefillOnly, RepairDecodeOnly}

Live == {"Active", "Draining"}
Answer(command, ok) == [command |-> command, ok |-> ok]
Observation(epochs, desired, routed) == [epochs |-> epochs, desired |-> desired, routed |-> routed]

VARIABLES board, switchAnswer, observed
vars == <<board, switchAnswer, observed>>

Init ==
    /\ board = [active |-> InitialAggregate,
                phase |-> [v \in Versions |->
                    CASE v = A1 -> "Active" [] v = D1 -> "Preparing" [] OTHER -> "Ready"],
                grants |-> Versions, pending |-> {}, completed |-> {}]
    /\ switchAnswer = Answer(Change(Empty, Empty), FALSE)
    /\ observed = Observation(Empty, InitialAggregate, FALSE)

ExpectedMatches(c, g) == board.active[g] = c.expected[g]

CanActivate(c, g) ==
    LET target == c.target[g]
    IN IF target \in {None, c.expected[g]} THEN TRUE
       ELSE target = Latest[g] /\ board.phase[<<g, target>>] = "Ready"

CanSwitch(c) == \A g \in Groups : ExpectedMatches(c, g) /\ CanActivate(c, g)

SetPhase(c, v) ==
    LET g == v[1]
        epoch == v[2]
    IN CASE c.expected[g] = c.target[g] -> board.phase[v]
         [] epoch = c.target[g] -> "Active"
         [] epoch = c.expected[g] -> "Draining"
         [] OTHER -> board.phase[v]

\* Validate every group before changing phases or pointers.
ApplySwitch(c) ==
    LET accepted == CanSwitch(c)
    IN /\ board' = IF accepted THEN [board EXCEPT !.active = c.target,
                       !.phase = [v \in Versions |-> SetPhase(c, v)]] ELSE board
       /\ switchAnswer' = Answer(c, accepted)
       /\ UNCHANGED observed

DecodeReady ==
    /\ board.phase[D1] = "Preparing"
    /\ board' = [board EXCEPT !.phase[D1] = "Ready"]
    /\ UNCHANGED <<switchAnswer, observed>>

\* ObserveActiveSet returns pointers for the requested groups in one state.
\* The router allows the desired layout only when every required epoch matches.
MatchesLayout(epochs, desired) ==
    \A g \in Groups : desired[g] # None => epochs[g] = desired[g]

Observe(desired) ==
    /\ observed' = Observation(board.active, desired, MatchesLayout(board.active, desired))
    /\ UNCHANGED <<board, switchAnswer>>

PlannedVersions(desired) == {v \in Versions : desired[v[1]] = v[2]}

\* Track at most one admission per version. A successfully admitted
\* request retains its epoch while the routing pointer advances.
Admit(desired) ==
    LET required == PlannedVersions(desired)
    IN /\ observed.routed /\ observed.desired = desired
       /\ MatchesLayout(board.active, desired)
       /\ required \subseteq board.grants \ (board.pending \cup board.completed)
       /\ board' = [board EXCEPT !.pending = @ \cup required]
       /\ UNCHANGED <<switchAnswer, observed>>

Finish(v) ==
    /\ v \in board.grants \cap board.pending /\ board.phase[v] \in Live
    /\ board' = [board EXCEPT !.pending = @ \ {v}, !.completed = @ \cup {v}]
    /\ UNCHANGED <<switchAnswer, observed>>

\* ReportFailure can revoke one peer group separately. It can also invalidate
\* a pending candidate. Revocation fences work before releasing its grant.
ReportFailure(v) ==
    /\ board.phase[v] \in Live \cup {"Preparing", "Ready"}
    /\ LET g == v[1] IN
       board' = [board EXCEPT !.active[g] = IF @ = v[2] THEN None ELSE @,
                              !.phase[v] = "Revoked", !.pending = @ \ {v}]
    /\ UNCHANGED <<switchAnswer, observed>>

\* A grant represents all member grants of its version. The integration must
\* wait for a clean drain before issuing their final release.
Release(v) ==
    /\ board.phase[v] \in {"Draining", "Revoked"}
    /\ v \in board.grants \ board.pending
    /\ board' = [board EXCEPT !.grants = @ \ {v}, !.phase[v] = "Retired"]
    /\ UNCHANGED <<switchAnswer, observed>>

Next ==
    \/ \E c \in Commands : ApplySwitch(c)
    \/ DecodeReady
    \/ \E desired \in Layouts : Observe(desired) \/ Admit(desired)
    \/ \E v \in Versions : Finish(v) \/ ReportFailure(v) \/ Release(v)

Spec == Init /\ [][Next]_vars

ActivePointersSound ==
    /\ \A g \in Groups : board.active[g] # None => <<g, board.active[g]>> \in Versions
    /\ \A v \in Versions :
         /\ (board.phase[v] = "Active") = (board.active[v[1]] = v[2])
         /\ (v \in board.grants) = (board.phase[v] # "Retired")

NoMixedLayouts == board.active[Aggregate] # None =>
    board.active[Prefill] = None /\ board.active[Decode] = None

\* Check the transition boundary, including attempts that fail without writes.
\* The postcondition is stated per version, independently of SetPhase.
Eligible(c) ==
    /\ c.expected = board.active
    /\ \A v \in PlannedVersions(c.target) \ PlannedVersions(board.active) :
        v[2] = Latest[v[1]] /\ board.phase[v] = "Ready"

SwitchPostcondition(c) ==
    /\ switchAnswer' = Answer(c, TRUE)
    /\ board'.active = c.target
    /\ board'.grants = board.grants /\ board'.pending = board.pending
    /\ board'.completed = board.completed
    /\ \A v \in Versions :
        LET g == v[1] IN
        IF board.active[g] = c.target[g] THEN board'.phase[v] = board.phase[v]
        ELSE CASE v[2] = board.active[g] -> board'.phase[v] = "Draining"
               [] v[2] = c.target[g] -> board'.phase[v] = "Active"
               [] OTHER -> board'.phase[v] = board.phase[v]

AtomicCompareAndSwitch ==
    \A c \in Commands :
        /\ ENABLED ApplySwitch(c)
        /\ IF Eligible(c) THEN ~ENABLED (ApplySwitch(c) /\ ~SwitchPostcondition(c))
           ELSE ~ENABLED (ApplySwitch(c) /\ (board' # board \/ switchAnswer'.ok))

ObservationIsOneCut ==
    \A desired \in Layouts :
        /\ ENABLED Observe(desired)
        /\ ~ENABLED (Observe(desired) /\ (observed'.epochs # board.active \/ board' # board))

RouterRequiresAllGroups ==
    observed.routed = (\A v \in PlannedVersions(observed.desired) : observed.epochs[v[1]] = v[2])

DrainingKeepsExistingWork ==
    \A v \in board.pending :
        /\ board.phase[v] \in Live /\ v \in board.grants
        /\ ENABLED Finish(v)
        /\ ~ENABLED Release(v)
====
