---- MODULE PlaitAbortFailedSet ----
\* One ordered control state machine. A successful two-group preparation is
\* already recorded. Its first version owns two devices on the same host.
\* A subsequent preparation can reuse one released device while the other is
\* still held. Abort changes candidate phases, and only exact Release returns
\* resources. Standby activation and replica recovery are abstracted.
EXTENDS Naturals, Sequences

Prefill == 1
Decode == 2
Aggregate == 3
Groups == {Prefill, Decode, Aggregate}
P1 == <<Prefill, 1>>
D1 == <<Decode, 1>>
A1 == <<Aggregate, 1>>
P2 == <<Prefill, 2>>
Versions == {P1, D1, A1, P2}
Devices == 1..4
Members(v) == CASE v = P1 -> {1, 2} [] v = D1 -> {3} [] v = A1 -> {4} [] OTHER -> {1}
NoVersion == <<0, 0>>
Inactive == 0
Candidates == {"Preparing", "Ready"}
Terminal == {"Revoked", "Retired"}

Charge(bytes, pinned) == [bytes |-> bytes, pinned |-> pinned]
Add(a, b) == Charge(a.bytes + b.bytes, a.pinned + b.pinned)
Subtract(a, b) == Charge(a.bytes - b.bytes, a.pinned - b.pinned)
Demand(v) == IF v = P1 THEN Charge(4, 2) ELSE Charge(2, 1)

Grant(version, device, fence) == [version |-> version, device |-> device, incarnation |-> 1, fence |-> fence]
NoGrant == Grant(NoVersion, 0, 0)
Lease(v, d) == Grant(v, d, v[2])

OldPrepare == 11
NewPrepare == 12
FailedPrepare == 13
OtherOperation == 14
PrepareIds == {OldPrepare, NewPrepare, FailedPrepare, OtherOperation}
OldVersions == <<P1, D1>>
NewVersions == <<P2>>
PrepareReceipt(kind, versions) == [kind |-> kind, versions |-> versions]
NoPrepare == PrepareReceipt("Missing", <<>>)
Prepared(versions) == PrepareReceipt("PrepareSet", versions)
Abort(id, prepare, versions) == [id |-> id, prepare |-> prepare, versions |-> versions]
OldAbort == Abort(1, OldPrepare, OldVersions)
NewIdOldAbort == Abort(2, OldPrepare, OldVersions)
AbortRequests == {OldAbort, NewIdOldAbort, Abort(1, NewPrepare, NewVersions),
    Abort(1, NewPrepare, OldVersions), Abort(1, OldPrepare, <<P2, D1>>),
    Abort(1, OldPrepare, <<D1>>), Abort(1, OldPrepare, <<D1, P1>>),
    Abort(1, FailedPrepare, OldVersions), Abort(1, OtherOperation, OldVersions)}
AbortReceipt(command, outcome) == [command |-> command, outcome |-> outcome]
NoAbortReceipt == AbortReceipt(Abort(0, 0, <<>>), "Pending")
Aborted == "Aborted"
Rejected == "Rejected"
SequenceMembers(versions) == {versions[i] : i \in DOMAIN versions}

VARIABLES state, receipts, reply
vars == <<state, receipts, reply>>

Init ==
    /\ state = [phase |-> [v \in Versions |-> CASE v = A1 -> "Active" [] v = P2 -> "Absent" [] OTHER -> "Preparing"],
                 latest |-> <<1, 1, 1>>, active |-> <<Inactive, Inactive, 1>>, activated |-> {A1},
                 released |-> [v \in Versions |-> {}],
                 owner |-> <<Lease(P1, 1), Lease(P1, 2), Lease(D1, 3), Lease(A1, 4)>>,
                 fence |-> <<1, 1, 1, 1>>, reserved |-> Charge(8, 4),
                 origin |-> [v \in Versions |-> IF v \in {P1, D1} THEN OldPrepare ELSE 0],
                 prepares |-> [id \in PrepareIds |->
                     CASE id = OldPrepare -> Prepared(OldVersions)
                       [] id = FailedPrepare -> PrepareReceipt("Failed", <<>>)
                       [] id = OtherOperation -> PrepareReceipt("Other", OldVersions)
                       [] OTHER -> NoPrepare]]
    /\ receipts = [id \in 1..2 |-> NoAbortReceipt] /\ reply = NoAbortReceipt

PrepareSuccessor ==
    /\ state.phase[P2] = "Absent" /\ state.phase[P1] \in Terminal /\ state.owner[1] = NoGrant
    /\ state' = [state EXCEPT !.phase[P2] = "Preparing", !.latest[Prefill] = 2,
                  !.origin[P2] = NewPrepare, !.prepares[NewPrepare] = Prepared(NewVersions),
                  !.owner[1] = Lease(P2, 1), !.fence[1] = @ + 1, !.reserved = Add(@, Demand(P2))]
    /\ UNCHANGED <<receipts, reply>>

Ready(v) ==
    /\ state.phase[v] = "Preparing"
    /\ state' = [state EXCEPT !.phase[v] = "Ready"]
    /\ UNCHANGED <<receipts, reply>>

CanActivate(v) == state.phase[v] = "Ready" /\ state.latest[v[1]] = v[2]

Activate(v) ==
    /\ CanActivate(v) /\ state.active[v[1]] = Inactive
    /\ state' = [state EXCEPT !.phase[v] = "Active", !.active[v[1]] = v[2], !.activated = @ \cup {v}]
    /\ UNCHANGED <<receipts, reply>>

SwitchActiveSet ==
    /\ state.active = <<Inactive, Inactive, 1>> /\ CanActivate(P1) /\ CanActivate(D1)
    /\ state' = [state EXCEPT !.phase[P1] = "Active", !.phase[D1] = "Active", !.phase[A1] = "Draining",
                  !.active = <<1, 1, Inactive>>, !.activated = @ \cup {P1, D1}]
    /\ UNCHANGED <<receipts, reply>>

Drain(v) ==
    /\ state.phase[v] = "Active"
    /\ state' = [state EXCEPT !.phase[v] = "Draining", !.active[v[1]] = Inactive]
    /\ UNCHANGED <<receipts, reply>>

ReportFailure(v) ==
    /\ state.phase[v] \in Candidates \cup {"Active", "Draining"}
    /\ \E d \in Members(v) : state.owner[d] = Lease(v, d)
    /\ state' = [state EXCEPT !.phase[v] = "Revoked",
                   !.active[v[1]] = IF @ = v[2] THEN Inactive ELSE @]
    /\ UNCHANGED <<receipts, reply>>

CanAbort(c) ==
    LET prior == state.prepares[c.prepare]
        members == SequenceMembers(c.versions)
    IN /\ prior = Prepared(c.versions)
       /\ members \cap state.activated = {}
       /\ \A v \in members : state.phase[v] \in Candidates \cup Terminal
       /\ \E v \in members : state.phase[v] \in Terminal

ApplyAbort(c) ==
    LET prior == receipts[c.id]
    IN IF prior # NoAbortReceipt THEN
           /\ reply' = IF prior.command = c THEN prior ELSE AbortReceipt(c, "OperationIdReused")
           /\ UNCHANGED <<state, receipts>>
       ELSE LET allowed == CanAbort(c)
                result == AbortReceipt(c, IF allowed THEN Aborted ELSE Rejected)
            IN /\ state' = IF allowed THEN [state EXCEPT !.phase = [v \in Versions |->
                       IF v \in SequenceMembers(c.versions) /\ state.phase[v] \in Candidates
                       THEN "Revoked" ELSE state.phase[v]]] ELSE state
               /\ receipts' = [receipts EXCEPT ![c.id] = result]
               /\ reply' = result

Release(v, grant) ==
    /\ grant.device \in Members(v) /\ grant = Lease(v, grant.device)
    /\ IF grant.device \in state.released[v] THEN UNCHANGED state
       ELSE /\ state.phase[v] \in {"Draining", "Revoked"}
            /\ state.owner[grant.device] = grant
            /\ LET released == state.released[v] \cup {grant.device}
                   last == released = Members(v)
               IN state' = [state EXCEPT !.owner[grant.device] = NoGrant, !.released[v] = released,
                            !.reserved = IF last THEN Subtract(@, Demand(v)) ELSE @,
                            !.phase[v] = IF last THEN "Retired" ELSE @]
    /\ UNCHANGED <<receipts, reply>>

ReleaseInputs(v) ==
    {Grant(owner, d, fence) : owner \in {w \in Versions : Members(v) \cap Members(w) # {}},
                              d \in Members(v), fence \in 1..2}

PreparedView(prepare, members, activated) == [prepare |-> prepare, members |-> members, activated |-> activated]
NoPreparedView == PreparedView(0, <<>>, FALSE)

LookupLatest(group) ==
    LET version == <<group, state.latest[group]>>
        prepare == state.origin[version]
    IN IF prepare = 0 THEN NoPreparedView
       ELSE LET versions == state.prepares[prepare].versions
            IN PreparedView(prepare, [i \in DOMAIN versions |-> <<versions[i], state.phase[versions[i]]>>],
                            SequenceMembers(versions) \cap state.activated # {})

Next ==
    \/ PrepareSuccessor \/ SwitchActiveSet
    \/ \E v \in Versions : Ready(v) \/ Activate(v) \/ Drain(v) \/ ReportFailure(v)
    \/ \E v \in Versions : \E g \in ReleaseInputs(v) : Release(v, g)
    \/ \E c \in AbortRequests : ApplyAbort(c)

Spec == Init /\ [][Next]_vars

\* A whole-state postcondition, separate from the tests in ApplyAbort. Phase
\* changes outside the matched receipt, or any eager ownership release, fail.
ExpectedAbort(c) ==
    /\ state.prepares[c.prepare].kind = "PrepareSet"
    /\ c.versions = state.prepares[c.prepare].versions
    /\ \A i \in DOMAIN c.versions :
           c.versions[i] \notin state.activated /\ state.phase[c.versions[i]] \in Candidates \cup Terminal
    /\ \E i \in DOMAIN c.versions : state.phase[c.versions[i]] \in Terminal

AbortPostcondition(c) ==
    /\ state' = [state EXCEPT !.phase = [v \in Versions |->
           IF state.phase[v] \notin Terminal /\ \E i \in DOMAIN c.versions : c.versions[i] = v
           THEN "Revoked" ELSE state.phase[v]]]
    /\ reply' = AbortReceipt(c, Aborted)

ExactAtomicAbort ==
    \A c \in AbortRequests :
        /\ ENABLED ApplyAbort(c)
        /\ receipts[c.id] = NoAbortReceipt =>
               /\ ~ENABLED (ApplyAbort(c) /\ receipts'[c.id] # reply')
               /\ IF ExpectedAbort(c) THEN ~ENABLED (ApplyAbort(c) /\ ~AbortPostcondition(c))
                  ELSE ~ENABLED (ApplyAbort(c) /\ (state' # state \/ reply'.outcome # Rejected))

RetryPreservesTerminal ==
    \A c \in AbortRequests :
        LET prior == receipts[c.id] IN
        prior # NoAbortReceipt =>
            ~ENABLED (ApplyAbort(c) /\ (state' # state \/ receipts' # receipts \/
               IF prior.command = c THEN reply' # prior ELSE reply'.outcome # "OperationIdReused"))

LatestReturnsItsPrepare ==
    LET old == PreparedView(OldPrepare, <<<<P1, state.phase[P1]>>, <<D1, state.phase[D1]>>>>,
                            P1 \in state.activated \/ D1 \in state.activated)
        newer == PreparedView(NewPrepare, <<<<P2, state.phase[P2]>>>>, P2 \in state.activated)
    IN /\ LookupLatest(Prefill) = IF state.phase[P2] = "Absent" THEN old ELSE newer
       /\ LookupLatest(Decode) = old
       /\ LookupLatest(Aggregate) = NoPreparedView
       /\ state.prepares[OldPrepare] = Prepared(OldVersions)

ActiveHistoryIsKept ==
    /\ \A v \in Versions :
           /\ (state.active[v[1]] = v[2]) = (state.phase[v] = "Active")
           /\ state.phase[v] \in {"Active", "Draining"} => v \in state.activated
    /\ \A v \in Versions :
           ~ENABLED (Activate(v) /\ v \notin state'.activated)
    /\ ~ENABLED (SwitchActiveSet /\ ~({P1, D1} \subseteq state'.activated))
    /\ \A v \in state.activated : ~ENABLED (ReportFailure(v) /\ v \notin state'.activated)

OwnershipTracksRelease ==
    /\ \A v \in Versions :
           LET owned == {d \in Members(v) : state.owner[d] = Lease(v, d)} IN
           /\ state.phase[v] # "Absent" => owned = Members(v) \ state.released[v]
           /\ state.phase[v] = "Retired" => state.released[v] = Members(v)
    /\ \A d \in Devices :
           state.owner[d] # NoGrant => state.owner[d].fence = state.fence[d]
    /\ \A v \in Versions : \A g \in ReleaseInputs(v) :
           g.device \in state.released[v] => ~ENABLED (Release(v, g) /\ state' # state)

StillReserved(v) ==
    IF state.phase[v] # "Absent" /\ state.released[v] # Members(v) THEN Demand(v) ELSE Charge(0, 0)

HostBudgetIsHeld ==
    state.reserved = Add(Add(StillReserved(P1), StillReserved(D1)), Add(StillReserved(A1), StillReserved(P2)))
====
