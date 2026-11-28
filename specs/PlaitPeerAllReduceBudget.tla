---- MODULE PlaitPeerAllReduceBudget ----
EXTENDS Naturals, Sequences

\* The control-plane manifest stays independent of the startup option.
\* Digests are symbolic fingerprints of their authenticated inputs.
ActiveEpoch == 1
CandidateEpoch == 2
Epochs == {ActiveEpoch, CandidateEpoch}
Layouts == {"stable", "alternate"}
Default == [build |-> "new", option |-> <<>>]
Legacy == [build |-> "old", option |-> <<>>]
Explicit(limit) == [build |-> "new", option |-> <<limit>>]
KiB64 == 65536
MiB == 1048576
IntMax == 2147483647
Limits == {0, KiB64, MiB, IntMax}
Policies == {Legacy, Default} \cup {Explicit(limit) : limit \in Limits}

VARIABLES world, views, links, ready, aborted
vars == <<world, views, links, ready, aborted>>
Slots == 1..world
Listeners == Epochs \X Slots

Policy(listener) ==
    IF listener[1] = ActiveEpoch THEN Legacy ELSE views[listener[2]].policy
Layout(listener) ==
    IF listener[1] = ActiveEpoch THEN "stable" ELSE views[listener[2]].layout
WireOption(policy) == policy.option
ControlManifest(epoch) == <<epoch>>
BaseFingerprint(epoch, layout) ==
    LET manifest == ControlManifest(epoch)
    IN IF layout = "stable" THEN manifest ELSE Append(manifest, <<"slot-layout", layout>>)
Handshake(epoch, layout, policy) ==
    LET base == BaseFingerprint(epoch, layout)
        option == WireOption(policy)
    IN IF option = <<>> THEN base ELSE Append(base, <<"replicated-egress", option[1]>>)
Fingerprint(listener) == Handshake(listener[1], Layout(listener), Policy(listener))

Init ==
    /\ world \in {2, 3}
    /\ views \in [Slots -> [layout: Layouts, policy: Policies]]
    /\ links = {} /\ ready = {ActiveEpoch} /\ aborted = FALSE

\* Old-epoch greetings may still arrive while the candidate bootstraps.
\* A link stands for both wire lanes, each of which checks the fingerprint.
Hello(left, right) ==
    /\ ~aborted /\ CandidateEpoch \notin ready
    /\ CandidateEpoch \in {left[1], right[1]}
    /\ left[2] < right[2]
    /\ <<left, right>> \notin links
    /\ Fingerprint(left) = Fingerprint(right)
    /\ links' = links \cup {<<left, right>>}
    /\ UNCHANGED <<world, views, ready, aborted>>

CandidateMesh ==
    \A a, b \in Slots : a < b => <<<<CandidateEpoch, a>>, <<CandidateEpoch, b>>>> \in links

MarkReady ==
    /\ ~aborted /\ CandidateEpoch \notin ready /\ CandidateMesh
    /\ ready' = ready \cup {CandidateEpoch}
    /\ UNCHANGED <<world, views, links, aborted>>

AbortCandidate ==
    /\ ~aborted /\ CandidateEpoch \notin ready
    /\ aborted' = TRUE /\ links' = {}
    /\ UNCHANGED <<world, views, ready>>

Next == MarkReady \/ AbortCandidate \/ \E left, right \in Listeners : Hello(left, right)
Spec == Init /\ [][Next]_vars

DefaultLimit(workers) == IF workers = 2 THEN MiB ELSE KiB64
Limit(workers, policy) ==
    LET option == WireOption(policy)
    IN IF option = <<>> THEN DefaultLimit(workers) ELSE option[1]

\* Dividing the limit by the fan-out is equivalent to comparing exact total
\* egress, including at a machine integer limit. A singleton sends no data.
Plan(bytes, workers, policy) ==
    IF workers = 1 THEN "Replicated"
    ELSE IF bytes <= Limit(workers, policy) \div (workers - 1)
    THEN "Replicated"
    ELSE "ScatterGather"

Payloads == {0, 1, KiB64 \div 2, KiB64 \div 2 + 1, KiB64, KiB64 + 1,
             MiB, MiB + 1, 1073741824, IntMax}

ScopedLinks ==
    \A link \in links:
        /\ link[1][1] = CandidateEpoch /\ link[2][1] = CandidateEpoch
        /\ Layout(link[1]) = Layout(link[2])
        /\ WireOption(Policy(link[1])) = WireOption(Policy(link[2]))

CandidateReadyAgrees ==
    CandidateEpoch \in ready =>
        /\ ~aborted /\ CandidateMesh
        /\ \A a, b \in Slots:
              /\ Layout(<<CandidateEpoch, a>>) = Layout(<<CandidateEpoch, b>>)
              /\ WireOption(Policy(<<CandidateEpoch, a>>)) = WireOption(Policy(<<CandidateEpoch, b>>))
              /\ \A bytes \in Payloads:
                   Plan(bytes, world, Policy(<<CandidateEpoch, a>>)) =
                   Plan(bytes, world, Policy(<<CandidateEpoch, b>>))

ActiveEpochIsolated ==
    /\ ActiveEpoch \in ready
    /\ \A slot \in Slots:
          Fingerprint(<<ActiveEpoch, slot>>) = ControlManifest(ActiveEpoch)

WireCompatibility ==
    \A epoch \in Epochs, layout \in Layouts:
        /\ Handshake(epoch, layout, Legacy) = BaseFingerprint(epoch, layout)
        /\ Handshake(epoch, layout, Default) = BaseFingerprint(epoch, layout)
        /\ \A limit \in Limits:
              Handshake(epoch, layout, Explicit(limit)) # BaseFingerprint(epoch, layout)

DefaultBoundaries ==
    \A policy \in {Legacy, Default}:
        /\ Plan(MiB, 2, policy) = "Replicated"
        /\ Plan(MiB + 1, 2, policy) = "ScatterGather"
        /\ \A workers \in {3, 5, 16}:
              LET perPeer == KiB64 \div (workers - 1)
              IN /\ Plan(perPeer, workers, policy) = "Replicated"
                 /\ Plan(perPeer + 1, workers, policy) = "ScatterGather"

ExplicitBoundaries ==
    /\ \A workers \in {2, 3, 5, 16}:
          /\ Plan(0, workers, Explicit(0)) = "Replicated"
          /\ Plan(1, workers, Explicit(0)) = "ScatterGather"
          /\ \A bytes \in Payloads \ {1073741824, IntMax}, limit \in Limits:
                (Plan(bytes, workers, Explicit(limit)) = "Replicated") = (bytes * (workers - 1) <= limit)
    /\ Plan(IntMax, 1, Explicit(0)) = "Replicated"
    /\ Plan(IntMax, 2, Explicit(IntMax)) = "Replicated"
    /\ Plan(1073741824, 3, Explicit(0)) = "ScatterGather"
    /\ Plan(1073741824, 3, Explicit(IntMax)) = "ScatterGather"
====
