---- MODULE PlaitGroupReplan ----
EXTENDS Naturals

CONSTANTS CheckSamples, CheckEpoch
Planners == {"a", "b"}
NoProposal == [epoch |-> 0, winner |-> 0, revision |-> 0]

VARIABLES latest, winner, revision, seen, count, submitted, receipt, accepted, early
vars == <<latest, winner, revision, seen, count, submitted, receipt, accepted, early>>

Snapshot == [epoch |-> latest, winner |-> winner, revision |-> revision]
SamePlacement(a, b) == a.epoch = b.epoch /\ a.winner = b.winner

Init ==
    /\ latest = 1 /\ winner = 0 /\ revision = 1
    /\ seen = [p \in Planners |-> NoProposal]
    /\ count = [p \in Planners |-> 0]
    /\ submitted = [p \in Planners |-> NoProposal]
    /\ receipt = [p \in Planners |-> "none"]
    /\ accepted = 0 /\ early = FALSE

Refresh(target) ==
    /\ revision < 2
    /\ winner' = target /\ revision' = revision + 1
    /\ UNCHANGED <<latest, seen, count, submitted, receipt, accepted, early>>

Read(p) ==
    /\ submitted[p] = NoProposal
    /\ IF accepted > 0 \/ winner = 0
          THEN /\ seen' = [seen EXCEPT ![p] = NoProposal]
               /\ count' = [count EXCEPT ![p] = 0]
          ELSE /\ seen' = [seen EXCEPT ![p] = Snapshot]
               /\ count' = [count EXCEPT ![p] =
                                    IF ~SamePlacement(seen[p], Snapshot) THEN 1 ELSE 2]
    /\ UNCHANGED <<latest, winner, revision, submitted, receipt, accepted, early>>

Submit(p) ==
    /\ submitted[p] = NoProposal /\ seen[p] # NoProposal
    /\ CheckSamples => count[p] = 2
    /\ submitted' = [submitted EXCEPT ![p] = seen[p]]
    /\ early' = (early \/ count[p] < 2)
    /\ UNCHANGED <<latest, winner, revision, seen, count, receipt, accepted>>

Apply(p) ==
    /\ submitted[p] # NoProposal /\ receipt[p] = "none"
    /\ IF submitted[p].winner = winner
          /\ submitted[p].revision = revision
          /\ (CheckEpoch => submitted[p].epoch = latest)
          THEN /\ receipt' = [receipt EXCEPT ![p] = "accepted"]
               /\ latest' = latest + 1 /\ accepted' = accepted + 1
          ELSE /\ receipt' = [receipt EXCEPT ![p] = "rejected"]
               /\ UNCHANGED <<latest, accepted>>
    /\ UNCHANGED <<winner, revision, seen, count, submitted, early>>

Acknowledge(p) ==
    /\ receipt[p] # "none"
    /\ submitted' = [submitted EXCEPT ![p] = NoProposal]
    /\ receipt' = [receipt EXCEPT ![p] = "none"]
    /\ seen' = [seen EXCEPT ![p] = NoProposal]
    /\ count' = [count EXCEPT ![p] = 0]
    /\ UNCHANGED <<latest, winner, revision, accepted, early>>

ResetObservations(p) ==
    /\ submitted[p] = NoProposal
    /\ seen' = [seen EXCEPT ![p] = NoProposal]
    /\ count' = [count EXCEPT ![p] = 0]
    /\ UNCHANGED <<latest, winner, revision, submitted, receipt, accepted, early>>

(* Between Apply and Acknowledge, retrying returns the same applied receipt.
   These retries leave all modeled state unchanged. *)
Next ==
    \/ \E target \in 0..2 : Refresh(target)
    \/ \E p \in Planners :
           Read(p) \/ Submit(p) \/ Apply(p) \/ Acknowledge(p) \/ ResetObservations(p)

Spec == Init /\ [][Next]_vars
StableBeforeSubmit == ~early
OneReservation == accepted <= 1
====
