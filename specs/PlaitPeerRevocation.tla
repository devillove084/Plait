---- MODULE PlaitPeerRevocation ----
EXTENDS Naturals

CONSTANTS BindEpoch, RequireTerminal, PreserveDrain
Epochs == 1..2
Voters == 1..2
VARIABLES phase, released, views, closed, pending
vars == <<phase, released, views, closed, pending>>

Ended(epoch) == phase[epoch] = "Revoked" \/ epoch \in released

AppliedStatus(epoch) ==
    IF Ended(epoch) \/ (~PreserveDrain /\ phase[epoch] = "Draining")
    THEN "Ended"
    ELSE "Held"

Init ==
    /\ phase = [epoch \in Epochs |-> IF epoch = 1 THEN "Active" ELSE "Ready"]
    /\ released = {} /\ closed = {} /\ pending = {}
    /\ views = [voter \in Voters |-> [epoch \in Epochs |-> "Unknown"]]

ActivateSuccessor ==
    /\ phase[2] = "Ready"
    /\ phase' = [phase EXCEPT ![1] = IF @ = "Active" THEN "Draining" ELSE @,
                             ![2] = "Active"]
    /\ UNCHANGED <<released, views, closed, pending>>

Drain(epoch) ==
    /\ phase[epoch] = "Active"
    /\ phase' = [phase EXCEPT ![epoch] = "Draining"]
    /\ UNCHANGED <<released, views, closed, pending>>

Revoke(epoch) ==
    /\ phase[epoch] # "Revoked"
    /\ phase' = [phase EXCEPT ![epoch] = "Revoked"]
    /\ UNCHANGED <<released, views, closed, pending>>

Release(epoch) ==
    /\ phase[epoch] \in {"Draining", "Revoked"}
    /\ released' = released \cup {epoch}
    /\ UNCHANGED <<phase, views, closed, pending>>

Apply(voter, epoch) ==
    /\ views' = [views EXCEPT ![voter][epoch] = AppliedStatus(epoch)]
    /\ UNCHANGED <<phase, released, closed, pending>>

WaitForPeer(epoch) ==
    /\ epoch \notin closed
    /\ pending' = pending \cup {epoch}
    /\ UNCHANGED <<phase, released, views, closed>>

Observe(local, observed, voter) ==
    /\ BindEpoch => local = observed
    /\ IF RequireTerminal THEN views[voter][observed] = "Ended"
                           ELSE views[voter][observed] # "Held"
    /\ closed' = closed \cup {local}
    /\ pending' = pending \ {local}
    /\ UNCHANGED <<phase, released, views>>

Next ==
    \/ ActivateSuccessor
    \/ \E epoch \in Epochs : Drain(epoch) \/ Revoke(epoch) \/ Release(epoch) \/ WaitForPeer(epoch)
    \/ \E voter \in Voters, epoch \in Epochs : Apply(voter, epoch)
    \/ \E local, observed \in Epochs, voter \in Voters : Observe(local, observed, voter)

Spec == Init /\ [][Next]_vars
OnlyEndedGrantsClose == \A epoch \in closed : Ended(epoch)
ClosedCallsWake == closed \cap pending = {}
====
