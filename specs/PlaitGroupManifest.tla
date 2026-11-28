---- MODULE PlaitGroupManifest ----
EXTENDS Naturals

CONSTANTS CheckDigest, RequireAllReady, FenceRelease
Versions == 1..2
Devices == 1..3
Members(g) == IF g = 1 THEN {1, 2} ELSE {2, 3}

VARIABLES phase, endpoints, confirmed, badProof, released, owner
vars == <<phase, endpoints, confirmed, badProof, released, owner>>

Init ==
    /\ phase = [g \in Versions |-> "Absent"]
    /\ endpoints = [g \in Versions |-> {}]
    /\ confirmed = [g \in Versions |-> {}]
    /\ badProof = [g \in Versions |-> {}]
    /\ released = [g \in Versions |-> {}]
    /\ owner = [d \in Devices |-> 0]

Prepare(g) ==
    /\ phase[g] = "Absent"
    /\ g = 1 \/ phase[1] = "Revoked"
    /\ \A d \in Members(g) : owner[d] = 0
    /\ phase' = [phase EXCEPT ![g] = "Preparing"]
    /\ owner' = [d \in Devices |-> IF d \in Members(g) THEN g ELSE owner[d]]
    /\ UNCHANGED <<endpoints, confirmed, badProof, released>>

Publish(g, d) ==
    /\ phase[g] = "Preparing"
    /\ d \in Members(g) \ endpoints[g]
    /\ endpoints' = [endpoints EXCEPT ![g] = @ \cup {d}]
    /\ UNCHANGED <<phase, confirmed, badProof, released, owner>>

Ready(g, d, correct) ==
    /\ phase[g] = "Preparing"
    /\ endpoints[g] = Members(g)
    /\ d \in Members(g) \ confirmed[g]
    /\ CheckDigest => correct
    /\ confirmed' = [confirmed EXCEPT ![g] = @ \cup {d}]
    /\ badProof' = [badProof EXCEPT ![g] = IF correct THEN @ ELSE @ \cup {d}]
    /\ phase' = IF ~RequireAllReady \/ confirmed'[g] = Members(g)
                THEN [phase EXCEPT ![g] = "Ready"] ELSE phase
    /\ UNCHANGED <<endpoints, released, owner>>

Activate(g) ==
    /\ phase[g] = "Ready"
    /\ phase' = [phase EXCEPT ![g] = "Active"]
    /\ UNCHANGED <<endpoints, confirmed, badProof, released, owner>>

Revoke(g) ==
    /\ phase[g] \in {"Preparing", "Ready", "Active"}
    /\ phase' = [phase EXCEPT ![g] = "Revoked"]
    /\ UNCHANGED <<endpoints, confirmed, badProof, released, owner>>

Release(g, d) ==
    /\ phase[g] = "Revoked"
    /\ d \in Members(g)
    /\ released' = [released EXCEPT ![g] = @ \cup {d}]
    /\ owner' = IF FenceRelease /\ d \in released[g] THEN owner ELSE [owner EXCEPT ![d] = 0]
    /\ UNCHANGED <<phase, endpoints, confirmed, badProof>>

Next ==
    \E g \in Versions :
        \/ Prepare(g) \/ Activate(g) \/ Revoke(g)
        \/ \E d \in Devices :
               \/ Publish(g, d) \/ Release(g, d)
               \/ \E correct \in BOOLEAN : Ready(g, d, correct)

Spec == Init /\ [][Next]_vars

ReadyIsComplete ==
    \A g \in Versions : phase[g] \in {"Ready", "Active"} =>
        /\ endpoints[g] = Members(g)
        /\ confirmed[g] = Members(g)
        /\ badProof[g] = {}

GrantsSurviveOldRelease ==
    \A g \in Versions : phase[g] # "Absent" =>
        \A d \in Members(g) \ released[g] : owner[d] = g
====
