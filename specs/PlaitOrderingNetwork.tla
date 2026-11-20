---- MODULE PlaitOrderingNetwork ----
\* Explicit message-buffer model for the ordering fast/slow paths.
\* Messages are a set, so delivery may be delayed, duplicated, or reordered;
\* dropping a message is represented by simply not delivering it.
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS Replicas, Commands, Quorums, MaxBallot

None == 0

VARIABLES proposed, preCmd, preBallot, accCmd, accBallot, committed, ballot, msgs

vars == <<proposed, preCmd, preBallot, accCmd, accBallot, committed, ballot, msgs>>

Init ==
    /\ proposed = None
    /\ preCmd = [r \in Replicas |-> None]
    /\ preBallot = [r \in Replicas |-> 0]
    /\ accCmd = [r \in Replicas |-> None]
    /\ accBallot = [r \in Replicas |-> 0]
    /\ committed = [r \in Replicas |-> None]
    /\ ballot = 1
    /\ msgs = {}

Propose(c) ==
    /\ proposed = None
    /\ c \in Commands
    /\ proposed' = c
    /\ UNCHANGED <<preCmd, preBallot, accCmd, accBallot, committed, ballot, msgs>>

SendPre(to) ==
    /\ proposed # None
    /\ msgs' = msgs \cup {[kind |-> "pre", to |-> to,
                            cmd |-> proposed, ballot |-> ballot]}
    /\ UNCHANGED <<proposed, preCmd, preBallot, accCmd, accBallot, committed, ballot>>

DeliverPre(m) ==
    /\ m \in msgs
    /\ m.kind = "pre"
    /\ m.cmd \in Commands
    /\ preCmd' = [preCmd EXCEPT ![m.to] = m.cmd]
    /\ preBallot' = [preBallot EXCEPT ![m.to] = m.ballot]
    /\ UNCHANGED <<proposed, accCmd, accBallot, committed, ballot, msgs>>

SendAccept(to) ==
    /\ \E from \in Replicas : preCmd[from] # None
    /\ msgs' = msgs \cup {[kind |-> "acc", to |-> to,
                            cmd |-> preCmd[CHOOSE from \in Replicas : preCmd[from] # None], ballot |-> ballot]}
    /\ UNCHANGED <<proposed, preCmd, preBallot, accCmd, accBallot, committed, ballot>>

DeliverAccept(m) ==
    /\ m \in msgs
    /\ m.kind = "acc"
    /\ m.cmd \in Commands
    /\ accCmd' = [accCmd EXCEPT ![m.to] = m.cmd]
    /\ accBallot' = [accBallot EXCEPT ![m.to] = m.ballot]
    /\ UNCHANGED <<proposed, preCmd, preBallot, committed, ballot, msgs>>

FastCommit ==
    /\ proposed # None
    /\ \E Q \in Quorums :
        \A r \in Q : preCmd[r] = proposed /\ preBallot[r] = ballot
    /\ committed' = [r \in Replicas |-> proposed]
    /\ UNCHANGED <<proposed, preCmd, preBallot, accCmd, accBallot, ballot, msgs>>

SlowCommit(c) ==
    /\ c \in Commands
    /\ \E Q \in Quorums :
        \A r \in Q : accCmd[r] = c
    /\ committed' = [r \in Replicas |-> c]
    /\ UNCHANGED <<proposed, preCmd, preBallot, accCmd, accBallot, ballot, msgs>>

Next ==
    \/ \E c \in Commands : Propose(c)
    \/ \E to \in Replicas : SendPre(to)
    \/ \E m \in msgs : DeliverPre(m)
    \/ \E to \in Replicas : SendAccept(to)
    \/ \E m \in msgs : DeliverAccept(m)
    \/ FastCommit
    \/ \E c \in Commands : SlowCommit(c)

Spec == Init /\ [][Next]_vars

Agreement ==
    \A r1, r2 \in Replicas :
        committed[r1] # None /\ committed[r2] # None =>
            committed[r1] = committed[r2]

CommittedStable ==
    \A r1, r2 \in Replicas :
        committed[r1] # None =>
            committed[r2] = None \/ committed[r2] = committed[r1]

THEOREM Spec => []Agreement
====
