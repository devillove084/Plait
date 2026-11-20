---- MODULE PlaitEPaxos ----
\* Finite EPaxos safety model for the Plait control plane.
\*
\* This model abstracts messages and dependency metadata to the properties that
\* matter for agreement: proposal, preaccept, accept, fast/slow commit, and
\* recovery with a committed-value guard. It is intentionally finite so TLC can
\* enumerate every reachable state.
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS Replicas, Instances, Commands, MaxBallot

None == 0
NoneRecord == [cmd |-> 0, ballot |-> 0, deps |-> {}, seq |-> 0]
Quorums == {Q \in SUBSET Replicas : Cardinality(Q) * 2 > Cardinality(Replicas)}
Leader(i) == CHOOSE r \in Replicas : TRUE

VARIABLES proposed, preaccepted, accepted, committed, ballot

vars == <<proposed, preaccepted, accepted, committed, ballot>>

Init ==
    /\ proposed = [i \in Instances |-> None]
    /\ preaccepted = [r \in Replicas |-> [i \in Instances |-> NoneRecord]]
    /\ accepted = [r \in Replicas |-> [i \in Instances |-> NoneRecord]]
    /\ committed = [r \in Replicas |-> [i \in Instances |-> None]]
    /\ ballot = [i \in Instances |-> 1]

Propose(i, c) ==
    /\ proposed[i] = None
    /\ c \in Commands
    /\ proposed' = [proposed EXCEPT ![i] = c]
    /\ UNCHANGED <<preaccepted, accepted, committed, ballot>>

PreAccept(r, i) ==
    /\ proposed[i] # None
    /\ preaccepted[r][i] = NoneRecord
    /\ accepted[r][i] = NoneRecord
    /\ preaccepted' =
        [preaccepted EXCEPT ![r][i] =
            [cmd |-> proposed[i], ballot |-> ballot[i], deps |-> {}, seq |-> 1]]
    /\ UNCHANGED <<proposed, accepted, committed, ballot>>

Accept(r, i) ==
    /\ preaccepted[r][i] # NoneRecord
    /\ accepted[r][i] = NoneRecord
    /\ accepted' =
        [accepted EXCEPT ![r][i] =
            [cmd |-> preaccepted[r][i].cmd,
             ballot |-> ballot[i],
             deps |-> preaccepted[r][i].deps,
             seq |-> preaccepted[r][i].seq]]
    /\ UNCHANGED <<proposed, preaccepted, committed, ballot>>

FastCommit(i) ==
    /\ proposed[i] # None
    /\ \E Q \in Quorums :
        \A r \in Q :
            preaccepted[r][i] # NoneRecord
            /\ preaccepted[r][i].cmd = proposed[i]
            /\ preaccepted[r][i].ballot = ballot[i]
    /\ committed' = [r \in Replicas |-> [committed[r] EXCEPT ![i] = proposed[i]]]
    /\ UNCHANGED <<proposed, preaccepted, accepted, ballot>>

SlowCommit(i) ==
    \E c \in Commands :
        /\ \E Q \in Quorums :
            \A r \in Q :
                accepted[r][i] # NoneRecord
                /\ accepted[r][i].cmd = c
        /\ committed' = [r \in Replicas |-> [committed[r] EXCEPT ![i] = c]]
        /\ UNCHANGED <<proposed, preaccepted, accepted, ballot>>

\* A recovering leader may only choose a value that is either already committed
\* at some replica, or accepted by a quorum. Otherwise it keeps the proposal.
ChosenValue(i) ==
    LET committed_value ==
            IF \E r \in Replicas : committed[r][i] # None
            THEN LET rr == CHOOSE r \in Replicas : committed[r][i] # None
                 IN committed[rr][i]
            ELSE None
        accepted_quorum(c) ==
            \E Q \in Quorums :
                \A r \in Q : accepted[r][i] # NoneRecord /\ accepted[r][i].cmd = c
    IN IF committed_value # None
       THEN committed_value
       ELSE IF \E c \in Commands : accepted_quorum(c)
            THEN CHOOSE c \in Commands : accepted_quorum(c)
            ELSE proposed[i]

Recover(i) ==
    LET value == ChosenValue(i)
    IN /\ ballot[i] < MaxBallot
       /\ value # None
       /\ ballot' = [ballot EXCEPT ![i] = ballot[i] + 1]
       /\ proposed' = [proposed EXCEPT ![i] = value]
       /\ preaccepted' =
           [preaccepted EXCEPT ![Leader(i)][i] =
               [cmd |-> value, ballot |-> ballot[i] + 1, deps |-> {}, seq |-> 1]]
       /\ UNCHANGED <<accepted, committed>>

Next ==
    \/ \E i \in Instances, c \in Commands : Propose(i, c)
    \/ \E r \in Replicas, i \in Instances : PreAccept(r, i)
    \/ \E r \in Replicas, i \in Instances : Accept(r, i)
    \/ \E i \in Instances : FastCommit(i)
    \/ \E i \in Instances : SlowCommit(i)
    \/ \E i \in Instances : Recover(i)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Agreement: two replicas never commit different commands for one instance.
Agreement ==
    \A i \in Instances :
        \A r1, r2 \in Replicas :
            committed[r1][i] # None /\ committed[r2][i] # None =>
                committed[r1][i] = committed[r2][i]

\* A committed instance does not change value.
CommittedStable ==
    \A i \in Instances, r \in Replicas :
        committed[r][i] # None =>
            \A s \in Replicas : committed[s][i] = None \/ committed[s][i] = committed[r][i]

Live ==
    \A i \in Instances :
        (proposed[i] # None) ~> (\E r \in Replicas : committed[r][i] # None)

THEOREM Spec => []Agreement
====
