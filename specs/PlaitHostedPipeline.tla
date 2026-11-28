---- MODULE PlaitHostedPipeline ----
EXTENDS Naturals

Versions == {1, 2}
Lanes == {1, 2}
Workers == {1, 2, 3, 4}
Channels == Versions \X Lanes

VARIABLES active, ready, admitted, described, queued, received, complete, retired
vars == <<active, ready, admitted, described, queued, received, complete, retired>>

Init ==
    /\ active = 1 /\ ready = {} /\ admitted = {}
    /\ described = {} /\ queued = {} /\ complete = {}
    /\ received = [channel \in Channels |-> 0] /\ retired = FALSE

PrepareWorker(w) ==
    /\ w \in Workers \ ready
    /\ ready' = ready \cup {w}
    /\ UNCHANGED <<active, admitted, described, queued, received, complete, retired>>

Publish ==
    /\ active = 1 /\ ready = Workers
    /\ active' = 2
    /\ UNCHANGED <<ready, admitted, described, queued, received, complete, retired>>

Admit ==
    /\ active \notin admitted
    /\ admitted' = admitted \cup {active}
    /\ UNCHANGED <<active, ready, described, queued, received, complete, retired>>

Describe(ch) ==
    /\ ch[1] \in admitted /\ ch \notin described
    /\ described' = described \cup {ch}
    /\ UNCHANGED <<active, ready, admitted, queued, received, complete, retired>>

Send(ch) ==
    /\ ch \in described \ queued
    /\ queued' = queued \cup {ch}
    /\ UNCHANGED <<active, ready, admitted, described, received, complete, retired>>

Receive(ch, src) ==
    /\ ch \in described /\ src \in queued
    /\ ch[1] = src[1] /\ ch[2] = src[2]
    /\ received[ch] = 0
    /\ received' = [received EXCEPT ![ch] = src[1]]
    /\ UNCHANGED <<active, ready, admitted, described, queued, complete, retired>>

Result(v) ==
    /\ v \in admitted \ complete
    /\ \A l \in Lanes : received[<<v, l>>] = v
    /\ complete' = complete \cup {v}
    /\ UNCHANGED <<active, ready, admitted, described, queued, received, retired>>

Retire ==
    /\ active = 2 /\ ~retired
    /\ (1 \notin admitted \/ 1 \in complete)
    /\ retired' = TRUE
    /\ UNCHANGED <<active, ready, admitted, described, queued, received, complete>>

Next == Publish \/ Admit \/ Retire
        \/ \E w \in Workers : PrepareWorker(w)
        \/ \E v \in Versions : Result(v)
        \/ \E ch \in Channels : Describe(ch) \/ Send(ch)
                                   \/ \E src \in Channels : Receive(ch, src)

Spec == Init /\ [][Next]_vars

ReadyBeforeActivation == active = 2 => ready = Workers
NoCrossVersionActivations ==
    \A ch \in Channels : received[ch] # 0 => ch \in queued /\ received[ch] = ch[1]
GatherBeforeResult == \A v \in complete : \A l \in Lanes : received[<<v, l>>] = v
RetiredAfterDrain == retired => active = 2 /\ (1 \notin admitted \/ 1 \in complete)
====
