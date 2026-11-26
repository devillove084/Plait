---- MODULE PlaitHostLaunch ----
EXTENDS Naturals

\* Two epochs may overlap on a host. The resource ledger has already
\* assigned disjoint device grants and sufficient host memory. Only the
\* manifest's HTTP host publishes the endpoint, after its engine is healthy.
\* The HTTP gate reports a clean stop after its last admitted request.
Epochs == 1..2
Hosts == 1..3
Sessions == 1..2
NoSession == 0
Unstarted == 3
Members(e) == IF e = 1 THEN {1, 2} ELSE {2, 3}
HttpHost(e) == IF e = 1 THEN 1 ELSE 3
Quiescing == {"Draining", "Revoked"}

VARIABLES phase, owner, ready, published, stopped, active, admitted, shares, delivered
vars == <<phase, owner, ready, published, stopped, active, admitted, shares, delivered>>
ReadyHosts(e) == {h \in Members(e) : ready[e][h] # NoSession}
StoppedHosts(e) == {h \in Members(e) : stopped[e][h] # NoSession}

Init ==
    /\ phase = [e \in Epochs |-> IF e = 1 THEN "Preparing" ELSE "Absent"]
    /\ owner = [e \in Epochs |-> [h \in Hosts |-> NoSession]]
    /\ ready = owner
    /\ stopped = owner
    /\ published = {}
    /\ active = 0
    /\ admitted = {}
    /\ shares = [e \in Epochs |-> {}]
    /\ delivered = {}

Prepare ==
    /\ active = 1 /\ phase[2] = "Absent"
    /\ phase' = [phase EXCEPT ![2] = "Preparing"]
    /\ UNCHANGED <<owner, ready, published, stopped, active, admitted, shares, delivered>>

Claim(e, h, s) ==
    /\ phase[e] = "Preparing" /\ h \in Members(e) /\ owner[e][h] = NoSession
    /\ owner' = [owner EXCEPT ![e][h] = s]
    /\ UNCHANGED <<phase, ready, published, stopped, active, admitted, shares, delivered>>

Ready(e, h, s) ==
    /\ phase[e] = "Preparing" /\ h \in Members(e) /\ ready[e][h] = NoSession
    /\ owner[e][h] = s
    /\ ready' = [ready EXCEPT ![e][h] = s]
    /\ published' = IF h = HttpHost(e) THEN published \cup {e} ELSE published
    /\ phase' = IF ReadyHosts(e)' = Members(e) /\ e \in published'
                THEN [phase EXCEPT ![e] = "Ready"] ELSE phase
    /\ UNCHANGED <<owner, stopped, active, admitted, shares, delivered>>

Activate(e) ==
    /\ phase[e] = "Ready" /\ e > active
    /\ phase' = [x \in Epochs |->
        IF x = e THEN "Active" ELSE IF x = active THEN "Draining" ELSE phase[x]]
    /\ active' = e
    /\ UNCHANGED <<owner, ready, published, stopped, admitted, shares, delivered>>

Admit(e) ==
    /\ e = active /\ phase[e] = "Active" /\ e \notin admitted \cup delivered
    /\ admitted' = admitted \cup {e}
    /\ UNCHANGED <<phase, owner, ready, published, stopped, active, shares, delivered>>

PublishShard(e, h) ==
    /\ e \in admitted /\ h \in Members(e) \ shares[e]
    /\ shares' = [shares EXCEPT ![e] = @ \cup {h}]
    /\ UNCHANGED <<phase, owner, ready, published, stopped, active, admitted, delivered>>

Finish(e) ==
    /\ e \in admitted /\ shares[e] = Members(e)
    /\ admitted' = admitted \ {e}
    /\ delivered' = delivered \cup {e}
    /\ UNCHANGED <<phase, owner, ready, published, stopped, active, shares>>

Revoke(e) ==
    /\ phase[e] \in {"Preparing", "Ready", "Active", "Draining"}
    /\ phase' = [phase EXCEPT ![e] = "Revoked"]
    /\ active' = IF active = e THEN 0 ELSE active
    /\ admitted' = admitted \ {e}
    /\ UNCHANGED <<owner, ready, published, stopped, shares, delivered>>

Stop(e, h, s) ==
    /\ phase[e] \in Quiescing /\ h \in Members(e) \ StoppedHosts(e)
    /\ IF s = Unstarted THEN phase[e] = "Revoked" /\ owner[e][h] = NoSession
                        ELSE owner[e][h] = s
    /\ phase[e] = "Draining" =>
        IF h = HttpHost(e) THEN e \notin admitted ELSE HttpHost(e) \in StoppedHosts(e)
    /\ stopped' = [stopped EXCEPT ![e][h] = s]
    /\ UNCHANGED <<phase, owner, ready, published, active, admitted, shares, delivered>>

Retire(e) ==
    /\ phase[e] \in Quiescing /\ StoppedHosts(e) = Members(e)
    /\ phase' = [phase EXCEPT ![e] = "Retired"]
    /\ UNCHANGED <<owner, ready, published, stopped, active, admitted, shares, delivered>>

Next == Prepare
    \/ (\E e \in Epochs : Activate(e) \/ Admit(e) \/ Finish(e) \/ Revoke(e) \/ Retire(e))
    \/ (\E e \in Epochs, h \in Hosts :
            PublishShard(e, h)
            \/ (\E s \in Sessions : Claim(e, h, s) \/ Ready(e, h, s))
            \/ (\E s \in Sessions \cup {Unstarted} : Stop(e, h, s)))

Spec == Init /\ [][Next]_vars

ActiveReady ==
    \A e \in Epochs : phase[e] = "Active" =>
        e = active /\ ReadyHosts(e) = Members(e) /\ e \in published

ReportsMatchOwner ==
    \A e \in Epochs : \A h \in Members(e) :
        /\ ready[e][h] # NoSession => ready[e][h] = owner[e][h]
        /\ stopped[e][h] # NoSession =>
            IF stopped[e][h] = Unstarted THEN owner[e][h] = NoSession
                                        ELSE stopped[e][h] = owner[e][h]

FollowersStopAfterHttp ==
    \A e \in Epochs : phase[e] = "Draining" /\ StoppedHosts(e) # {} =>
        HttpHost(e) \in StoppedHosts(e)
OutstandingWorkHasWorkers == \A e \in admitted : StoppedHosts(e) = {}
RetiredRequiresAllStops == \A e \in Epochs : phase[e] = "Retired" => StoppedHosts(e) = Members(e)
NewCompletionIgnoresStalledOld ==
    (1 \in admitted /\ shares[1] # Members(1)
        /\ 2 \in admitted /\ shares[2] = Members(2))
    => ENABLED Finish(2)
====
