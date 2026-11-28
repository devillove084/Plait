---- MODULE PlaitNativeServingCutover ----
EXTENDS Naturals

CONSTANTS CheckOrigin, CheckEngine, PinStreams
Workers == {"a", "b"}
Ingresses == {"west", "east"}

VARIABLES ready, healthy, origins, active, observed, stream, admitted, reply
vars == <<ready, healthy, origins, active, observed, stream, admitted, reply>>

Init ==
    /\ ready = {} /\ healthy = FALSE
    /\ origins = {1} /\ active = 1
    /\ observed = [i \in Ingresses |-> 1]
    /\ stream = [i \in Ingresses |-> 0]
    /\ admitted = [i \in Ingresses |-> 0]
    /\ reply = FALSE

TransportReady(w) ==
    /\ w \notin ready /\ ready' = ready \cup {w}
    /\ UNCHANGED <<healthy, origins, active, observed, stream, admitted, reply>>

EngineReady ==
    /\ ready = Workers /\ ~healthy /\ healthy' = TRUE
    /\ UNCHANGED <<ready, origins, active, observed, stream, admitted, reply>>

Publish ==
    /\ CheckEngine => healthy
    /\ 2 \notin origins /\ origins' = origins \cup {2}
    /\ UNCHANGED <<ready, healthy, active, observed, stream, admitted, reply>>

Activate ==
    /\ active = 1 /\ ready = Workers
    /\ CheckEngine => healthy
    /\ CheckOrigin => 2 \in origins
    /\ active' = 2
    /\ UNCHANGED <<ready, healthy, origins, observed, stream, admitted, reply>>

(* An ambiguous reply can be replayed after activation. *)
Acknowledge ==
    /\ active = 2 /\ ~reply /\ reply' = TRUE
    /\ UNCHANGED <<ready, healthy, origins, active, observed, stream, admitted>>

Observe(i) ==
    /\ observed[i] < active
    /\ observed' = [observed EXCEPT ![i] = active]
    /\ IF PinStreams \/ stream[i] = 0
          THEN UNCHANGED stream
          ELSE stream' = [stream EXCEPT ![i] = active]
    /\ UNCHANGED <<ready, healthy, origins, active, admitted, reply>>

Admit(i) ==
    /\ stream[i] = 0 /\ observed[i] \in origins
    /\ stream' = [stream EXCEPT ![i] = observed[i]]
    /\ admitted' = [admitted EXCEPT ![i] = observed[i]]
    /\ UNCHANGED <<ready, healthy, origins, active, observed, reply>>

Finish(i) ==
    /\ stream[i] # 0
    /\ stream' = [stream EXCEPT ![i] = 0]
    /\ UNCHANGED <<ready, healthy, origins, active, observed, admitted, reply>>

Next ==
    \/ \E w \in Workers : TransportReady(w)
    \/ EngineReady \/ Publish \/ Activate \/ Acknowledge
    \/ \E i \in Ingresses : Observe(i) \/ Admit(i) \/ Finish(i)

Spec == Init /\ [][Next]_vars
ActiveHasOrigin == active = 2 => 2 \in origins
ActiveEngineReady == active = 2 => ready = Workers /\ healthy
StreamKeepsVersion ==
    \A i \in Ingresses : stream[i] # 0 => stream[i] = admitted[i]
====
