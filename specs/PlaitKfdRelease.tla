---- MODULE PlaitKfdRelease ----
EXTENDS FiniteSets

Target == "device-a"
Other == "device-b"
Empty == [complete |-> FALSE, identity |-> "none", busy |-> {}]

VARIABLES phase, identity, queues, vram, snapshot, proof
vars == <<phase, identity, queues, vram, snapshot, proof>>

Init ==
    /\ phase = "serving" /\ identity = Target
    /\ queues = {Target, Other} /\ vram = {Target, Other}
    /\ snapshot = Empty /\ proof = Empty

Drain ==
    /\ phase = "serving" /\ phase' = "drained"
    /\ UNCHANGED <<identity, queues, vram, snapshot, proof>>

CloseQueue ==
    /\ phase = "drained" /\ Target \in queues
    /\ queues' = queues \ {Target}
    /\ UNCHANGED <<phase, identity, vram, snapshot, proof>>

FreeMemory ==
    /\ phase = "drained" /\ Target \in vram
    /\ vram' = vram \ {Target}
    /\ UNCHANGED <<phase, identity, queues, snapshot, proof>>

ChangeMapping ==
    /\ phase = "drained" /\ identity = Target /\ identity' = Other
    /\ UNCHANGED <<phase, queues, vram, snapshot, proof>>

Observe(complete) ==
    /\ phase = "drained"
    /\ snapshot' = [complete |-> complete, identity |-> identity,
                    busy |-> IF complete THEN queues \cup vram ELSE {Other}]
    /\ UNCHANGED <<phase, identity, queues, vram, proof>>

Release ==
    /\ phase = "drained" /\ snapshot.complete
    /\ snapshot.identity = Target /\ identity = Target
    /\ Target \notin snapshot.busy
    /\ phase' = "released" /\ proof' = snapshot
    /\ UNCHANGED <<identity, queues, vram, snapshot>>

Reuse ==
    /\ phase = "released" /\ phase' = "reused"
    /\ UNCHANGED <<identity, queues, vram, snapshot, proof>>

Next == Drain \/ CloseQueue \/ FreeMemory \/ ChangeMapping \/ Release \/ Reuse
        \/ \E complete \in BOOLEAN : Observe(complete)
Spec == Init /\ [][Next]_vars

ReleaseEvidence == phase \in {"released", "reused"} =>
    /\ proof.complete /\ proof.identity = Target /\ identity = Target
    /\ Target \notin (queues \cup vram)

OtherDeviceIndependent ==
    (phase = "drained" /\ snapshot.complete /\ snapshot.identity = Target
     /\ identity = Target /\ snapshot.busy = {Other}) => ENABLED Release
====
