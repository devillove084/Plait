---- MODULE PlaitHostedPdTransition ----
EXTENDS Naturals

VARIABLES phase, revision, replacementReady, oldStopped, oldRetired,
          deferred, owner, targetReady, targetHost
vars == <<phase, revision, replacementReady, oldStopped, oldRetired,
          deferred, owner, targetReady, targetHost>>

Init ==
    /\ phase = "initial" /\ revision = 1
    /\ replacementReady = FALSE /\ oldStopped = FALSE /\ oldRetired = FALSE
    /\ deferred = FALSE /\ owner = "oldD" /\ targetReady = FALSE /\ targetHost = TRUE

Stage ==
    /\ phase = "initial" /\ phase' = "preparingBridge"
    /\ deferred' = TRUE
    /\ UNCHANGED <<revision, replacementReady, oldStopped, oldRetired,
                   owner, targetReady, targetHost>>

ReadyBridge ==
    /\ phase = "preparingBridge" /\ replacementReady' = TRUE
    /\ UNCHANGED <<phase, revision, oldStopped, oldRetired, deferred,
                   owner, targetReady, targetHost>>

PublishBridge ==
    /\ phase = "preparingBridge" /\ replacementReady
    /\ phase' = "waiting" /\ revision' = 2
    /\ UNCHANGED <<replacementReady, oldStopped, oldRetired, deferred,
                   owner, targetReady, targetHost>>

StopOld ==
    /\ revision = 2 /\ ~oldRetired
    /\ oldStopped' = TRUE
    /\ UNCHANGED <<phase, revision, replacementReady, oldRetired,
                   deferred, owner, targetReady, targetHost>>

RetireOld ==
    /\ oldStopped /\ owner = "oldD"
    /\ oldRetired' = TRUE /\ owner' = "free"
    /\ UNCHANGED <<phase, revision, replacementReady, oldStopped,
                   deferred, targetReady, targetHost>>

ReserveTarget ==
    /\ phase = "waiting" /\ targetHost
    /\ oldRetired /\ owner = "free"
    /\ phase' = "preparingTarget" /\ owner' = "newP"
    /\ deferred' = FALSE
    /\ UNCHANGED <<revision, replacementReady, oldStopped, oldRetired,
                   targetReady, targetHost>>

CompetingReserve ==
    /\ owner = "free" /\ ~deferred
    /\ owner' = "other"
    /\ UNCHANGED <<phase, revision, replacementReady, oldStopped,
                   oldRetired, deferred, targetReady, targetHost>>

ReadyTarget ==
    /\ phase = "preparingTarget" /\ targetHost /\ targetReady' = TRUE
    /\ UNCHANGED <<phase, revision, replacementReady, oldStopped,
                   oldRetired, deferred, owner, targetHost>>

PublishTarget ==
    /\ phase = "preparingTarget" /\ targetReady /\ targetHost
    /\ phase' = "published" /\ revision' = 3
    /\ UNCHANGED <<replacementReady, oldStopped, oldRetired, deferred,
                   owner, targetReady, targetHost>>

HostLost ==
    /\ phase \in {"preparingBridge", "waiting", "preparingTarget"}
    /\ targetHost' = FALSE
    /\ UNCHANGED <<phase, revision, replacementReady, oldStopped,
                   oldRetired, deferred, owner, targetReady>>

Cancel ==
    /\ phase \in {"preparingBridge", "waiting", "preparingTarget"}
    /\ phase' = "cancelled" /\ deferred' = FALSE
    /\ UNCHANGED <<revision, replacementReady, oldStopped, oldRetired,
                   owner, targetReady, targetHost>>

Next == Stage \/ ReadyBridge \/ PublishBridge \/ StopOld \/ RetireOld
        \/ ReserveTarget \/ CompetingReserve \/ ReadyTarget \/ PublishTarget
        \/ HostLost \/ Cancel

Spec == Init /\ [][Next]_vars

PreserveBridge == revision >= 2 => replacementReady
NoReuseBeforeRetirement == owner \in {"newP", "other"} => oldRetired /\ oldStopped
HoldForTarget == deferred => owner \in {"oldD", "free"}
PublishAfterTargetReady == revision = 3 =>
    phase = "published" /\ owner = "newP" /\ targetReady /\ targetHost
====
