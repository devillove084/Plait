---- MODULE PlaitPeerFailover ----
EXTENDS Naturals

CONSTANTS ExactGrant, NeedReady, ExcludeAffected
Epochs == 1..2
Fences == 1..2
VARIABLES phases, active, armed, held, badReport, badPromotion
vars == <<phases, active, armed, held, badReport, badPromotion>>

Init ==
    /\ phases = [epoch \in Epochs |-> IF epoch = 1 THEN "Active" ELSE "Preparing"]
    /\ active = 1 /\ armed = FALSE /\ held = Epochs
    /\ badReport = FALSE /\ badPromotion = FALSE

Ready ==
    /\ phases[2] = "Preparing"
    /\ phases' = [phases EXCEPT ![2] = "Ready"]
    /\ UNCHANGED <<active, armed, held, badReport, badPromotion>>

Arm ==
    /\ active = 1 /\ 2 \in held
    /\ phases[2] \in {"Preparing", "Ready"}
    /\ (NeedReady => phases[2] = "Ready")
    /\ armed' = TRUE
    /\ UNCHANGED <<phases, active, held, badReport, badPromotion>>

Disarm ==
    /\ armed /\ armed' = FALSE
    /\ UNCHANGED <<phases, active, held, badReport, badPromotion>>

ManualActivate ==
    /\ phases[2] = "Ready"
    /\ phases' = [phases EXCEPT ![1] = IF @ = "Active" THEN "Draining" ELSE @,
                                ![2] = "Active"]
    /\ active' = 2 /\ armed' = FALSE
    /\ UNCHANGED <<held, badReport, badPromotion>>

End(epoch, fault, sharedHost) ==
    LET promote == /\ fault /\ active = 1 /\ epoch = 1 /\ armed
                   /\ (NeedReady => phases[2] = "Ready")
                   /\ (ExcludeAffected => ~sharedHost)
    IN
    /\ phases[epoch] \in {"Preparing", "Ready", "Active", "Draining"}
    /\ phases' = [version \in Epochs |->
            IF version = epoch THEN "Revoked"
            ELSE IF promote THEN "Active"
            ELSE IF epoch = 1 /\ sharedHost /\ version = 2 /\ phases[2] # "Retired" THEN "Revoked"
            ELSE phases[version]]
    /\ active' = IF promote THEN 2
                  ELSE IF active = epoch \/ (sharedHost /\ active = 2) THEN 0
                  ELSE active
    /\ armed' = IF epoch = active \/ epoch = 2 THEN FALSE ELSE armed
    /\ badPromotion' = (badPromotion \/ (promote /\ (phases[2] # "Ready" \/ sharedHost)))
    /\ UNCHANGED held

Report(epoch, fence) ==
    /\ epoch \in held
    /\ (ExactGrant => fence = epoch)
    /\ End(epoch, TRUE, FALSE)
    /\ badReport' = (badReport \/ fence # epoch)

HostLoss(sharedHost) ==
    /\ End(1, TRUE, sharedHost)
    /\ UNCHANGED badReport

ManualRevoke(epoch) ==
    /\ End(epoch, FALSE, FALSE)
    /\ UNCHANGED badReport

Release(epoch) ==
    /\ phases[epoch] \in {"Revoked", "Draining"}
    /\ phases' = [phases EXCEPT ![epoch] = "Retired"]
    /\ held' = held \ {epoch}
    /\ UNCHANGED <<active, armed, badReport, badPromotion>>

Next ==
    \/ Ready \/ Arm \/ Disarm \/ ManualActivate
    \/ \E epoch \in Epochs : ManualRevoke(epoch) \/ Release(epoch)
    \/ \E epoch \in Epochs, fence \in Fences : Report(epoch, fence)
    \/ \E sharedHost \in BOOLEAN : HostLoss(sharedHost)
Spec == Init /\ [][Next]_vars
OnlyExactReports == ~badReport
EligiblePromotions == ~badPromotion
UniqueActive == {epoch \in Epochs : phases[epoch] = "Active"} = ({active} \cap Epochs)
RetainUntilRelease == held = {epoch \in Epochs : phases[epoch] # "Retired"}
====
