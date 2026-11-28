---- MODULE PlaitPeerDeliveryFence ----

CONSTANTS CheckCancellation, RequireVerifiedStop

VARIABLES revoked, cancelled, stopped, oldGrant, newGrant, result,
          claimedAfterCancellation, wroteUnderNewGrant
vars == <<revoked, cancelled, stopped, oldGrant, newGrant, result,
          claimedAfterCancellation, wroteUnderNewGrant>>

Init ==
    /\ revoked = FALSE /\ cancelled = FALSE /\ stopped = FALSE
    /\ oldGrant = TRUE /\ newGrant = FALSE /\ result = "Buffered"
    /\ claimedAfterCancellation = FALSE /\ wroteUnderNewGrant = FALSE

CommitRevocation ==
    /\ ~revoked /\ revoked' = TRUE
    /\ UNCHANGED <<cancelled, stopped, oldGrant, newGrant, result,
                   claimedAfterCancellation, wroteUnderNewGrant>>

ObserveCancellation ==
    /\ revoked /\ ~cancelled /\ cancelled' = TRUE
    /\ UNCHANGED <<revoked, stopped, oldGrant, newGrant, result,
                   claimedAfterCancellation, wroteUnderNewGrant>>

ClaimUpload ==
    /\ ~stopped /\ oldGrant /\ result = "Buffered"
    /\ CheckCancellation => ~cancelled
    /\ result' = "Uploading"
    /\ claimedAfterCancellation' = (claimedAfterCancellation \/ cancelled)
    /\ UNCHANGED <<revoked, cancelled, stopped, oldGrant, newGrant, wroteUnderNewGrant>>

CompleteUpload ==
    /\ result = "Uploading"
    /\ result' = "Written"
    /\ wroteUnderNewGrant' = (wroteUnderNewGrant \/ newGrant)
    /\ UNCHANGED <<revoked, cancelled, stopped, oldGrant, newGrant, claimedAfterCancellation>>

VerifyContextStopped ==
    /\ revoked /\ ~stopped /\ stopped' = TRUE
    /\ result' = IF result \in {"Buffered", "Uploading"} THEN "Discarded" ELSE result
    /\ UNCHANGED <<revoked, cancelled, oldGrant, newGrant,
                   claimedAfterCancellation, wroteUnderNewGrant>>

ReturnOldGrant ==
    /\ revoked /\ oldGrant
    /\ RequireVerifiedStop => stopped
    /\ oldGrant' = FALSE
    /\ UNCHANGED <<revoked, cancelled, stopped, newGrant, result,
                   claimedAfterCancellation, wroteUnderNewGrant>>

GrantSuccessor ==
    /\ ~oldGrant /\ ~newGrant /\ newGrant' = TRUE
    /\ UNCHANGED <<revoked, cancelled, stopped, oldGrant, result,
                   claimedAfterCancellation, wroteUnderNewGrant>>

Next ==
    \/ CommitRevocation \/ ObserveCancellation \/ ClaimUpload
    \/ CompleteUpload \/ VerifyContextStopped \/ ReturnOldGrant \/ GrantSuccessor

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ \A b \in {revoked, cancelled, stopped, oldGrant, newGrant,
                 claimedAfterCancellation, wroteUnderNewGrant} : b \in BOOLEAN
    /\ result \in {"Buffered", "Uploading", "Written", "Discarded"}

\* ClaimUpload is the last liveness check, before the backend copy.
NoClaimAfterLocalCancellation == ~claimedAfterCancellation
ReleasedRequiresStopped == ~oldGrant => stopped
InFlightUploadKeepsGrant == result = "Uploading" => oldGrant
NoWriteUnderSuccessorGrant == ~wroteUnderNewGrant
====
