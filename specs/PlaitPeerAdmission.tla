---- MODULE PlaitPeerAdmission ----
EXTENDS Naturals, FiniteSets

CONSTANTS CheckQueueBeforeDevice, ReturnSlotOnByteRejection, AdvanceOnlyOnPublish

Calls == 1..3
Cost(call) == IF call = 3 THEN 2 ELSE 1
Slots == 1
Bytes == 1

VARIABLES phase, touched, sequenced
vars == <<phase, touched, sequenced>>
Init ==
    /\ phase = [call \in Calls |-> "new"]
    /\ touched = {} /\ sequenced = {}

SlotHolders == {call \in Calls :
    phase[call] \in {"slot", "admitted", "staged", "queued", "leaked"}}
ByteHolders == {call \in Calls :
    phase[call] \in {"admitted", "staged", "queued", "running"}}
HeldBytes == Cardinality(ByteHolders) + Cardinality(ByteHolders \cap {3})

TrySlot(call) ==
    /\ phase[call] = "new"
    /\ LET sufficient == Cardinality(SlotHolders) < Slots
       IN /\ phase' = [phase EXCEPT ![call] = IF sufficient THEN "slot" ELSE "busy"]
          /\ touched' = IF ~sufficient /\ ~CheckQueueBeforeDevice
                        THEN touched \cup {call} ELSE touched
          /\ sequenced' = IF ~sufficient /\ ~AdvanceOnlyOnPublish
                          THEN sequenced \cup {call} ELSE sequenced

TryBytes(call) ==
    /\ phase[call] = "slot"
    /\ LET sufficient == HeldBytes + Cost(call) <= Bytes
       IN phase' = [phase EXCEPT ![call] =
           IF sufficient THEN "admitted"
           ELSE IF ReturnSlotOnByteRejection THEN "busy" ELSE "leaked"]
    /\ UNCHANGED <<touched, sequenced>>

Stage(call) ==
    /\ phase[call] = "admitted"
    /\ phase' = [phase EXCEPT ![call] = "staged"]
    /\ touched' = touched \cup {call}
    /\ UNCHANGED sequenced

Publish(call) ==
    /\ phase[call] = "staged"
    /\ phase' = [phase EXCEPT ![call] = "queued"]
    /\ sequenced' = sequenced \cup {call}
    /\ UNCHANGED touched

AbortStage(call) ==
    /\ phase[call] \in {"admitted", "staged"}
    /\ phase' = [phase EXCEPT ![call] = "failed"]
    /\ UNCHANGED <<touched, sequenced>>

Dequeue(call) ==
    /\ phase[call] = "queued"
    /\ phase' = [phase EXCEPT ![call] = "running"]
    /\ UNCHANGED <<touched, sequenced>>

Finish(call) ==
    /\ phase[call] = "running"
    /\ phase' = [phase EXCEPT ![call] = "done"]
    /\ UNCHANGED <<touched, sequenced>>

Next == \E call \in Calls :
    TrySlot(call) \/ TryBytes(call) \/ Stage(call) \/ Publish(call)
    \/ AbortStage(call) \/ Dequeue(call) \/ Finish(call)
Spec == Init /\ [][Next]_vars
BusyDoesNotTouchDevice == \A call \in Calls : phase[call] = "busy" => call \notin touched
SequenceOnlyOnPublish == \A call \in sequenced : phase[call] \in {"queued", "running", "done"}
NoLeakedSlot == \A call \in Calls : phase[call] # "leaked"
WithinLimits == Cardinality(SlotHolders) <= Slots /\ HeldBytes <= Bytes
====
