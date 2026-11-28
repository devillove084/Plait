---- MODULE PlaitHostFoldOrder ----
\* One vector block and one tail element stand for the two FP16 sum loops.
\* Slot traces represent FP32 adds before a single FP16 conversion. Rust
\* bit-pattern tests check the arithmetic and encoding on actual inputs.
EXTENDS Naturals, Sequences

CONSTANTS KeepSlotOrder, NarrowAfterAllSlots, DetectNonFinite

Blocks == 1..2
Slots == 1..4
Canonical == <<1, 2, 3, 4>>
SpecialLocations == {<<0, 0>>} \cup (Blocks \X Slots)

VARIABLES special, block, trace, narrowed, route
vars == <<special, block, trace, narrowed, route>>

Init ==
    /\ special \in SpecialLocations
    /\ block = 1
    /\ trace = <<>> /\ narrowed = <<>>
    /\ route = "fast"

ReadSlot(slot) ==
    /\ route = "fast" /\ block \in Blocks
    /\ slot \notin {trace[i] : i \in DOMAIN trace}
    /\ KeepSlotOrder => slot = Len(trace) + 1
    /\ IF DetectNonFinite /\ special = <<block, slot>>
       THEN /\ route' = "fallback"
            /\ trace' = <<>> /\ narrowed' = <<>>
       ELSE /\ trace' = Append(trace, slot)
            /\ UNCHANGED <<route, narrowed>>
    /\ UNCHANGED <<block, special>>

Narrow ==
    /\ route = "fast" /\ block \in Blocks
    /\ Len(trace) = Len(Canonical) \/ (~NarrowAfterAllSlots /\ Len(trace) > 0)
    /\ narrowed' = Append(narrowed, trace)
    /\ block' = block + 1
    /\ trace' = <<>>
    /\ UNCHANGED <<route, special>>

Publish ==
    /\ route = "fast" /\ block = 3
    /\ route' = "vector"
    /\ UNCHANGED <<special, block, trace, narrowed>>

Scalar ==
    /\ route = "fallback"
    /\ route' = "scalar"
    /\ UNCHANGED <<special, block, trace, narrowed>>

Next == (\E slot \in Slots : ReadSlot(slot)) \/ Narrow \/ Publish \/ Scalar
Spec == Init /\ [][Next]_vars

SlotOrder == trace = SubSeq(Canonical, 1, Len(trace))
NarrowedOnceAfterFullFold ==
    \A i \in DOMAIN narrowed : narrowed[i] = Canonical
PublishedIsComplete ==
    /\ route = "vector" => Len(narrowed) = 2 /\ special = <<0, 0>>
    /\ route \in {"fallback", "scalar"} => narrowed = <<>> /\ trace = <<>>
UnpublishedCanAdvance == route \in {"fast", "fallback"} => ENABLED Next
====
