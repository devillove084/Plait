---- MODULE PlaitW4A4Check ----
EXTENDS PlaitW4A4, FiniteSets

VARIABLE checked
Init == checked = FALSE
Next == checked' = TRUE
Spec == Init /\ [][Next]_<<checked>>

AllSingleTerms ==
    \A a, w, z \in Nibbles :
        LogicalTerm(a, w, z) = NativeTerm(a, w) + (8 - z) * SignedNibble(a)

\* Mapping from PTX m16n8k64 S4 registers into the logical tile.
LaneGroup(lane) == lane \div 4
LaneInGroup(lane) == lane % 4
AFragment(lane, reg, code) ==
    <<LaneGroup(lane) + 8 * (reg % 2),
      8 * LaneInGroup(lane) + 32 * (reg \div 2) + code>>
BFragment(lane, reg, code) ==
    <<8 * LaneInGroup(lane) + 32 * reg + code, LaneGroup(lane)>>
DFragment(lane, reg) == <<LaneGroup(lane) + 8 * (reg \div 2), 2 * LaneInGroup(lane) + (reg % 2)>>

\* The operand register files have the same cardinalities as the matrices.
\* Their images therefore establish coverage and unique ownership.
Tile(rows, cols) == (0..(rows - 1)) \X (0..(cols - 1))
Lanes == 0..31
FragmentCoverage ==
    /\ {AFragment(lane, reg, code) : lane \in Lanes, reg \in 0..3, code \in 0..7}
        = Tile(16, 64)
    /\ {BFragment(lane, reg, code) : lane \in Lanes, reg \in 0..1, code \in 0..7}
        = Tile(64, 8)
    /\ {DFragment(lane, reg) : lane \in Lanes, reg \in 0..3} = Tile(16, 8)

\* Exercise cutpoints on either side of packed-register and native K-tile
\* boundaries, including odd group starts.
TailSafety ==
    \A k \in {1, 7, 8, 9, 63, 64, 65, 127, 128, 129} :
        \A begin \in {0, 1, 3, 7}, count \in {0, 1, 7, 8, 9, 63, 64, 65},
           offset \in {0, 1, 7, 8, 15, 32, 63, 64}, nibble \in 0..7 :
            begin + count <= k =>
                /\ GuardedRead(count, offset, nibble) =>
                    PackedAddress(begin, offset, nibble) \in 0..(PackedStride(k) - 1)
                /\ ~GuardedRead(count, offset, nibble) =>
                    NativeTerm(MaskedActivation(7, count, offset, nibble), 15) = 0
                /\ AlignedRead(begin, count, offset) =>
                    \A b \in 0..3 : (((begin + offset) \div 2) + b) \in 0..(PackedStride(k) - 1)

RepresentableAtAbiLimit == NativeMagnitudeBound(AbiGroupLimit) <= 2147483647
====
