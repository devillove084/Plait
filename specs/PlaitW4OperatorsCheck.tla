---- MODULE PlaitW4OperatorsCheck ----
EXTENDS PlaitW4Operators

W4A4 == INSTANCE PlaitW4A4

VARIABLE checked
Init == checked = FALSE
Next == checked' = TRUE
Spec == Init /\ [][Next]_<<checked>>

Nibble == 0..15

NoBorrowA8Byte ==
    \A code, zero \in Nibble :
        /\ 128 + code - zero \in 0..255
        /\ A8WeightHalfword(<<code>>, zero) = LogicalWeights(<<code>>, zero)

A8PackedWeights ==
    \A first, second, zero \in Nibble :
        /\ SubtractWeightZero(<<first, second>>, zero) \in 0..65535
        /\ NativeA8Weights(<<first, second, 15, 0>>, zero) =
            LogicalWeights(<<first, second, 15, 0>>, zero)
        /\ NativeA8Weights(<<15, 0, first, second>>, zero) =
            LogicalWeights(<<15, 0, first, second>>, zero)

A8Accumulator ==
    \A a, b \in {0, 1, 127, 128, 255}, w, z \in Nibble :
        A8NativeDot(<<a, b, 128, 127>>, <<w, 15 - w, 0, 15>>, z) =
        A8LogicalDot(<<a, b, 128, 127>>, <<w, 15 - w, 0, 15>>, z)

A8TermRange ==
    /\ \A byte \in 0..255, code, zero \in Nibble :
           Kernel!DecodeSigned(byte, 8) * (code - zero) \in (-1920)..1920
    /\ W4A4!AbiGroupLimit = 2147483647 \div 1920

\* Odd K, a group boundary inside a packed byte, and independent bias.
A16OddTail ==
    \A first, second, last \in Nibble :
        LET codes == <<first, second, last>>
            packed == <<Kernel!PackCodes(<<first, second>>, 4), Kernel!PackCodes(<<last, 15>>, 4)>>
            a == <<-3, 2, -4>>
            scales == <<2, 5, 7>>
            zeros == <<3, 8, 12>>
        IN \A groupSize \in 1..3, bias \in {-1, 0, 2} :
               A16PackedDot(a, packed, groupSize, scales, zeros, bias) =
               A16LogicalDot(a, codes, groupSize, scales, zeros, bias)
====
