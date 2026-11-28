---- MODULE PlaitKernelPrimitivesCheck ----
EXTENDS PlaitKernelPrimitives

VARIABLE checked
Init == checked = FALSE
Next == checked' = TRUE
Spec == Init /\ [][Next]_<<checked>>

DecodeCoversSignedRange ==
    \A bits \in {1, 4, 8} :
        /\ {DecodeSigned(code, bits) : code \in UnsignedCodes(bits)} = SignedValues(bits)
        /\ \A code \in UnsignedCodes(bits) :
            DecodeSigned(code, bits) % (2 ^ bits) = code

PackedCoordinates ==
    \A count \in 0..65, slotsPerWord \in {1, 2, 4, 8} :
        LET words == PackedWordCount(count, slotsPerWord) IN
        /\ count <= words * slotsPerWord
        /\ words = 0 \/ (words - 1) * slotsPerWord < count
        /\ \A column \in 0..(count - 1) :
            LET word == PackedWordIndex(column, slotsPerWord)
                slot == PackedSlot(column, slotsPerWord)
            IN /\ word \in 0..(words - 1)
               /\ slot \in 0..(slotsPerWord - 1)
               /\ column = word * slotsPerWord + slot

NibbleWordsRoundTrip ==
    \A low, high \in UnsignedCodes(4) :
        LET word == PackCodes(<<low, high>>, 4) IN
        /\ word \in UnsignedCodes(8)
        /\ word = 16 * high + low
        /\ ExtractPackedCode(word, 0, 4) = low
        /\ ExtractPackedCode(word, 1, 4) = high

PackedTailIsZero ==
    \A code \in UnsignedCodes(4) :
        LET word == PackCodes(<<code>>, 4) IN
        /\ ExtractPackedCode(word, 0, 4) = code
        /\ ExtractPackedCode(word, 1, 4) = 0

SignedDotComposition ==
    \A a, b, c, d \in {-128, -8, -1, 0, 1, 7, 127} :
        /\ IntDot(<<a, b>>, <<c, d>>) = (a * c) + (b * d)
        /\ IntDot(<<a, b>>, <<c, d>>) = IntDot(<<a>>, <<c>>) + IntDot(<<b>>, <<d>>)

AccumulatorChecksEveryPrefix ==
    /\ IntDot(<<>>, <<>>) = 0
    /\ IntDotPrefixesFit(<<-8, 7>>, <<1, 1>>, 4)
    /\ FitsSigned(IntDot(<<7, 1, 1>>, <<1, 1, -1>>), 4)
    /\ ~IntDotPrefixesFit(<<7, 1, 1>>, <<1, 1, -1>>, 4)
    /\ ~IntDotPrefixesFit(<<-8, -1>>, <<1, 1>>, 4)
====
