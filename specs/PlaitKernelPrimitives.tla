---- MODULE PlaitKernelPrimitives ----
EXTENDS Integers, Sequences

\* Widths are positive integers and codes are in the corresponding unsigned
\* range. Signed operands use two's complement.
UnsignedCodes(bits) == 0..(2 ^ bits - 1)
SignedValues(bits) == (0 - 2 ^ (bits - 1))..(2 ^ (bits - 1) - 1)

\* Radix is the number of unsigned codes, a power of two greater than one.
DecodeSignedWithRadix(code, radix) ==
    IF code < radix \div 2 THEN code ELSE code - radix

DecodeSigned(code, bits) == DecodeSignedWithRadix(code, 2 ^ bits)

FitsSigned(value, bits) == value \in SignedValues(bits)

\* Words store a positive integral number of equal-width codes, with the
\* lowest logical column in the least significant bits. Column and slot
\* coordinates are zero-based. Sequences of codes use TLA+'s one-based index.
PackedWordCount(elements, slotsPerWord) ==
    (elements + slotsPerWord - 1) \div slotsPerWord

PackedWordIndex(column, slotsPerWord) == column \div slotsPerWord
PackedSlot(column, slotsPerWord) == column % slotsPerWord

ExtractPackedCode(word, slot, codeBits) ==
    (word \div (2 ^ (slot * codeBits))) % (2 ^ codeBits)

PackPrefix(codes, codeBits, count) ==
    LET prefix[i \in 0..count] ==
        IF i = 0 THEN 0
        ELSE prefix[i - 1] + codes[i] * (2 ^ ((i - 1) * codeBits))
    IN prefix[count]

\* The caller supplies at most one word of representable codes.
\* Any remaining high bits in the result are zero.
PackCodes(codes, codeBits) == PackPrefix(codes, codeBits, Len(codes))

\* Inputs have equal length and contain already decoded integers.
\* Dot products use exact integer arithmetic. PrefixesFit expresses the
\* additional obligation for a backend that accumulates in the given order.
IntDotPrefix(left, right, count) ==
    LET prefix[i \in 0..count] ==
        IF i = 0 THEN 0
        ELSE prefix[i - 1] + left[i] * right[i]
    IN prefix[count]

IntDot(left, right) == IntDotPrefix(left, right, Len(left))

IntDotPrefixesFit(left, right, accumulatorBits) ==
    \A count \in 0..Len(left) :
        FitsSigned(IntDotPrefix(left, right, count), accumulatorBits)
====
