---- MODULE PlaitW4Operators ----
EXTENDS Integers, Sequences

Kernel == INSTANCE PlaitKernelPrimitives

\* W4A8 consumes signed bytes. Widened W4 codes are biased by 128 in
\* each byte before subtracting the zero point. A lane cannot borrow.
\* Model a native four-byte word as two halfwords to stay within TLC's
\* signed integer range. The no-borrow condition applies at the split.
BiasedWeights(codes) == Kernel!PackCodes([i \in DOMAIN codes |-> 128 + codes[i]], 8)
RepeatedZero(codes, zero) == Kernel!PackCodes([i \in DOMAIN codes |-> zero], 8)
SubtractWeightZero(codes, zero) == BiasedWeights(codes) - RepeatedZero(codes, zero)

UnbiasSignedByte(code) == (code + 128) % 256
A8WeightHalfword(codes, zero) ==
    LET word == SubtractWeightZero(codes, zero)
    IN [i \in DOMAIN codes |->
        Kernel!DecodeSigned(UnbiasSignedByte(Kernel!ExtractPackedCode(word, i - 1, 8)), 8)]

NativeA8Weights(codes, zero) ==
    A8WeightHalfword(SubSeq(codes, 1, 2), zero) \o
    A8WeightHalfword(SubSeq(codes, 3, 4), zero)

SignedActivations(codes) == [i \in DOMAIN codes |-> Kernel!DecodeSigned(codes[i], 8)]
LogicalWeights(codes, zero) == [i \in DOMAIN codes |-> codes[i] - zero]
\* One register block. Both operands contain four codes, with an
\* activation code of zero in each unused lane of a tail block.
A8NativeDot(a, w, zero) == Kernel!IntDot(SignedActivations(a), NativeA8Weights(w, zero))
A8LogicalDot(a, w, zero) == Kernel!IntDot(SignedActivations(a), LogicalWeights(w, zero))

\* Finite W4A16 values use a common exact scale for activations and another
\* for weight scales. The bias is in the resulting product units. The
\* model checks indexing and dequantization before floating-point rounding.
\* The group size is positive. The caller supplies one packed code per
\* activation and a scale and zero point for every represented group.
WeightGroup(column, groupSize) == (column \div groupSize) + 1
DecodeWeight(packed, column) ==
    Kernel!ExtractPackedCode(
        packed[Kernel!PackedWordIndex(column, 2) + 1], Kernel!PackedSlot(column, 2), 4)

ScaledWeight(code, group, scales, zeros) == (code - zeros[group]) * scales[group]
A16PackedDot(a, packed, groupSize, scales, zeros, bias) ==
    bias + Kernel!IntDot(a,
        [i \in DOMAIN a |-> ScaledWeight(DecodeWeight(packed, i - 1),
            WeightGroup(i - 1, groupSize), scales, zeros)])

A16LogicalDot(a, codes, groupSize, scales, zeros, bias) ==
    LET groupOf(i) == CHOOSE group \in DOMAIN scales :
            (group - 1) * groupSize < i /\ i <= group * groupSize
        weights == [i \in DOMAIN codes |-> (codes[i] - zeros[groupOf(i)]) * scales[groupOf(i)]]
    IN bias + Kernel!IntDot(a, weights)
====
