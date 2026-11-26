---- MODULE PlaitW4A4 ----
EXTENDS Integers, Sequences

Kernel == INSTANCE PlaitKernelPrimitives

\* Logical codes are ordered by K. The low half-byte belongs to the even
\* zero-based column. Arithmetic here is the INT32 stage before scaling.
Nibbles == 0..15
SignedNibble(code) == Kernel!DecodeSignedWithRadix(code, 16)
CenteredCode(code) == SignedNibble((code + 8) % 16)
LogicalTerm(activation, weight, zero) == SignedNibble(activation) * (weight - zero)
NativeTerm(activation, weight) == SignedNibble(activation) * CenteredCode(weight)

PrefixSums(length, terms) ==
    LET sums[n \in 0..length] == IF n = 0 THEN 0 ELSE sums[n - 1] + terms[n]
    IN sums
LogicalPrefix(a, w, zero, count) ==
    PrefixSums(Len(a), [n \in 1..Len(a) |-> LogicalTerm(a[n], w[n], zero)])[count]
CenteredPrefix(a, w, count) ==
    PrefixSums(Len(a), [n \in 1..Len(a) |-> NativeTerm(a[n], w[n])])[count]
ActivationPrefix(a, count) ==
    PrefixSums(Len(a), [n \in 1..Len(a) |-> SignedNibble(a[n])])[count]

LogicalDot(a, w, zero) == LogicalPrefix(a, w, zero, Len(a))
NativeDot(a, w, zero) ==
    CenteredPrefix(a, w, Len(a)) + (8 - zero) * ActivationPrefix(a, Len(a))

\* The caller guarantees begin + count <= k. A complete aligned register
\* may read four packed bytes. An odd start or tail checks each nibble.
PackedStride(k) == Kernel!PackedWordCount(k, 2)
GuardedRead(count, offset, nibble) == offset < count /\ nibble < count - offset
AlignedRead(begin, count, offset) == (begin + offset) % 2 = 0 /\ offset + 8 <= count
PackedAddress(begin, offset, nibble) == Kernel!PackedWordIndex(begin + offset + nibble, 2)
MaskedActivation(code, count, offset, nibble) ==
    IF GuardedRead(count, offset, nibble) THEN code ELSE 0

\* In the native S4 path |x * (q - 8)| <= 64 and |(8-z) * x| <= 64.
\* The ABI cap is shared with A8 and avoids wrapped signed accumulation.
AbiGroupLimit == 1118481
NativeMagnitudeBound(count) == 128 * count
====
