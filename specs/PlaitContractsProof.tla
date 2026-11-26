---- MODULE PlaitContractsProof ----
EXTENDS PlaitW4A4, PlaitDeviceCompletion, NaturalsInduction, TLAPS

THEOREM CenteredNibble ==
    ASSUME NEW code \in Nibbles
    PROVE CenteredCode(code) = code - 8
BY SMT DEF CenteredCode, SignedNibble, Kernel!DecodeSignedWithRadix, Nibbles

THEOREM ZeroPointCorrection ==
    ASSUME NEW activation \in Nibbles, NEW weight \in Nibbles,
           NEW zero \in Nibbles
    PROVE LogicalTerm(activation, weight, zero) =
          NativeTerm(activation, weight) + (8 - zero) * SignedNibble(activation)
<1>1. CenteredCode(weight) = weight - 8
    BY CenteredNibble
<1>2. SignedNibble(activation) \in Int
    BY SMT DEF SignedNibble, Kernel!DecodeSignedWithRadix, Nibbles
<1> QED BY <1>1, <1>2, SMT DEF LogicalTerm, NativeTerm, Nibbles

LOCAL THEOREM PrefixRecurrence ==
    ASSUME NEW length \in Nat, NEW terms
    PROVE PrefixSums(length, terms) =
          [n \in 0..length |-> IF n = 0 THEN 0 ELSE PrefixSums(length, terms)[n - 1] + terms[n]]
<1> DEFINE Step(v, n) == v + terms[n]
<1>1. FiniteNatInductiveDefHypothesis(PrefixSums(length, terms), 0, Step, 0, length)
    BY DEF FiniteNatInductiveDefHypothesis, PrefixSums, Step
<1>2. FiniteNatInductiveDefConclusion(PrefixSums(length, terms), 0, Step, 0, length)
    BY <1>1, FiniteNatInductiveDef
<1> QED BY <1>2 DEF FiniteNatInductiveDefConclusion, Step

LOCAL THEOREM PrefixBase ==
    ASSUME NEW length \in Nat, NEW terms
    PROVE PrefixSums(length, terms)[0] = 0
BY PrefixRecurrence, SMT

LOCAL THEOREM PrefixStep ==
    ASSUME NEW length \in Nat, NEW terms, NEW n \in Nat, n + 1 <= length
    PROVE PrefixSums(length, terms)[n + 1] = PrefixSums(length, terms)[n] + terms[n + 1]
BY PrefixRecurrence, SMT

LOCAL THEOREM PrefixType ==
    ASSUME NEW length \in Nat, NEW terms, \A n \in 1..length: terms[n] \in Int
    PROVE \A n \in 0..length: PrefixSums(length, terms)[n] \in Int
<1> DEFINE S == PrefixSums(length, terms)
           P(n) == n <= length => S[n] \in Int
<1>1. P(0) BY PrefixBase, SMT DEF P, S
<1>2. ASSUME NEW n \in Nat, P(n)
       PROVE P(n + 1)
    <2>1. SUFFICES ASSUME n + 1 <= length
                   PROVE S[n + 1] \in Int
        BY DEF P
    <2>2. n + 1 \in 1..length /\ n <= length
        BY <2>1, SMT
    <2>3. S[n + 1] = S[n] + terms[n + 1]
        BY <2>1, PrefixStep DEF S
    <2> QED BY <2>2, <2>3, P(n), SMT DEF P
<1>3. \A n \in Nat: P(n)
    BY <1>1, <1>2, NatInduction
<1> QED BY <1>3, SMT DEF P, S

LOCAL THEOREM PrefixLinearity ==
    ASSUME NEW length \in Nat, NEW left, NEW base, NEW correction, NEW scale \in Int,
           \A i \in 1..length:
              /\ left[i] \in Int /\ base[i] \in Int /\ correction[i] \in Int
              /\ left[i] = base[i] + scale * correction[i]
    PROVE PrefixSums(length, left)[length] =
          PrefixSums(length, base)[length] + scale * PrefixSums(length, correction)[length]
<1> DEFINE L == PrefixSums(length, left)
           B == PrefixSums(length, base)
           C == PrefixSums(length, correction)
           P(n) == n <= length => L[n] = B[n] + scale * C[n]
<1>1. \A n \in 0..length: L[n] \in Int /\ B[n] \in Int /\ C[n] \in Int
    BY PrefixType, SMT DEF L, B, C
<1>2. L[0] = 0 /\ B[0] = 0 /\ C[0] = 0
    BY PrefixBase, SMT DEF L, B, C
<1>3. P(0) BY <1>2, SMT DEF P
<1>4. ASSUME NEW n \in Nat, P(n)
       PROVE P(n + 1)
    <2>1. SUFFICES ASSUME n + 1 <= length
                   PROVE L[n + 1] = B[n + 1] + scale * C[n + 1]
        BY DEF P
    <2>2. n \in 0..length /\ (n + 1) \in 1..length
        BY <2>1, SMT
    <2>3. L[n] = B[n] + scale * C[n]
        BY <2>2, P(n), SMT DEF P
    <2>4. PrefixSums(length, left)[n + 1] = PrefixSums(length, left)[n] + left[n + 1]
        BY <2>1, PrefixStep
    <2>5. PrefixSums(length, base)[n + 1] = PrefixSums(length, base)[n] + base[n + 1]
        BY <2>1, PrefixStep
    <2>6. PrefixSums(length, correction)[n + 1] =
          PrefixSums(length, correction)[n] + correction[n + 1]
        BY <2>1, PrefixStep
    <2>7. /\ L[n] \in Int /\ B[n] \in Int /\ C[n] \in Int
          /\ left[n + 1] \in Int /\ base[n + 1] \in Int /\ correction[n + 1] \in Int
          /\ left[n + 1] = base[n + 1] + scale * correction[n + 1]
        BY <1>1, <2>2, SMT
    <2> QED BY <2>3, <2>4, <2>5, <2>6, <2>7, SMT DEF L, B, C
<1>5. \A n \in Nat: P(n)
    BY <1>3, <1>4, NatInduction
<1> QED BY <1>5, SMT DEF P, L, B, C

THEOREM DotZeroPointCorrection ==
    ASSUME NEW a, NEW w, NEW zero \in Nibbles,
           Len(a) \in Nat, Len(a) = Len(w),
           \A i \in 1..Len(a): a[i] \in Nibbles /\ w[i] \in Nibbles
    PROVE LogicalDot(a, w, zero) = NativeDot(a, w, zero)
<1>1. \A i \in 1..Len(a):
         LogicalTerm(a[i], w[i], zero) =
             NativeTerm(a[i], w[i]) + (8 - zero) * SignedNibble(a[i])
    BY ZeroPointCorrection, SMT
<1>2. \A i \in 1..Len(a):
         /\ LogicalTerm(a[i], w[i], zero) \in Int
         /\ NativeTerm(a[i], w[i]) \in Int
         /\ SignedNibble(a[i]) \in Int
    BY CenteredNibble, SMT DEF Nibbles, LogicalTerm, NativeTerm, SignedNibble, Kernel!DecodeSignedWithRadix
<1> QED BY <1>1, <1>2, PrefixLinearity, SMT
    DEF LogicalDot, NativeDot, LogicalPrefix, CenteredPrefix, ActivationPrefix, Nibbles

THEOREM GuardedPackedAddress ==
    ASSUME NEW k \in Nat, NEW begin \in Nat, NEW count \in Nat,
           NEW offset \in Nat, NEW nibble \in Nat, begin + count <= k,
           GuardedRead(count, offset, nibble)
    PROVE PackedAddress(begin, offset, nibble) \in 0..(PackedStride(k) - 1)
BY SMT DEF GuardedRead, PackedAddress, PackedStride, Kernel!PackedWordIndex, Kernel!PackedWordCount

LOCAL THEOREM ContiguousPackedAddress ==
    ASSUME NEW begin \in Nat, NEW offset \in Nat, NEW byte \in Nat
    PROVE PackedAddress(begin, offset, 0) + byte = PackedAddress(begin, offset, 2 * byte)
BY SMT DEF PackedAddress, Kernel!PackedWordIndex

THEOREM AlignedPackedAddress ==
    ASSUME NEW k \in Nat, NEW begin \in Nat, NEW count \in Nat,
           NEW offset \in Nat, NEW byte \in 0..3, begin + count <= k,
           AlignedRead(begin, count, offset)
    PROVE /\ PackedAddress(begin, offset, 0) + byte \in 0..(PackedStride(k) - 1)
          /\ PackedAddress(begin, offset, 0) + byte =
             PackedAddress(begin, offset, 2 * byte)
<1>1. /\ byte \in Nat
      /\ 2 * byte \in Nat
      /\ GuardedRead(count, offset, 2 * byte)
    BY SMT DEF AlignedRead, GuardedRead
<1>2. PackedAddress(begin, offset, 2 * byte) \in 0..(PackedStride(k) - 1)
    BY <1>1, GuardedPackedAddress
<1>3. PackedAddress(begin, offset, 0) + byte = PackedAddress(begin, offset, 2 * byte)
    BY <1>1, ContiguousPackedAddress
<1> QED BY <1>2, <1>3

THEOREM MaskedTail ==
    ASSUME NEW count \in Nat, NEW offset \in Nat, NEW nibble \in Nat, NEW code,
           offset + nibble >= count
    PROVE MaskedActivation(code, count, offset, nibble) = 0
BY SMT DEF GuardedRead, MaskedActivation

LOCAL DeviceJobRecord(job) ==
    job = [phase |-> job.phase, stamp |-> job.stamp, event |-> job.event,
           output |-> job.output, snapshot |-> job.snapshot]

THEOREM QueuePreservesSoundness ==
    \A stamp, expected: /\ DeviceJobRecord(QueueDevice(stamp))
                        /\ DeviceJobSound(QueueDevice(stamp), stamp, expected)
BY SMT DEF QueueDevice, EmptyDeviceJob, DeviceJobRecord, DeviceJobSound, DevicePhases

THEOREM WritePreservesSoundness ==
    ASSUME NEW job, NEW stamp, NEW expected,
           DeviceJobRecord(job), DeviceJobSound(job, stamp, expected), CanWriteDevice(job)
    PROVE /\ DeviceJobRecord(WriteDevice(job, expected))
          /\ DeviceJobSound(WriteDevice(job, expected), stamp, expected)
BY SMT DEF CanWriteDevice, WriteDevice, DeviceJobRecord, DeviceJobSound, DevicePhases

THEOREM SignalPreservesSoundness ==
    ASSUME NEW job, NEW stamp, NEW expected,
           DeviceJobRecord(job), DeviceJobSound(job, stamp, expected), CanSignalDevice(job, stamp)
    PROVE /\ DeviceJobRecord(SignalDevice(job, stamp))
          /\ DeviceJobSound(SignalDevice(job, stamp), stamp, expected)
BY SMT DEF CanSignalDevice, SignalDevice, DeviceJobRecord, DeviceJobSound, DevicePhases

THEOREM CopyPreservesSoundness ==
    ASSUME NEW job, NEW stamp, NEW expected,
           DeviceJobRecord(job), DeviceJobSound(job, stamp, expected), CanCopyDevice(job, stamp)
    PROVE /\ DeviceJobRecord(CopyDevice(job))
          /\ DeviceJobSound(CopyDevice(job), stamp, expected)
BY SMT DEF CanCopyDevice, CopyDevice, DeviceJobRecord, DeviceJobSound, DevicePhases

THEOREM DifferentStampRejected ==
    ASSUME NEW job, NEW stamp, NEW stale, NEW expected,
           DeviceJobSound(job, stamp, expected), stale # stamp
    PROVE /\ ~CanSignalDevice(job, stale)
          /\ ~CanCopyDevice(job, stale)
          /\ ~CanPublishDevice(job, stale)
BY SMT DEF CanSignalDevice, CanCopyDevice, CanPublishDevice, DeviceJobSound

THEOREM PublicationRequiresCopy ==
    ASSUME NEW job, NEW stamp, NEW expected,
           DeviceJobSound(job, stamp, expected), CanPublishDevice(job, stamp)
    PROVE /\ job.phase = "Copied"
          /\ job.event = stamp
          /\ job.snapshot = expected
BY SMT DEF CanPublishDevice, DeviceJobSound
====
