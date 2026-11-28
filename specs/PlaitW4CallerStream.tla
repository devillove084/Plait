---- MODULE PlaitW4CallerStream ----
\* Raw callers own the buffers and synchronization. Borrowed inputs and
\* outputs remain live and disjoint from scratch. Host calls sharing scratch
\* on one stream enqueue contiguous pairs. The ABI does not enforce these
\* obligations. The model uses byte ranges and an odd-sized, two-row fixture.
\* Finite quantization uses signed integer fixtures with unit scales. Driver
\* synchronization and floating-point behavior are abstracted at completion.
EXTENDS Integers, Sequences

Kernel == INSTANCE PlaitKernelPrimitives

Jobs == 1..3
Streams == 1..2
Rows == 1..2
K == 5
Groups == 1..2
ByteAddresses == 0..59
NoOwner == 0

Shape(m, n, k, groupSize, out, fits) ==
    [m |-> m, n |-> n, k |-> k, groupSize |-> groupSize, out |-> out, fits |-> fits]
Request(shape, bits, base, capacity, stream) ==
    [shape |-> shape, bits |-> bits, base |-> base, capacity |-> capacity, stream |-> stream]
A4 == Request(Shape(2, 1, K, 3, 16, TRUE), 4, 0, 24, 1)
A8 == Request(Shape(2, 1, K, 3, 32, TRUE), 8, 0, 28, 1)
Requests(j) == CASE j = 1 -> {A4} [] j = 2 -> {A8}
                 [] OTHER -> {[A4 EXCEPT !.base = base, !.stream = 2] : base \in {0, 4, 32}}
Invalid == {[A4 EXCEPT !.capacity = 23], [A4 EXCEPT !.bits = 16], [A4 EXCEPT !.shape.m = 0],
            [A4 EXCEPT !.shape.groupSize = 0], [A4 EXCEPT !.shape.out = 8],
            [A4 EXCEPT !.shape.fits = FALSE], [A4 EXCEPT !.base = 2]}

RoundUp(n, d) == ((n + d - 1) \div d) * d
CodeBytes(r) == r.shape.m * (IF r.bits = 4 THEN (r.shape.k + 1) \div 2 ELSE r.shape.k)
ScaleOffset(r) == RoundUp(CodeBytes(r), 4)
ScaleBytes(r) == 4 * r.shape.m * ((r.shape.k + r.shape.groupSize - 1) \div r.shape.groupSize)
Required(r) == ScaleOffset(r) + ScaleBytes(r)
Footprint(r) == r.base..(r.base + Required(r) - 1)
CodesRegion(r) == r.base..(r.base + CodeBytes(r) - 1)
ScalesRegion(r) == (r.base + ScaleOffset(r))..(r.base + Required(r) - 1)
Touched(r) == CodesRegion(r) \cup ScalesRegion(r)

StaticValid(r) ==
    /\ r.shape.m > 0 /\ r.shape.n > 0 /\ r.shape.k > 0 /\ r.shape.groupSize > 0
    /\ r.shape.fits /\ r.shape.out \in {16, 32} /\ r.bits \in {4, 8}
    /\ r.base % 4 = 0 /\ Required(r) <= r.capacity

GroupOf(column) == ((column - 1) \div 3) + 1
BadGroup(j, row, g) == (j = 1 /\ row = 1 /\ g = 2) \/ (j = 2 /\ row = 2 /\ g = 1)
FiniteCode(j, r, row, column) ==
    LET code == ((j + row + column) % 7) + 1
        magnitude == code * (IF r.bits = 4 THEN 1 ELSE 13)
    IN IF column % 2 = 0 THEN 0 - magnitude ELSE magnitude

Unsigned(value, bits) == (value + 2 ^ bits) % (2 ^ bits)
Packed(weights) == [i \in 1..((Len(weights) + 1) \div 2) |->
    Kernel!PackCodes(<<weights[2 * i - 1], IF 2 * i <= Len(weights) THEN weights[2 * i] ELSE 0>>, 4)]

Scales(j) == [row \in Rows |-> [g \in Groups |-> IF BadGroup(j, row, g) THEN "NaN" ELSE "Unit"]]
Codes(j, r, row) == LET codes == [i \in 1..K |->
    IF BadGroup(j, row, GroupOf(i)) THEN 0 ELSE Unsigned(FiniteCode(j, r, row, i), r.bits)]
    IN IF r.bits = 4 THEN Packed(codes) ELSE codes
QuantizeImage(j, r) == [codes |-> [row \in Rows |-> Codes(j, r, row)], scales |-> Scales(j)]
EmptyImage == [codes |-> [row \in Rows |-> <<>>], scales |-> [row \in Rows |-> [g \in Groups |-> "None"]]]
ReadCode(codes, bits, i) ==
    IF bits = 4 THEN Kernel!ExtractPackedCode(codes[((i - 1) \div 2) + 1], (i - 1) % 2, 4) ELSE codes[i]
Weights == <<9, 2, 7, 11, 4>>
CanonicalWeights == Packed(Weights)
Zero == <<3, 6>>
RowResult(nan, value) == [nan |-> nan, value |-> value]
NoOutput == [row \in Rows |-> RowResult(FALSE, 0)]

GemmImage(r, image) == [row \in Rows |->
    IF \E g \in Groups : image.scales[row][g] = "NaN" THEN RowResult(TRUE, 0)
    ELSE RowResult(FALSE, Kernel!IntDot(
        [i \in 1..K |-> Kernel!DecodeSigned(ReadCode(image.codes[row], r.bits, i), r.bits)],
        [i \in 1..K |-> ReadCode(CanonicalWeights, 4, i) - Zero[GroupOf(i)]]))]

Reference(j, r) == [row \in Rows |->
    IF \E g \in Groups : BadGroup(j, row, g) THEN RowResult(TRUE, 0)
    ELSE RowResult(FALSE, Kernel!IntDot([i \in 1..K |-> FiniteCode(j, r, row, i)],
                                      [i \in 1..K |-> Weights[i] - Zero[GroupOf(i)]]))]

Job(phase, request, prefixOnly) == [phase |-> phase, request |-> request, prefixOnly |-> prefixOnly,
    image |-> EmptyImage, output |-> NoOutput, deviceRead |-> FALSE, hostRead |-> FALSE]

VARIABLES jobs, queues, memory
vars == <<jobs, queues, memory>>

Init ==
    /\ jobs = [j \in Jobs |-> Job("Idle", A4, FALSE)]
    /\ queues = [s \in Streams |-> <<>>]
    /\ memory = [address \in ByteAddresses |-> NoOwner]

Held(j) == jobs[j].phase \in {"Queued", "Quantized", "Done"}
MayBorrow(request) ==
    \A p \in Jobs : Held(p) /\ jobs[p].request.stream # request.stream =>
                    Footprint(request) \cap Footprint(jobs[p].request) = {}

Submit(j, request, prefixOnly) ==
    /\ jobs[j].phase = "Idle" /\ StaticValid(request) /\ MayBorrow(request)
    /\ jobs' = [jobs EXCEPT ![j] = Job("Queued", request, prefixOnly)]
    /\ queues' = [queues EXCEPT ![request.stream] = Append(@, j)]
    /\ UNCHANGED memory

AtHead(j) == LET q == queues[jobs[j].request.stream] IN Len(q) > 0 /\ q[1] = j

Quantize(j) ==
    /\ jobs[j].phase = "Queued" /\ AtHead(j)
    /\ LET request == jobs[j].request IN
       /\ jobs' = [jobs EXCEPT ![j].phase = "Quantized", ![j].image = QuantizeImage(j, request)]
       /\ memory' = [address \in ByteAddresses |-> IF address \in Touched(request) THEN j ELSE memory[address]]
    /\ UNCHANGED queues

WorkspaceWritten(j) == \A address \in Touched(jobs[j].request) : memory[address] = j

\* A failed GEMM submission may leave a quantizer queued. It drains through
\* the same FIFO and holds its scratch lease until observed completion.
Execute(j) ==
    /\ jobs[j].phase = "Quantized" /\ AtHead(j) /\ WorkspaceWritten(j)
    /\ LET request == jobs[j].request IN
       /\ jobs' = [jobs EXCEPT ![j].phase = "Done", ![j].output =
                     IF jobs[j].prefixOnly THEN NoOutput ELSE GemmImage(request, jobs[j].image)]
       /\ queues' = [queues EXCEPT ![request.stream] = Tail(@)]
    /\ UNCHANGED memory

Observe(j, stream) ==
    /\ jobs[j].phase = "Done" /\ jobs[j].request.stream = stream
    /\ jobs' = [jobs EXCEPT ![j].phase = "Observed"]
    /\ UNCHANGED <<queues, memory>>

ReadOnStream(j, stream) ==
    /\ ~jobs[j].prefixOnly /\ ~jobs[j].deviceRead
    /\ jobs[j].phase = "Observed" \/ (jobs[j].phase = "Done" /\ stream = jobs[j].request.stream)
    /\ jobs' = [jobs EXCEPT ![j].deviceRead = TRUE]
    /\ UNCHANGED <<queues, memory>>

ReadOnHost(j) ==
    /\ ~jobs[j].prefixOnly /\ ~jobs[j].hostRead /\ jobs[j].phase = "Observed"
    /\ jobs' = [jobs EXCEPT ![j].hostRead = TRUE]
    /\ UNCHANGED <<queues, memory>>

Next ==
    \/ \E j \in Jobs : Quantize(j) \/ Execute(j) \/ ReadOnHost(j)
        \/ \E stream \in Streams : Observe(j, stream) \/ ReadOnStream(j, stream)
        \/ \E request \in Requests(j) \cup Invalid, prefixOnly \in BOOLEAN : Submit(j, request, prefixOnly)

Spec == Init /\ [][Next]_vars

InvalidSubmissionsNeverQueue ==
    \A j \in Jobs, r \in Invalid, prefixOnly \in BOOLEAN : ~ENABLED Submit(j, r, prefixOnly)

SubmissionOnlyAppends ==
    \A j \in Jobs : \A r \in Requests(j), prefixOnly \in BOOLEAN :
        ~ENABLED (Submit(j, r, prefixOnly) /\
            (memory' # memory \/ jobs'[j].phase # "Queued" \/ jobs'[j].output # NoOutput \/
             queues'[r.stream] # Append(queues[r.stream], j)))

WorkspaceFitsAndIsAligned ==
    \A j \in Jobs : jobs[j].phase # "Idle" =>
        LET r == jobs[j].request IN
        /\ Footprint(r) \subseteq ByteAddresses /\ Touched(r) \subseteq Footprint(r)
        /\ CodesRegion(r) \cap ScalesRegion(r) = {}
        /\ (r.base + ScaleOffset(r)) % 4 = 0
        /\ Required(r) = IF r.bits = 4 THEN 24 ELSE 28
        /\ Required(r) <= r.capacity

QuantizeWritesOnlyItsRegions ==
    \A j \in Jobs :
        ~ENABLED (Quantize(j) /\
            \E address \in ByteAddresses \ Touched(jobs[j].request) : memory'[address] # memory[address])

CrossStreamScratchIsExclusive ==
    \A j, p \in Jobs : Held(j) /\ Held(p) /\ jobs[j].request.stream # jobs[p].request.stream =>
        Footprint(jobs[j].request) \cap Footprint(jobs[p].request) = {}

StreamRunsInOrder ==
    \A j \in Jobs :
        /\ jobs[j].phase = "Quantized" => AtHead(j) /\ WorkspaceWritten(j)
        /\ IF AtHead(j) THEN TRUE ELSE ~ENABLED Quantize(j) /\ ~ENABLED Execute(j)
        /\ ~ENABLED (Execute(j) /\ (jobs[j].phase # "Quantized" \/ ~WorkspaceWritten(j)))

BadGroupZerosAndPropagates ==
    \A j \in Jobs :
        LET step == jobs[j] IN
        step.phase \in {"Quantized", "Done", "Observed"} =>
            /\ \A row \in Rows, g \in Groups :
                  /\ (step.image.scales[row][g] = "NaN") = BadGroup(j, row, g)
                  /\ BadGroup(j, row, g) => \A i \in 1..K : GroupOf(i) = g =>
                               ReadCode(step.image.codes[row], step.request.bits, i) = 0
            /\ step.phase \in {"Done", "Observed"} /\ ~step.prefixOnly => step.output = Reference(j, step.request)

ConsumptionNeedsSynchronization ==
    \A j \in Jobs :
        /\ jobs[j].hostRead => jobs[j].phase = "Observed" /\ ~jobs[j].prefixOnly
        /\ jobs[j].deviceRead => jobs[j].phase \in {"Done", "Observed"} /\ ~jobs[j].prefixOnly
        /\ \A s \in Streams :
               /\ s # jobs[j].request.stream /\ jobs[j].phase # "Observed" => ~ENABLED ReadOnStream(j, s)
               /\ s # jobs[j].request.stream \/ jobs[j].phase # "Done" => ~ENABLED Observe(j, s)
====
