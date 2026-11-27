---- MODULE PlaitW4CallerStreamCANN ----
\* The caller-stream buffer protocol with CANN enqueue stages. Each
\* successful launch contributes to a contiguous FIFO prefix. A launch error
\* stops enqueueing; previously queued stages continue to borrow the region.
\* The memory map tags intervals with their latest writer. It abstracts tile
\* contents, device instructions and cache coherence, and does not enumerate
\* bytes. Address arithmetic uses finite fixtures with valid pointer widths.
EXTENDS PlaitW4CallerStream

Alignment == 1024
TileM == 16
TileN == 32
TileK == 32
CubeBudget == 64 * 1024 * 1024
CubeGroupLimit == 256
CeilDiv(n, d) == (n + d - 1) \div d
ActiveK(r) == IF r.shape.k < r.shape.groupSize THEN r.shape.k ELSE r.shape.groupSize
GroupCount(r) == CeilDiv(r.shape.k, r.shape.groupSize)

Plan(r) ==
    LET m == RoundUp(r.shape.m, TileM)
        n == RoundUp(r.shape.n, TileN)
        k == RoundUp(ActiveK(r), TileK)
        g == GroupCount(r)
    IN [a |-> RoundUp(g * m * k, Alignment), b |-> RoundUp(g * n * k, Alignment),
        dots |-> g * m * n * 4, budgetPerGroup |-> (m + n) * k + m * n * 4]

UseCube(r) == ActiveK(r) <= CubeGroupLimit /\
    Plan(r).budgetPerGroup <= (CubeBudget - 3 * Alignment) \div GroupCount(r)

CubeCapacity(r) ==
    LET p == Plan(r) IN p.a + p.b + p.dots + Alignment - 1

\* Required is the common codes/scales prefix length. The real cube data
\* starts at an aligned absolute address inside the optional reserved suffix.
CANNRequired(r) == Required(r) + (IF UseCube(r) THEN CubeCapacity(r) ELSE 0)
AlignedCube(r) == RoundUp(r.base + Required(r), Alignment)
Span(first, limit) == [first |-> first, limit |-> limit]
Overlap(a, b) == a.first < b.limit /\ b.first < a.limit
Inside(a, b) == a.first >= b.first /\ a.limit <= b.limit
CANNFootprint(r) == Span(r.base, r.base + CANNRequired(r))
CodeSpan(r) == Span(r.base, r.base + CodeBytes(r))
ScaleSpan(r) == Span(r.base + ScaleOffset(r), r.base + Required(r))
PackedA(r) == Span(AlignedCube(r), AlignedCube(r) + Plan(r).a)
PackedB(r) == Span(PackedA(r).limit, PackedA(r).limit + Plan(r).b)
DotSpan(r) == Span(PackedB(r).limit, PackedB(r).limit + Plan(r).dots)
CubeSpans(r) == {PackedA(r), PackedB(r), DotSpan(r)}

CubeA4 == Request(Shape(2, 1, 5, 3, 16, TRUE), 4, 4, 8215, 1)
ScalarA8 == Request(Shape(2, 1, 257, 257, 32, TRUE), 8, 4, 524, 1)
CubeA8 == Request(Shape(2, 1, 5, 3, 32, TRUE), 8, 4096, 8219, 2)
DisjointCube == [CubeA8 EXCEPT !.base = 12288]
CANNRequests(j) == CASE j = 1 -> {CubeA4} [] j = 2 -> {ScalarA8} [] OTHER -> {CubeA8, DisjointCube}
BadCANNRequests == {[CubeA4 EXCEPT !.capacity = 8214], [CubeA4 EXCEPT !.base = 2],
                    [ScalarA8 EXCEPT !.bits = 16], [ScalarA8 EXCEPT !.shape.groupSize = 0]}
CANNStatic(r) == StaticValid(r) /\ CANNRequired(r) <= r.capacity

Schedule(r) == IF UseCube(r) THEN <<"Scales", "Codes", "Pack", "Cube", "Finish">>
                            ELSE <<"Scales", "Codes", "Finish">>

\* The first job exercises every failure cutpoint. Its peers model successful
\* enqueues on the same and a competing stream.
AcceptedCounts(j, r) == IF j = 1 THEN 0..Len(Schedule(r)) ELSE {Len(Schedule(r))}

Writes(r, stage) == CASE stage = "Scales" -> {ScaleSpan(r)}
                      [] stage = "Codes" -> {CodeSpan(r)}
                      [] stage = "Pack" -> {PackedA(r), PackedB(r)}
                      [] stage = "Cube" -> {DotSpan(r)}
                      [] OTHER -> {}

Reads(r, stage) == CASE stage = "Codes" -> {ScaleSpan(r)}
                     [] stage = "Pack" -> {CodeSpan(r)}
                     [] stage = "Cube" -> {PackedA(r), PackedB(r)}
                     [] stage = "Finish" -> {ScaleSpan(r), IF UseCube(r) THEN DotSpan(r) ELSE CodeSpan(r)}
                     [] OTHER -> {}

ModelRequests == UNION {CANNRequests(j) : j \in Jobs}
Spans == UNION { {CodeSpan(r), ScaleSpan(r)} \cup (IF UseCube(r) THEN CubeSpans(r) ELSE {}) : r \in ModelRequests}
InputsPresent(j, r, stage) == \A interval \in Reads(r, stage) : memory[interval] = j

CANNJob(phase, r, enqueued) ==
    [phase |-> phase, request |-> r, enqueued |-> enqueued, executed |-> <<>>,
     prefixOnly |-> enqueued < Len(Schedule(r)), outputWritten |-> FALSE,
     deviceRead |-> FALSE, hostRead |-> FALSE]

CANNInit ==
    /\ jobs = [j \in Jobs |-> CANNJob("Idle", CubeA4, 0)]
    /\ queues = [s \in Streams |-> <<>>]
    /\ memory = [interval \in Spans |-> NoOwner]

CANNMayBorrow(r) ==
    \A p \in Jobs : Held(p) /\ jobs[p].request.stream # r.stream =>
        ~Overlap(CANNFootprint(r), CANNFootprint(jobs[p].request))

\* Zero accepted stages model a failure before the first enqueue. For an
\* error after a prefix, the ABI returns before those stages finish.
CANNSubmit(j, r, accepted) ==
    /\ jobs[j].phase = "Idle" /\ CANNStatic(r)
    /\ accepted \in 0..Len(Schedule(r))
    /\ IF accepted = 0 THEN
           /\ jobs' = [jobs EXCEPT ![j] = CANNJob("NoWork", r, 0)]
           /\ UNCHANGED queues
       ELSE /\ CANNMayBorrow(r)
            /\ jobs' = [jobs EXCEPT ![j] = CANNJob("Queued", r, accepted)]
            /\ queues' = [queues EXCEPT ![r.stream] = Append(@, j)]
    /\ UNCHANGED memory

JobStage(j) == Schedule(jobs[j].request)[Len(jobs[j].executed) + 1]
Runnable(j) == jobs[j].phase = "Queued" /\ AtHead(j) /\ Len(jobs[j].executed) < jobs[j].enqueued

RunStage(j) ==
    /\ Runnable(j)
    /\ LET r == jobs[j].request
           stage == JobStage(j)
           written == Writes(r, stage)
           last == Len(jobs[j].executed) + 1 = jobs[j].enqueued
       IN /\ InputsPresent(j, r, stage)
          /\ jobs' = [jobs EXCEPT ![j].executed = Append(@, stage), ![j].outputWritten = stage = "Finish",
                       ![j].phase = IF last THEN "Done" ELSE "Queued"]
          /\ queues' = IF last THEN [queues EXCEPT ![r.stream] = Tail(@)] ELSE queues
          /\ memory' = [interval \in Spans |->
                  IF interval \in written THEN j
                  ELSE IF \E other \in written : Overlap(interval, other) THEN NoOwner ELSE memory[interval]]

CANNNext ==
    \E j \in Jobs :
        \/ RunStage(j)
        \/ ReadOnHost(j)
        \/ (\E s \in Streams : Observe(j, s) \/ ReadOnStream(j, s))
        \/ (\E r \in CANNRequests(j) : \E accepted \in AcceptedCounts(j, r) : CANNSubmit(j, r, accepted))
        \/ (\E invalid \in BadCANNRequests : CANNSubmit(j, invalid, 1))

CANNSpec == CANNInit /\ [][CANNNext]_vars

CANNLayout ==
    /\ CANNRequired(CubeA4) = 8215 /\ CANNRequired(CubeA8) = 8219 /\ CANNRequired(ScalarA8) = 524
    /\ ~UseCube(ScalarA8) /\ UseCube(CubeA4) /\ UseCube(CubeA8)
    /\ \A r \in ModelRequests :
        /\ ScaleSpan(r).first % 4 = 0 /\ CodeSpan(r).limit <= ScaleSpan(r).first
        /\ Inside(CodeSpan(r), CANNFootprint(r)) /\ Inside(ScaleSpan(r), CANNFootprint(r))
        /\ UseCube(r) =>
              /\ ScaleSpan(r).limit <= PackedA(r).first
              /\ \A a \in CubeSpans(r) :
                    /\ a.first % Alignment = 0 /\ Inside(a, CANNFootprint(r))
                    /\ \A b \in CubeSpans(r) \ {a} : ~Overlap(a, b)

CANNInputChecksAndLayout ==
    /\ CANNLayout
    /\ \A j \in Jobs : jobs[j].phase # "Idle" => jobs[j].request \in CANNRequests(j)

CANNWorkIsEnqueuedByPrefix ==
    /\ \A j \in Jobs :
        LET work == jobs[j]
            plan == Schedule(work.request)
        IN /\ Len(work.executed) <= work.enqueued /\ work.enqueued <= Len(plan)
           /\ work.executed = SubSeq(plan, 1, Len(work.executed))
           /\ work.phase \in {"Done", "Observed"} => Len(work.executed) = work.enqueued
           /\ work.phase = "Queued" => Len(work.executed) < work.enqueued
           /\ \A r \in CANNRequests(j) : \A accepted \in AcceptedCounts(j, r) \ {0} :
                 ~ENABLED (CANNSubmit(j, r, accepted) /\
                    (memory' # memory \/ queues'[r.stream] # Append(queues[r.stream], j)))
           /\ \A r \in CANNRequests(j) : ~ENABLED (CANNSubmit(j, r, 0) /\ (memory' # memory \/ queues' # queues))

NoOverlapWhileBorrowed ==
    \A j, p \in Jobs : Held(j) /\ Held(p) /\ jobs[j].request.stream # jobs[p].request.stream =>
        ~Overlap(CANNFootprint(jobs[j].request), CANNFootprint(jobs[p].request))

EveryStageReadsItsPredecessors ==
    \A j \in Jobs :
        /\ Runnable(j) => InputsPresent(j, jobs[j].request, JobStage(j)) /\ ENABLED RunStage(j)
        /\ ~AtHead(j) => ~ENABLED RunStage(j)
        /\ ~ENABLED (RunStage(j) /\
            \E interval \in Spans : memory'[interval] = j /\ memory[interval] # j
                                /\ ~Inside(interval, CANNFootprint(jobs[j].request)))

OnlyFinishedOutputIsReadable ==
    /\ ConsumptionNeedsSynchronization
    /\ \A j \in Jobs :
         /\ jobs[j].outputWritten = ("Finish" \in {jobs[j].executed[i] : i \in DOMAIN jobs[j].executed})
         /\ jobs[j].hostRead \/ jobs[j].deviceRead => jobs[j].outputWritten
         /\ jobs[j].prefixOnly => ~jobs[j].outputWritten
====
