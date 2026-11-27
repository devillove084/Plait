---- MODULE PlaitGroupPrepareSet ----
\* One shared host. Commands contain one or two preparations. A recorded
\* candidate represents a new Preparing version holding its reservation.
\* Per-plan checks read the ordered state. Staged grants and host totals live
\* in one temporary value and are published only with the whole prepare set.
\* Two operation IDs race for candidates or devices. Each replays its first
\* result. Resource releases and limits on integer widths are outside this slice.
EXTENDS Naturals, Sequences, FiniteSets

First == 1
Second == 2
Incumbent == 3
Other == 4
Groups == {First, Second, Incumbent, Other}
HeldDevice == 1
FirstDevice == 2
SecondDevice == 3
Devices == {HeldDevice, FirstDevice, SecondDevice}
Ids == {1, 2}
None == 0
NoGuard == 2

Memory(bytes, pinned) == [bytes |-> bytes, pinned |-> pinned]
Add(a, b) == Memory(a.bytes + b.bytes, a.pinned + b.pinned)
Fits(use, cap) == use.bytes <= cap.bytes /\ use.pinned <= cap.pinned
Capacity == Memory(8, 4)
IncumbentMemory == Memory(2, 1)

Plan(group, epoch, active, device, incarnation, memory, recipe, probe, rate) ==
    [group |-> group, epoch |-> epoch, active |-> active, device |-> device,
     incarnation |-> incarnation, memory |-> memory, recipe |-> recipe, probe |-> probe, rate |-> rate]
NoPlan == Plan(None, 0, 0, None, 0, Memory(0, 0), 0, 0, 0)
Fast == Plan(First, 0, None, FirstDevice, 1, Memory(3, 1), 1, 0, 0)
Measured == Plan(Second, 0, None, SecondDevice, 1, Memory(3, 1), 2, 1, 1)
Unmeasured == [Measured EXCEPT !.recipe = 0, !.probe = 0, !.rate = 0]

Command(id, plans, guard) == [id |-> id, plans |-> plans, guard |-> guard]
Prepare == Command(1, <<Fast, Measured>>, 1)
PrepareUnmeasured == Command(1, <<Fast, Unmeasured>>, NoGuard)
PinOverflow == Command(1, <<[Fast EXCEPT !.memory.pinned = 2], [Measured EXCEPT !.memory.pinned = 2]>>, 1)
HostOverflow == Command(1, <<[Fast EXCEPT !.memory.bytes = 4], [Measured EXCEPT !.memory.bytes = 4]>>, 1)
DuplicateDevice == Command(1, <<Fast, [Unmeasured EXCEPT !.device = FirstDevice]>>, 1)
DuplicateGroup == Command(1, <<Fast, [Unmeasured EXCEPT !.group = First]>>, 1)
WrongActive == Command(1, <<Fast, [Unmeasured EXCEPT !.active = 1]>>, 1)
WrongEpoch == Command(1, <<Fast, [Unmeasured EXCEPT !.epoch = 1]>>, 1)
CompetingEpoch == Command(2, <<Unmeasured>>, NoGuard)
CompetingDevice == Command(2, <<[Unmeasured EXCEPT !.group = Other]>>, NoGuard)
Commands == {Prepare, PrepareUnmeasured, PinOverflow, HostOverflow, DuplicateDevice,
             DuplicateGroup, WrongActive, WrongEpoch, CompetingEpoch, CompetingDevice}

Grant(group, epoch, incarnation, fence) ==
    [group |-> group, epoch |-> epoch, incarnation |-> incarnation, fence |-> fence]
NoGrant == Grant(None, 0, 0, 0)
BaseFence(d) == IF d = HeldDevice THEN 1 ELSE 3
Receipt(command, outcome, versions) == [command |-> command, outcome |-> outcome, versions |-> versions]
NoReceipt == Receipt(Command(0, <<>>, NoGuard), "NotApplied", <<>>)
Accepted == "Prepared"

VARIABLES ledger, receipts, reply
vars == <<ledger, receipts, reply>>

Init ==
    /\ ledger = [latest |-> [g \in Groups |-> IF g = Incumbent THEN 1 ELSE 0],
                  active |-> [g \in Groups |-> IF g = Incumbent THEN 1 ELSE None],
                  candidate |-> [g \in Groups |-> NoPlan],
                  fence |-> [d \in Devices |-> BaseFence(d)],
                  grant |-> [d \in Devices |-> IF d = HeldDevice THEN Grant(Incumbent, 1, 1, 1) ELSE NoGrant],
                  incarnation |-> [d \in Devices |-> 1],
                  rate |-> [d \in Devices |-> 1], probe |-> 1,
                  eligible |-> Devices, hostUp |-> TRUE, reserved |-> IncumbentMemory]
    /\ receipts = [id \in Ids |-> NoReceipt] /\ reply = NoReceipt

Check(g) ==
    CASE g.active # ledger.active[g.group] -> "ActiveChanged"
      [] g.epoch # ledger.latest[g.group] -> "StaleEpoch"
      [] ledger.candidate[g.group] # NoPlan -> "CandidateExists"
      [] g.probe # 0 /\ g.probe # ledger.probe -> "StaleProbe"
      [] g.rate # 0 /\ g.rate # ledger.rate[g.device] -> "StaleRate"
      [] g.device \notin ledger.eligible -> "DeviceDraining"
      [] ledger.incarnation[g.device] # g.incarnation -> "StaleIncarnation"
      [] ledger.grant[g.device] # NoGrant -> "WorkerBusy"
      [] ~ledger.hostUp -> "HostDraining"
      [] OTHER -> Accepted

NewScratch == [outcome |-> Accepted, reserved |-> ledger.reserved, devices |-> {},
               groups |-> {}, grants |-> [d \in Devices |-> NoGrant]]

Stage(g, scratch) ==
    LET total == Add(scratch.reserved, g.memory)
        checked == Check(g)
    IN CASE scratch.outcome # Accepted -> scratch
         [] g.group \in scratch.groups -> [scratch EXCEPT !.outcome = "DuplicateGroup"]
         [] checked # Accepted -> [scratch EXCEPT !.outcome = checked]
         [] g.device \in scratch.devices -> [scratch EXCEPT !.outcome = "WorkerBusy"]
         [] ~Fits(total, Capacity) -> [scratch EXCEPT !.outcome = "HostCapacity"]
         [] OTHER ->
              [scratch EXCEPT !.reserved = total, !.devices = @ \cup {g.device},
                       !.groups = @ \cup {g.group},
                       !.grants[g.device] = Grant(g.group, g.epoch + 1, g.incarnation, ledger.fence[g.device] + 1)]

Staged(c) ==
    LET head == Stage(c.plans[1], NewScratch)
    IN IF c.guard # NoGuard /\ c.guard # ledger.active[Incumbent]
       THEN [NewScratch EXCEPT !.outcome = "ActiveSetChanged"]
       ELSE IF Len(c.plans) = 1 THEN head ELSE Stage(c.plans[2], head)

CommandGroups(c) == {c.plans[i].group : i \in DOMAIN c.plans}
CommandDevices(c) == {c.plans[i].device : i \in DOMAIN c.plans}
Versions(c) == [i \in DOMAIN c.plans |-> <<c.plans[i].group, c.plans[i].epoch + 1>>]

Install(c, staged) ==
    [ledger EXCEPT !.reserved = staged.reserved,
        !.latest = [g \in Groups |-> IF g \in staged.groups THEN ledger.latest[g] + 1 ELSE ledger.latest[g]],
        !.candidate = [g \in Groups |-> IF g \in staged.groups
                        THEN c.plans[CHOOSE i \in DOMAIN c.plans : c.plans[i].group = g] ELSE ledger.candidate[g]],
        !.grant = [d \in Devices |-> IF d \in staged.devices THEN staged.grants[d] ELSE ledger.grant[d]],
        !.fence = [d \in Devices |-> IF d \in staged.devices THEN staged.grants[d].fence ELSE ledger.fence[d]]]

Apply(c) ==
    LET old == receipts[c.id]
    IN IF old # NoReceipt THEN
           /\ reply' = IF old.command = c THEN old ELSE Receipt(c, "OperationIdReused", <<>>)
           /\ UNCHANGED <<ledger, receipts>>
       ELSE LET staged == Staged(c)
                success == staged.outcome = Accepted
                result == Receipt(c, staged.outcome, IF success THEN Versions(c) ELSE <<>>)
            IN /\ ledger' = IF success THEN Install(c, staged) ELSE ledger
               /\ receipts' = [receipts EXCEPT ![c.id] = result]
               /\ reply' = result

DrainIncumbent ==
    /\ ledger.active[Incumbent] = 1
    /\ ledger' = [ledger EXCEPT !.active[Incumbent] = None]
    /\ UNCHANGED <<receipts, reply>>

AdvanceProbe ==
    /\ ledger.probe = 1
    /\ ledger' = [ledger EXCEPT !.probe = 2]
    /\ UNCHANGED <<receipts, reply>>

AdvanceRate ==
    /\ ledger.rate[SecondDevice] = 1
    /\ ledger' = [ledger EXCEPT !.rate[SecondDevice] = 2]
    /\ UNCHANGED <<receipts, reply>>

ReregisterSecond ==
    /\ ledger.grant[SecondDevice] = NoGrant /\ ledger.incarnation[SecondDevice] = 1
    /\ ledger' = [ledger EXCEPT !.incarnation[SecondDevice] = 2, !.rate[SecondDevice] = 2]
    /\ UNCHANGED <<receipts, reply>>

WithdrawSecond ==
    /\ SecondDevice \in ledger.eligible
    /\ ledger' = [ledger EXCEPT !.eligible = @ \ {SecondDevice}]
    /\ UNCHANGED <<receipts, reply>>

WithdrawHost ==
    /\ ledger.hostUp
    /\ ledger' = [ledger EXCEPT !.hostUp = FALSE]
    /\ UNCHANGED <<receipts, reply>>

Next ==
    \/ \E c \in Commands : Apply(c)
    \/ DrainIncumbent \/ AdvanceProbe \/ AdvanceRate \/ ReregisterSecond \/ WithdrawSecond \/ WithdrawHost

Spec == Init /\ [][Next]_vars

\* The reference checks the whole request directly. It does not run Stage or
\* recycle its intermediate host totals, grants, or result code.
Need(c) == IF Len(c.plans) = 1 THEN c.plans[1].memory ELSE Add(c.plans[1].memory, c.plans[2].memory)
Legal(c) ==
    /\ Cardinality(CommandGroups(c)) = Len(c.plans)
    /\ Cardinality(CommandDevices(c)) = Len(c.plans)
    /\ ledger.hostUp /\ Fits(Add(ledger.reserved, Need(c)), Capacity)
    /\ c.guard = NoGuard \/ c.guard = ledger.active[Incumbent]
    /\ \A i \in DOMAIN c.plans :
         LET g == c.plans[i] IN
         /\ ledger.active[g.group] = g.active /\ ledger.latest[g.group] = g.epoch
         /\ ledger.candidate[g.group] = NoPlan
         /\ g.device \in ledger.eligible /\ ledger.grant[g.device] = NoGrant
         /\ ledger.incarnation[g.device] = g.incarnation
         /\ g.probe \in {0, ledger.probe} /\ g.rate \in {0, ledger.rate[g.device]}

SuccessState(c) ==
    /\ reply' = Receipt(c, Accepted, Versions(c))
    /\ ledger'.reserved = Add(ledger.reserved, Need(c))
    /\ \A i \in DOMAIN c.plans :
          LET g == c.plans[i] IN
          /\ ledger'.candidate[g.group] = g /\ ledger'.latest[g.group] = g.epoch + 1
          /\ ledger'.grant[g.device] = Grant(g.group, g.epoch + 1, g.incarnation, ledger.fence[g.device] + 1)
          /\ ledger'.fence[g.device] = ledger.fence[g.device] + 1
    /\ \A g \in Groups \ CommandGroups(c) :
          ledger'.candidate[g] = ledger.candidate[g] /\ ledger'.latest[g] = ledger.latest[g]
    /\ \A d \in Devices \ CommandDevices(c) :
          ledger'.grant[d] = ledger.grant[d] /\ ledger'.fence[d] = ledger.fence[d]

AtomicPrepare ==
    \A c \in Commands :
      /\ ENABLED Apply(c)
      /\ ~ENABLED (Apply(c) /\ ledger'.active # ledger.active)
      /\ receipts[c.id] = NoReceipt =>
          /\ ~ENABLED (Apply(c) /\ receipts'[c.id] # reply')
          /\ IF Legal(c) THEN ~ENABLED (Apply(c) /\ ~SuccessState(c))
             ELSE ~ENABLED (Apply(c) /\ (ledger' # ledger \/ reply'.outcome = Accepted \/ reply'.versions # <<>>))

RetryKeepsReceipt ==
    \A c \in Commands :
      LET old == receipts[c.id] IN
      old # NoReceipt =>
         ~ENABLED (Apply(c) /\ (ledger' # ledger \/ receipts' # receipts \/
             IF old.command = c THEN reply' # old ELSE reply'.outcome # "OperationIdReused"))

GrantsFenceCandidates ==
    /\ ledger.grant[HeldDevice] = Grant(Incumbent, 1, 1, 1) /\ ledger.fence[HeldDevice] = 1
    /\ \A d \in Devices \ {HeldDevice} :
         LET grant == ledger.grant[d] IN
         IF grant = NoGrant THEN ledger.fence[d] = BaseFence(d)
         ELSE /\ grant.group \in Groups /\ ledger.candidate[grant.group].device = d
              /\ grant.epoch = ledger.latest[grant.group] /\ grant.incarnation = ledger.incarnation[d]
              /\ grant.fence = BaseFence(d) + 1 /\ ledger.fence[d] = grant.fence
    /\ \A g \in Groups :
         LET prepared == ledger.candidate[g] IN
         prepared # NoPlan => ledger.grant[prepared.device].group = g /\ ledger.latest[g] = prepared.epoch + 1

LedgerMatchesReservations ==
    /\ Fits(ledger.reserved, Capacity)
    /\ ledger.reserved = Add(IncumbentMemory,
           Add(ledger.candidate[First].memory, Add(ledger.candidate[Second].memory, ledger.candidate[Other].memory)))
====
