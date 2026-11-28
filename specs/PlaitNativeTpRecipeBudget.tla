---- MODULE PlaitNativeTpRecipeBudget ----
EXTENDS Naturals

\* A symbolic ID has the same equality as its hash preimage. Configuration
\* validation also preserves the rest of a recipe, such as its compute work.
Recipe(base, width, option, work) ==
    [base |-> base, width |-> width, option |-> option, work |-> work]
Original == Recipe("original", 2, <<>>, 30)
Replicate == Recipe("replicated", 3, <<16>>, 1)
Scatter == Recipe("scatter-gather", 2, <<0>>, 1)
Edited == [Original EXCEPT !.work = @ + 1]
EditedReplica == [Replicate EXCEPT !.work = @ + 1]
Profiles == {Original, Replicate, Scatter, Edited, EditedReplica}
RawId(recipe) == <<recipe.base>>
Id(recipe) ==
    IF recipe.option = <<>> THEN RawId(recipe)
    ELSE <<recipe.base, "peer-all-reduce-egress", recipe.width, recipe.option[1]>>

Unique(profiles) ==
    \A a, b \in profiles : a # b => RawId(a) # RawId(b) /\ Id(a) # Id(b)
Lookup(profiles, id) == CHOOSE recipe \in profiles : Id(recipe) = id

\* Expired and fresh observations can contain the same measured numbers.
Evidence(fresh, band) == [fresh |-> fresh, band |-> band]
Bands == {"latency-sensitive", "bandwidth-sensitive"}
LatencySensitive == Evidence(TRUE, "latency-sensitive")
BandwidthSensitive == Evidence(TRUE, "bandwidth-sensitive")
Payload == 8
Limit(recipe) ==
    IF recipe.option # <<>> THEN recipe.option[1]
    ELSE IF recipe.width = 2 THEN 1048576 ELSE 65536
Exchange(recipe) ==
    IF Payload <= Limit(recipe) \div (recipe.width - 1)
    THEN "Replicated" ELSE "ScatterGather"

\* Small sample costs separate serialization from one or two exchange rounds.
Cost(recipe, observation) ==
    LET latency == IF observation.band = "latency-sensitive" THEN 12 ELSE 2
        perByte == IF observation.band = "latency-sensitive" THEN 1 ELSE 4
    IN recipe.work +
       IF Exchange(recipe) = "Replicated" THEN latency + perByte * Payload * (recipe.width - 1)
       ELSE 2 * latency + perByte * Payload
Best(profiles, current, observation) ==
    {r \in profiles : /\ Id(r) # current.id
                     /\ Cost(r, observation) + 1 < Cost(Lookup(profiles, current.id), observation)
                     /\ \A q \in profiles : Id(q) # current.id => Cost(r, observation) <= Cost(q, observation)}

Binding(epoch, recipe) ==
    [epoch |-> epoch, id |-> Id(recipe), width |-> recipe.width, option |-> recipe.option]
NoBinding == [epoch |-> 0, id |-> <<>>, width |-> 0, option |-> <<>>]
Slots == 1..3
Unresolved == [s \in Slots |-> NoBinding]
NoCandidate == [bind |-> NoBinding, reports |-> Unresolved,
                communicator |-> NoBinding, warmed |-> FALSE,
                evidence |-> Evidence(FALSE, "latency-sensitive"),
                selectionCatalog |-> {}, previous |-> NoBinding]
Quote(binding, recipe, observation) ==
    [bind |-> binding, mode |-> Exchange(recipe), cost |-> Cost(recipe, observation),
     evidence |-> observation]
NoQuote == [bind |-> NoBinding, mode |-> "none", cost |-> 0,
            evidence |-> Evidence(FALSE, "latency-sensitive")]

VARIABLES catalog, retained, evidence, latest, candidate, active, publication, quote
vars == <<catalog, retained, evidence, latest, candidate, active, publication, quote>>

Init ==
    /\ catalog = {Original} /\ retained = catalog
    /\ evidence = Evidence(FALSE, "latency-sensitive")
    /\ latest = 1 /\ candidate = NoCandidate
    /\ active = Binding(latest, Original)
    /\ publication = [bind |-> active, warmed |-> TRUE]
    /\ quote = NoQuote

Reload(incoming) ==
    /\ catalog \subseteq incoming /\ Unique(incoming)
    /\ catalog' = incoming /\ retained' = retained \cup incoming
    /\ UNCHANGED <<evidence, latest, candidate, active, publication, quote>>

Observe(band) ==
    /\ evidence' = Evidence(TRUE, band)
    /\ UNCHANGED <<catalog, retained, latest, candidate, active, publication, quote>>

Expire ==
    /\ evidence.fresh /\ evidence' = [evidence EXCEPT !.fresh = FALSE]
    /\ UNCHANGED <<catalog, retained, latest, candidate, active, publication, quote>>

PriceIncumbent ==
    /\ evidence.fresh
    /\ quote' = Quote(active, Lookup(catalog, active.id), evidence)
    /\ UNCHANGED <<catalog, retained, evidence, latest, candidate, active, publication>>

Reserve(recipe) ==
    /\ evidence.fresh /\ candidate = NoCandidate /\ latest < 3
    /\ recipe \in Best(catalog, active, evidence)
    /\ latest' = latest + 1
    /\ candidate' = [NoCandidate EXCEPT !.bind = Binding(latest', recipe),
                       !.evidence = evidence, !.selectionCatalog = catalog,
                       !.previous = active]
    /\ UNCHANGED <<catalog, retained, evidence, active, publication, quote>>

\* Launchers may read a stale epoch or resolve a different local startup
\* option or width. The accepted report must match the committed binding.
Reports ==
    LET bind == candidate.bind
    IN {bind, [bind EXCEPT !.epoch = active.epoch], [bind EXCEPT !.id = active.id],
        [bind EXCEPT !.width = 5 - @], [bind EXCEPT !.option = <<>>]}

Report(slot, observed) ==
    /\ candidate # NoCandidate /\ slot \in 1..candidate.bind.width
    /\ candidate.reports[slot] = NoBinding
    /\ candidate' = [candidate EXCEPT !.reports[slot] = observed]
    /\ UNCHANGED <<catalog, retained, evidence, latest, active, publication, quote>>

CreateCommunicator ==
    /\ candidate # NoCandidate /\ candidate.communicator = NoBinding
    /\ \A s \in 1..candidate.bind.width : candidate.reports[s] = candidate.bind
    /\ candidate' = [candidate EXCEPT !.communicator = candidate.reports[1]]
    /\ UNCHANGED <<catalog, retained, evidence, latest, active, publication, quote>>

Warm ==
    /\ candidate.communicator # NoBinding /\ ~candidate.warmed
    /\ candidate' = [candidate EXCEPT !.warmed = TRUE]
    /\ UNCHANGED <<catalog, retained, evidence, latest, active, publication, quote>>

Abort ==
    /\ candidate # NoCandidate /\ candidate' = NoCandidate
    /\ UNCHANGED <<catalog, retained, evidence, latest, active, publication, quote>>

Activate ==
    /\ candidate.warmed /\ candidate.previous = active
    /\ active' = candidate.bind
    /\ publication' = [bind |-> candidate.communicator, warmed |-> candidate.warmed]
    /\ candidate' = NoCandidate /\ quote' = NoQuote
    /\ UNCHANGED <<catalog, retained, evidence, latest>>

Next ==
    \/ Observe("latency-sensitive") \/ Observe("bandwidth-sensitive") \/ Expire
    \/ PriceIncumbent \/ CreateCommunicator \/ Warm \/ Abort \/ Activate
    \/ \E incoming \in SUBSET Profiles : Reload(incoming)
    \/ \E recipe \in Profiles : Reserve(recipe)
    \/ \E slot \in Slots, observed \in Reports : Report(slot, observed)

Spec == Init /\ [][Next]_vars

IdentityBindsWidthAndBudget ==
    LET legacy == Original
        zero == [legacy EXCEPT !.option = <<0>>]
        wider == [zero EXCEPT !.width = 3]
        higher == [zero EXCEPT !.option = <<16>>]
    IN /\ Id(legacy) = RawId(legacy)
       /\ Id(zero) # Id(legacy) /\ Id(zero) # Id(wider) /\ Id(zero) # Id(higher)

AppendOnlyCatalog == catalog = retained /\ Unique(catalog)

IncumbentUsesCommittedRecipe ==
    /\ \E r \in catalog : active = Binding(active.epoch, r)
    /\ quote # NoQuote =>
          /\ quote.evidence.fresh
          /\ quote = Quote(active, Lookup(catalog, active.id), quote.evidence)

CandidateUsesFreshSelection ==
    candidate # NoCandidate =>
        /\ candidate.evidence.fresh /\ candidate.selectionCatalog \subseteq catalog
        /\ candidate.previous = active
        /\ \E r \in Best(candidate.selectionCatalog, candidate.previous, candidate.evidence) :
                 candidate.bind = Binding(candidate.bind.epoch, r)

LinkEvidenceChangesRanking ==
    /\ Best({Original, Replicate, Scatter}, Binding(1, Original), LatencySensitive) = {Replicate}
    /\ Best({Original, Replicate, Scatter}, Binding(1, Original), BandwidthSensitive) = {Scatter}

CommunicatorsUseMatchedBindings ==
    /\ candidate.warmed => candidate.communicator # NoBinding
    /\ candidate.communicator # NoBinding =>
          /\ candidate.communicator = candidate.bind
          /\ \A s \in 1..candidate.bind.width : candidate.reports[s] = candidate.bind

ActiveRequiresSuccessfulWarmup ==
    publication.bind = active /\ publication.warmed
====
