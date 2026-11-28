---- MODULE PlaitGroupRecipe ----
EXTENDS Naturals

CONSTANTS CheckEpoch, BindOnReserve
Versions == 1..2
Workers == {"a", "b"}
NoLaunch == [epoch |-> 0, recipe |-> "none"]

VARIABLES link, observed, count, latest, recipes, grantsAt, recipeAt, launch
vars == <<link, observed, count, latest, recipes, grantsAt, recipeAt, launch>>

Init ==
    /\ link = "fast" /\ observed = "none" /\ count = 0 /\ latest = 0
    /\ recipes = [e \in Versions |-> "none"]
    /\ grantsAt = [w \in Workers |-> 0]
    /\ recipeAt = [w \in Workers |-> 0]
    /\ launch = [w \in Workers |-> NoLaunch]

Recommend == IF link = "fast" THEN "tp" ELSE "pp"

ChangeLink ==
    /\ link' = IF link = "fast" THEN "slow" ELSE "fast"
    /\ UNCHANGED <<observed, count, latest, recipes, grantsAt, recipeAt, launch>>

Poll ==
    /\ observed' = Recommend
    /\ count' = IF observed = Recommend THEN 2 ELSE 1
    /\ UNCHANGED <<link, latest, recipes, grantsAt, recipeAt, launch>>

Reserve ==
    /\ count = 2 /\ latest < 2
    /\ latest' = latest + 1
    /\ recipes' = IF BindOnReserve
                  THEN [recipes EXCEPT ![latest'] = observed]
                  ELSE recipes
    /\ count' = 0 /\ observed' = "none"
    /\ UNCHANGED <<link, grantsAt, recipeAt, launch>>

ReadGrants(w, e) ==
    /\ e <= latest /\ grantsAt' = [grantsAt EXCEPT ![w] = e]
    /\ UNCHANGED <<link, observed, count, latest, recipes, recipeAt, launch>>

ReadRecipe(w, e) ==
    /\ e <= latest /\ recipeAt' = [recipeAt EXCEPT ![w] = e]
    /\ UNCHANGED <<link, observed, count, latest, recipes, grantsAt, launch>>

Launch(w) ==
    /\ grantsAt[w] # 0 /\ recipeAt[w] # 0
    /\ CheckEpoch => grantsAt[w] = recipeAt[w]
    /\ launch' = [launch EXCEPT ![w] = [epoch |-> grantsAt[w],
                                        recipe |-> recipes[recipeAt[w]]]]
    /\ UNCHANGED <<link, observed, count, latest, recipes, grantsAt, recipeAt>>

Next ==
    \/ ChangeLink \/ Poll \/ Reserve
    \/ \E w \in Workers, e \in Versions : ReadGrants(w, e) \/ ReadRecipe(w, e)
    \/ \E w \in Workers : Launch(w)

Spec == Init /\ [][Next]_vars
EveryGrantHasRecipe == \A e \in 1..latest : recipes[e] # "none"
LaunchUsesItsEpochRecipe ==
    \A w \in Workers : launch[w] # NoLaunch =>
        launch[w].recipe = recipes[launch[w].epoch]
====
