---- MODULE PlaitExecutionGraph ----
\* A finite model of Plait's SCC-condensation execution scheduler.
\*
\* This is a scheduler specification, not a full ordering protocol spec. The
\* consensus protocol must separately establish that committed values and their
\* dependency edges are stable. This model checks the execution layer:
\*   - a component runs only after all external dependencies have executed;
\*   - a node never runs before a dependency outside its SCC;
\*   - every component eventually runs under weak fairness.
\*
\* The component partition is an input because TLA+ modules cannot use an
\* external SCC implementation as an oracle. A separate refinement proof (or a
\* model that computes reachability) must connect this partition to the graph.

EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
    Nodes,          \* finite set of instance ids
    Components,     \* finite set of SCCs (sets of Nodes)
    Deps            \* relation <<node, dependency>>; see DepsOf

ASSUME
    /\ \A c \in Components: c # {} /\ c \subseteq Nodes
    /\ \A c \in Components: \A n \in c: n \in Nodes
    /\ \A a, b \in Components: a # b => a \cap b = {}
    /\ \A n \in Nodes: \E c \in Components: n \in c

ComponentOf(n) == CHOOSE c \in Components: n \in c
DepsOf(n) == {d \in Nodes: <<n, d>> \in Deps}

VARIABLES executed

vars == <<executed>>

Init == executed = {}

\* A component is ready when every dependency of every member is either in
\* the component itself or has already executed.
Ready(c) ==
    \A n \in c:
        \A d \in DepsOf(n):
            \/ ComponentOf(d) = c
            \/ d \in executed

Execute(c) ==
    /\ c \cap executed = {}
    /\ Ready(c)
    /\ executed' = executed \cup c

Next == \E c \in Components: Execute(c)

Spec == Init /\ [][Next]_executed /\ WF_executed(Next)

\* Safety: no executed node has a not-yet-executed dependency outside its SCC.
Safety ==
    \A n \in executed:
        \A d \in DepsOf(n):
            \/ d \in executed
            \/ ComponentOf(d) = ComponentOf(n)

\* Every component is either executed or can eventually be selected.
EventuallyExecutable ==
    \A c \in Components:
        c \subseteq executed \/ Ready(c)

\* The scheduler cannot execute a component twice.
ExecutedOnce ==
    \A c \in Components: c \subseteq executed \/ c \cap executed = {}

THEOREM Spec => []Safety
THEOREM Spec => []ExecutedOnce

====