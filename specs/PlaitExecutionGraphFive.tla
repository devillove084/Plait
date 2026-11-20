---- MODULE PlaitExecutionGraphFive ----
\* Concrete 5-node instance of the execution scheduler model.
EXTENDS Naturals, FiniteSets

Nodes == {1, 2, 3, 4, 5}
Components == {{1, 2, 3}, {4}, {5}}
Deps == {<<1, 2>>, <<2, 1>>, <<3, 1>>, <<4, 2>>, <<5, 4>>}

VARIABLE executed

I == INSTANCE PlaitExecutionGraph

Spec == I!Spec
Safety == I!Safety
ExecutedOnce == I!ExecutedOnce
====
