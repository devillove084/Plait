---- MODULE PlaitExecutionGraphSmall ----
\* Concrete 3-node instance of the execution scheduler model.
EXTENDS Naturals, FiniteSets

Nodes == {1, 2, 3}
Components == {{1, 2}, {3}}
Deps == {<<1, 2>>, <<2, 1>>, <<3, 1>>}

VARIABLE executed

I == INSTANCE PlaitExecutionGraph

Spec == I!Spec
Safety == I!Safety
ExecutedOnce == I!ExecutedOnce
====
