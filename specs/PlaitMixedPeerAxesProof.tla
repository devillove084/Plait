---- MODULE PlaitMixedPeerAxesProof ----
EXTENDS PlaitMixedPeerAxes, TLAPS

LOCAL THEOREM ReconstructSlot ==
    ASSUME NEW slot \in Nat, NEW width \in Nat \ {0}
    PROVE slot = Stage(slot, width) * width + TensorSlot(slot, width)
BY SMT DEF Stage, TensorSlot

THEOREM DistinctWorkersShareAtMostOneAxis ==
    ASSUME NEW width \in Nat \ {0}, NEW left \in Nat, NEW right \in Nat,
           left # right
    PROVE ~(Stage(left, width) = Stage(right, width) /\
            TensorSlot(left, width) = TensorSlot(right, width))
<1>1. left = Stage(left, width) * width + TensorSlot(left, width)
    BY ReconstructSlot
<1>2. right = Stage(right, width) * width + TensorSlot(right, width)
    BY ReconstructSlot
<1> QED BY <1>1, <1>2

THEOREM CoordinatesFitLayout ==
    ASSUME NEW width \in Nat \ {0}, NEW stages \in Nat \ {0},
           NEW slot \in 0..(width * stages - 1)
    PROVE /\ Stage(slot, width) \in 0..(stages - 1)
          /\ TensorSlot(slot, width) \in 0..(width - 1)
<1>1. slot \in Nat /\ 0 < width /\ slot < width * stages BY SMT
<1>2. Stage(slot, width) \in Nat BY <1>1, SMT DEF Stage
<1>3. TensorSlot(slot, width) \in 0..(width - 1)
    BY <1>1, SMT DEF TensorSlot
<1>4. slot = Stage(slot, width) * width + TensorSlot(slot, width)
    BY <1>1, ReconstructSlot
<1>5. Stage(slot, width) * width < stages * width
    BY <1>1, <1>2, <1>3, <1>4, SMT
<1>6. Stage(slot, width) < stages
    BY <1>1, <1>2, <1>5, SMT
<1> QED BY <1>2, <1>3, <1>6, SMT
====
