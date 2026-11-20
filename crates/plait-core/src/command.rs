use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::identity::NodeId;
use crate::tensor::{ShardId, TensorId, TensorKey};

/// A command's replication identity. Operator ids are minted by a node, not by
/// a rank: `(origin, counter)` is unique for the lifetime of that node.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct OpId {
    pub origin: NodeId,
    pub counter: u64,
}

impl OpId {
    pub const fn new(origin: NodeId, counter: u64) -> Self {
        Self { origin, counter }
    }
}

impl std::fmt::Debug for OpId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OpId(#{:?}@{})", self.counter, self.origin.to_hex())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FuseKind {
    /// Element-wise sum of every contributor's partial.
    Sum,
    /// Element-wise maximum of every contributor's partial.
    Max,
    /// Concatenation in contributor-id order.
    Gather,
    /// Merge each contributor's local top-k candidate list. A contribution is
    /// a flat `[score, index, score, index, ...]` list; the fused result is the
    /// selected indices in descending score order.
    TopK { k: u32 },
    /// Combine online-softmax partials encoded as `[max, sum, accumulator...]`.
    /// This is associative, so context-parallel attention partials fuse
    /// without a global softmax pass.
    OnlineSoftmax,
}

/// The replicated description of a fused operator. Contributors are named
/// explicitly; there is no implicit rank or device ordering.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpSpec {
    pub id: OpId,
    pub kind: FuseKind,
    pub contributors: BTreeSet<NodeId>,
    pub tensor: TensorKey,
    pub epoch: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Mode {
    Read,
    Write,
}

/// A key plus the access mode used for EPaxos conflict detection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Access {
    pub key: Key,
    pub mode: Mode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Key {
    /// Replicated membership configuration. Every command reads it; membership
    /// changes write it, so reconfiguration is serialized with all commands.
    Config,
    /// Whole-operator state. Contributions to one operator conflict and are
    /// executed in consensus order; different operators run concurrently.
    Op(OpId),
    /// A materialized tensor shard.
    Tensor(TensorKey),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MembershipChange {
    /// Admit a node as a non-voting learner and record its address. The new
    /// node can receive a snapshot but does not vote until `CommitAdd`.
    StageAdd { node: NodeId, addr: String },
    /// Promote a staged learner. This instance must be acknowledged by every
    /// old voter and the new node, so no node silently joins the quorum.
    CommitAdd { node: NodeId },
    /// Begin removing a voter. After this executes it is excluded from new
    /// configurations but remains available to acknowledge the commit phase.
    StageRemove { node: NodeId },
    /// Remove a staged node. Every voter in the old configuration must ack.
    CommitRemove { node: NodeId },
}

impl MembershipChange {
    pub fn is_stage(&self) -> bool {
        matches!(self, Self::StageAdd { .. } | Self::StageRemove { .. })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Command {
    /// A no-op that still exercises consensus. Useful for heartbeats and tests.
    Nop,
    /// Establish the contributor set and combination rule for a fused op.
    DeclareOp(OpSpec),
    /// Publish one node's local partial. The partial travels inside the
    /// consensus command, so replication and reduction are one mechanism.
    Contribute {
        op: OpId,
        contributor: NodeId,
        values: Vec<f64>,
    },
    /// Install a materialized shard, replacing any previous version.
    Publish {
        key: TensorKey,
        version: u64,
        values: Vec<f64>,
    },
    /// Pipeline-stage handoff. The command reads the previous stage's tensor
    /// (if any) and writes the next stage's tensor. EPaxos therefore orders a
    /// stage after its input without a global step, while different
    /// microbatches remain independent.
    Stage {
        input: Option<TensorKey>,
        output: TensorKey,
        version: u64,
        values: Vec<f64>,
    },
    Membership(MembershipChange),
}

impl Command {
    /// Keys used by EPaxos conflict detection. The read of `Key::Config` means
    /// ordinary commands only conflict with a concurrent membership write.
    pub fn accesses(&self) -> Vec<Access> {
        let read_config = Access {
            key: Key::Config,
            mode: Mode::Read,
        };
        match self {
            Self::Nop => vec![read_config],
            Self::DeclareOp(spec) => vec![
                read_config,
                Access {
                    key: Key::Op(spec.id),
                    mode: Mode::Write,
                },
            ],
            Self::Contribute { op, .. } => vec![
                read_config,
                Access {
                    key: Key::Op(*op),
                    mode: Mode::Write,
                },
            ],
            Self::Publish { key, .. } => vec![
                read_config,
                Access {
                    key: Key::Tensor(*key),
                    mode: Mode::Write,
                },
            ],
            Self::Stage { input, output, .. } => {
                let mut accesses = vec![read_config];
                if let Some(input) = input {
                    accesses.push(Access {
                        key: Key::Tensor(*input),
                        mode: Mode::Read,
                    });
                }
                accesses.push(Access {
                    key: Key::Tensor(*output),
                    mode: Mode::Write,
                });
                accesses
            }
            Self::Membership(_) => vec![Access {
                key: Key::Config,
                mode: Mode::Write,
            }],
        }
    }

    pub fn is_membership(&self) -> bool {
        matches!(self, Self::Membership(_))
    }

    /// EPaxos conflict rule: same key and at least one side writes.
    pub fn conflicts_with(&self, other: &Self) -> bool {
        let mine = self.accesses();
        let theirs = other.accesses();
        mine.iter().any(|left| {
            theirs.iter().any(|right| {
                left.key == right.key && (left.mode == Mode::Write || right.mode == Mode::Write)
            })
        })
    }

    pub fn membership(&self) -> Option<&MembershipChange> {
        match self {
            Self::Membership(change) => Some(change),
            _ => None,
        }
    }

    pub fn summary(&self) -> String {
        match self {
            Self::Nop => "nop".to_owned(),
            Self::DeclareOp(spec) => format!(
                "declare {:?} op {:?} on {:?} epoch {}",
                spec.kind, spec.id, spec.tensor, spec.epoch
            ),
            Self::Contribute {
                op,
                contributor,
                values,
            } => {
                format!(
                    "contribute {op:?} from {contributor} ({} values)",
                    values.len()
                )
            }
            Self::Publish {
                key,
                version,
                values,
            } => {
                format!("publish {key:?} v{version} ({} values)", values.len())
            }
            Self::Stage {
                input,
                output,
                version,
                values,
            } => format!(
                "stage {input:?} -> {output:?} v{version} ({} values)",
                values.len()
            ),
            Self::Membership(change) => format!("membership {change:?}"),
        }
    }
}

/// Declare an operator whose contributors are known locally. This is a
/// convenience for callers; the wire format remains [`OpSpec`].
pub fn op_spec(
    id: OpId,
    kind: FuseKind,
    contributors: impl IntoIterator<Item = NodeId>,
    tensor: TensorId,
    shard: ShardId,
    epoch: u64,
) -> OpSpec {
    OpSpec {
        id,
        kind,
        contributors: contributors.into_iter().collect(),
        tensor: TensorKey::new(tensor, shard),
        epoch,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(label: &str) -> NodeId {
        NodeId::derive(label)
    }

    #[test]
    fn contributions_to_one_op_conflict() {
        let a = Command::Contribute {
            op: OpId::new(node("a"), 1),
            contributor: node("a"),
            values: vec![1.0],
        };
        let b = Command::Contribute {
            op: OpId::new(node("a"), 1),
            contributor: node("b"),
            values: vec![2.0],
        };
        assert!(a.conflicts_with(&b));
    }

    #[test]
    fn independent_ops_do_not_conflict() {
        let a = Command::Contribute {
            op: OpId::new(node("a"), 1),
            contributor: node("a"),
            values: vec![1.0],
        };
        let b = Command::Contribute {
            op: OpId::new(node("a"), 2),
            contributor: node("a"),
            values: vec![2.0],
        };
        assert!(!a.conflicts_with(&b));
    }

    #[test]
    fn membership_writes_the_configuration_key() {
        let command = Command::Membership(MembershipChange::StageAdd {
            node: node("new"),
            addr: "127.0.0.1:9000".to_owned(),
        });
        assert!(
            command
                .accesses()
                .iter()
                .any(|access| { access.key == Key::Config && access.mode == Mode::Write })
        );
    }
}
