use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::command::{Command, MembershipChange};
use crate::identity::NodeId;

/// Replicated membership. There is no bootstrap rank and no permanent
/// coordinator: every voter may propose, and every configuration change is
/// itself an EPaxos instance.
///
/// Reconfiguration is deliberately conservative:
///
/// * a stage command is acknowledged by **every** current voter;
/// * a commit command is acknowledged by every old voter and, for an addition,
///   by the joining node as well.
///
/// This gives up availability during a membership change but removes the
/// split-brain window between two disjoint majorities. Steady-state commands
/// still commit with an ordinary majority and take the EPaxos fast path when
/// their dependency sets agree.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClusterConfig {
    voters: BTreeSet<NodeId>,
    staged_add: BTreeSet<NodeId>,
    staged_remove: BTreeSet<NodeId>,
    version: u64,
}

impl ClusterConfig {
    pub fn new(voters: impl IntoIterator<Item = NodeId>) -> Result<Self, ConfigError> {
        let voters: BTreeSet<NodeId> = voters.into_iter().collect();
        if voters.is_empty() {
            return Err(ConfigError::Empty);
        }
        Ok(Self {
            voters,
            staged_add: BTreeSet::new(),
            staged_remove: BTreeSet::new(),
            version: 0,
        })
    }

    pub fn bootstrap(voters: impl IntoIterator<Item = NodeId>) -> Self {
        Self::new(voters).expect("bootstrap requires at least one voter")
    }

    pub fn voters(&self) -> &BTreeSet<NodeId> {
        &self.voters
    }

    pub fn staged_add(&self) -> &BTreeSet<NodeId> {
        &self.staged_add
    }

    pub fn staged_remove(&self) -> &BTreeSet<NodeId> {
        &self.staged_remove
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn is_voter(&self, node: NodeId) -> bool {
        self.voters.contains(&node)
    }

    pub fn is_member(&self, node: NodeId) -> bool {
        self.voters.contains(&node)
            || self.staged_add.contains(&node)
            || self.staged_remove.contains(&node)
    }

    /// Quorum policy for a command. Ordinary commands use a majority (and a
    /// larger fast quorum); membership commands require the exact set that
    /// must observe them.
    pub fn quorum_for(&self, command: &Command) -> Result<QuorumPolicy, ConfigError> {
        let change = match command.membership() {
            None => return Ok(QuorumPolicy::majority(&self.voters)),
            Some(change) => change,
        };

        match change {
            MembershipChange::StageAdd { node, .. } => {
                if self.voters.contains(node) {
                    return Err(ConfigError::AlreadyVoter(*node));
                }
                if self.staged_add.contains(node) {
                    return Err(ConfigError::AlreadyStaged(*node));
                }
                Ok(QuorumPolicy::unanimous(self.voters.clone()))
            }
            MembershipChange::CommitAdd { node } => {
                if !self.staged_add.contains(node) {
                    return Err(ConfigError::NotStaged(*node));
                }
                let mut required = self.voters.clone();
                required.insert(*node);
                Ok(QuorumPolicy::unanimous(required))
            }
            MembershipChange::StageRemove { node } => {
                if !self.voters.contains(node) {
                    return Err(ConfigError::NotAVoter(*node));
                }
                if self.voters.len() == 1 {
                    return Err(ConfigError::LastVoter);
                }
                if self.staged_remove.contains(node) {
                    return Err(ConfigError::AlreadyStaged(*node));
                }
                Ok(QuorumPolicy::unanimous(self.voters.clone()))
            }
            MembershipChange::CommitRemove { node } => {
                if !self.staged_remove.contains(node) {
                    return Err(ConfigError::NotStaged(*node));
                }
                if self.voters.len() == 1 {
                    return Err(ConfigError::LastVoter);
                }
                Ok(QuorumPolicy::unanimous(self.voters.clone()))
            }
        }
    }

    /// Apply a membership command at execution time. Returns `true` when the
    /// configuration changed. Replaying a snapshot is idempotent.
    pub fn apply(&mut self, change: &MembershipChange) -> Result<bool, ConfigError> {
        match change {
            MembershipChange::StageAdd { node, .. } => {
                if self.voters.contains(node) {
                    return Ok(false);
                }
                if self.staged_add.insert(*node) {
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            MembershipChange::CommitAdd { node } => {
                if self.voters.contains(node) {
                    return Ok(false);
                }
                if !self.staged_add.remove(node) {
                    return Err(ConfigError::NotStaged(*node));
                }
                self.voters.insert(*node);
                self.version = self.version.saturating_add(1);
                Ok(true)
            }
            MembershipChange::StageRemove { node } => {
                if !self.voters.contains(node) {
                    return Err(ConfigError::NotAVoter(*node));
                }
                if self.voters.len() == 1 {
                    return Err(ConfigError::LastVoter);
                }
                if self.staged_remove.insert(*node) {
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            MembershipChange::CommitRemove { node } => {
                if !self.voters.contains(node) {
                    return Ok(false);
                }
                if !self.staged_remove.remove(node) {
                    return Err(ConfigError::NotStaged(*node));
                }
                self.voters.remove(node);
                self.version = self.version.saturating_add(1);
                Ok(true)
            }
        }
    }
}

/// The acknowledgement set for one consensus instance. `ack_required` is the
/// number of distinct voters needed to decide; `fast_acks` is the larger EPaxos
/// fast quorum whose members must have identical attributes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuorumPolicy {
    pub voters: BTreeSet<NodeId>,
    pub ack_required: usize,
    pub fast_acks: usize,
}

impl QuorumPolicy {
    pub fn majority(voters: &BTreeSet<NodeId>) -> Self {
        let n = voters.len();
        let majority = n / 2 + 1;
        // EPaxos fast quorum. For 3 voters this is 2, for 5 it is 3, for 7 it
        // is 5. Two fast quorums always intersect, and every fast quorum
        // intersects every recovery (majority) quorum.
        let fast = majority.max(3 * n / 4);
        Self {
            voters: voters.clone(),
            ack_required: majority,
            fast_acks: fast,
        }
    }

    pub fn unanimous(voters: BTreeSet<NodeId>) -> Self {
        let size = voters.len();
        Self {
            voters,
            ack_required: size,
            fast_acks: size,
        }
    }

    pub fn is_voter(&self, node: NodeId) -> bool {
        self.voters.contains(&node)
    }

    pub fn is_unanimous(&self) -> bool {
        self.ack_required == self.voters.len() && self.fast_acks == self.voters.len()
    }

    pub fn acked(&self, acks: &BTreeSet<NodeId>) -> bool {
        acks.iter()
            .filter(|node| self.voters.contains(node))
            .count()
            >= self.ack_required
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfigError {
    Empty,
    AlreadyVoter(NodeId),
    NotAVoter(NodeId),
    AlreadyStaged(NodeId),
    NotStaged(NodeId),
    LastVoter,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => f.write_str("membership needs at least one voter"),
            Self::AlreadyVoter(node) => write!(f, "{node} is already a voter"),
            Self::NotAVoter(node) => write!(f, "{node} is not a voter"),
            Self::AlreadyStaged(node) => write!(f, "{node} is already staged"),
            Self::NotStaged(node) => write!(f, "{node} is not staged"),
            Self::LastVoter => f.write_str("refusing to remove the last voter"),
        }
    }
}

impl std::error::Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(label: &str) -> NodeId {
        NodeId::derive(label)
    }

    #[test]
    fn quorum_sizes_keep_intersections() {
        for n in 1..=9usize {
            let voters: BTreeSet<_> = (0..n).map(|i| NodeId::derive(&format!("n{i}"))).collect();
            let quorum = QuorumPolicy::majority(&voters);
            assert_eq!(quorum.ack_required, n / 2 + 1);
            assert!(quorum.fast_acks >= quorum.ack_required);
            assert!(2 * quorum.fast_acks > n, "fast quorums must intersect");
            assert!(quorum.fast_acks + quorum.ack_required > n);
        }
    }

    #[test]
    fn add_requires_old_voters_and_joiner() {
        let a = node("a");
        let b = node("b");
        let c = node("c");
        let mut config = ClusterConfig::bootstrap([a, b]);
        config
            .apply(&MembershipChange::StageAdd {
                node: c,
                addr: "127.0.0.1:9000".into(),
            })
            .unwrap();
        let quorum = config
            .quorum_for(&Command::Membership(MembershipChange::CommitAdd {
                node: c,
            }))
            .unwrap();
        assert_eq!(quorum.voters, BTreeSet::from([a, b, c]));
        assert_eq!(quorum.ack_required, 3);
        config
            .apply(&MembershipChange::CommitAdd { node: c })
            .unwrap();
        assert!(config.is_voter(c));
        assert_eq!(config.version(), 1);
    }

    #[test]
    fn remove_requires_all_old_voters() {
        let a = node("a");
        let b = node("b");
        let c = node("c");
        let mut config = ClusterConfig::bootstrap([a, b, c]);
        config
            .apply(&MembershipChange::StageRemove { node: c })
            .unwrap();
        let quorum = config
            .quorum_for(&Command::Membership(MembershipChange::CommitRemove {
                node: c,
            }))
            .unwrap();
        assert_eq!(quorum.ack_required, 3);
        config
            .apply(&MembershipChange::CommitRemove { node: c })
            .unwrap();
        assert!(!config.is_voter(c));
        assert_eq!(config.voters(), &BTreeSet::from([a, b]));
    }

    #[test]
    fn last_voter_cannot_be_removed() {
        let a = node("a");
        let mut config = ClusterConfig::bootstrap([a]);
        assert_eq!(
            config
                .apply(&MembershipChange::StageRemove { node: a })
                .unwrap_err(),
            ConfigError::LastVoter
        );
    }

    #[test]
    fn replayed_membership_is_idempotent() {
        let a = node("a");
        let b = node("b");
        let mut config = ClusterConfig::bootstrap([a]);
        let add = MembershipChange::CommitAdd { node: b };
        // CommitAdd is only accepted after the stage, so a replay of a stale
        // commit with no staged entry is an error, while a replay of an
        // already-applied add is a no-op.
        assert_eq!(config.apply(&add), Err(ConfigError::NotStaged(b)));
        config
            .apply(&MembershipChange::StageAdd {
                node: b,
                addr: "b:1".into(),
            })
            .unwrap();
        assert_eq!(config.apply(&add), Ok(true));
        assert_eq!(config.apply(&add), Ok(false));
    }
}
