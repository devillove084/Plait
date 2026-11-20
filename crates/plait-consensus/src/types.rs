use std::collections::{BTreeMap, BTreeSet};

use plait_core::{ClusterConfig, Command, ConfigError, NodeId, QuorumPolicy};
use serde::{Deserialize, Serialize};

/// Globally unique identity of one consensus instance. There is no global
/// sequence number: the proposing node owns the counter.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct InstanceId {
    pub proposer: NodeId,
    pub counter: u64,
}

impl std::fmt::Debug for InstanceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}#{}", self.proposer.to_hex(), self.counter)
    }
}

/// Paxos ballot. Rounds are comparable; the node id breaks ties
/// deterministically without making that node a leader.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Serialize, Deserialize)]
pub struct Ballot {
    pub round: u64,
    pub node: NodeId,
}

impl Ballot {
    pub const fn new(round: u64, node: NodeId) -> Self {
        Self { round, node }
    }

    pub const fn zero(node: NodeId) -> Self {
        Self { round: 0, node }
    }

    /// A ballot strictly greater than this one, owned by `node`.
    pub fn next(self, node: NodeId) -> Self {
        Self {
            round: self.round.saturating_add(1),
            node,
        }
    }
}

/// EPaxos attributes: a dependency set plus a sequence number that gives a
/// deterministic order inside strongly connected components.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attrs {
    pub seq: u64,
    pub deps: BTreeSet<InstanceId>,
}

impl Attrs {
    pub fn new(seq: u64, deps: BTreeSet<InstanceId>) -> Self {
        Self { seq, deps }
    }

    /// Union dependencies and keep the larger sequence number.
    pub fn merge(&mut self, other: &Self) {
        self.seq = self.seq.max(other.seq);
        self.deps.extend(other.deps.iter().copied());
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Value {
    pub command: Command,
    pub attrs: Attrs,
    pub ballot: Ballot,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Record {
    pub value: Value,
    pub phase: Phase,
    pub quorum: QuorumPolicy,
    pub config_version: u64,
    pub updated_ms: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    PreAccepted,
    Accepted,
    Committed,
    Executed,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ValueState {
    pub value: Value,
    pub phase: Phase,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Message {
    /// Phase 1 of the fast path. `quorum` is replicated with the instance so
    /// every acceptor validates against the same acknowledgement set.
    PreAccept {
        id: InstanceId,
        command: Command,
        attrs: Attrs,
        ballot: Ballot,
        quorum: QuorumPolicy,
        config_version: u64,
    },
    PreAcceptReply {
        id: InstanceId,
        attrs: Attrs,
        ballot: Ballot,
        state: Option<ValueState>,
        config_version: u64,
    },
    /// Slow path after dependency sets differed.
    Accept {
        id: InstanceId,
        command: Command,
        attrs: Attrs,
        ballot: Ballot,
        quorum: QuorumPolicy,
        config_version: u64,
    },
    AcceptReply {
        id: InstanceId,
        accepted: bool,
        ballot: Ballot,
        state: Option<ValueState>,
        config_version: u64,
    },
    /// Recovery cycle 2: merge conflicts without accepting yet.
    TryPreAccept {
        id: InstanceId,
        command: Command,
        attrs: Attrs,
        ballot: Ballot,
        quorum: QuorumPolicy,
        config_version: u64,
    },
    TryPreAcceptReply {
        id: InstanceId,
        attrs: Attrs,
        ballot: Ballot,
        state: Option<ValueState>,
        config_version: u64,
    },
    Commit {
        id: InstanceId,
        value: Value,
    },
    /// Recovery cycle 1. Every replica may initiate this; ballots keep
    /// concurrent recoveries safe.
    Recover {
        id: InstanceId,
        ballot: Ballot,
        config_version: u64,
    },
    RecoverReply {
        id: InstanceId,
        state: Option<ValueState>,
        config_version: u64,
    },
}

#[derive(Clone, Debug)]
pub struct Outbound {
    pub to: NodeId,
    pub message: Message,
}

/// A complete transferable image of one replica. The joining node replays the
/// executed prefix and can then acknowledge membership commits.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Snapshot {
    pub config: ClusterConfig,
    pub counters: BTreeMap<NodeId, u64>,
    pub records: Vec<(InstanceId, Record)>,
    pub executed_order: Vec<InstanceId>,
}

#[derive(Clone, Debug)]
pub struct Options {
    /// Time to wait for the fast quorum before retrying preaccept.
    pub fast_timeout_ms: u64,
    /// Base retry interval for accept and recovery messages.
    pub retry_ms: u64,
    /// Age at which a preaccepted/accepted instance becomes recoverable by
    /// any replica.
    pub recovery_after_ms: u64,
    /// Retries before a stuck proposer enters recovery.
    pub retries_before_recovery: u32,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            fast_timeout_ms: 150,
            retry_ms: 300,
            recovery_after_ms: 1_000,
            retries_before_recovery: 2,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProposeError {
    NotVoter(NodeId),
    Config(ConfigError),
    CounterOverflow(NodeId),
}

impl std::fmt::Display for ProposeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotVoter(node) => write!(f, "{node} is not a voter in the current configuration"),
            Self::Config(error) => write!(f, "{error}"),
            Self::CounterOverflow(node) => write!(f, "instance counter overflow for {node}"),
        }
    }
}

impl std::error::Error for ProposeError {}

impl From<ConfigError> for ProposeError {
    fn from(value: ConfigError) -> Self {
        Self::Config(value)
    }
}
