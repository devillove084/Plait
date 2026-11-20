use std::collections::{BTreeMap, BTreeSet};

use rayon::prelude::*;

use plait_core::{Access, ClusterConfig, Command, Key, Mode, NodeId, QuorumPolicy};

use crate::execution::ExecutionGraph;
use crate::types::{
    Attrs, Ballot, InstanceId, Message, Options, Outbound, Phase, ProposeError, Record, Snapshot,
    Value, ValueState,
};

/// The replicated state machine. The consensus layer is generic over it so the
/// same protocol can drive tensor fusion, placement metadata, or a future
/// device-control plane.
pub trait StateMachine: Send + Sync {
    fn apply(&mut self, instance: InstanceId, command: &Command);

    /// Whether independent SCC batches may run concurrently. A state machine
    /// that returns true must implement `fork_batch`/`merge_batch` so the final
    /// state is the same as sequential execution in batch order.
    fn supports_parallel(&self) -> bool {
        false
    }

    fn fork_batch(&self) -> Option<Self>
    where
        Self: Sized,
    {
        None
    }

    fn merge_batch(&mut self, _other: Self)
    where
        Self: Sized,
    {
    }
}

#[derive(Clone, Debug)]
struct Active {
    command: Command,
    quorum: QuorumPolicy,
    config_version: u64,
    initial_attrs: Attrs,
    ballot: Ballot,
    phase: ActivePhase,
    deadline_ms: u64,
    attempts: u32,
}

#[derive(Clone, Debug)]
enum ActivePhase {
    PreAccept {
        replies: BTreeMap<NodeId, Attrs>,
    },
    Accept {
        value: Value,
        acks: BTreeSet<NodeId>,
    },
    TryPreAccept {
        value: Value,
        replies: BTreeMap<NodeId, Attrs>,
    },
    Recover {
        ballot: Ballot,
        replies: BTreeMap<NodeId, Option<ValueState>>,
    },
}

/// One EPaxos replica. There is no distinguished replica: [`propose`] works on
/// every voter, and every replica runs the same recovery watchdog.
pub struct Replica<S: StateMachine> {
    id: NodeId,
    config: ClusterConfig,
    machine: S,
    options: Options,
    records: BTreeMap<InstanceId, Record>,
    record_index: BTreeMap<Key, BTreeSet<InstanceId>>,
    indexed: BTreeSet<InstanceId>,
    executed: BTreeSet<InstanceId>,
    execution: ExecutionGraph,
    executed_order: Vec<InstanceId>,
    counters: BTreeMap<NodeId, u64>,
    active: BTreeMap<InstanceId, Active>,
    outbox: Vec<Outbound>,
    violations: Vec<String>,
    clock_ms: u64,
}

impl<S: StateMachine> Replica<S> {
    pub fn new(id: NodeId, config: ClusterConfig, machine: S, options: Options) -> Self {
        Self {
            id,
            config,
            machine,
            options,
            records: BTreeMap::new(),
            record_index: BTreeMap::new(),
            indexed: BTreeSet::new(),
            executed: BTreeSet::new(),
            execution: ExecutionGraph::new(),
            executed_order: Vec::new(),
            counters: BTreeMap::new(),
            active: BTreeMap::new(),
            outbox: Vec::new(),
            violations: Vec::new(),
            clock_ms: 0,
        }
    }

    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn config(&self) -> &ClusterConfig {
        &self.config
    }

    pub fn machine(&self) -> &S {
        &self.machine
    }

    pub fn machine_mut(&mut self) -> &mut S {
        &mut self.machine
    }

    pub fn executed_count(&self) -> usize {
        self.executed.len()
    }

    pub fn violations(&self) -> &[String] {
        &self.violations
    }

    pub fn take_outbox(&mut self) -> Vec<Outbound> {
        std::mem::take(&mut self.outbox)
    }

    /// Propose a command from this node. Every voter may do this; the counter
    /// is local to this node's identity, not a global rank.
    pub fn propose(&mut self, command: Command, now_ms: u64) -> Result<InstanceId, ProposeError> {
        self.clock_ms = now_ms;
        if !self.config.is_voter(self.id) {
            return Err(ProposeError::NotVoter(self.id));
        }
        let quorum = self.config.quorum_for(&command)?;
        let counter = self
            .counters
            .get(&self.id)
            .copied()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or(ProposeError::CounterOverflow(self.id))?;
        self.counters.insert(self.id, counter);
        let id = InstanceId {
            proposer: self.id,
            counter,
        };
        let attrs = self.merge_attrs(Attrs::new(1, BTreeSet::new()), &command, id);
        let ballot = Ballot::zero(self.id);
        let value = Value {
            command: command.clone(),
            attrs: attrs.clone(),
            ballot,
        };
        let record = Record {
            value: value.clone(),
            phase: Phase::PreAccepted,
            quorum: quorum.clone(),
            config_version: self.config.version(),
            updated_ms: now_ms,
        };
        self.insert_record(id, record);

        let mut replies = BTreeMap::new();
        if quorum.is_voter(self.id) {
            replies.insert(self.id, attrs.clone());
        }
        self.active.insert(
            id,
            Active {
                command: command.clone(),
                quorum: quorum.clone(),
                config_version: self.config.version(),
                initial_attrs: attrs.clone(),
                ballot,
                phase: ActivePhase::PreAccept { replies },
                deadline_ms: now_ms + self.options.fast_timeout_ms,
                attempts: 0,
            },
        );
        self.send_preaccept(id, &value, &quorum, self.config.version());
        self.evaluate_preaccept(id, now_ms);
        Ok(id)
    }

    pub fn handle(&mut self, from: NodeId, message: Message, now_ms: u64) {
        self.clock_ms = now_ms;
        match message {
            Message::PreAccept {
                id,
                command,
                attrs,
                ballot,
                quorum,
                config_version,
            } => self.on_preaccept(
                from,
                id,
                command,
                attrs,
                ballot,
                quorum,
                config_version,
                now_ms,
            ),
            Message::PreAcceptReply {
                id,
                attrs,
                ballot,
                state,
                config_version,
            } => self.on_preaccept_reply(from, id, attrs, ballot, state, config_version, now_ms),
            Message::Accept {
                id,
                command,
                attrs,
                ballot,
                quorum,
                config_version,
            } => self.on_accept(
                from,
                id,
                command,
                attrs,
                ballot,
                quorum,
                config_version,
                now_ms,
            ),
            Message::AcceptReply {
                id,
                accepted,
                ballot,
                state,
                config_version,
            } => self.on_accept_reply(from, id, accepted, ballot, state, config_version, now_ms),
            Message::TryPreAccept {
                id,
                command,
                attrs,
                ballot,
                quorum,
                config_version,
            } => self.on_try_preaccept(
                from,
                id,
                command,
                attrs,
                ballot,
                quorum,
                config_version,
                now_ms,
            ),
            Message::TryPreAcceptReply {
                id,
                attrs,
                ballot,
                state,
                config_version,
            } => {
                self.on_try_preaccept_reply(from, id, attrs, ballot, state, config_version, now_ms)
            }
            Message::Commit { id, value } => self.commit_value(id, value, now_ms),
            Message::Recover {
                id,
                ballot,
                config_version,
            } => self.on_recover(from, id, ballot, config_version),
            Message::RecoverReply {
                id,
                state,
                config_version,
            } => self.on_recover_reply(from, id, state, config_version, now_ms),
        }
    }

    pub fn tick(&mut self, now_ms: u64) {
        self.clock_ms = now_ms;
        let ids: Vec<InstanceId> = self.active.keys().copied().collect();
        for id in ids {
            let Some(active) = self.active.get(&id).cloned() else {
                continue;
            };
            if now_ms < active.deadline_ms {
                continue;
            }
            match &active.phase {
                ActivePhase::PreAccept { .. } => {
                    if active.attempts >= self.options.retries_before_recovery {
                        self.start_recovery(id, now_ms);
                    } else {
                        self.retry_preaccept(id, now_ms);
                    }
                }
                ActivePhase::Accept { .. } => {
                    if active.attempts >= self.options.retries_before_recovery {
                        self.start_recovery(id, now_ms);
                    } else {
                        self.retry_accept(id, now_ms);
                    }
                }
                ActivePhase::TryPreAccept { .. } => {
                    if active.attempts >= self.options.retries_before_recovery {
                        self.start_recovery(id, now_ms);
                    } else {
                        self.retry_try_preaccept(id, now_ms);
                    }
                }
                ActivePhase::Recover { .. } => {
                    self.retry_recover(id, now_ms);
                }
            }
        }

        // Any replica may rescue an instance that stopped making progress.
        // Jitter avoids a synchronized recovery stampede without assigning a
        // privileged recovery leader.
        let stale: Vec<InstanceId> = self
            .records
            .iter()
            .filter(|(id, record)| {
                matches!(record.phase, Phase::PreAccepted | Phase::Accepted)
                    && !self.executed.contains(*id)
                    && !self.active.contains_key(*id)
                    && now_ms.saturating_sub(record.updated_ms)
                        >= self.options.recovery_after_ms + self.recovery_jitter(**id)
            })
            .map(|(id, _)| *id)
            .collect();
        for id in stale {
            self.start_recovery(id, now_ms);
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            config: self.config.clone(),
            counters: self.counters.clone(),
            records: self
                .records
                .iter()
                .map(|(id, record)| (*id, record.clone()))
                .collect(),
            executed_order: self.executed_order.clone(),
        }
    }

    fn recovery_jitter(&self, id: InstanceId) -> u64 {
        if self.options.retry_ms == 0 {
            return 0;
        }
        let bytes = self.id.as_bytes();
        let node = u64::from(bytes[0]) << 8 | u64::from(bytes[1]);
        (id.counter.wrapping_mul(31).wrapping_add(node)) % self.options.retry_ms
    }

    fn send(&mut self, to: NodeId, message: Message) {
        if to == self.id {
            return;
        }
        self.outbox.push(Outbound { to, message });
    }

    fn send_preaccept(
        &mut self,
        id: InstanceId,
        value: &Value,
        quorum: &QuorumPolicy,
        config_version: u64,
    ) {
        for voter in &quorum.voters {
            if *voter == self.id {
                continue;
            }
            self.send(
                *voter,
                Message::PreAccept {
                    id,
                    command: value.command.clone(),
                    attrs: value.attrs.clone(),
                    ballot: value.ballot,
                    quorum: quorum.clone(),
                    config_version,
                },
            );
        }
    }

    fn send_accept(
        &mut self,
        id: InstanceId,
        value: &Value,
        quorum: &QuorumPolicy,
        config_version: u64,
    ) {
        for voter in &quorum.voters {
            if *voter == self.id {
                continue;
            }
            self.send(
                *voter,
                Message::Accept {
                    id,
                    command: value.command.clone(),
                    attrs: value.attrs.clone(),
                    ballot: value.ballot,
                    quorum: quorum.clone(),
                    config_version,
                },
            );
        }
    }

    fn send_try_preaccept(
        &mut self,
        id: InstanceId,
        value: &Value,
        quorum: &QuorumPolicy,
        config_version: u64,
    ) {
        for voter in &quorum.voters {
            if *voter == self.id {
                continue;
            }
            self.send(
                *voter,
                Message::TryPreAccept {
                    id,
                    command: value.command.clone(),
                    attrs: value.attrs.clone(),
                    ballot: value.ballot,
                    quorum: quorum.clone(),
                    config_version,
                },
            );
        }
    }

    fn broadcast_commit(&mut self, id: InstanceId, value: &Value) {
        let mut recipients: BTreeSet<NodeId> = self.config.voters().clone();
        recipients.extend(self.config.staged_add().iter().copied());
        recipients.extend(self.config.staged_remove().iter().copied());
        for node in recipients {
            if node == self.id {
                continue;
            }
            self.send(
                node,
                Message::Commit {
                    id,
                    value: value.clone(),
                },
            );
        }
    }

    fn insert_record(&mut self, id: InstanceId, record: Record) {
        let keys: Vec<Key> = record
            .value
            .command
            .accesses()
            .iter()
            .map(|access| access.key)
            .collect();
        self.records.insert(id, record);
        if self.indexed.insert(id) {
            for key in keys {
                self.record_index.entry(key).or_default().insert(id);
            }
        }
    }

    /// Merge the dependencies visible to this replica into an incoming
    /// attribute set. Adding the newest instance per proposer is standard
    /// EPaxos conflict tracking; it is transitively sufficient.
    fn merge_attrs(&self, mut base: Attrs, command: &Command, exclude: InstanceId) -> Attrs {
        let mut candidates: BTreeSet<InstanceId> = BTreeSet::new();
        for access in command.accesses() {
            let Some(ids) = self.record_index.get(&access.key) else {
                continue;
            };
            for candidate in ids {
                if *candidate == exclude || self.executed.contains(candidate) {
                    continue;
                }
                let Some(record) = self.records.get(candidate) else {
                    continue;
                };
                if Self::conflicts_on(&record.value.command, &access) {
                    candidates.insert(*candidate);
                }
            }
        }

        let mut latest: BTreeMap<NodeId, InstanceId> = BTreeMap::new();
        for candidate in candidates {
            latest
                .entry(candidate.proposer)
                .and_modify(|current| {
                    if current.counter < candidate.counter {
                        *current = candidate;
                    }
                })
                .or_insert(candidate);
        }
        for candidate in latest.values() {
            base.deps.insert(*candidate);
            if let Some(record) = self.records.get(candidate) {
                base.seq = base.seq.max(record.value.attrs.seq.saturating_add(1));
            }
        }
        base
    }

    fn conflicts_on(command: &Command, access: &Access) -> bool {
        if access.mode != Mode::Read {
            command
                .accesses()
                .iter()
                .any(|candidate| candidate.key == access.key)
        } else {
            command
                .accesses()
                .iter()
                .any(|candidate| candidate.key == access.key && candidate.mode != Mode::Read)
        }
    }

    fn reply_state(&self, id: InstanceId) -> Option<ValueState> {
        self.records.get(&id).map(|record| ValueState {
            value: record.value.clone(),
            phase: match record.phase {
                Phase::Executed => Phase::Committed,
                other => other,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn on_preaccept(
        &mut self,
        from: NodeId,
        id: InstanceId,
        command: Command,
        attrs: Attrs,
        ballot: Ballot,
        quorum: QuorumPolicy,
        config_version: u64,
        now_ms: u64,
    ) {
        if config_version != self.config.version() || !quorum.is_voter(self.id) {
            self.send(
                from,
                Message::PreAcceptReply {
                    id,
                    attrs,
                    ballot,
                    state: None,
                    config_version: self.config.version(),
                },
            );
            return;
        }

        let existing = self.records.get(&id).cloned();
        let (merged, state) = match existing {
            Some(record) if matches!(record.phase, Phase::Committed | Phase::Executed) => (
                record.value.attrs.clone(),
                Some(ValueState {
                    value: record.value,
                    phase: record.phase,
                }),
            ),
            Some(record) if record.value.ballot > ballot => (
                record.value.attrs.clone(),
                Some(ValueState {
                    value: record.value,
                    phase: record.phase,
                }),
            ),
            Some(record) if record.phase == Phase::Accepted => (
                record.value.attrs.clone(),
                Some(ValueState {
                    value: record.value,
                    phase: record.phase,
                }),
            ),
            Some(record) => {
                if record.value.command != command {
                    self.violations
                        .push(format!("{id:?}: conflicting commands for one instance"));
                    (
                        record.value.attrs.clone(),
                        Some(ValueState {
                            value: record.value,
                            phase: record.phase,
                        }),
                    )
                } else {
                    let mut base = record.value.attrs.clone();
                    base.merge(&attrs);
                    let merged = self.merge_attrs(base, &command, id);
                    if let Some(record) = self.records.get_mut(&id) {
                        record.value.attrs = merged.clone();
                        record.value.ballot = record.value.ballot.max(ballot);
                        record.phase = Phase::PreAccepted;
                        record.updated_ms = now_ms;
                    }
                    (merged, None)
                }
            }
            None => {
                let merged = self.merge_attrs(attrs, &command, id);
                let record = Record {
                    value: Value {
                        command,
                        attrs: merged.clone(),
                        ballot,
                    },
                    phase: Phase::PreAccepted,
                    quorum,
                    config_version,
                    updated_ms: now_ms,
                };
                self.insert_record(id, record);
                (merged, None)
            }
        };

        let record = self.records.get(&id).expect("record was just inserted");
        self.send(
            from,
            Message::PreAcceptReply {
                id,
                attrs: merged,
                ballot: record.value.ballot,
                state,
                config_version: self.config.version(),
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn on_preaccept_reply(
        &mut self,
        from: NodeId,
        id: InstanceId,
        attrs: Attrs,
        ballot: Ballot,
        state: Option<ValueState>,
        config_version: u64,
        now_ms: u64,
    ) {
        if config_version != self.config.version() {
            return;
        }
        let Some(active) = self.active.get(&id).cloned() else {
            return;
        };
        if !active.quorum.is_voter(from) {
            return;
        }
        if let Some(state) = state {
            match state.phase {
                Phase::Committed | Phase::Executed => {
                    self.commit_value(id, state.value, now_ms);
                    return;
                }
                Phase::Accepted => {
                    // Somebody accepted a value while we were preaccepting.
                    // Let the recovery path pick the highest ballot instead of
                    // guessing.
                    self.start_recovery(id, now_ms);
                    return;
                }
                Phase::PreAccepted => {}
            }
        }
        if ballot > active.ballot {
            // A recovery raised the ballot. A fast commit at our lower ballot
            // could race a higher-ballot decision, so abandon the fast path.
            self.start_recovery(id, now_ms);
            return;
        }
        let ActivePhase::PreAccept { replies } = &mut self
            .active
            .get_mut(&id)
            .expect("active checked above")
            .phase
        else {
            return;
        };
        replies.insert(from, attrs);
        self.evaluate_preaccept(id, now_ms);
    }

    fn evaluate_preaccept(&mut self, id: InstanceId, now_ms: u64) {
        let Some(active) = self.active.get(&id).cloned() else {
            return;
        };
        let ActivePhase::PreAccept { replies } = &active.phase else {
            return;
        };
        let count = replies
            .keys()
            .filter(|node| active.quorum.is_voter(**node))
            .count();
        if count < active.quorum.fast_acks {
            return;
        }
        let all_equal = replies.values().all(|attrs| attrs == &active.initial_attrs);
        if all_equal {
            let value = Value {
                command: active.command.clone(),
                attrs: active.initial_attrs.clone(),
                ballot: active.ballot,
            };
            self.commit_value(id, value, now_ms);
        } else {
            let mut merged = Attrs {
                seq: 1,
                deps: BTreeSet::new(),
            };
            for attrs in replies.values() {
                merged.merge(attrs);
            }
            let merged = self.merge_attrs(merged, &active.command, id);
            let ballot = active.ballot.next(self.id);
            self.begin_accept(id, active.command.clone(), merged, ballot, now_ms);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn on_accept(
        &mut self,
        from: NodeId,
        id: InstanceId,
        command: Command,
        attrs: Attrs,
        ballot: Ballot,
        quorum: QuorumPolicy,
        config_version: u64,
        now_ms: u64,
    ) {
        if config_version != self.config.version() || !quorum.is_voter(self.id) {
            self.send(
                from,
                Message::AcceptReply {
                    id,
                    accepted: false,
                    ballot,
                    state: None,
                    config_version: self.config.version(),
                },
            );
            return;
        }
        let existing = self.records.get(&id).cloned();
        let reply = match existing {
            Some(record) if matches!(record.phase, Phase::Committed | Phase::Executed) => {
                Message::AcceptReply {
                    id,
                    accepted: true,
                    ballot: record.value.ballot,
                    state: Some(ValueState {
                        value: record.value,
                        phase: record.phase,
                    }),
                    config_version: self.config.version(),
                }
            }
            Some(record) if record.value.ballot > ballot => Message::AcceptReply {
                id,
                accepted: false,
                ballot: record.value.ballot,
                state: Some(ValueState {
                    value: record.value,
                    phase: record.phase,
                }),
                config_version: self.config.version(),
            },
            Some(mut record) => {
                if record.value.command != command {
                    self.violations
                        .push(format!("{id:?}: accept for a different command"));
                }
                record.value = Value {
                    command,
                    attrs,
                    ballot,
                };
                record.phase = Phase::Accepted;
                record.quorum = quorum;
                record.config_version = config_version;
                record.updated_ms = now_ms;
                self.records.insert(id, record);
                Message::AcceptReply {
                    id,
                    accepted: true,
                    ballot,
                    state: None,
                    config_version: self.config.version(),
                }
            }
            None => {
                let record = Record {
                    value: Value {
                        command,
                        attrs,
                        ballot,
                    },
                    phase: Phase::Accepted,
                    quorum,
                    config_version,
                    updated_ms: now_ms,
                };
                self.insert_record(id, record);
                Message::AcceptReply {
                    id,
                    accepted: true,
                    ballot,
                    state: None,
                    config_version: self.config.version(),
                }
            }
        };
        self.send(from, reply);
    }

    fn begin_accept(
        &mut self,
        id: InstanceId,
        command: Command,
        attrs: Attrs,
        ballot: Ballot,
        now_ms: u64,
    ) {
        let Some(active) = self.active.get(&id).cloned() else {
            return;
        };
        let quorum = active.quorum.clone();
        let config_version = active.config_version;
        if let Some(record) = self.records.get(&id) {
            if record.phase == Phase::Accepted && record.value.ballot > ballot {
                // A higher-ballot acceptor already owns this instance.
                self.start_recovery(id, now_ms);
                return;
            }
        }
        let value = Value {
            command,
            attrs,
            ballot,
        };
        if let Some(record) = self.records.get_mut(&id) {
            if !matches!(record.phase, Phase::Committed | Phase::Executed) {
                record.value = value.clone();
                record.phase = Phase::Accepted;
                record.quorum = quorum.clone();
                record.updated_ms = now_ms;
            }
        } else {
            let record = Record {
                value: value.clone(),
                phase: Phase::Accepted,
                quorum: quorum.clone(),
                config_version,
                updated_ms: now_ms,
            };
            self.insert_record(id, record);
        }
        let mut acks = BTreeSet::new();
        if quorum.is_voter(self.id) {
            acks.insert(self.id);
        }
        if let Some(active) = self.active.get_mut(&id) {
            active.phase = ActivePhase::Accept {
                value: value.clone(),
                acks,
            };
            active.attempts = 0;
            active.ballot = ballot;
            active.deadline_ms = now_ms + self.options.retry_ms;
        }
        self.send_accept(id, &value, &quorum, config_version);
        self.check_accept_quorum(id, now_ms);
    }

    fn check_accept_quorum(&mut self, id: InstanceId, now_ms: u64) {
        let Some(active) = self.active.get(&id).cloned() else {
            return;
        };
        let ActivePhase::Accept { value, acks } = active.phase else {
            return;
        };
        if active.quorum.acked(&acks) {
            self.commit_value(id, value, now_ms);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn on_accept_reply(
        &mut self,
        from: NodeId,
        id: InstanceId,
        accepted: bool,
        _ballot: Ballot,
        state: Option<ValueState>,
        config_version: u64,
        now_ms: u64,
    ) {
        if config_version != self.config.version() {
            return;
        }
        let Some(active) = self.active.get(&id).cloned() else {
            return;
        };
        if !active.quorum.is_voter(from) {
            return;
        }
        if let Some(state) = state {
            match state.phase {
                Phase::Committed | Phase::Executed => {
                    self.commit_value(id, state.value, now_ms);
                    return;
                }
                Phase::Accepted if !accepted => {
                    self.start_recovery(id, now_ms);
                    return;
                }
                _ => {}
            }
        }
        if accepted {
            if let Some(active) = self.active.get_mut(&id) {
                if let ActivePhase::Accept { acks, .. } = &mut active.phase {
                    acks.insert(from);
                }
            }
            self.check_accept_quorum(id, now_ms);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn on_try_preaccept(
        &mut self,
        from: NodeId,
        id: InstanceId,
        command: Command,
        attrs: Attrs,
        ballot: Ballot,
        quorum: QuorumPolicy,
        config_version: u64,
        now_ms: u64,
    ) {
        if config_version != self.config.version() || !quorum.is_voter(self.id) {
            self.send(
                from,
                Message::TryPreAcceptReply {
                    id,
                    attrs,
                    ballot,
                    state: None,
                    config_version: self.config.version(),
                },
            );
            return;
        }
        let existing = self.records.get(&id).cloned();
        let (merged, state) = match existing {
            Some(record) if matches!(record.phase, Phase::Committed | Phase::Executed) => (
                record.value.attrs.clone(),
                Some(ValueState {
                    value: record.value,
                    phase: record.phase,
                }),
            ),
            Some(record) if record.phase == Phase::Accepted => (
                record.value.attrs.clone(),
                Some(ValueState {
                    value: record.value,
                    phase: record.phase,
                }),
            ),
            Some(record) if record.value.ballot > ballot => (
                record.value.attrs.clone(),
                Some(ValueState {
                    value: record.value,
                    phase: record.phase,
                }),
            ),
            Some(record) => {
                if record.value.command != command {
                    self.violations
                        .push(format!("{id:?}: try-preaccept for a different command"));
                }
                let mut base = record.value.attrs.clone();
                base.merge(&attrs);
                let merged = self.merge_attrs(base, &command, id);
                if let Some(record) = self.records.get_mut(&id) {
                    record.value.attrs = merged.clone();
                    record.value.ballot = record.value.ballot.max(ballot);
                    record.phase = Phase::PreAccepted;
                    record.updated_ms = now_ms;
                }
                (merged, None)
            }
            None => {
                let merged = self.merge_attrs(attrs, &command, id);
                let record = Record {
                    value: Value {
                        command,
                        attrs: merged.clone(),
                        ballot,
                    },
                    phase: Phase::PreAccepted,
                    quorum,
                    config_version,
                    updated_ms: now_ms,
                };
                self.insert_record(id, record);
                (merged, None)
            }
        };
        let record = self.records.get(&id).expect("record exists");
        self.send(
            from,
            Message::TryPreAcceptReply {
                id,
                attrs: merged,
                ballot: record.value.ballot,
                state,
                config_version: self.config.version(),
            },
        );
    }

    fn begin_try_preaccept(
        &mut self,
        id: InstanceId,
        command: Command,
        attrs: Attrs,
        ballot: Ballot,
        now_ms: u64,
    ) {
        let Some(active) = self.active.get(&id).cloned() else {
            return;
        };
        let quorum = active.quorum.clone();
        let config_version = active.config_version;
        let value = Value {
            command,
            attrs: attrs.clone(),
            ballot,
        };
        if let Some(record) = self.records.get_mut(&id) {
            if !matches!(record.phase, Phase::Committed | Phase::Executed) {
                record.value = value.clone();
                record.phase = Phase::PreAccepted;
                record.quorum = quorum.clone();
                record.updated_ms = now_ms;
            }
        } else {
            let record = Record {
                value: value.clone(),
                phase: Phase::PreAccepted,
                quorum: quorum.clone(),
                config_version,
                updated_ms: now_ms,
            };
            self.insert_record(id, record);
        }
        let mut replies = BTreeMap::new();
        if quorum.is_voter(self.id) {
            replies.insert(self.id, attrs.clone());
        }
        if let Some(active) = self.active.get_mut(&id) {
            active.phase = ActivePhase::TryPreAccept {
                value: value.clone(),
                replies,
            };
            active.attempts = 0;
            active.ballot = ballot;
            active.deadline_ms = now_ms + self.options.retry_ms;
        }
        self.send_try_preaccept(id, &value, &quorum, config_version);
        self.evaluate_try_preaccept(id, now_ms);
    }

    #[allow(clippy::too_many_arguments)]
    fn on_try_preaccept_reply(
        &mut self,
        from: NodeId,
        id: InstanceId,
        attrs: Attrs,
        ballot: Ballot,
        state: Option<ValueState>,
        config_version: u64,
        now_ms: u64,
    ) {
        if config_version != self.config.version() {
            return;
        }
        let Some(active) = self.active.get(&id).cloned() else {
            return;
        };
        if !active.quorum.is_voter(from) {
            return;
        }
        if let Some(state) = state {
            match state.phase {
                Phase::Committed | Phase::Executed => {
                    self.commit_value(id, state.value, now_ms);
                    return;
                }
                Phase::Accepted => {
                    // An accepted value appeared while recovery was probing.
                    // It must be carried forward with a higher ballot.
                    let ballot = active.ballot.max(state.value.ballot).next(self.id);
                    self.begin_accept(id, state.value.command, state.value.attrs, ballot, now_ms);
                    return;
                }
                Phase::PreAccepted => {}
            }
        }
        if ballot > active.ballot {
            self.start_recovery(id, now_ms);
            return;
        }
        let Some(active) = self.active.get(&id).cloned() else {
            return;
        };
        let ActivePhase::TryPreAccept { value, .. } = &active.phase else {
            return;
        };
        if let Some(active) = self.active.get_mut(&id) {
            if let ActivePhase::TryPreAccept { replies, .. } = &mut active.phase {
                replies.insert(from, attrs);
            }
        }
        self.evaluate_try_preaccept(id, now_ms);
        let _ = value;
    }

    fn evaluate_try_preaccept(&mut self, id: InstanceId, now_ms: u64) {
        let Some(active) = self.active.get(&id).cloned() else {
            return;
        };
        let ActivePhase::TryPreAccept { value, replies } = &active.phase else {
            return;
        };
        let count = replies
            .keys()
            .filter(|node| active.quorum.is_voter(**node))
            .count();
        if count < active.quorum.fast_acks {
            return;
        }
        let all_equal = replies.values().all(|attrs| attrs == &value.attrs);
        if all_equal {
            let committed = Value {
                command: active.command.clone(),
                attrs: value.attrs.clone(),
                ballot: active.ballot,
            };
            self.commit_value(id, committed, now_ms);
        } else {
            let mut merged = Attrs {
                seq: value.attrs.seq,
                deps: BTreeSet::new(),
            };
            for attrs in replies.values() {
                merged.merge(attrs);
            }
            let merged = self.merge_attrs(merged, &active.command, id);
            let ballot = active.ballot.next(self.id);
            self.begin_accept(id, active.command.clone(), merged, ballot, now_ms);
        }
    }

    fn commit_value(&mut self, id: InstanceId, value: Value, now_ms: u64) {
        let newly_committed = match self.records.get(&id) {
            None => true,
            Some(record) => !matches!(record.phase, Phase::Committed | Phase::Executed),
        };
        if let Some(record) = self.records.get_mut(&id) {
            if matches!(record.phase, Phase::Committed | Phase::Executed) {
                if record.value.command != value.command {
                    self.violations
                        .push(format!("{id:?}: conflicting committed commands"));
                }
                self.execute_ready(now_ms);
                return;
            }
            record.value = value.clone();
            record.phase = Phase::Committed;
            record.updated_ms = now_ms;
        } else {
            let record = Record {
                value: value.clone(),
                phase: Phase::Committed,
                quorum: QuorumPolicy::majority(self.config.voters()),
                config_version: self.config.version(),
                updated_ms: now_ms,
            };
            self.insert_record(id, record);
        }
        self.active.remove(&id);
        self.execution.commit(
            id,
            value.command.clone(),
            value.attrs.seq,
            &value.attrs.deps,
        );
        if newly_committed {
            self.broadcast_commit(id, &value);
        }
        self.execute_ready(now_ms);
    }

    fn on_recover(&mut self, from: NodeId, id: InstanceId, _ballot: Ballot, config_version: u64) {
        self.send(
            from,
            Message::RecoverReply {
                id,
                state: self.reply_state(id),
                config_version: self.config.version(),
            },
        );
        let _ = config_version;
    }

    fn start_recovery(&mut self, id: InstanceId, now_ms: u64) {
        let Some(record) = self.records.get(&id).cloned() else {
            return;
        };
        if matches!(record.phase, Phase::Committed | Phase::Executed) {
            return;
        }
        if let Some(active) = self.active.get(&id) {
            if matches!(active.phase, ActivePhase::Recover { .. }) {
                return;
            }
        }
        let (quorum, config_version, prior_ballot) = match self.active.get(&id) {
            Some(active) => (active.quorum.clone(), active.config_version, active.ballot),
            None => (
                record.quorum.clone(),
                record.config_version,
                record.value.ballot,
            ),
        };
        let ballot = prior_ballot.max(record.value.ballot).next(self.id);
        let mut replies = BTreeMap::new();
        if quorum.is_voter(self.id) {
            replies.insert(
                self.id,
                Some(ValueState {
                    value: record.value.clone(),
                    phase: record.phase,
                }),
            );
        }
        self.active.insert(
            id,
            Active {
                command: record.value.command.clone(),
                quorum: quorum.clone(),
                config_version,
                initial_attrs: record.value.attrs.clone(),
                ballot,
                phase: ActivePhase::Recover { ballot, replies },
                deadline_ms: now_ms + self.options.retry_ms,
                attempts: 0,
            },
        );
        for voter in &quorum.voters {
            if *voter == self.id {
                continue;
            }
            self.send(
                *voter,
                Message::Recover {
                    id,
                    ballot,
                    config_version,
                },
            );
        }
        self.evaluate_recovery(id, now_ms);
    }

    fn on_recover_reply(
        &mut self,
        from: NodeId,
        id: InstanceId,
        state: Option<ValueState>,
        config_version: u64,
        now_ms: u64,
    ) {
        if config_version != self.config.version() {
            return;
        }
        let Some(active) = self.active.get(&id).cloned() else {
            return;
        };
        if !active.quorum.is_voter(from) {
            return;
        }
        if let Some(state) = &state {
            if matches!(state.phase, Phase::Committed | Phase::Executed) {
                self.commit_value(id, state.value.clone(), now_ms);
                return;
            }
        }
        if let Some(active) = self.active.get_mut(&id) {
            if let ActivePhase::Recover { replies, .. } = &mut active.phase {
                replies.insert(from, state);
            }
        }
        self.evaluate_recovery(id, now_ms);
    }

    fn evaluate_recovery(&mut self, id: InstanceId, now_ms: u64) {
        let Some(active) = self.active.get(&id).cloned() else {
            return;
        };
        let ActivePhase::Recover { ballot, replies } = &active.phase else {
            return;
        };
        let count = replies
            .keys()
            .filter(|node| active.quorum.is_voter(**node))
            .count();
        if count < active.quorum.ack_required {
            return;
        }

        let mut accepted: Option<Value> = None;
        let mut preaccepted: Option<Value> = None;
        for state in replies.values().flatten() {
            match state.phase {
                Phase::Committed | Phase::Executed => {
                    self.commit_value(id, state.value.clone(), now_ms);
                    return;
                }
                Phase::Accepted => {
                    let replace = accepted.as_ref().is_none_or(|current| {
                        state.value.ballot > current.ballot
                            || (state.value.ballot == current.ballot
                                && state.value.attrs.seq > current.attrs.seq)
                    });
                    if replace {
                        accepted = Some(state.value.clone());
                    }
                }
                Phase::PreAccepted => {
                    let replace = preaccepted.as_ref().is_none_or(|current| {
                        state.value.attrs.seq > current.attrs.seq
                            || (state.value.attrs.seq == current.attrs.seq
                                && state.value.ballot > current.ballot)
                    });
                    if replace {
                        preaccepted = Some(state.value.clone());
                    }
                }
            }
        }

        if let Some(value) = accepted {
            let next_ballot = value.ballot.max(*ballot).next(self.id);
            self.begin_accept(id, value.command, value.attrs, next_ballot, now_ms);
        } else if let Some(value) = preaccepted {
            let next_ballot = value.ballot.max(*ballot).next(self.id);
            self.begin_try_preaccept(id, value.command, value.attrs, next_ballot, now_ms);
        } else {
            // A majority has no record for this instance; there is nothing
            // safe to recover.
            self.active.remove(&id);
        }
    }

    fn retry_preaccept(&mut self, id: InstanceId, now_ms: u64) {
        let Some(active) = self.active.get(&id).cloned() else {
            return;
        };
        let ActivePhase::PreAccept { replies } = &active.phase else {
            return;
        };
        for voter in &active.quorum.voters {
            if *voter == self.id {
                continue;
            }
            if replies.contains_key(voter) {
                continue;
            }
            self.send(
                *voter,
                Message::PreAccept {
                    id,
                    command: active.command.clone(),
                    attrs: active.initial_attrs.clone(),
                    ballot: active.ballot,
                    quorum: active.quorum.clone(),
                    config_version: active.config_version,
                },
            );
        }
        if let Some(active) = self.active.get_mut(&id) {
            active.attempts += 1;
            active.deadline_ms = now_ms + self.options.retry_ms;
        }
    }

    fn retry_accept(&mut self, id: InstanceId, now_ms: u64) {
        let Some(active) = self.active.get(&id).cloned() else {
            return;
        };
        let ActivePhase::Accept { value, .. } = &active.phase else {
            return;
        };
        self.send_accept(id, value, &active.quorum, active.config_version);
        if let Some(active) = self.active.get_mut(&id) {
            active.attempts += 1;
            active.deadline_ms = now_ms + self.options.retry_ms;
        }
    }

    fn retry_try_preaccept(&mut self, id: InstanceId, now_ms: u64) {
        let Some(active) = self.active.get(&id).cloned() else {
            return;
        };
        let ActivePhase::TryPreAccept { value, .. } = &active.phase else {
            return;
        };
        self.send_try_preaccept(id, value, &active.quorum, active.config_version);
        if let Some(active) = self.active.get_mut(&id) {
            active.attempts += 1;
            active.deadline_ms = now_ms + self.options.retry_ms;
        }
    }

    fn retry_recover(&mut self, id: InstanceId, now_ms: u64) {
        let Some(active) = self.active.get(&id).cloned() else {
            return;
        };
        let ActivePhase::Recover { .. } = active.phase else {
            return;
        };
        let ballot = active.ballot.next(self.id);
        if let Some(active) = self.active.get_mut(&id) {
            active.ballot = ballot;
            active.phase = ActivePhase::Recover {
                ballot,
                replies: BTreeMap::new(),
            };
            if let Some(record) = self.records.get(&id) {
                if active.quorum.is_voter(self.id) {
                    if let ActivePhase::Recover { replies, .. } = &mut active.phase {
                        replies.insert(
                            self.id,
                            Some(ValueState {
                                value: record.value.clone(),
                                phase: record.phase,
                            }),
                        );
                    }
                }
            }
        }
        for voter in &active.quorum.voters {
            if *voter == self.id {
                continue;
            }
            self.send(
                *voter,
                Message::Recover {
                    id,
                    ballot,
                    config_version: active.config_version,
                },
            );
        }
        if let Some(active) = self.active.get_mut(&id) {
            active.deadline_ms = now_ms + self.options.retry_ms;
            active.attempts += 1;
        }
    }

    fn execute_ready(&mut self, now_ms: u64) {
        loop {
            let batches = self.execution.ready_batches();
            if batches.is_empty() {
                return;
            }
            if batches.len() > 1
                && self.machine.supports_parallel()
                && !batches.iter().flatten().any(|id| {
                    self.records
                        .get(id)
                        .map(|record| record.value.command.is_membership())
                        .unwrap_or(false)
                })
            {
                self.execute_batches_parallel(batches, now_ms);
            } else {
                for batch in batches {
                    for id in batch {
                        self.execute_instance(id, now_ms);
                    }
                }
            }
        }
    }

    fn execute_batches_parallel(&mut self, batches: Vec<Vec<InstanceId>>, now_ms: u64)
    where
        S: Send + Sync,
    {
        let commands: Vec<Vec<(InstanceId, Command)>> = batches
            .iter()
            .map(|batch| {
                batch
                    .iter()
                    .filter_map(|id| {
                        self.execution
                            .command(*id)
                            .cloned()
                            .map(|command| (*id, command))
                    })
                    .collect()
            })
            .collect();
        let results: Vec<(Vec<InstanceId>, S)> = commands
            .into_par_iter()
            .map(|commands| {
                let mut machine = self
                    .machine
                    .fork_batch()
                    .expect("supports_parallel implies fork_batch");
                for (id, command) in &commands {
                    machine.apply(*id, command);
                }
                (commands.iter().map(|(id, _)| *id).collect(), machine)
            })
            .collect();
        for (batch, machine) in results {
            self.machine.merge_batch(machine);
            for id in batch {
                self.finish_instance(id, now_ms);
            }
        }
    }

    fn execute_instance(&mut self, id: InstanceId, now_ms: u64) {
        if self.executed.contains(&id) {
            return;
        }
        let Some(record) = self.records.get(&id) else {
            return;
        };
        let command = record.value.command.clone();
        self.apply_membership(id, &command);
        self.machine.apply(id, &command);
        self.mark_instance_executed(id, now_ms);
    }

    fn finish_instance(&mut self, id: InstanceId, now_ms: u64) {
        if self.executed.contains(&id) {
            return;
        }
        let Some(record) = self.records.get(&id) else {
            return;
        };
        let command = record.value.command.clone();
        self.apply_membership(id, &command);
        self.mark_instance_executed(id, now_ms);
    }

    fn apply_membership(&mut self, id: InstanceId, command: &Command) {
        if let Some(change) = command.membership() {
            if let Err(error) = self.config.apply(change) {
                self.violations
                    .push(format!("{id:?}: membership apply failed: {error}"));
            }
            // Proposals that used the old configuration cannot be counted
            // against the new quorum. They stay in the log and can recover
            // under their recorded quorum.
            let version = self.config.version();
            self.active
                .retain(|_, active| active.config_version == version);
        }
    }

    fn mark_instance_executed(&mut self, id: InstanceId, now_ms: u64) {
        if let Some(record) = self.records.get_mut(&id) {
            record.phase = Phase::Executed;
            record.updated_ms = now_ms;
        }
        self.executed.insert(id);
        self.execution.mark_executed(id);
        self.executed_order.push(id);
        self.active.remove(&id);
    }
}

impl<S: StateMachine + Default> Replica<S> {
    pub fn from_snapshot(id: NodeId, snapshot: Snapshot, options: Options) -> Self {
        let mut replica = Self::new(id, snapshot.config, S::default(), options);
        replica.counters = snapshot.counters;
        for (instance, record) in snapshot.records {
            replica.insert_record(instance, record);
        }
        replica.executed = snapshot.executed_order.iter().copied().collect();
        replica.executed_order = snapshot.executed_order;
        for (instance, record) in replica.records.clone() {
            if matches!(record.phase, Phase::Committed | Phase::Executed) {
                replica.execution.commit(
                    instance,
                    record.value.command.clone(),
                    record.value.attrs.seq,
                    &record.value.attrs.deps,
                );
            }
        }
        for instance in replica.executed_order.clone() {
            replica.execution.mark_executed(instance);
            let Some(record) = replica.records.get(&instance) else {
                replica.violations.push(format!(
                    "{instance:?}: executed snapshot entry has no record"
                ));
                continue;
            };
            let command = record.value.command.clone();
            replica.machine.apply(instance, &command);
            if let Some(record) = replica.records.get_mut(&instance) {
                record.phase = Phase::Executed;
            }
        }
        replica
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Default)]
    struct ParallelProbe {
        seen: BTreeSet<InstanceId>,
        current: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        max: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl StateMachine for ParallelProbe {
        fn apply(&mut self, instance: InstanceId, _command: &Command) {
            use std::sync::atomic::Ordering;
            let current = self.current.fetch_add(1, Ordering::SeqCst) + 1;
            self.max.fetch_max(current, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(20));
            self.seen.insert(instance);
            self.current.fetch_sub(1, Ordering::SeqCst);
        }

        fn supports_parallel(&self) -> bool {
            true
        }

        fn fork_batch(&self) -> Option<Self> {
            Some(self.clone())
        }

        fn merge_batch(&mut self, other: Self) {
            self.seen.extend(other.seen);
        }
    }

    #[test]
    fn independent_sccs_run_in_parallel() {
        let node = NodeId::derive("parallel-probe");
        let voters = BTreeSet::from([node]);
        let config = ClusterConfig::bootstrap(voters.iter().copied());
        let instance = |counter| InstanceId {
            proposer: node,
            counter,
        };
        let command = |value: f64, tensor: u128| Command::Publish {
            key: plait_core::TensorKey::new(plait_core::TensorId(tensor), plait_core::ShardId(0)),
            version: 1,
            values: vec![value],
        };
        let records = vec![
            (
                instance(1),
                Record {
                    value: Value {
                        command: command(1.0, 1),
                        attrs: Attrs::new(1, BTreeSet::new()),
                        ballot: Ballot::zero(node),
                    },
                    phase: Phase::Committed,
                    quorum: QuorumPolicy::majority(&voters),
                    config_version: 0,
                    updated_ms: 1,
                },
            ),
            (
                instance(2),
                Record {
                    value: Value {
                        command: command(2.0, 2),
                        attrs: Attrs::new(2, BTreeSet::new()),
                        ballot: Ballot::zero(node),
                    },
                    phase: Phase::Committed,
                    quorum: QuorumPolicy::majority(&voters),
                    config_version: 0,
                    updated_ms: 1,
                },
            ),
        ];
        let mut replica: Replica<ParallelProbe> = Replica::from_snapshot(
            node,
            Snapshot {
                config,
                counters: BTreeMap::new(),
                records,
                executed_order: Vec::new(),
            },
            Options::default(),
        );
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("test pool");
        pool.install(|| replica.execute_ready(2));
        assert_eq!(replica.executed_count(), 2);
        assert!(
            replica
                .machine()
                .max
                .load(std::sync::atomic::Ordering::SeqCst)
                >= 2,
            "independent SCCs should overlap"
        );
    }

    #[test]
    fn fast_quorum_intersects_majority() {
        for n in 1usize..=9 {
            let voters: BTreeSet<NodeId> = (0..n)
                .map(|index| NodeId::derive(&format!("quorum-{index}")))
                .collect();
            let quorum = QuorumPolicy::majority(&voters);
            assert!(quorum.fast_acks + quorum.ack_required > n);
        }
    }
}
