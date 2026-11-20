use std::collections::{BTreeMap, BTreeSet, VecDeque};

use plait_consensus::{
    Attrs, Ballot, InstanceId, Message, Options, Phase, Record, Replica, Snapshot, StateMachine,
    Value,
};
use plait_core::{
    ClusterConfig, Command, MembershipChange, NodeId, QuorumPolicy, ShardId, TensorId, TensorKey,
};

#[derive(Default, Clone)]
struct Book {
    value: i64,
    applied: BTreeSet<InstanceId>,
    order: Vec<InstanceId>,
}

impl StateMachine for Book {
    fn apply(&mut self, instance: InstanceId, command: &Command) {
        if !self.applied.insert(instance) {
            return;
        }
        if let Command::Publish { values, .. } = command {
            self.value = values.first().copied().unwrap_or_default() as i64;
        }
        self.order.push(instance);
    }
}

struct Sim {
    nodes: BTreeMap<NodeId, Replica<Book>>,
    queue: VecDeque<(NodeId, NodeId, Message)>,
    blocked: BTreeSet<(NodeId, NodeId)>,
    now: u64,
    accepts: usize,
    recovers: usize,
    options: Options,
}

impl Sim {
    fn new(ids: &[NodeId]) -> Self {
        let options = Options {
            fast_timeout_ms: 20,
            retry_ms: 30,
            recovery_after_ms: 100,
            retries_before_recovery: 2,
        };
        let config = ClusterConfig::bootstrap(ids.iter().copied());
        let nodes = ids
            .iter()
            .map(|id| {
                (
                    *id,
                    Replica::new(*id, config.clone(), Book::default(), options.clone()),
                )
            })
            .collect();
        Self {
            nodes,
            queue: VecDeque::new(),
            blocked: BTreeSet::new(),
            now: 1,
            accepts: 0,
            recovers: 0,
            options,
        }
    }

    fn drain_outbox(&mut self) {
        let ids: Vec<NodeId> = self.nodes.keys().copied().collect();
        for from in ids {
            if let Some(replica) = self.nodes.get_mut(&from) {
                for outbound in replica.take_outbox() {
                    if matches!(outbound.message, Message::Accept { .. }) {
                        self.accepts += 1;
                    }
                    if matches!(outbound.message, Message::Recover { .. }) {
                        self.recovers += 1;
                    }
                    self.queue.push_back((from, outbound.to, outbound.message));
                }
            }
        }
    }

    fn propose(&mut self, node: NodeId, command: Command) -> InstanceId {
        self.now += 1;
        let id = self
            .nodes
            .get_mut(&node)
            .expect("node exists")
            .propose(command, self.now)
            .expect("proposal accepted");
        self.drain_outbox();
        id
    }

    fn deliver(&mut self, from: NodeId, to: NodeId, message: Message) {
        if self.blocked.contains(&(from, to)) {
            return;
        }
        if let Some(replica) = self.nodes.get_mut(&to) {
            replica.handle(from, message, self.now);
        }
    }

    fn pump(&mut self) {
        let mut steps = 0usize;
        while let Some((from, to, message)) = self.queue.pop_front() {
            self.deliver(from, to, message);
            self.drain_outbox();
            steps += 1;
            assert!(steps < 200_000, "message loop did not quiesce");
        }
    }

    fn deliver_where(&mut self, mut predicate: impl FnMut(&NodeId, &NodeId, &Message) -> bool) {
        let mut kept = VecDeque::new();
        while let Some((from, to, message)) = self.queue.pop_front() {
            if predicate(&from, &to, &message) {
                self.deliver(from, to, message);
                self.drain_outbox();
            } else {
                kept.push_back((from, to, message));
            }
        }
        self.queue = kept;
    }

    fn tick_all(&mut self, delta_ms: u64) {
        self.now += delta_ms;
        let ids: Vec<NodeId> = self.nodes.keys().copied().collect();
        for id in ids {
            self.nodes.get_mut(&id).expect("node exists").tick(self.now);
        }
        self.drain_outbox();
    }

    fn book(&self, node: NodeId) -> &Book {
        self.nodes.get(&node).expect("node exists").machine()
    }

    fn assert_consistent(&self, expected_applied: usize) {
        let mut books = self.nodes.values().map(|replica| replica.machine());
        let first = books.next().expect("at least one node");
        assert_eq!(first.order.len(), expected_applied);
        for book in books {
            assert_eq!(book.order, first.order, "execution orders diverged");
            assert_eq!(book.value, first.value, "state diverged");
        }
        for (id, replica) in &self.nodes {
            assert!(
                replica.violations().is_empty(),
                "{id}: {:?}",
                replica.violations()
            );
        }
    }
}

fn node(label: &str) -> NodeId {
    NodeId::derive(label)
}

fn publish(value: i64) -> Command {
    Command::Publish {
        key: TensorKey::new(TensorId(42), ShardId(0)),
        version: value.unsigned_abs(),
        values: vec![value as f64],
    }
}

#[test]
fn single_proposal_commits_everywhere_without_accept() {
    let ids = [node("a"), node("b"), node("c")];
    let mut sim = Sim::new(&ids);
    sim.propose(ids[0], publish(7));
    sim.pump();
    sim.assert_consistent(1);
    assert_eq!(sim.book(ids[0]).value, 7);
    assert_eq!(
        sim.accepts, 0,
        "uncontended proposal should take the fast path"
    );
}

#[test]
fn concurrent_conflicting_proposals_converge() {
    let ids = [node("a"), node("b"), node("c")];
    let mut sim = Sim::new(&ids);
    for (index, id) in ids.iter().enumerate() {
        sim.propose(*id, publish(index as i64 + 1));
    }
    sim.pump();
    sim.assert_consistent(3);
    assert!(sim.accepts > 0, "conflicts should force the slow path");
    let value = sim.book(ids[0]).value;
    assert!((1..=3).contains(&value));
    assert!(ids.iter().all(|id| sim.book(*id).value == value));
}

#[test]
fn minority_cannot_commit_but_majority_can() {
    let ids = [node("a"), node("b"), node("c"), node("d"), node("e")];
    let mut sim = Sim::new(&ids);
    sim.propose(ids[0], publish(9));

    // A's fast quorum needs three replicas. Cut A off from C, D, and E, so
    // only B's leg counts and no decision is possible.
    for peer in [ids[2], ids[3], ids[4]] {
        sim.blocked.insert((ids[0], peer));
    }
    sim.pump();
    assert_eq!(sim.nodes[&ids[0]].executed_count(), 0);
    assert_eq!(sim.accepts, 0);

    // Restore the link. A's retry reaches C, supplying the third fast-quorum
    // reply.
    for peer in [ids[2], ids[3], ids[4]] {
        sim.blocked.remove(&(ids[0], peer));
    }
    sim.tick_all(25);
    sim.pump();
    sim.assert_consistent(1);
    assert_eq!(sim.book(ids[0]).value, 9);
}

#[test]
fn recovery_after_proposer_stops() {
    let ids = [node("a"), node("b"), node("c")];
    let mut sim = Sim::new(&ids);
    sim.propose(ids[0], publish(11));

    // Only B receives the preaccept. A and C are then partitioned away.
    sim.deliver_where(|from, to, message| {
        *from == ids[0] && *to == ids[1] && matches!(message, Message::PreAccept { .. })
    });
    sim.blocked.insert((ids[0], ids[1]));
    sim.blocked.insert((ids[1], ids[0]));
    sim.blocked.insert((ids[0], ids[2]));
    sim.blocked.insert((ids[2], ids[0]));

    for _ in 0..6 {
        sim.tick_all(120);
        sim.pump();
    }
    assert_eq!(sim.book(ids[1]).value, 11, "B must recover the value");
    assert_eq!(sim.book(ids[1]).order.len(), 1);
    assert_eq!(sim.book(ids[2]).order.len(), 1);
    assert!(sim.recovers > 0);

    // Heal A. Its own watchdog should discover the committed value.
    sim.blocked.clear();
    sim.tick_all(1_000);
    sim.pump();
    sim.assert_consistent(1);
    assert_eq!(sim.book(ids[0]).value, 11);
}

#[test]
fn membership_add_uses_snapshot_and_promotes_to_voter() {
    let ids = [node("a"), node("b"), node("c")];
    let joining = node("d");
    let mut sim = Sim::new(&ids);

    sim.propose(
        ids[0],
        Command::Membership(MembershipChange::StageAdd {
            node: joining,
            addr: "127.0.0.1:9004".to_owned(),
        }),
    );
    sim.pump();
    for id in &ids {
        assert!(sim.nodes[id].config().staged_add().contains(&joining));
        assert!(!sim.nodes[id].config().is_voter(joining));
    }

    // A snapshot carries the stage and the executed prefix.
    let snapshot = sim.nodes[&ids[0]].snapshot();
    let joined = Replica::<Book>::from_snapshot(joining, snapshot, sim.options.clone());
    sim.nodes.insert(joining, joined);

    sim.propose(
        ids[0],
        Command::Membership(MembershipChange::CommitAdd { node: joining }),
    );
    sim.pump();
    for id in &ids {
        assert!(sim.nodes[id].config().is_voter(joining));
    }
    assert!(sim.nodes[&joining].config().is_voter(joining));

    // The new voter can propose; four voters require a three-node quorum.
    sim.propose(joining, publish(5));
    sim.pump();
    sim.assert_consistent(3);
    assert_eq!(sim.book(joining).value, 5);
}

#[test]
fn membership_remove_excludes_node_from_quorum() {
    let ids = [node("a"), node("b"), node("c")];
    let mut sim = Sim::new(&ids);

    sim.propose(
        ids[0],
        Command::Membership(MembershipChange::StageRemove { node: ids[2] }),
    );
    sim.pump();
    sim.propose(
        ids[0],
        Command::Membership(MembershipChange::CommitRemove { node: ids[2] }),
    );
    sim.pump();

    for id in [ids[0], ids[1]] {
        assert!(!sim.nodes[&id].config().is_voter(ids[2]));
    }
    assert!(!sim.nodes[&ids[2]].config().is_voter(ids[2]));

    sim.propose(ids[1], publish(3));
    sim.pump();
    assert_eq!(sim.book(ids[0]).value, 3);
    assert_eq!(sim.book(ids[1]).value, 3);
    // The removed node was not part of the quorum and receives no commit.
    assert_eq!(sim.book(ids[2]).order.len(), 2);
}

#[test]
fn recovery_preserves_an_accepted_value() {
    let ids = [node("a"), node("b"), node("c")];
    let config = ClusterConfig::bootstrap(ids.iter().copied());
    let instance = InstanceId {
        proposer: ids[0],
        counter: 1,
    };
    let value = Value {
        command: publish(42),
        attrs: Attrs::new(1, BTreeSet::new()),
        ballot: Ballot::zero(ids[0]),
    };
    let record = Record {
        value,
        phase: Phase::Accepted,
        quorum: QuorumPolicy::majority(config.voters()),
        config_version: 0,
        updated_ms: 0,
    };
    let snapshot = Snapshot {
        config,
        counters: BTreeMap::from([(ids[0], 1)]),
        records: vec![(instance, record)],
        executed_order: Vec::new(),
    };
    let mut sim = Sim::new(&ids);
    sim.nodes.clear();
    for id in ids {
        sim.nodes.insert(
            id,
            Replica::<Book>::from_snapshot(id, snapshot.clone(), sim.options.clone()),
        );
    }
    for _ in 0..8 {
        sim.tick_all(200);
        sim.pump();
    }
    sim.assert_consistent(1);
    assert_eq!(sim.book(ids[0]).value, 42);
    assert!(sim.accepts > 0);
}
