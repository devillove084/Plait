//! Local replica journal. Only the final, incomplete frame can be discarded.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};

use bincode::Options as _;
use plait_consensus::{InstanceId, Record, ReplicaDelta, Snapshot};
use plait_core::NodeId;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::JoinTarget;

const MAGIC: &[u8] = b"PLAIT-WAL\x01";
const HEADER_BYTES: u64 = 16;
const CHECKSUM_BYTES: u64 = 32;

#[derive(Clone, Debug)]
pub enum StorageMode {
    Initialize(PathBuf),
    Resume(PathBuf),
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Image {
    pub id: NodeId,
    pub snapshot: Snapshot,
    pub peers: BTreeMap<NodeId, String>,
    /// Present until a joining replica has durably installed its first cluster image.
    pub pending_join: Option<JoinTarget>,
}

#[derive(Serialize, Deserialize)]
enum Frame {
    Image(Image),
    Delta(ReplicaDelta),
}

pub(crate) struct Journal {
    file: File,
    _lock: File,
}

impl Journal {
    pub fn open(mode: StorageMode, id: NodeId) -> io::Result<(Self, Option<Image>)> {
        let (dir, initialize) = match mode {
            StorageMode::Initialize(path) => (path, true),
            StorageMode::Resume(path) => (path, false),
        };
        if initialize {
            make_state_directory(&dir)?;
        }
        let lock = lock_directory(&dir)?;
        let file = OpenOptions::new()
            .read(true)
            .append(true)
            .create_new(initialize)
            .open(dir.join("replica.wal"))?;
        let mut journal = Self { file, _lock: lock };
        let image = if initialize {
            journal.file.write_all(MAGIC)?;
            // Make the newly created file discoverable on restart.
            File::open(&dir)?.sync_all()?;
            None
        } else {
            Some(journal.replay(id)?)
        };
        Ok((journal, image))
    }

    pub fn install(&mut self, image: Image) -> io::Result<()> {
        self.append(&Frame::Image(image))
    }

    pub fn record(&mut self, delta: ReplicaDelta) -> io::Result<()> {
        self.append(&Frame::Delta(delta))
    }

    fn append(&mut self, frame: &Frame) -> io::Result<()> {
        let payload = codec().serialize(frame).map_err(invalid)?;
        let length = payload.len() as u64;
        self.file.write_all(&length.to_le_bytes())?;
        self.file.write_all(&(!length).to_le_bytes())?;
        self.file.write_all(&payload)?;
        self.file.write_all(&Sha256::digest(&payload))?;
        self.file.sync_all()
    }

    fn replay(&mut self, id: NodeId) -> io::Result<Image> {
        let bytes = self.file.metadata()?.len();
        let mut reader = BufReader::new(&self.file);
        let mut magic = [0; MAGIC.len()];
        reader.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(invalid("unknown replica journal format"));
        }
        let mut offset = MAGIC.len() as u64;
        let mut replay = None;
        while bytes - offset >= HEADER_BYTES {
            let mut header = [0; HEADER_BYTES as usize];
            reader.read_exact(&mut header)?;
            let length = u64::from_le_bytes(header[..8].try_into().unwrap());
            let inverse = u64::from_le_bytes(header[8..].try_into().unwrap());
            if length == 0 || length != !inverse {
                return Err(invalid("invalid journal frame header"));
            }
            let available = bytes - offset - HEADER_BYTES;
            if available < CHECKSUM_BYTES || length > available - CHECKSUM_BYTES {
                break;
            }
            let capacity = usize::try_from(length).map_err(invalid)?;
            let mut payload = Vec::new();
            payload
                .try_reserve_exact(capacity)
                .map_err(io::Error::other)?;
            payload.resize(capacity, 0);
            reader.read_exact(&mut payload)?;
            let mut checksum = [0; CHECKSUM_BYTES as usize];
            reader.read_exact(&mut checksum)?;
            if checksum != <[u8; 32]>::from(Sha256::digest(&payload)) {
                return Err(invalid("replica journal checksum mismatch"));
            }
            match codec()
                .with_limit(length)
                .deserialize(&payload)
                .map_err(invalid)?
            {
                Frame::Image(image) => {
                    if image.id != id {
                        return Err(invalid("replica journal belongs to a different node ID"));
                    }
                    replay = Some(Replay::new(image));
                }
                Frame::Delta(delta) => replay
                    .as_mut()
                    .ok_or_else(|| invalid("delta before image"))?
                    .apply(delta),
            }
            offset += HEADER_BYTES + length + CHECKSUM_BYTES;
        }
        let image = replay
            .ok_or_else(|| invalid("replica journal has no complete initial image"))?
            .finish();
        if offset < bytes {
            self.file.set_len(offset)?;
            self.file.sync_all()?;
        }
        Ok(image)
    }
}

struct Replay {
    image: Image,
    records: BTreeMap<InstanceId, Record>,
}

impl Replay {
    fn new(mut image: Image) -> Self {
        let records = std::mem::take(&mut image.snapshot.records)
            .into_iter()
            .collect();
        Self { image, records }
    }

    fn apply(&mut self, delta: ReplicaDelta) {
        self.image.snapshot.config = delta.config;
        self.image.snapshot.counters = delta.counters;
        self.image
            .snapshot
            .executed_order
            .extend(delta.executed_tail);
        self.records.extend(delta.records);
    }

    fn finish(mut self) -> Image {
        self.image.snapshot.records = self.records.into_iter().collect();
        self.image
    }
}

fn codec() -> impl bincode::Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .reject_trailing_bytes()
}

fn invalid(error: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error.to_string())
}

fn make_state_directory(path: &Path) -> io::Result<()> {
    let path = std::env::current_dir()?.join(path);
    let mut parents = Vec::new();
    for directory in path.ancestors() {
        if directory.try_exists()? {
            break;
        }
        if let Some(parent) = directory.parent() {
            parents.push(parent);
        }
    }
    std::fs::create_dir_all(&path)?;
    for parent in parents {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}

#[cfg(unix)]
fn lock_directory(dir: &Path) -> io::Result<File> {
    use std::os::fd::AsRawFd;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(dir.join("writer.lock"))?;
    // SAFETY: the descriptor remains owned by `file` until the journal closes.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(file)
}

#[cfg(not(unix))]
fn lock_directory(_dir: &Path) -> io::Result<File> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "replica journal requires an OS file lock",
    ))
}

#[cfg(all(test, unix))]
mod tests {
    use plait_consensus::{Attrs, Ballot, Message, Options, Phase, Replica};
    use plait_core::{ClusterConfig, Command};
    use plait_fabric::Runtime;

    use super::*;

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            Self(std::env::temp_dir().join(format!("plait-journal-{}", NodeId::random())))
        }

        fn resume(&self) -> StorageMode {
            StorageMode::Resume(self.0.clone())
        }

        fn initialize(&self, replica: &Replica<Runtime>) -> Journal {
            let (mut journal, _) =
                Journal::open(StorageMode::Initialize(self.0.clone()), replica.id()).unwrap();
            journal
                .install(Image {
                    id: replica.id(),
                    snapshot: replica.snapshot(),
                    peers: BTreeMap::new(),
                    pending_join: None,
                })
                .unwrap();
            journal
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn restart_preserves_an_uncommitted_acceptor_vote_and_proposer_counter() {
        let id = NodeId::derive("durable-local");
        let peer = NodeId::derive("durable-peer");
        let config = ClusterConfig::bootstrap([id, peer]);
        let quorum = config.quorum_for(&Command::Nop).unwrap();
        let mut replica = Replica::new(id, config, Runtime::new(), Options::default());
        let dir = TestDirectory::new();
        let mut journal = dir.initialize(&replica);
        replica.track_deltas();
        let own = replica.propose(Command::Nop, 42_000).unwrap();
        let remote = InstanceId {
            proposer: peer,
            counter: 1,
        };
        let accept = |round| Message::Accept {
            id: remote,
            command: Command::Nop,
            attrs: Attrs::new(1, Default::default()),
            ballot: Ballot::new(round, peer),
            quorum: quorum.clone(),
            config_version: 0,
        };
        replica.handle(peer, accept(7), 42_000);
        journal.record(replica.take_delta().unwrap()).unwrap();
        assert!(replica.take_delta().is_none());
        assert!(Journal::open(dir.resume(), id).is_err());
        drop(journal);

        let (_, image) = Journal::open(dir.resume(), id).unwrap();
        let image = image.unwrap().snapshot;
        assert!(
            image
                .records
                .iter()
                .any(|(instance, record)| *instance == remote && record.phase == Phase::Accepted)
        );
        let mut replica = Replica::<Runtime>::from_snapshot(id, image, Options::default());
        replica.handle(peer, accept(6), 10);
        assert!(replica.take_outbox().iter().any(|reply| matches!(reply.message,
            Message::AcceptReply {id, accepted: false, ballot, ..} if id == remote && ballot.round == 7
        )));
        assert!(replica.propose(Command::Nop, 11).unwrap().counter > own.counter);
    }

    #[test]
    fn repairs_an_incomplete_tail_but_rejects_corruption_of_a_complete_frame() {
        let id = NodeId::derive("durable-replay");
        let mut replica = Replica::new(
            id,
            ClusterConfig::bootstrap([id]),
            Runtime::new(),
            Options::default(),
        );
        let dir = TestDirectory::new();
        let mut journal = dir.initialize(&replica);
        replica.track_deltas();
        let instance = replica.propose(Command::Nop, 10).unwrap();
        journal.record(replica.take_delta().unwrap()).unwrap();
        let length = journal.file.metadata().unwrap().len();
        journal.file.write_all(&32u64.to_le_bytes()).unwrap();
        journal.file.write_all(&(!32u64).to_le_bytes()).unwrap();
        journal.file.write_all(b"torn tail").unwrap();
        drop(journal);

        assert!(Journal::open(dir.resume(), NodeId::derive("other")).is_err());
        let (mut journal, image) = Journal::open(dir.resume(), id).unwrap();
        assert_eq!(image.unwrap().snapshot.executed_order, [instance]);
        assert_eq!(journal.file.metadata().unwrap().len(), length);
        journal.file.write_all(&1u64.to_le_bytes()).unwrap();
        journal.file.write_all(&(!1u64).to_le_bytes()).unwrap();
        journal.file.write_all(&[0; 33]).unwrap();
        drop(journal);
        assert!(Journal::open(dir.resume(), id).is_err());
    }

    #[tokio::test]
    async fn retired_decoder_history_replays_without_admitting_new_operations() {
        use plait_core::OpId;
        use plait_core::group::{GroupAction, GroupOperation, HostMemory};
        use plait_core::serving::{ServingAction, ServingLookup, ServingOperation, ServingView};

        use crate::{Client, NodeConfig, RunningNode};

        let id = NodeId::derive("durable-legacy-decoder");
        let action = ServingAction::PublishDecoderRoutes {
            model: "legacy".into(),
            expected_revision: 0,
            paths: Vec::new(),
            activate: None,
        };
        let previous = ServingOperation {
            id: OpId::new(id, 1),
            action: action.clone(),
        };
        let mut replica = Replica::new(
            id,
            ClusterConfig::bootstrap([id]),
            Runtime::new(),
            Options::default(),
        );
        let dir = TestDirectory::new();
        let mut journal = dir.initialize(&replica);
        replica.track_deltas();
        replica
            .propose(Command::Serving(Box::new(previous.clone())), 1)
            .unwrap();
        let receipt = replica.machine().serving_receipt(previous.id).unwrap();
        let expected = receipt.result.clone();
        assert!(expected.is_ok());
        journal.record(replica.take_delta().unwrap()).unwrap();
        drop(journal);

        let mut config = NodeConfig::bootstrap(id, "127.0.0.1:0", BTreeMap::new());
        config.advertise_devices = false;
        let node = RunningNode::start_durable(config, dir.resume())
            .await
            .unwrap();
        let mut client = Client::new().await.unwrap();
        client.add_peer(id, node.addr());
        assert_eq!(client.apply_serving(id, &previous).await.unwrap(), expected);

        let new = ServingOperation {
            id: OpId::new(id, 2),
            action,
        };
        let rejected = client.apply_serving(id, &new).await.unwrap_err();
        assert!(rejected.to_string().contains("native decoder is retired"));
        assert_eq!(
            client
                .query_serving(id, ServingLookup::Operation(new.id))
                .await
                .unwrap(),
            ServingView::Operation(None)
        );
        let register = GroupOperation {
            id: OpId::new(id, 3),
            action: GroupAction::SetHostMemory {
                host: id,
                capacity: HostMemory {
                    bytes: 64,
                    pinned_bytes: 8,
                },
            },
        };
        assert!(client.apply_group(id, &register).await.unwrap().is_ok());
    }

    #[tokio::test]
    async fn observation_reply_survives_a_replica_restart() {
        use plait_core::OpId;
        use plait_core::serving::{
            GroupId, ServingAction, ServingError, ServingLookup, ServingOperation, ServingReceipt,
            ServingView,
        };

        use crate::{Client, NodeConfig, RunningNode};

        let id = NodeId::derive("durable-observer");
        let group = GroupId(NodeId::derive("durable-observed-group"));
        let mut config = NodeConfig::bootstrap(id, "127.0.0.1:0", BTreeMap::new());
        config.advertise_devices = false;
        let dir = TestDirectory::new();
        let mut node =
            RunningNode::start_durable(config.clone(), StorageMode::Initialize(dir.0.clone()))
                .await
                .unwrap();
        let mut client = Client::new().await.unwrap();
        client.add_peer(id, node.addr());
        let operation = OpId::new(NodeId::derive("durable-observation"), 1);
        assert_eq!(
            client
                .observe_serving_group(id, operation, group, None)
                .await
                .unwrap(),
            Err(ServingError::NoActiveGroup(group))
        );
        node.stop.notify_one();
        node.wait().await.unwrap();

        let node = RunningNode::start_durable(config, dir.resume())
            .await
            .unwrap();
        let mut client = Client::new().await.unwrap();
        client.add_peer(id, node.addr());
        assert_eq!(
            client
                .query_serving(id, ServingLookup::Operation(operation))
                .await
                .unwrap(),
            ServingView::Operation(Some(ServingReceipt {
                operation: ServingOperation {
                    id: operation,
                    action: ServingAction::ObserveGroup { group, epoch: None },
                },
                result: Err(ServingError::NoActiveGroup(group)),
            }))
        );
        assert_eq!(
            client
                .observe_serving_group(id, operation, group, Some(1))
                .await
                .unwrap(),
            Err(ServingError::OperationIdReused)
        );
    }
}
