use super::*;

const REPLICATED_BYTES_PER_RANK: usize = 64 * 1024;
const TWO_RANK_REPLICATED_LIMIT: usize = 1024 * 1024;
use crate::backend::BackendError;
use crate::native::{
    PLAIT_PEER_BUSY, PlaitPeerCollective, PlaitPeerRequest, plaitPeerCollectiveSubmit,
    plaitPeerRequestDestroy, plaitPeerRequestTest, plaitPeerRequestWait,
};
use crate::{
    Comm, DeviceRuntime, NCCL_INVALID_ARGUMENT, NCCL_SUCCESS, RanklessComm, Transport,
    ncclAllGather, ncclAllReduce, ncclBroadcast, ncclRecv, ncclReduce, ncclReduceScatter, ncclSend,
};
use std::ffi::c_void;

use plait_core::{NodeId, group::GroupId};

fn listeners(world: usize) -> (Vec<Listener>, Vec<Endpoint>) {
    let listeners: Vec<_> = (0..world)
        .map(|index| Listener::bind("127.0.0.1:0".parse().unwrap(), [index as u8; 16]).unwrap())
        .collect();
    let members = listeners.iter().map(Listener::endpoint).collect();
    (listeners, members)
}

fn groups(world: usize) -> Vec<Arc<PeerGroup>> {
    groups_with_timeout(world, Duration::from_secs(8))
}

#[test]
fn mismatched_group_epochs_cannot_connect_with_the_same_endpoint_manifest() {
    let group = GroupId(NodeId::derive("peer epoch"));
    let old = GroupVersion { group, epoch: 1 };
    let new = GroupVersion { group, epoch: 2 };
    let address = "127.0.0.1:0".parse().unwrap();
    let left = Listener::bind_in_epoch(address, [9; 16], old).unwrap();
    let right = Listener::bind_in_epoch(address, [10; 16], new).unwrap();
    let members = vec![left.endpoint(), right.endpoint()];
    let other_members = members.clone();
    let timeout = Duration::from_secs(2);
    thread::scope(|scope| {
        let peer = scope.spawn(move || left.connect(0, members, timeout).is_err());
        assert!(right.connect(1, other_members, timeout).is_err());
        assert!(peer.join().unwrap());
    });
}

#[test]
fn allreduce_bounds_the_replicated_exchange_and_keeps_the_same_fold() {
    let op = Collective::AllReduce(DataType::Float32, ReduceOp::Sum);
    assert!(use_replicated_exchange(
        op,
        TWO_RANK_REPLICATED_LIMIT,
        2,
        None
    ));
    assert!(!use_replicated_exchange(
        op,
        TWO_RANK_REPLICATED_LIMIT + 4,
        2,
        None
    ));
    assert!(use_replicated_exchange(
        op,
        REPLICATED_BYTES_PER_RANK / 4,
        5,
        None
    ));
    assert!(!use_replicated_exchange(
        op,
        REPLICATED_BYTES_PER_RANK / 4 + 4,
        5,
        None
    ));
    assert!(!use_replicated_exchange(
        Collective::ReduceScatter(DataType::Float32, ReduceOp::Sum),
        4,
        4,
        None
    ));

    thread::scope(|scope| {
        for (slot, group) in groups(4).into_iter().enumerate() {
            scope.spawn(move || {
                assert!(group.execute(op, &[]).unwrap().is_empty());
                // Sorting contributions by their arrival round would produce
                // different results on some slots with this input.
                let input = [1e20f32, -1e20, 1.0, 2.0][slot].to_ne_bytes();
                for count in [2048, 8192] {
                    let result = group.execute(op, &input.repeat(count)).unwrap();
                    assert_eq!(result, 3.0f32.to_ne_bytes().repeat(count));
                }
            });
        }
    });
}

#[test]
fn replicated_exchange_drains_two_writers_with_small_send_buffers() {
    use std::os::fd::AsRawFd;

    let group = groups(2);
    for (slot, peer) in group.iter().enumerate() {
        let capacity: libc::c_int = 16 * 1024;
        let fd = peer.link(1 - slot).collective.as_raw_fd();
        // The fd is live and capacity is a valid integer socket option.
        let result = unsafe {
            libc::setsockopt(
                fd,
                libc::SOL_SOCKET,
                libc::SO_SNDBUF,
                (&capacity as *const libc::c_int).cast(),
                std::mem::size_of_val(&capacity) as libc::socklen_t,
            )
        };
        assert_eq!(result, 0);
    }
    thread::scope(|scope| {
        for (slot, peer) in group.into_iter().enumerate() {
            scope.spawn(move || {
                let value = slot as u8 + 1;
                let result = peer
                    .execute(
                        Collective::AllReduce(DataType::Uint8, ReduceOp::Sum),
                        &vec![value; TWO_RANK_REPLICATED_LIMIT],
                    )
                    .unwrap();
                assert_eq!(result, vec![3; TWO_RANK_REPLICATED_LIMIT]);
                if slot == 1 {
                    thread::sleep(Duration::from_millis(20));
                }
                // The next sharded operation reuses the same collective lane.
                assert_eq!(
                    peer.execute(
                        Collective::ReduceScatter(DataType::Uint8, ReduceOp::Sum),
                        &[value, value],
                    )
                    .unwrap(),
                    [3]
                );
            });
        }
    });
}

#[test]
fn paired_scatter_sends_a_reduced_prefix_before_the_tail_arrives() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let local = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    local.set_nonblocking(true).unwrap();
    let (mut remote, _) = listener.accept().unwrap();
    let timeout = Duration::from_secs(3);
    configure(&remote, timeout).unwrap();
    let block = frame::PIPELINE_BLOCK_BYTES;
    let own = 2 * block + 2;
    let other = 2 * block;
    let mut output = vec![0; own + other];
    let (reduced, peer_reduced) = output.split_at_mut(own);
    let op = Collective::AllReduce(DataType::Float16, ReduceOp::Sum);
    let scatter = op.header(0, own + other, 0);
    let gather = op.header(0, own + other, 1);

    thread::scope(|scope| {
        let pipeline = scope.spawn(|| {
            frame::ScatterGather {
                peer: 1,
                stream: &local,
                scatter_header: scatter,
                gather_header: gather,
                scatter_send: &vec![7; other],
                scatter_recv: &mut vec![0; own],
                gather_send: reduced,
                gather_recv: peer_reduced,
            }
            .run(
                timeout,
                &mut Timer::new(false),
                |_, contribution, result| {
                    for (value, target) in contribution.iter().zip(result) {
                        *target = *value + 1;
                    }
                },
            )
            .unwrap();
        });

        frame::write(&remote, &scatter, &vec![2; block]).unwrap();
        let mut contribution = vec![0; other];
        frame::read(&remote, 0, &scatter, &mut contribution).unwrap();
        assert_eq!(contribution, vec![7; other]);
        let mut prefix = vec![0; block];
        frame::read(&remote, 0, &gather, &mut prefix).unwrap();
        assert_eq!(prefix, vec![3; block]);
        remote.write_all(&vec![4; own - block]).unwrap();
        frame::write(&remote, &gather, &vec![9; other]).unwrap();
        let mut tail = vec![0; own - block];
        remote.read_exact(&mut tail).unwrap();
        assert_eq!(tail, vec![5; own - block]);
        pipeline.join().unwrap();
    });
    assert_eq!(&output[..block], vec![3; block]);
    assert_eq!(&output[block..own], vec![5; own - block]);
    assert_eq!(&output[own..], vec![9; other]);
}

#[test]
fn paired_half_pipeline_keeps_uneven_and_nonfinite_sums_in_slot_order() {
    let op = Collective::AllReduce(DataType::Float16, ReduceOp::Sum);
    let words = TWO_RANK_REPLICATED_LIMIT / 2 + 1;
    let mut inputs = [vec![0; words * 2], vec![0; words * 2]];
    let patterns = [[0x7e03u16, 0x8000, 0x3800], [0x7e27u16, 0x8000, 0x3c00]];
    for (input, pattern) in inputs.iter_mut().zip(patterns) {
        for (i, bytes) in input.chunks_exact_mut(2).enumerate() {
            bytes.copy_from_slice(&pattern[i % pattern.len()].to_ne_bytes());
        }
    }
    let expected = op.reduce(&inputs);
    thread::scope(|scope| {
        for (peer, input) in groups(2).into_iter().zip(&inputs) {
            let expected = &expected;
            scope.spawn(move || assert_eq!(peer.execute(op, input).unwrap(), *expected));
        }
    });
}

#[test]
fn sharded_exchange_progresses_on_other_links_while_one_peer_waits() {
    fn sockets() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let local = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        local.set_nonblocking(true).unwrap();
        let remote = listener.accept().unwrap().0;
        remote
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        (local, remote)
    }

    let (slow, slow_remote) = sockets();
    let (fast, fast_remote) = sockets();
    let (release, wait) = std::sync::mpsc::channel();
    let op = Collective::ReduceScatter(DataType::Uint8, ReduceOp::Sum);
    let header = op.header(0, 4, 0);

    thread::scope(|scope| {
        scope.spawn(move || {
            let mut input = [0; 2];
            frame::read(&slow_remote, 0, &header, &mut input).unwrap();
            assert_eq!(input, [1, 2]);
            wait.recv_timeout(Duration::from_secs(3)).unwrap();
            frame::write(&slow_remote, &header, &[7]).unwrap();
        });
        scope.spawn(move || {
            let mut input = [0];
            frame::read(&fast_remote, 0, &header, &mut input).unwrap();
            assert_eq!(input, [3]);
            frame::write(&fast_remote, &header, &[8]).unwrap();
            release.send(()).unwrap();
        });
        let mut first = [0];
        let mut second = [0];
        frame::exchange_all(
            [
                (1, &slow, &[1, 2][..], first.as_mut_slice()),
                (2, &fast, &[3][..], second.as_mut_slice()),
            ]
            .into_iter(),
            &header,
            Duration::from_secs(3),
        )
        .unwrap();
        assert_eq!((first, second), ([7], [8]));
    });
}

#[test]
fn sharded_fp8_allreduce_preserves_uneven_output_ranges() {
    let op = Collective::AllReduceFp8ToF32;
    let count = REPLICATED_BYTES_PER_RANK / 2 + 1;
    let input: Vec<_> = (0..count)
        .map(|index| if index % 2 == 0 { 0x38 } else { 0x40 })
        .collect();
    assert!(!use_replicated_exchange(op, input.len(), 3, None));
    let expected: Vec<_> = (0..count)
        .flat_map(|index| (if index % 2 == 0 { 3f32 } else { 6f32 }).to_ne_bytes())
        .collect();
    thread::scope(|scope| {
        for group in groups(3) {
            let input = &input;
            let expected = &expected;
            scope.spawn(move || assert_eq!(&group.execute(op, input).unwrap(), expected));
        }
    });
}

fn groups_with_timeout(world: usize, timeout: Duration) -> Vec<Arc<PeerGroup>> {
    let (listeners, members) = listeners(world);
    thread::scope(|scope| {
        let jobs: Vec<_> = listeners
            .into_iter()
            .enumerate()
            .map(|(slot, listener)| {
                let members = members.clone();
                scope.spawn(move || listener.connect(slot, members, timeout).unwrap())
            })
            .collect();
        jobs.into_iter().map(|job| job.join().unwrap()).collect()
    })
}

#[test]
fn a_stalled_peer_times_out_while_another_group_runs() {
    let mut stalled = groups(2);
    let sleeping = stalled.pop().unwrap();
    let mut waiting = stalled.pop().unwrap();
    Arc::get_mut(&mut waiting).unwrap().timeout = Duration::from_millis(350);
    let healthy = groups(2);
    let op = Collective::AllReduce(DataType::Uint8, ReduceOp::Sum);
    thread::scope(|scope| {
        let stalled = scope.spawn(|| {
            let error = waiting.execute(op, &[1]).unwrap_err();
            assert!(matches!(
                error.kind(),
                io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
            ));
            assert!(waiting.execute(op, &[1]).is_err());
        });
        for group in healthy {
            scope.spawn(move || assert_eq!(group.execute(op, &[3]).unwrap(), [6]));
        }
        stalled.join().unwrap();
    });
    drop(sleeping);
}

struct HostDevice;

impl DeviceRuntime for HostDevice {
    fn copy_host_to_device(&self, target: u64, source: &[u8]) -> Result<(), BackendError> {
        unsafe {
            source
                .as_ptr()
                .copy_to_nonoverlapping(target as *mut u8, source.len())
        };
        Ok(())
    }

    fn copy_device_to_host(&self, source: u64, target: &mut [u8]) -> Result<(), BackendError> {
        unsafe { (source as *const u8).copy_to_nonoverlapping(target.as_mut_ptr(), target.len()) };
        Ok(())
    }

    fn synchronize_stream(&self, stream: u64) -> Result<(), BackendError> {
        assert_eq!(stream, 7);
        Ok(())
    }
}

#[test]
fn collectives_use_public_entry_points_with_odd_and_uneven_groups() {
    for world in [1, 2, 3, 4, 8, 16] {
        thread::scope(|scope| {
            for (slot, group) in groups(world).into_iter().enumerate() {
                scope.spawn(move || {
                    let mut comm = Comm::with_peers(group, slot, world, slot, Box::new(HostDevice));
                    let stream = 7 as *mut c_void;
                    let dtype = DataType::Int32 as i32;
                    let sum = ReduceOp::Sum as i32;
                    let slot = slot as i32;
                    let input = [slot, slot + 1, slot + 2, slot + 3, slot + 4];
                    let send = input.as_ptr().cast();
                    let mut reduced = [0; 5];
                    let result = ncclAllReduce(
                        send,
                        reduced.as_mut_ptr().cast(),
                        5,
                        dtype,
                        sum,
                        &mut comm,
                        stream,
                    );
                    assert_eq!(result, NCCL_SUCCESS);
                    let offset: i32 = (0..world as i32).sum();
                    assert_eq!(
                        reduced,
                        std::array::from_fn(|i| offset + (i * world) as i32)
                    );

                    let mut gathered = vec![0i32; world * 5];
                    let result = ncclAllGather(
                        send,
                        gathered.as_mut_ptr().cast(),
                        5,
                        dtype,
                        &mut comm,
                        stream,
                    );
                    assert_eq!(result, NCCL_SUCCESS);
                    for (sender, payload) in gathered.chunks_exact(5).enumerate() {
                        assert_eq!(
                            payload,
                            std::array::from_fn::<_, 5, _>(|i| (sender + i) as i32)
                        );
                    }

                    let input = (0..world as i32).map(|i| i + slot).collect::<Vec<_>>();
                    let mut shard = 0i32;
                    let result = ncclReduceScatter(
                        input.as_ptr().cast(),
                        (&mut shard as *mut i32).cast(),
                        1,
                        dtype,
                        sum,
                        &mut comm,
                        stream,
                    );
                    assert_eq!(result, NCCL_SUCCESS);
                    assert_eq!(shard, offset + slot * world as i32);

                    let root = world as i32 - 1;
                    let mut broadcast = [0; 5];
                    let source = if slot == root { send } else { std::ptr::null() };
                    let result = ncclBroadcast(
                        source,
                        broadcast.as_mut_ptr().cast(),
                        5,
                        dtype,
                        root,
                        &mut comm,
                        stream,
                    );
                    assert_eq!(result, NCCL_SUCCESS);
                    assert_eq!(broadcast, std::array::from_fn(|i| root + i as i32));

                    let mut result_at_root = [0; 5];
                    let dest = if slot == root {
                        result_at_root.as_mut_ptr().cast()
                    } else {
                        std::ptr::null_mut()
                    };
                    let result = ncclReduce(send, dest, 5, dtype, sum, root, &mut comm, stream);
                    assert_eq!(result, NCCL_SUCCESS);
                    if slot == root {
                        assert_eq!(result_at_root, reduced);
                    }
                });
            }
        });
    }
}

#[test]
fn simultaneous_sends_bypass_the_idle_participant() {
    let mut group = groups(3);
    let idle = group.remove(0);
    let completed = std::sync::Barrier::new(2);
    thread::scope(|scope| {
        for (index, group) in group.into_iter().enumerate() {
            let completed = &completed;
            scope.spawn(move || {
                let local = index + 1;
                let peer = 3 - local;
                let mut comm = Comm::with_peers(group, local, 3, local, Box::new(HostDevice));
                let outgoing = vec![local as u8; 4 * 1024 * 1024];
                let dtype = DataType::Uint8 as i32;
                let stream = 7 as *mut c_void;
                let sent = ncclSend(
                    outgoing.as_ptr().cast(),
                    outgoing.len(),
                    dtype,
                    peer as i32,
                    &mut comm,
                    stream,
                );
                assert_eq!(sent, NCCL_SUCCESS);
                let mut incoming = vec![0u8; outgoing.len()];
                let received = ncclRecv(
                    incoming.as_mut_ptr().cast(),
                    incoming.len(),
                    dtype,
                    peer as i32,
                    &mut comm,
                    stream,
                );
                assert_eq!(received, NCCL_SUCCESS);
                assert!(incoming.iter().all(|item| *item == peer as u8));
                completed.wait();
            });
        }
    });
    drop(idle);
}

#[test]
fn rank_zero_loss_isolated_to_its_communicator() {
    let mut failed = groups(3);
    drop(failed.remove(0));
    // A distinct communicator reuses the slots but has independent sockets.
    let healthy = groups(3);
    let op = Collective::AllReduce(DataType::Int8, ReduceOp::Sum);
    thread::scope(|scope| {
        for group in failed {
            scope.spawn(move || {
                assert!(group.execute(op, &[3]).is_err());
                assert_eq!(
                    group.execute(op, &[3]).unwrap_err().to_string(),
                    "peer communicator has failed"
                );
            });
        }
        for group in healthy {
            scope.spawn(move || {
                assert_eq!(group.execute(op, &[2]).unwrap(), [6]);
                assert_eq!(group.execute(op, &[1, 3]).unwrap(), [3, 9]);
            });
        }
    });
}

#[test]
fn mismatched_collectives_fail_before_any_payload_is_delivered() {
    let group = groups(2);
    thread::scope(|scope| {
        let jobs: Vec<_> = group
            .into_iter()
            .enumerate()
            .map(|(slot, group)| {
                scope.spawn(move || {
                    let op = if slot == 0 {
                        Collective::AllGather(DataType::Uint8)
                    } else {
                        Collective::AllReduce(DataType::Uint8, ReduceOp::Sum)
                    };
                    let error = group.execute(op, &[1]).unwrap_err();
                    assert!(group.execute(op, &[1]).is_err());
                    error.to_string()
                })
            })
            .collect();
        let errors: Vec<_> = jobs.into_iter().map(|job| job.join().unwrap()).collect();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("different collective descriptor"))
        );
    });
}

#[test]
fn a_stale_endpoint_ticket_cannot_join() {
    let (mut listeners, members) = listeners(2);
    let right = listeners.pop().unwrap();
    let left = listeners.pop().unwrap();
    let mut stale = members.clone();
    stale[1].ticket = [7; 16];
    thread::scope(|scope| {
        let timeout = Duration::from_secs(2);
        let left = scope.spawn(move || left.connect(0, stale, timeout).is_err());
        assert!(right.connect(1, members, timeout).is_err());
        assert!(left.join().unwrap());
    });
}

#[test]
fn participants_reject_partial_membership_views() {
    let (mut listeners, mut members) = listeners(3);
    // The two live participants have valid endpoint tickets but disagree
    // about whether another participant belongs to this group.
    let _unused = listeners.pop().unwrap();
    let right = listeners.pop().unwrap();
    let left = listeners.pop().unwrap();
    let full = members.clone();
    members.pop();
    thread::scope(|scope| {
        let timeout = Duration::from_secs(2);
        let short = scope.spawn(move || left.connect(0, members, timeout).is_err());
        assert!(right.connect(1, full, timeout).is_err());
        assert!(short.join().unwrap());
    });
}

fn rankless_listeners(
    version: GroupVersion,
    devices: &[DeviceId],
) -> (Vec<RanklessListener>, Vec<RanklessMember>) {
    let listeners: Vec<_> = devices
        .iter()
        .enumerate()
        .map(|(index, &device)| {
            RanklessListener::bind(
                "127.0.0.1:0".parse().unwrap(),
                [index as u8; 16],
                plait_core::group::WorkerGrant {
                    version,
                    device,
                    incarnation: 1,
                    fence: version.epoch,
                },
            )
            .unwrap()
        })
        .collect();
    let members = listeners.iter().map(RanklessListener::member).collect();
    (listeners, members)
}

fn rankless_comms(
    version: GroupVersion,
    devices: &[DeviceId],
    order: Option<&[DeviceId]>,
) -> Vec<RanklessComm> {
    let (listeners, members) = rankless_listeners(version, devices);
    thread::scope(|scope| {
        let jobs: Vec<_> = listeners
            .into_iter()
            .enumerate()
            .map(|(index, listener)| {
                let mut members = members.clone();
                members.rotate_left(index);
                scope.spawn(move || {
                    Comm::connect_rankless_with_runtime(
                        listener,
                        members,
                        order,
                        devices[index].ordinal as usize,
                        Box::new(HostDevice),
                        Duration::from_secs(2),
                        Duration::from_secs(2),
                    )
                    .unwrap()
                })
            })
            .collect();
        jobs.into_iter().map(|job| job.join().unwrap()).collect()
    })
}

#[test]
fn explicit_slot_layout_preserves_engine_allgather_order() {
    let [a, b, c] = [1, 2, 3].map(|id| DeviceId::new(NodeId::from_bytes([id; 16]), 0));
    let version = GroupVersion {
        group: GroupId(NodeId::derive("slot layout")),
        epoch: 1,
    };
    let peers = rankless_comms(version, &[a, b, c], Some(&[c, a, b]));
    thread::scope(|scope| {
        for (peer, value) in peers.iter().zip([10i32, 20, 30]) {
            scope.spawn(move || {
                assert_eq!(peer.peer_slot(c), Some(0));
                assert_eq!(peer.peer_slot(a), Some(1));
                assert_eq!(peer.peer_slot(b), Some(2));
                let mut output = [0i32; 3];
                assert_eq!(
                    ncclAllGather(
                        std::ptr::from_ref(&value).cast(),
                        output.as_mut_ptr().cast(),
                        1,
                        DataType::Int32 as i32,
                        peer.as_ptr(),
                        7 as *mut c_void,
                    ),
                    NCCL_SUCCESS
                );
                assert_eq!(output, [30, 10, 20]);
            });
        }
    });
}

#[test]
fn different_slot_orders_cannot_connect_with_the_same_grants() {
    let devices = [1, 2].map(|id| DeviceId::new(NodeId::from_bytes([id; 16]), 0));
    let version = GroupVersion {
        group: GroupId(NodeId::derive("slot mismatch")),
        epoch: 1,
    };
    let (mut listeners, members) = rankless_listeners(version, &devices);
    let right = listeners.pop().unwrap();
    let left = listeners.pop().unwrap();
    thread::scope(|scope| {
        let timeout = Duration::from_millis(500);
        let left_members = members.clone();
        let dialer =
            scope.spawn(move || left.connect(left_members, None, timeout, timeout).is_err());
        let reversed = [devices[1], devices[0]];
        assert!(
            right
                .connect(members, Some(&reversed), timeout, timeout)
                .is_err()
        );
        assert!(dialer.join().unwrap());
    });
}

#[test]
fn explicit_replicated_egress_limit_must_match_on_both_peers() {
    let devices = [1, 2].map(|id| DeviceId::new(NodeId::from_bytes([id; 16]), 0));
    let version = GroupVersion {
        group: GroupId(NodeId::derive("egress limit agreement")),
        epoch: 1,
    };
    for other in [None, Some(0), Some(384 * 1024)] {
        let (mut listeners, members) = rankless_listeners(version, &devices);
        let mut right = listeners.pop().unwrap();
        let mut left = listeners.pop().unwrap();
        let limit = 384 * 1024;
        left.set_replicated_egress_bytes(limit);
        if let Some(limit) = other {
            right.set_replicated_egress_bytes(limit);
        }
        thread::scope(|scope| {
            let timeout = Duration::from_secs(2);
            let left_members = members.clone();
            let dialer = scope.spawn(move || left.connect(left_members, None, timeout, timeout));
            let right = right.connect(members, None, timeout, timeout);
            for group in [dialer.join().unwrap(), right] {
                assert_eq!(group.is_ok(), other == Some(limit));
                if let Ok(group) = group {
                    assert_eq!(group.replicated_egress_bytes, Some(limit));
                }
            }
        });
    }
}

fn all_reduce_i32(comm: &Comm, value: i32, output: &mut i32) -> i32 {
    ncclAllReduce(
        (&value as *const i32).cast(),
        (output as *mut i32).cast(),
        1,
        DataType::Int32 as i32,
        ReduceOp::Sum as i32,
        std::ptr::from_ref(comm).cast_mut(),
        7 as *mut c_void,
    )
}

#[test]
fn rankless_replacement_preserves_device_identity_and_isolates_the_old_failure() {
    let [a, b, c, replacement] = [
        DeviceId::new(NodeId::from_bytes([1; 16]), 0),
        DeviceId::new(NodeId::from_bytes([2; 16]), 0),
        DeviceId::new(NodeId::from_bytes([2; 16]), 1),
        DeviceId::new(NodeId::from_bytes([3; 16]), 0),
    ];
    let group = GroupId(NodeId::derive("native group"));
    let old = GroupVersion { group, epoch: 1 };
    let new = GroupVersion { group, epoch: 2 };
    let independent = GroupVersion {
        group: GroupId(NodeId::derive("native independent group")),
        epoch: 1,
    };
    let mut original = rankless_comms(old, &[c, a, b], None);
    let running = rankless_comms(independent, &[c, b], None);
    assert_eq!(original[0].peer_slot(a), Some(0));
    assert_eq!(original[0].peer_slot(b), Some(1));
    assert_eq!(original[0].peer_slot(c), Some(2));
    drop(original.remove(1));

    thread::scope(|scope| {
        for comm in &original {
            scope.spawn(move || {
                let mut output = -1;
                assert_ne!(all_reduce_i32(comm, 1, &mut output), NCCL_SUCCESS);
                assert_eq!(output, -1);
                assert_eq!(comm.peers_failed(), Some(true));
            });
        }
        for comm in &running {
            scope.spawn(move || {
                let mut output = -1;
                assert_eq!(all_reduce_i32(comm, 4, &mut output), NCCL_SUCCESS);
                assert_eq!(output, 8);
                assert_eq!(comm.peers_failed(), Some(false));
            });
        }
    });

    let next = rankless_comms(new, &[replacement, c, b], None);
    thread::scope(|scope| {
        for comm in &next {
            scope.spawn(move || {
                assert_eq!(comm.peer_slot(a), None);
                assert_eq!(comm.peer_slot(b), Some(0));
                let mut output = -1;
                assert_eq!(all_reduce_i32(comm, 3, &mut output), NCCL_SUCCESS);
                assert_eq!(output, 9);
            });
        }
    });
    for comm in original {
        let mut output = -1;
        assert_ne!(all_reduce_i32(&comm, 1, &mut output), NCCL_SUCCESS);
        assert_eq!(output, -1);
    }
}

#[test]
fn peer_handshake_rejects_different_grants_with_identical_endpoints() {
    let version = GroupVersion {
        group: GroupId(NodeId::derive("native mismatched grants")),
        epoch: 1,
    };
    let devices = [1, 2].map(|index| DeviceId::new(NodeId::from_bytes([index; 16]), 0));
    let (mut listeners, members) = rankless_listeners(version, &devices);
    let right = listeners.pop().unwrap();
    let left = listeners.pop().unwrap();
    let mut stale = members.clone();
    stale[1].grant.fence += 1;
    thread::scope(|scope| {
        let timeout = Duration::from_secs(2);
        let left = scope.spawn(move || left.connect(stale, None, timeout, timeout).is_err());
        assert!(right.connect(members, None, timeout, timeout).is_err());
        assert!(left.join().unwrap());
    });
}

#[test]
fn cancel_wakes_an_active_peer_call_and_rejects_reuse() {
    let version = GroupVersion {
        group: GroupId(NodeId::derive("native cancellation")),
        epoch: 1,
    };
    let devices = [1, 2].map(|index| DeviceId::new(NodeId::from_bytes([index; 16]), 0));
    let mut peers = rankless_comms(version, &devices, None);
    let idle = peers.pop().unwrap();
    let waiting = peers.pop().unwrap();
    let Transport::Peers(remote) = &idle.transport else {
        unreachable!()
    };
    thread::scope(|scope| {
        let call = scope.spawn(|| {
            let mut output = -1;
            let result = all_reduce_i32(&waiting, 1, &mut output);
            assert_ne!(result, NCCL_SUCCESS);
            assert_eq!(output, -1);
        });
        // Read the operation header before cancelling so the call is in flight.
        let mut stream = &remote.link(0).collective;
        stream.set_nonblocking(false).unwrap();
        let mut header = [0; 24];
        stream.read_exact(&mut header).unwrap();
        assert!(waiting.cancel_peers());
        assert_eq!(waiting.peers_failed(), Some(true));
        call.join().unwrap();
    });
    let mut output = -1;
    assert_ne!(all_reduce_i32(&waiting, 1, &mut output), NCCL_SUCCESS);
    assert_eq!(output, -1);
}

fn submit_i32(kind: i32, count: usize, send: &[i32], comm: &RanklessComm) -> *mut PlaitPeerRequest {
    try_submit_i32(kind, count, send, comm).unwrap()
}

fn try_submit_i32(
    kind: i32,
    count: usize,
    send: &[i32],
    comm: &RanklessComm,
) -> Result<*mut PlaitPeerRequest, i32> {
    let desc = PlaitPeerCollective {
        kind,
        datatype: DataType::Int32 as i32,
        reduction: ReduceOp::Sum as i32,
        root: 1,
        count,
    };
    let send = if send.is_empty() {
        std::ptr::null()
    } else {
        send.as_ptr().cast()
    };
    let mut request = std::ptr::null_mut();
    let status = unsafe {
        plaitPeerCollectiveSubmit(comm.as_ptr(), &desc, send, 7 as *mut c_void, &mut request)
    };
    if status != NCCL_SUCCESS {
        assert!(request.is_null());
        return Err(status);
    }
    Ok(request)
}

fn wait_i32(comm: &RanklessComm, request: &mut *mut PlaitPeerRequest, output: &mut [i32]) -> i32 {
    unsafe {
        plaitPeerRequestWait(
            comm.as_ptr(),
            request,
            output.as_mut_ptr().cast(),
            std::mem::size_of_val(output),
            7 as *mut c_void,
        )
    }
}

#[test]
fn asynchronous_staging_budget_covers_buffered_and_abandoned_results() {
    let version = GroupVersion {
        group: GroupId(NodeId::derive("asynchronous peer staging budget")),
        epoch: 1,
    };
    let devices = [1, 2].map(|index| DeviceId::new(NodeId::from_bytes([index; 16]), 0));
    let mut peers = rankless_comms(version, &devices, None);
    let [left, right] = peers.as_mut_slice() else {
        unreachable!()
    };
    let Transport::Peers(group) = &mut left.0.transport else {
        unreachable!()
    };
    // One i32 reduction requires four bytes of input and four of output.
    Arc::get_mut(group).unwrap().requests = requests::Dispatcher::new(8);
    let Transport::Peers(group) = &left.transport else {
        unreachable!()
    };
    assert_eq!(try_submit_i32(2, 1, &[1], left), Err(PLAIT_PEER_BUSY));
    assert_eq!(group.order.lock().unwrap().submitted, 0);

    let mut buffered = submit_i32(1, 1, &[1], left);
    let mut output = [-1];
    assert_eq!(all_reduce_i32(right, 4, &mut output[0]), NCCL_SUCCESS);
    assert_eq!(output, [5]);
    let mut ready = 0;
    let deadline = Instant::now() + Duration::from_secs(2);
    while ready == 0 {
        assert_eq!(
            unsafe { plaitPeerRequestTest(buffered, &mut ready) },
            NCCL_SUCCESS
        );
        assert!(Instant::now() < deadline);
        thread::yield_now();
    }
    assert_eq!(try_submit_i32(1, 1, &[2], left), Err(PLAIT_PEER_BUSY));
    assert_eq!(group.order.lock().unwrap().submitted, 1);

    // A worker under local pressure may run the very same operation through
    // the blocking path while its peer accepts an asynchronous request.
    let mut remote = submit_i32(1, 1, &[10], right);
    assert_eq!(all_reduce_i32(left, 2, &mut output[0]), NCCL_SUCCESS);
    assert_eq!(output, [12]);
    assert_eq!(wait_i32(right, &mut remote, &mut output), NCCL_SUCCESS);
    assert_eq!(output, [12]);
    assert_eq!(wait_i32(left, &mut buffered, &mut output), NCCL_SUCCESS);
    assert_eq!(output, [5]);

    let abandoned = submit_i32(1, 1, &[7], left);
    assert_eq!(unsafe { plaitPeerRequestDestroy(abandoned) }, NCCL_SUCCESS);
    assert_eq!(try_submit_i32(1, 1, &[6], left), Err(PLAIT_PEER_BUSY));
    assert_eq!(all_reduce_i32(right, 1, &mut output[0]), NCCL_SUCCESS);
    assert_eq!(output, [8]);
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        match try_submit_i32(1, 1, &[6], left) {
            Ok(mut reused) => {
                assert_eq!(all_reduce_i32(right, 3, &mut output[0]), NCCL_SUCCESS);
                assert_eq!(wait_i32(left, &mut reused, &mut output), NCCL_SUCCESS);
                assert_eq!(output, [9]);
                break;
            }
            Err(PLAIT_PEER_BUSY) => {
                assert!(Instant::now() < deadline);
                thread::yield_now();
            }
            Err(error) => panic!("staging retry failed: {error}"),
        }
    }
}

#[test]
fn asynchronous_collectives_stage_inputs_and_share_blocking_order() {
    let version = GroupVersion {
        group: GroupId(NodeId::derive("ordered asynchronous peer requests")),
        epoch: 1,
    };
    let devices = [1, 2].map(|index| DeviceId::new(NodeId::from_bytes([index; 16]), 0));
    let peers = rankless_comms(version, &devices, None);
    let [left, right] = peers.as_slice() else {
        unreachable!()
    };
    let mut input = vec![7];
    let mut first = submit_i32(1, 1, &input, left);
    input[0] = 89;
    drop(input);
    let mut ready = 1;
    assert_eq!(
        unsafe { plaitPeerRequestTest(first, &mut ready) },
        NCCL_SUCCESS
    );
    assert_eq!(ready, 0);
    let mut output = [-1];
    assert_eq!(
        wait_i32(right, &mut first, &mut output),
        NCCL_INVALID_ARGUMENT
    );
    assert!(!first.is_null());

    thread::scope(|scope| {
        let blocking = scope.spawn(|| {
            let mut received = -1;
            assert_eq!(all_reduce_i32(left, 11, &mut received), NCCL_SUCCESS);
            assert_eq!(received, 24);
        });
        let mut received = -1;
        assert_eq!(all_reduce_i32(right, 5, &mut received), NCCL_SUCCESS);
        assert_eq!(received, 12);
        assert_eq!(all_reduce_i32(right, 13, &mut received), NCCL_SUCCESS);
        assert_eq!(received, 24);
        blocking.join().unwrap();
    });
    assert_eq!(wait_i32(left, &mut first, &mut output), NCCL_SUCCESS);
    assert_eq!(output, [12]);
    assert!(first.is_null());

    let abandoned = submit_i32(4, 1, &[], left);
    unsafe { plaitPeerRequestDestroy(abandoned) };
    let mut root = submit_i32(4, 1, &[3], right);
    assert_eq!(wait_i32(right, &mut root, &mut output), NCCL_SUCCESS);
    assert_eq!(output, [3]);

    let mut first = submit_i32(3, 1, &[2, 3], left);
    let mut second = submit_i32(3, 1, &[5, 7], right);
    assert_eq!(wait_i32(left, &mut first, &mut output), NCCL_SUCCESS);
    assert_eq!(output, [7]);
    assert_eq!(wait_i32(right, &mut second, &mut output), NCCL_SUCCESS);
    assert_eq!(output, [10]);

    let mut first = submit_i32(2, 1, &[1], left);
    let mut second = submit_i32(2, 1, &[2], right);
    let mut output = [-1, -1];
    for (comm, request) in [(left, &mut first), (right, &mut second)] {
        assert_eq!(wait_i32(comm, request, &mut output), NCCL_SUCCESS);
        assert_eq!(output, [1, 2]);
    }

    let mut first = submit_i32(5, 1, &[5], left);
    let mut second = submit_i32(5, 1, &[6], right);
    assert_eq!(wait_i32(left, &mut first, &mut []), NCCL_SUCCESS);
    assert_eq!(wait_i32(right, &mut second, &mut output[..1]), NCCL_SUCCESS);
    assert_eq!(output[0], 11);

    let mut pending = submit_i32(1, 1, &[1], left);
    let mut remote = submit_i32(1, 1, &[2], right);
    assert_eq!(wait_i32(right, &mut remote, &mut output[..1]), NCCL_SUCCESS);
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        assert_eq!(
            unsafe { plaitPeerRequestTest(pending, &mut ready) },
            NCCL_SUCCESS
        );
        if ready != 0 {
            break;
        }
        assert!(Instant::now() < deadline, "left result did not complete");
        thread::yield_now();
    }
    left.cancel_peers();
    output[0] = -1;
    assert_ne!(wait_i32(left, &mut pending, &mut output[..1]), NCCL_SUCCESS);
    assert_eq!(output[0], -1);
}

#[test]
fn cancellation_drains_pending_requests_without_writing_output() {
    let version = GroupVersion {
        group: GroupId(NodeId::derive("native asynchronous cancellation")),
        epoch: 1,
    };
    let devices = [1, 2].map(|index| DeviceId::new(NodeId::from_bytes([index; 16]), 0));
    let peers = rankless_comms(version, &devices, None);
    let [waiting, idle] = peers.as_slice() else {
        unreachable!()
    };
    let mut first = submit_i32(1, 1, &[1], waiting);
    let mut queued = submit_i32(1, 1, &[2], waiting);
    let Transport::Peers(remote) = &idle.transport else {
        unreachable!()
    };
    let mut header = [0; 24];
    let mut stream = &remote.link(0).collective;
    stream.set_nonblocking(false).unwrap();
    stream.read_exact(&mut header).unwrap();
    let Transport::Peers(group) = &waiting.transport else {
        unreachable!()
    };
    let op = Collective::AllReduce(DataType::Int32, ReduceOp::Sum);
    let submit = || {
        let staging = group.requests.reserve_staging(8).unwrap();
        requests::Dispatcher::submit(group, op, vec![0; 4], staging)
    };
    let remaining = (0..requests::QUEUE_DEPTH - 1)
        .map(|_| submit().unwrap())
        .collect::<Vec<_>>();
    let submitted = group.order.lock().unwrap().submitted;
    let Err(error) = submit() else {
        panic!("the pending collective queue exceeded capacity");
    };
    assert_eq!(error.kind(), io::ErrorKind::WouldBlock);
    assert_eq!(group.order.lock().unwrap().submitted, submitted);
    assert!(waiting.cancel_peers());
    let mut output = [-1];
    for request in [&mut first, &mut queued] {
        assert_ne!(wait_i32(waiting, request, &mut output), NCCL_SUCCESS);
        assert!(request.is_null());
        assert_eq!(output, [-1]);
    }
    for completion in remaining {
        assert!(completion.wait().is_err());
    }
}
