use super::*;
use crate::backend::BackendError;
use crate::{
    Comm, DeviceRuntime, NCCL_SUCCESS, ncclAllGather, ncclAllReduce, ncclBroadcast, ncclRecv,
    ncclReduce, ncclReduceScatter, ncclSend,
};
use std::ffi::c_void;

fn listeners(world: usize) -> (Vec<Listener>, Vec<Endpoint>) {
    let listeners: Vec<_> = (0..world)
        .map(|index| Listener::bind("127.0.0.1:0".parse().unwrap(), [index as u8; 16]).unwrap())
        .collect();
    let members = listeners.iter().map(Listener::endpoint).collect();
    (listeners, members)
}

fn groups(world: usize) -> Vec<Arc<PeerGroup>> {
    groups_with_timeout(world, Duration::from_secs(2))
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
    let mut stalled = groups_with_timeout(2, Duration::from_millis(350));
    let sleeping = stalled.pop().unwrap();
    let waiting = stalled.pop().unwrap();
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
