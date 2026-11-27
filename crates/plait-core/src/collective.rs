//! Exchange choices shared by the host-staged peer runtime and placement.

const REPLICATED_BYTES_PER_WORKER: u64 = 64 * 1024;
const TWO_WORKER_REPLICATED_LIMIT: u64 = 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeerAllReducePlan {
    Replicated,
    ScatterGather,
}

pub fn peer_all_reduce_plan(bytes: u64, workers: usize) -> PeerAllReducePlan {
    let replicated = if workers == 2 {
        // A full exchange moves the same payload as scatter plus gather with
        // two workers. Cap the amount the host reduction has to hold.
        bytes <= TWO_WORKER_REPLICATED_LIMIT
    } else {
        bytes.saturating_mul(workers.saturating_sub(1) as u64) <= REPLICATED_BYTES_PER_WORKER
    };
    if replicated {
        PeerAllReducePlan::Replicated
    } else {
        PeerAllReducePlan::ScatterGather
    }
}

/// The first `elements % workers` slots own one additional element.
/// The communicator supplies a positive worker count and a slot in that range.
pub fn peer_shard_elements(elements: u64, workers: usize, slot: usize) -> u64 {
    let workers = workers as u64;
    elements / workers + u64::from((slot as u64) < elements % workers)
}
