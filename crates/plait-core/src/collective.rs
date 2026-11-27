//! Exchange choices shared by the host-staged peer runtime and placement.

use sha2::{Digest, Sha256};

use crate::NodeId;

const REPLICATED_BYTES_PER_WORKER: u64 = 64 * 1024;
const TWO_WORKER_REPLICATED_LIMIT: u64 = 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeerAllReducePlan {
    Replicated,
    ScatterGather,
}

pub fn peer_all_reduce_plan(bytes: u64, workers: usize) -> PeerAllReducePlan {
    let egress_limit = if workers == 2 {
        // A full exchange moves the same payload as scatter plus gather with
        // two workers. Cap the amount the host reduction has to hold.
        TWO_WORKER_REPLICATED_LIMIT
    } else {
        REPLICATED_BYTES_PER_WORKER
    };
    peer_all_reduce_plan_with_egress(bytes, workers, egress_limit)
}

/// Limit the payload sent by each worker during one replicated exchange.
pub fn peer_all_reduce_plan_with_egress(
    bytes: u64,
    workers: usize,
    egress_limit: u64,
) -> PeerAllReducePlan {
    let replicated = bytes
        .checked_mul(workers.saturating_sub(1) as u64)
        .is_some_and(|egress| egress <= egress_limit);
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

/// Bind the caller's launch recipe to the TP width and replicated payload cap.
pub fn bind_peer_all_reduce_recipe(recipe: NodeId, width: u64, egress_bytes: u64) -> NodeId {
    let mut digest = Sha256::new();
    digest.update(b"plait-peer-all-reduce-recipe-v1");
    digest.update(recipe.as_bytes());
    digest.update(width.to_le_bytes());
    digest.update(egress_bytes.to_le_bytes());
    NodeId::from_bytes(digest.finalize()[..16].try_into().expect("16 digest bytes"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replicated_egress_limit_charges_each_other_slot_and_rejects_overflow() {
        use PeerAllReducePlan::{Replicated, ScatterGather};

        assert_eq!(peer_all_reduce_plan_with_egress(128, 4, 384), Replicated);
        assert_eq!(peer_all_reduce_plan_with_egress(129, 4, 384), ScatterGather);
        assert_eq!(peer_all_reduce_plan_with_egress(0, 4, 0), Replicated);
        assert_eq!(peer_all_reduce_plan_with_egress(1, 4, 0), ScatterGather);
        assert_eq!(
            peer_all_reduce_plan_with_egress(u64::MAX, 3, u64::MAX),
            ScatterGather
        );
    }
}
