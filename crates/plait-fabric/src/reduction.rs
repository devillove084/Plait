//! Versioned reductions over immutable contributions delivered by the data plane.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use plait_core::serving::{GroupVersion, GroupView};
use plait_core::tensor::KPartial;

#[derive(Debug, PartialEq, Eq)]
pub enum ReductionError {
    InvalidPlan,
    GrantUnavailable,
    InvalidContribution,
}

impl fmt::Display for ReductionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidPlan => {
                "K reduction requires a complete, aligned partition for one operation"
            }
            Self::GrantUnavailable => "a planned contribution has no live worker grant",
            Self::InvalidContribution => "missing, unexpected, or malformed K contribution",
        })
    }
}

impl std::error::Error for ReductionError {}

/// A row-major FP32 sum in logical K order, independent of arrival order or
/// physical placement. A K partition is tied to one group version and op.
pub struct KSum {
    partials: Vec<KPartial>,
    bytes: usize,
}

pub struct KReduction<'a, Payload> {
    plan: &'a KSum,
    next: usize,
    pending: BTreeMap<usize, Payload>,
    total: Vec<f32>,
}

impl KSum {
    pub fn new(
        k: u32,
        group_size: u32,
        mut partials: Vec<KPartial>,
    ) -> Result<Self, ReductionError> {
        let first = *partials.first().ok_or(ReductionError::InvalidPlan)?;
        let bytes = first.payload_bytes().ok_or(ReductionError::InvalidPlan)?;
        if group_size == 0 {
            return Err(ReductionError::InvalidPlan);
        }
        partials.sort_unstable_by_key(|partial| partial.k_start);
        let mut end = 0;
        let mut shards = BTreeSet::new();
        for partial in &partials {
            if partial.op != first.op
                || partial.grant.version != first.grant.version
                || partial.output_shape != first.output_shape
                || partial.k_start != end
                || partial.k_start % group_size != 0
                || partial.k_start >= partial.k_end
                || !shards.insert(partial.shard)
            {
                return Err(ReductionError::InvalidPlan);
            }
            end = partial.k_end;
        }
        if end != k {
            return Err(ReductionError::InvalidPlan);
        }
        Ok(Self { partials, bytes })
    }

    pub fn version(&self) -> GroupVersion {
        self.partials[0].grant.version
    }

    /// Check a current group view before dispatching and delivering a result.
    /// A draining group may finish work that was admitted while active.
    pub fn check_grants(&self, group: &GroupView) -> Result<(), ReductionError> {
        if self
            .partials
            .iter()
            .any(|partial| !group.allows_grant(partial.grant))
        {
            return Err(ReductionError::GrantUnavailable);
        }
        Ok(())
    }

    pub fn sum<'a>(
        &self,
        contributions: impl IntoIterator<Item = (KPartial, &'a [u8])>,
    ) -> Result<Vec<u8>, ReductionError> {
        let mut reduction = self.reduce();
        for (partial, payload) in contributions {
            reduction.push(partial, payload)?;
        }
        reduction.finish()
    }

    pub fn reduce<Payload: AsRef<[u8]>>(&self) -> KReduction<'_, Payload> {
        KReduction {
            plan: self,
            next: 0,
            pending: BTreeMap::new(),
            total: vec![0.0; self.bytes / 4],
        }
    }
}

impl<Payload: AsRef<[u8]>> KReduction<'_, Payload> {
    /// Accumulate a contiguous prefix and retain out-of-order contributions.
    pub fn push(&mut self, partial: KPartial, payload: Payload) -> Result<(), ReductionError> {
        let index = self
            .plan
            .partials
            .binary_search_by_key(&partial.k_start, |expected| expected.k_start)
            .map_err(|_| ReductionError::InvalidContribution)?;
        if index < self.next
            || self.plan.partials[index] != partial
            || payload.as_ref().len() != self.plan.bytes
            || self.pending.contains_key(&index)
        {
            return Err(ReductionError::InvalidContribution);
        }
        self.pending.insert(index, payload);
        while let Some(payload) = self.pending.remove(&self.next) {
            for (sum, chunk) in self.total.iter_mut().zip(payload.as_ref().chunks_exact(4)) {
                *sum += f32::from_le_bytes(chunk.try_into().expect("FP32 chunk"));
            }
            self.next += 1;
        }
        Ok(())
    }

    pub fn finish(self) -> Result<Vec<u8>, ReductionError> {
        if self.next != self.plan.partials.len() {
            return Err(ReductionError::InvalidContribution);
        }
        Ok(self.total.into_iter().flat_map(f32::to_le_bytes).collect())
    }
}

#[cfg(test)]
mod tests {
    use plait_core::serving::{GroupId, WorkerGrant};
    use plait_core::{DeviceId, NodeId, OpId, ShardId};

    use super::*;

    #[test]
    fn sums_in_k_order_and_requires_the_full_versioned_partition() {
        let host = NodeId::derive("device-host");
        let first = KPartial {
            grant: WorkerGrant {
                version: GroupVersion {
                    group: GroupId(host),
                    epoch: 2,
                },
                device: DeviceId::new(host, 3),
                incarnation: 4,
                fence: 5,
            },
            op: OpId::new(host, 2),
            shard: ShardId(7),
            k_start: 0,
            k_end: 3,
            output_shape: [1, 1],
        };
        let second = KPartial {
            shard: ShardId(5),
            k_start: 3,
            k_end: 9,
            ..first
        };
        let last = KPartial {
            shard: ShardId(2),
            k_start: 9,
            k_end: 17,
            ..first
        };
        let plan = KSum::new(17, 3, vec![last, second, first]).unwrap();
        let a = 16_777_216.0f32.to_le_bytes();
        let b = (-16_777_216.0f32).to_le_bytes();
        let c = 1.0f32.to_le_bytes();
        let contributions = [
            (last, c.as_slice()),
            (first, a.as_slice()),
            (second, b.as_slice()),
        ];
        assert_eq!(plan.sum(contributions).unwrap(), c);
        assert_eq!(
            plan.sum(contributions[..2].iter().copied()),
            Err(ReductionError::InvalidContribution)
        );
        let mut stale = first;
        stale.grant.version.epoch = 1;
        assert_eq!(
            plan.sum([
                (last, c.as_slice()),
                (stale, a.as_slice()),
                (second, b.as_slice())
            ]),
            Err(ReductionError::InvalidContribution)
        );
        assert!(matches!(
            KSum::new(17, 3, vec![first, last]),
            Err(ReductionError::InvalidPlan)
        ));
    }
}
