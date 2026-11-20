//! Rejoin policy for heterogeneous ranks.
//!
//! A strict Plait ordering recovery is not always the cheapest way to bring a rank
//! back. If the lost shard is small and deterministic, recomputing it can beat
//! replaying a snapshot or running a recovery ballot. This module makes that
//! choice explicit and testable; it does not weaken consensus for commands
//! that have already been committed.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RejoinAction {
    StrictRecover,
    SnapshotReplay,
    Recompute,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RejoinCost {
    pub recover_ms: f64,
    pub replay_ms: f64,
    pub recompute_ms: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RejoinDecision {
    pub action: RejoinAction,
    pub estimated_ms: f64,
}

impl RejoinCost {
    pub fn choose(self) -> Result<RejoinDecision, String> {
        let costs = [
            (RejoinAction::StrictRecover, self.recover_ms),
            (RejoinAction::SnapshotReplay, self.replay_ms),
            (RejoinAction::Recompute, self.recompute_ms),
        ];
        if costs
            .iter()
            .any(|(_, cost)| !cost.is_finite() || *cost < 0.0)
        {
            return Err("rejoin costs must be finite and non-negative".to_owned());
        }
        let (action, estimated_ms) = costs
            .into_iter()
            .min_by(|left, right| left.1.total_cmp(&right.1))
            .expect("three rejoin strategies");
        Ok(RejoinDecision {
            action,
            estimated_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cheap_deterministic_recompute_beats_replay_and_recovery() {
        let decision = RejoinCost {
            recover_ms: 120.0,
            replay_ms: 80.0,
            recompute_ms: 5.0,
        }
        .choose()
        .expect("costs");
        assert_eq!(decision.action, RejoinAction::Recompute);
        assert_eq!(decision.estimated_ms, 5.0);
    }

    #[test]
    fn snapshot_replay_wins_when_recompute_is_expensive() {
        let decision = RejoinCost {
            recover_ms: 120.0,
            replay_ms: 15.0,
            recompute_ms: 50.0,
        }
        .choose()
        .expect("costs");
        assert_eq!(decision.action, RejoinAction::SnapshotReplay);
    }

    #[test]
    fn strict_recovery_wins_when_it_is_cheapest() {
        let decision = RejoinCost {
            recover_ms: 10.0,
            replay_ms: 20.0,
            recompute_ms: 30.0,
        }
        .choose()
        .expect("costs");
        assert_eq!(decision.action, RejoinAction::StrictRecover);
    }

    #[test]
    fn invalid_cost_is_rejected() {
        assert!(
            RejoinCost {
                recover_ms: f64::NAN,
                replay_ms: 1.0,
                recompute_ms: 2.0,
            }
            .choose()
            .is_err()
        );
    }
}
