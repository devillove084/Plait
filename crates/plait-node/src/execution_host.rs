use std::collections::BTreeMap;

use plait_core::serving::{
    EXECUTION_HOST_TIMEOUT_MS, EXECUTION_PULSE_MS, ServingAction, ServingOperation,
};
use plait_core::{NodeId, OpId};
use plait_fabric::Runtime;

struct Observation {
    pulse: OpId,
    since: u64,
    retry_after: u64,
}

#[derive(Default)]
pub(crate) struct HostWatch {
    observations: BTreeMap<NodeId, Observation>,
    next_scan: u64,
}

impl HostWatch {
    pub(crate) fn expirations(&mut self, runtime: &Runtime, now: u64) -> Vec<ServingOperation> {
        if now < self.next_scan {
            return Vec::new();
        }
        self.next_scan = now + 250;
        let mut operations = Vec::new();
        for (host, health) in runtime.serving_execution_hosts() {
            if !health.online {
                self.observations.remove(&host);
                continue;
            }
            let observation = self
                .observations
                .entry(host)
                .or_insert_with(|| Observation {
                    pulse: health.pulse,
                    since: now,
                    retry_after: 0,
                });
            if observation.pulse != health.pulse {
                *observation = Observation {
                    pulse: health.pulse,
                    since: now,
                    retry_after: 0,
                };
            }
            if now - observation.since < EXECUTION_HOST_TIMEOUT_MS || now < observation.retry_after
            {
                continue;
            }
            observation.retry_after = now + EXECUTION_PULSE_MS;
            operations.push(ServingOperation {
                id: OpId::new(NodeId::random(), 1),
                action: ServingAction::ExpireExecutionHost {
                    host,
                    pulse: health.pulse,
                    groups: runtime.serving_execution_versions_on(host),
                },
            });
        }
        operations
    }
}
