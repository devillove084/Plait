use std::collections::btree_map::Entry;
use std::time::{Duration, Instant};

use plait_core::serving::{ServingError, ServingOperation, ServingOutcome};
use plait_core::{Command, NodeId};
use plait_fabric::Runtime;
use plait_net::{GroupObservation, Wire};

use crate::NodeState;

const MAX_PENDING: usize = 1024;
const OBSERVATION_TIMEOUT: Duration = Duration::from_secs(5);

pub(crate) struct PendingObservation {
    operation: ServingOperation,
    expires_at: Instant,
}

impl NodeState {
    pub(crate) async fn start_observation(
        &mut self,
        from: NodeId,
        request: u64,
        operation: ServingOperation,
        now: u64,
    ) {
        let room = self.observations.len() < MAX_PENDING;
        let Entry::Vacant(slot) = self.observations.entry((from, request)) else {
            return;
        };
        let result = if !self.joined || !self.replica.config().is_voter(self.id) {
            Err("this node is not a voter".to_owned())
        } else if !room {
            Err("too many pending group observations".to_owned())
        } else if self
            .replica
            .machine()
            .serving_receipt(operation.id)
            .is_some()
        {
            Ok(())
        } else {
            self.replica
                .propose(Command::Serving(Box::new(operation.clone())), now)
                .map(|_| ())
                .map_err(|error| error.to_string())
        };
        match result {
            Ok(()) => {
                slot.insert(PendingObservation {
                    operation,
                    expires_at: Instant::now() + OBSERVATION_TIMEOUT,
                });
            }
            Err(reason) => {
                self.send_observation(from, request, GroupObservation::Unavailable(reason))
                    .await;
            }
        }
    }

    /// Invoke only after persisting the replica changes from this event.
    pub(crate) async fn reply_to_observations(&mut self) {
        if self.observations.is_empty() {
            return;
        }
        let runtime = self.replica.machine();
        let now = Instant::now();
        let mut replies = Vec::new();
        self.observations.retain(|&(from, request), pending| {
            if pending.expires_at <= now {
                return false;
            }
            if let Some(response) = observed(runtime, &pending.operation) {
                replies.push((from, request, response));
                return false;
            }
            true
        });
        for (from, request, response) in replies {
            self.send_observation(from, request, response).await;
        }
    }

    async fn send_observation(&self, from: NodeId, request: u64, response: GroupObservation) {
        let _ = self
            .transport
            .send(from, Wire::ServingGroupObserved { request, response })
            .await;
    }
}

fn observed(runtime: &Runtime, operation: &ServingOperation) -> Option<GroupObservation> {
    let receipt = runtime.serving_receipt(operation.id)?;
    if &receipt.operation != operation {
        return Some(GroupObservation::Rejected(ServingError::OperationIdReused));
    }
    Some(match receipt.result {
        Ok(ServingOutcome::Group { version, .. }) => match runtime.serving_group(version) {
            Some(view) => GroupObservation::Applied(Box::new(view.clone())),
            None => GroupObservation::Rejected(ServingError::UnknownVersion(version)),
        },
        Ok(_) => GroupObservation::Unavailable("unexpected serving observation receipt".to_owned()),
        Err(error) => GroupObservation::Rejected(error),
    })
}
