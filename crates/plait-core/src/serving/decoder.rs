use std::net::SocketAddr;

use serde::{Deserialize, Serialize};

use super::{PdRoute, WorkerGrant};
use crate::{NodeId, OpId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecoderEndpoint {
    pub grant: WorkerGrant,
    pub addr: SocketAddr,
    pub fingerprint: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecoderRoute {
    pub prefill: DecoderEndpoint,
    pub decode: DecoderEndpoint,
}

impl DecoderRoute {
    pub fn path(self) -> PdRoute {
        PdRoute {
            prefill: self.prefill.grant.version,
            decode: self.decode.grant.version,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecoderDirectory {
    pub model: String,
    pub revision: u64,
    pub paths: Vec<DecoderRoute>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecoderAdmissionRoutes {
    pub revision: u64,
    pub paths: Vec<PdRoute>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecoderRouteWeight {
    pub path: PdRoute,
    pub weight: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecoderStandby {
    pub primary: PdRoute,
    pub alternatives: Vec<PdRoute>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecoderFailover {
    pub revision: u64,
    pub standby: DecoderStandby,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecoderSwitch {
    pub staged: OpId,
    pub revision: u64,
    pub previous: Vec<PdRoute>,
    pub paths: Vec<PdRoute>,
    pub standbys: Vec<DecoderStandby>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecoderPreparation {
    pub group: super::GroupId,
    pub expected_epoch: u64,
    pub expected_active: Option<u64>,
    pub plan: super::GroupPlan,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecoderRouteWork {
    Failover(DecoderFailover),
    Switch(DecoderSwitch),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecoderDrainStatus {
    pub pending: u64,
    pub next_expiry: Option<(u64, PdRoute)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecoderClaim {
    pub work: OpId,
    pub owner: OpId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecoderKvSource {
    pub addr: SocketAddr,
    pub transfer: NodeId,
    pub bytes: u64,
    pub sha256: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecoderOffer {
    pub source: DecoderKvSource,
    pub first_token: u32,
    pub prefix_tokens: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecoderCheckpoint {
    pub source: DecoderKvSource,
    pub start_offset: u32,
    pub end_offset: u32,
    pub last_token: u32,
}

pub const DECODER_REPLAY_MAX_TOKENS: u32 = 16_384;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecoderReplayState {
    Running,
    Complete,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecoderReplayPage {
    pub tokens: Vec<u32>,
    pub available: u32,
    pub state: DecoderReplayState,
    pub revision: u64,
    pub path: PdRoute,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecoderStep {
    StartPrefill,
    Offer {
        owner: OpId,
        kv: DecoderOffer,
    },
    ReadOffer,
    ClaimDecode {
        transfer: NodeId,
    },
    Finish {
        owner: OpId,
    },
    CheckOutput {
        owner: OpId,
    },
    CheckOutputCheckpoint {
        owner: OpId,
        checkpoint: DecoderCheckpoint,
    },
    ClaimDecodeCheckpoint {
        transfer: NodeId,
        offset: u32,
    },
    FinishPrefillToken {
        owner: OpId,
        token: u32,
    },
    FinishDecodeStream {
        owner: OpId,
        end_offset: u32,
    },
    StartPrefillEntry {
        entry: OpId,
    },
    ReadOfferEntry {
        entry: OpId,
    },
}
