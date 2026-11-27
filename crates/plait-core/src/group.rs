//! Versioned identities and accelerator grants shared by transports and runtimes.

use serde::{Deserialize, Serialize};

use crate::{DeviceId, NodeId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct GroupId(pub NodeId);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GroupVersion {
    pub group: GroupId,
    pub epoch: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerGrant {
    pub version: GroupVersion,
    pub device: DeviceId,
    pub incarnation: u64,
    pub fence: u64,
}
