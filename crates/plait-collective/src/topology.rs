//! GPU interconnect topology discovery.
//!
//! `nvidia-smi topo -m` reports the link class between every pair of GPUs.
//! Plait uses this only as a scheduling hint; a missing/unknown topology falls
//! back to the conservative ring path. The parser is intentionally small and
//! accepts the stable `GPU0 ...` matrix produced by the NVIDIA driver.

use std::process::Command;

use crate::AllReduceAlgorithm;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkClass {
    NvLink(u32),
    Pix,
    Pxb,
    Phb,
    Node,
    Sys,
    Unknown,
}

#[derive(Clone, Debug)]
pub struct GpuTopology {
    gpus: usize,
    links: Vec<Vec<LinkClass>>,
}

impl GpuTopology {
    pub fn detect() -> Result<Self, String> {
        let output = Command::new("nvidia-smi")
            .args(["topo", "-m"])
            .output()
            .map_err(|error| format!("nvidia-smi topo: {error}"))?;
        if !output.status.success() {
            return Err(format!("nvidia-smi topo exited with {}", output.status));
        }
        parse(&String::from_utf8_lossy(&output.stdout))
    }

    pub fn gpus(&self) -> usize {
        self.gpus
    }

    pub fn link(&self, left: usize, right: usize) -> LinkClass {
        self.links
            .get(left)
            .and_then(|row| row.get(right))
            .copied()
            .unwrap_or(LinkClass::Unknown)
    }

    pub fn has_nvlink(&self) -> bool {
        self.links
            .iter()
            .flatten()
            .any(|link| matches!(link, LinkClass::NvLink(_)))
    }

    /// Conservative selection used until tree algorithms are benchmarked.
    /// Small messages prefer the latency-oriented reduce+broadcast path;
    /// larger messages and NVLink-connected hosts use ring.
    pub fn recommended_allreduce(&self, bytes: usize) -> AllReduceAlgorithm {
        if bytes <= 32 * 1024 && !self.has_nvlink() {
            AllReduceAlgorithm::ReduceBroadcast
        } else {
            AllReduceAlgorithm::Ring
        }
    }
}

pub fn parse(text: &str) -> Result<GpuTopology, String> {
    let mut gpus = 0usize;
    let mut links: Vec<Vec<LinkClass>> = Vec::new();
    let mut in_matrix = false;
    for line in text.lines() {
        let columns: Vec<&str> = line.split_whitespace().collect();
        if columns.is_empty() {
            continue;
        }
        if !in_matrix && columns[0] == "GPU0" {
            gpus = columns
                .iter()
                .take_while(|column| column.starts_with("GPU"))
                .count();
            if gpus < 2 {
                return Err("topology matrix has fewer than two GPUs".to_owned());
            }
            in_matrix = true;
            continue;
        }
        if !in_matrix || !columns[0].starts_with("GPU") {
            continue;
        }
        // Row format: GPU0 X PIX ... [CPU Affinity] [NUMA Affinity] ...
        let row: Vec<LinkClass> = columns
            .iter()
            .skip(1)
            .take(gpus)
            .map(|value| parse_link(value))
            .collect();
        if row.len() != gpus {
            return Err(format!(
                "topology row has {} links, expected {gpus}",
                row.len()
            ));
        }
        links.push(row);
        if links.len() == gpus {
            break;
        }
    }
    if links.len() != gpus {
        return Err(format!(
            "parsed {} topology rows, expected {gpus}",
            links.len()
        ));
    }
    Ok(GpuTopology { gpus, links })
}

fn parse_link(value: &str) -> LinkClass {
    if value == "X" {
        return LinkClass::Unknown;
    }
    if let Some(count) = value.strip_prefix("NV") {
        return count
            .parse()
            .map(LinkClass::NvLink)
            .unwrap_or(LinkClass::Unknown);
    }
    match value {
        "PIX" => LinkClass::Pix,
        "PXB" => LinkClass::Pxb,
        "PHB" => LinkClass::Phb,
        "NODE" => LinkClass::Node,
        "SYS" => LinkClass::Sys,
        _ => LinkClass::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nvlink_and_pcie_classes() {
        let text = "\
        GPU0 GPU1 GPU2\n\
        GPU0 X NV4 SYS\n\
        GPU1 NV4 X NODE\n\
        GPU2 SYS NODE X\n";
        let topology = parse(text).expect("topology");
        assert_eq!(topology.gpus(), 3);
        assert!(topology.has_nvlink());
        assert_eq!(topology.link(0, 1), LinkClass::NvLink(4));
        assert_eq!(topology.link(0, 2), LinkClass::Sys);
        assert_eq!(topology.link(1, 2), LinkClass::Node);
        assert_eq!(
            topology.recommended_allreduce(1024),
            AllReduceAlgorithm::Ring
        );
    }

    #[test]
    fn small_pcie_message_prefers_reduce_broadcast() {
        let text = "\
        GPU0 GPU1\n\
        GPU0 X SYS\n\
        GPU1 SYS X\n";
        let topology = parse(text).expect("topology");
        assert_eq!(
            topology.recommended_allreduce(1024),
            AllReduceAlgorithm::ReduceBroadcast
        );
        assert_eq!(
            topology.recommended_allreduce(1 << 20),
            AllReduceAlgorithm::Ring
        );
    }
}
