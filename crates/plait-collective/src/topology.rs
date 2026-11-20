//! GPU interconnect topology discovery.
//!
//! `nvidia-smi topo -m` reports the link class between every pair of GPUs.
//! Plait uses this only as a scheduling hint; a missing/unknown topology falls
//! back to the conservative ring path. The parser is intentionally small and
//! accepts the stable `GPU0 ...` matrix produced by the NVIDIA driver.

use std::process::Command;

use crate::{AllReduceAlgorithm, Protocol};

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
        if !self.has_nvlink() {
            return AllReduceAlgorithm::Ring;
        }
        if bytes <= 32 * 1024 {
            AllReduceAlgorithm::ReduceBroadcast
        } else if bytes >= 64 * 1024 {
            AllReduceAlgorithm::Tree
        } else {
            AllReduceAlgorithm::Ring
        }
    }

    /// Pick a device protocol from message size and link class.
    ///
    /// LL/LL128 are latency protocols for NVLink. On PCIe their per-line flag
    /// traffic is slower than the simple ring, so PCIe always selects Simple.
    pub fn recommended_protocol(&self, bytes: usize) -> Protocol {
        if !self.has_nvlink() {
            return Protocol::Simple;
        }
        if bytes <= 32 * 1024 {
            Protocol::Ll
        } else if bytes >= 64 * 1024 {
            Protocol::Ll128
        } else {
            Protocol::Simple
        }
    }
}

pub fn parse(text: &str) -> Result<GpuTopology, String> {
    let mut gpus = 0usize;
    let mut links: Vec<Vec<LinkClass>> = Vec::new();
    let mut in_matrix = false;
    for line in text.lines() {
        let clean = strip_ansi(line);
        let columns: Vec<&str> = clean.split_whitespace().collect();
        if columns.is_empty() {
            continue;
        }
        let gpu_columns = columns
            .iter()
            .filter(|column| {
                column
                    .strip_prefix("GPU")
                    .is_some_and(|index| index.parse::<usize>().is_ok())
            })
            .count();
        if gpu_columns >= 2 {
            gpus = gpu_columns;
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

fn strip_ansi(line: &str) -> String {
    let mut output = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    while let Some(character) = chars.next() {
        if character == '\x1b' && chars.peek() == Some(&'[') {
            chars.next();
            for next in chars.by_ref() {
                if next == 'm' {
                    break;
                }
            }
        } else {
            output.push(character);
        }
    }
    output
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
    fn detects_the_host_or_reports_why_not() {
        match GpuTopology::detect() {
            Ok(topology) => println!(
                "topology gpus={} nvlink={} link01={:?}",
                topology.gpus(),
                topology.has_nvlink(),
                topology.link(0, 1)
            ),
            Err(error) => println!("topology unavailable: {error}"),
        }
    }

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
            AllReduceAlgorithm::ReduceBroadcast
        );
        assert_eq!(topology.recommended_protocol(1024), Protocol::Ll);
        assert_eq!(topology.recommended_protocol(1 << 20), Protocol::Ll128);
    }

    #[test]
    fn pcie_uses_ring_and_simple_protocol() {
        let text = "\
        GPU0 GPU1\n\
        GPU0 X SYS\n\
        GPU1 SYS X\n";
        let topology = parse(text).expect("topology");
        assert_eq!(
            topology.recommended_allreduce(1024),
            AllReduceAlgorithm::Ring
        );
        assert_eq!(
            topology.recommended_allreduce(1 << 20),
            AllReduceAlgorithm::Ring
        );
        assert_eq!(topology.recommended_protocol(1024), Protocol::Simple);
        assert_eq!(topology.recommended_protocol(1 << 20), Protocol::Simple);
    }
}
