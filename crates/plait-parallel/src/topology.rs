use std::collections::BTreeMap;

use plait_core::NodeId;
use serde::{Deserialize, Serialize};

/// The six parallelism axes. Plait treats all of them as placement
/// coordinates, not as a rank hierarchy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Axis {
    Data,
    Pipeline,
    Tensor,
    Expert,
    Context,
    Sequence,
}

pub const AXIS_ORDER: [Axis; 6] = [
    Axis::Data,
    Axis::Pipeline,
    Axis::Tensor,
    Axis::Expert,
    Axis::Context,
    Axis::Sequence,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AxisSizes {
    pub data: u32,
    pub pipeline: u32,
    pub tensor: u32,
    pub expert: u32,
    pub context: u32,
    pub sequence: u32,
}

impl AxisSizes {
    pub const fn new(
        data: u32,
        pipeline: u32,
        tensor: u32,
        expert: u32,
        context: u32,
        sequence: u32,
    ) -> Self {
        Self {
            data,
            pipeline,
            tensor,
            expert,
            context,
            sequence,
        }
    }

    pub const fn single_axis(axis: Axis, size: u32) -> Self {
        let mut sizes = Self::new(1, 1, 1, 1, 1, 1);
        match axis {
            Axis::Data => sizes.data = size,
            Axis::Pipeline => sizes.pipeline = size,
            Axis::Tensor => sizes.tensor = size,
            Axis::Expert => sizes.expert = size,
            Axis::Context => sizes.context = size,
            Axis::Sequence => sizes.sequence = size,
        }
        sizes
    }

    pub fn get(&self, axis: Axis) -> u32 {
        match axis {
            Axis::Data => self.data,
            Axis::Pipeline => self.pipeline,
            Axis::Tensor => self.tensor,
            Axis::Expert => self.expert,
            Axis::Context => self.context,
            Axis::Sequence => self.sequence,
        }
    }

    pub fn product(&self) -> Result<usize, TopologyError> {
        let mut product = 1usize;
        for axis in AXIS_ORDER {
            let size = self.get(axis) as usize;
            if size == 0 {
                return Err(TopologyError::ZeroAxis(axis));
            }
            product = product.checked_mul(size).ok_or(TopologyError::TooLarge)?;
        }
        Ok(product)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Coord {
    pub data: u32,
    pub pipeline: u32,
    pub tensor: u32,
    pub expert: u32,
    pub context: u32,
    pub sequence: u32,
}

impl Coord {
    pub fn get(&self, axis: Axis) -> u32 {
        match axis {
            Axis::Data => self.data,
            Axis::Pipeline => self.pipeline,
            Axis::Tensor => self.tensor,
            Axis::Expert => self.expert,
            Axis::Context => self.context,
            Axis::Sequence => self.sequence,
        }
    }

    fn set(&mut self, axis: Axis, value: u32) {
        match axis {
            Axis::Data => self.data = value,
            Axis::Pipeline => self.pipeline = value,
            Axis::Tensor => self.tensor = value,
            Axis::Expert => self.expert = value,
            Axis::Context => self.context = value,
            Axis::Sequence => self.sequence = value,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Topology {
    sizes: AxisSizes,
    nodes: Vec<NodeId>,
    strides: BTreeMap<Axis, usize>,
    coords: BTreeMap<NodeId, Coord>,
}

impl Topology {
    pub fn new(sizes: AxisSizes, nodes: Vec<NodeId>) -> Result<Self, TopologyError> {
        let product = sizes.product()?;
        if nodes.len() != product {
            return Err(TopologyError::NodeCount {
                expected: product,
                actual: nodes.len(),
            });
        }
        if nodes
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != nodes.len()
        {
            return Err(TopologyError::DuplicateNode);
        }
        let mut strides = BTreeMap::new();
        let mut stride = 1usize;
        for axis in AXIS_ORDER {
            strides.insert(axis, stride);
            stride = stride
                .checked_mul(sizes.get(axis) as usize)
                .ok_or(TopologyError::TooLarge)?;
        }
        let mut coords = BTreeMap::new();
        for (index, node) in nodes.iter().enumerate() {
            let mut remainder = index;
            let mut coord = Coord::default();
            for axis in AXIS_ORDER {
                let size = sizes.get(axis) as usize;
                coord.set(axis, (remainder % size) as u32);
                remainder /= size;
            }
            coords.insert(*node, coord);
        }
        Ok(Self {
            sizes,
            nodes,
            strides,
            coords,
        })
    }

    pub fn with_prefix(sizes: AxisSizes, prefix: &str) -> Result<Self, TopologyError> {
        let product = sizes.product()?;
        let nodes = (0..product)
            .map(|index| NodeId::derive(&format!("{prefix}-{index}")))
            .collect();
        Self::new(sizes, nodes)
    }

    pub fn sizes(&self) -> AxisSizes {
        self.sizes
    }

    pub fn nodes(&self) -> &[NodeId] {
        &self.nodes
    }

    pub fn node_index(&self, node: NodeId) -> Option<usize> {
        self.nodes.iter().position(|candidate| *candidate == node)
    }

    pub fn coord(&self, node: NodeId) -> Option<Coord> {
        self.coords.get(&node).copied()
    }

    pub fn node_at(&self, coord: Coord) -> Option<NodeId> {
        let mut index = 0usize;
        for axis in AXIS_ORDER {
            index += coord.get(axis) as usize * self.strides[&axis];
        }
        self.nodes.get(index).copied()
    }

    /// All nodes that differ only along `axis`, preserving the coordinate
    /// order of that axis. This is a communication group, not a rank range.
    pub fn group(&self, axis: Axis, coord: Coord) -> Vec<NodeId> {
        let mut group = Vec::with_capacity(self.sizes.get(axis) as usize);
        for value in 0..self.sizes.get(axis) {
            let mut member = coord;
            member.set(axis, value);
            if let Some(node) = self.node_at(member) {
                group.push(node);
            }
        }
        group
    }

    /// The group containing `node` along `axis`.
    pub fn group_of(&self, node: NodeId, axis: Axis) -> Vec<NodeId> {
        match self.coord(node) {
            Some(coord) => self.group(axis, coord),
            None => Vec::new(),
        }
    }

    pub fn axis_index(&self, node: NodeId, axis: Axis) -> Option<u32> {
        self.coord(node).map(|coord| coord.get(axis))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TopologyError {
    ZeroAxis(Axis),
    TooLarge,
    NodeCount { expected: usize, actual: usize },
    DuplicateNode,
}

impl std::fmt::Display for TopologyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroAxis(axis) => write!(f, "axis {axis:?} must be non-zero"),
            Self::TooLarge => f.write_str("topology dimensions overflow"),
            Self::NodeCount { expected, actual } => {
                write!(f, "expected {expected} nodes, got {actual}")
            }
            Self::DuplicateNode => f.write_str("node ids must be unique"),
        }
    }
}

impl std::error::Error for TopologyError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coordinates_round_trip() {
        let topology = Topology::with_prefix(AxisSizes::new(2, 2, 3, 1, 1, 1), "topo").unwrap();
        for node in topology.nodes() {
            let coord = topology.coord(*node).unwrap();
            assert_eq!(topology.node_at(coord), Some(*node));
        }
    }

    #[test]
    fn groups_differ_only_on_the_axis() {
        let topology = Topology::with_prefix(AxisSizes::new(1, 1, 2, 1, 3, 1), "group").unwrap();
        let node = topology.nodes()[0];
        let tp = topology.group_of(node, Axis::Tensor);
        assert_eq!(tp.len(), 2);
        for member in tp {
            let coord = topology.coord(member).unwrap();
            assert_eq!(coord.context, topology.coord(node).unwrap().context);
        }
        let cp = topology.group_of(node, Axis::Context);
        assert_eq!(cp.len(), 3);
        for member in cp {
            let coord = topology.coord(member).unwrap();
            assert_eq!(coord.tensor, topology.coord(node).unwrap().tensor);
        }
    }

    #[test]
    fn rejects_inconsistent_node_counts() {
        assert!(matches!(
            Topology::new(
                AxisSizes::new(1, 1, 2, 1, 1, 1),
                vec![NodeId::derive("only")]
            ),
            Err(TopologyError::NodeCount { .. })
        ));
    }
}
