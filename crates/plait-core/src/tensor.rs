use serde::{Deserialize, Serialize};

/// Identifies a logical tensor, independent of where its shards live.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TensorId(pub u128);

impl std::fmt::Debug for TensorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TensorId({:032x})", self.0)
    }
}

/// Identifies a shard of a tensor. Shard ids are opaque and chosen by the
/// producer; they are not ranks.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ShardId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TensorKey {
    pub tensor: TensorId,
    pub shard: ShardId,
}

impl TensorKey {
    pub const fn new(tensor: TensorId, shard: ShardId) -> Self {
        Self { tensor, shard }
    }
}

impl std::fmt::Debug for TensorKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}/shard{}", self.tensor, self.shard.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape {
    dimensions: Vec<u64>,
}

impl Shape {
    pub fn new(dimensions: impl Into<Vec<u64>>) -> Result<Self, ShapeError> {
        let dimensions = dimensions.into();
        if dimensions.is_empty() {
            return Err(ShapeError::Empty);
        }
        let mut elements: u128 = 1;
        for dimension in &dimensions {
            if *dimension == 0 {
                return Err(ShapeError::ZeroDimension);
            }
            elements = elements
                .checked_mul(u128::from(*dimension))
                .ok_or(ShapeError::Overflow)?;
        }
        if elements > u128::from(u64::MAX) {
            return Err(ShapeError::Overflow);
        }
        Ok(Self { dimensions })
    }

    pub fn dimensions(&self) -> &[u64] {
        &self.dimensions
    }

    pub fn element_count(&self) -> u64 {
        self.dimensions.iter().copied().product::<u64>()
    }

    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeError {
    Empty,
    ZeroDimension,
    Overflow,
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => f.write_str("a shape needs at least one dimension"),
            Self::ZeroDimension => f.write_str("shape dimensions must be non-zero"),
            Self::Overflow => f.write_str("shape element count overflows u64"),
        }
    }
}

impl std::error::Error for ShapeError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    F16,
    BF16,
    F32,
    F64,
}

impl DType {
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::F16 | Self::BF16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

/// Where a tensor's bytes are allowed to live. Placement is replicated data,
/// not a backend convention.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Placement {
    /// Exactly one owner holds the authoritative value.
    Local(crate::NodeId),
    /// Every listed node holds a full copy.
    Replicated(Vec<crate::NodeId>),
    /// The tensor is split along one axis into equal parts.
    Sharded { axis: u32, parts: u32 },
    /// Every listed node holds a partial contribution that must be combined.
    Partial(Vec<crate::NodeId>),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorSpec {
    pub id: TensorId,
    pub shape: Shape,
    pub dtype: DType,
    pub placement: Placement,
}

/// Monotonic version attached to a materialized view of a tensor shard.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize)]
pub struct TensorVersion(pub u64);

/// A materialized tensor value used by the fused-operator prototype. All
/// arithmetic currently uses `f64`; dtype conversion belongs in a device
/// adapter, not in the consensus log.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TensorView {
    pub key: TensorKey,
    pub version: TensorVersion,
    pub values: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_counts_elements() {
        let shape = Shape::new([2u64, 3, 4]).unwrap();
        assert_eq!(shape.element_count(), 24);
        assert_eq!(shape.rank(), 3);
    }

    #[test]
    fn shape_rejects_invalid_dimensions() {
        assert_eq!(
            Shape::new(Vec::<u64>::new()).unwrap_err(),
            ShapeError::Empty
        );
        assert_eq!(
            Shape::new([2u64, 0]).unwrap_err(),
            ShapeError::ZeroDimension
        );
        assert_eq!(Shape::new([u64::MAX, 2]).unwrap_err(), ShapeError::Overflow);
    }

    #[test]
    fn tensor_keys_are_ordered() {
        let key = TensorKey::new(TensorId(7), ShardId(3));
        assert_eq!(key, TensorKey::new(TensorId(7), ShardId(3)));
        assert!(key > TensorKey::new(TensorId(7), ShardId(2)));
    }
}
