use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A stable, position-independent node identity.
///
/// A `NodeId` never changes when membership changes and is never derived from
/// an array index. Two nodes that pick the same id are the same node as far as
/// the fabric is concerned; production deployments should persist the id and
/// authenticate it at the transport boundary. This prototype keeps the id
/// unauthenticated so the consensus and fusion layers can be exercised in
/// isolation.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeId([u8; 16]);

impl NodeId {
    pub const LEN: usize = 16;

    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// Derive a deterministic identity from a human-readable label. Useful for
    /// reproducible tests and for recovering an id from a persisted name.
    pub fn derive(label: &str) -> Self {
        let digest = Sha256::digest(label.as_bytes());
        let mut bytes = [0u8; 16];
        bytes.copy_from_slice(&digest[..16]);
        Self(bytes)
    }

    /// Generate a fresh identity from the operating system RNG.
    pub fn random() -> Self {
        Self(rand::random::<[u8; 16]>())
    }

    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    pub fn to_hex(self) -> String {
        let mut out = String::with_capacity(32);
        for byte in self.0 {
            use std::fmt::Write as _;
            write!(out, "{byte:02x}").expect("writing to a String cannot fail");
        }
        out
    }

    pub fn from_hex(text: &str) -> Result<Self, HexError> {
        let text = text.trim();
        if text.len() != 32 {
            return Err(HexError::Length(text.len()));
        }
        let mut bytes = [0u8; 16];
        for (index, chunk) in text.as_bytes().chunks_exact(2).enumerate() {
            let hi = hex_nibble(chunk[0])?;
            let lo = hex_nibble(chunk[1])?;
            bytes[index] = (hi << 4) | lo;
        }
        Ok(Self(bytes))
    }
}

fn hex_nibble(byte: u8) -> Result<u8, HexError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        other => Err(HexError::Character(other as char)),
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodeId({})", self.to_hex())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HexError {
    Length(usize),
    Character(char),
}

impl fmt::Display for HexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length(length) => write!(f, "expected 32 hex characters, got {length}"),
            Self::Character(ch) => write!(f, "invalid hex character {ch:?}"),
        }
    }
}

impl std::error::Error for HexError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_round_trip() {
        let id = NodeId::derive("alpha");
        assert_eq!(NodeId::from_hex(&id.to_hex()).unwrap(), id);
        assert_eq!(NodeId::from_hex(&id.to_hex().to_uppercase()).unwrap(), id);
    }

    #[test]
    fn rejects_bad_hex() {
        assert!(NodeId::from_hex("00").is_err());
        assert!(NodeId::from_hex(&"g".repeat(32)).is_err());
    }

    #[test]
    fn derived_ids_are_stable() {
        assert_eq!(NodeId::derive("alpha"), NodeId::derive("alpha"));
        assert_ne!(NodeId::derive("alpha"), NodeId::derive("beta"));
    }
}
