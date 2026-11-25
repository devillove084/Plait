//! Quantized linear operators with a shared layout and independent device backends.

pub mod decoder;
#[cfg(feature = "native")]
mod humming_layout;
#[cfg(feature = "cuda-marlin")]
pub mod marlin;
#[cfg(feature = "native")]
mod marlin_layout;
#[cfg(feature = "native")]
pub mod native;
#[cfg(any(feature = "native", test))]
mod packing;
mod quantized;
#[cfg(feature = "native")]
mod resident;
mod w4a16;

pub use quantized::{ActivationBits, QuantizedActivations};
pub use w4a16::{LinearOutput, OutputType, W4A16, W4Weights};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OperatorError {
    Shape(&'static str),
    Unsupported(&'static str),
    Backend(String),
}

impl std::fmt::Display for OperatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shape(message) | Self::Unsupported(message) => f.write_str(message),
            Self::Backend(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for OperatorError {}
