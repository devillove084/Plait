use std::io;
use std::path::{Path, PathBuf};

use plait_ops::decoder::{DecoderConfig, TokenHead};
use plait_ops::native::DeviceSession;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::{DeviceProjection, ProjectionExecution, W4WeightInput};

#[derive(Deserialize)]
pub(super) struct Config {
    vocab_size: u32,
    embeddings_f16: PathBuf,
    final_norm_f16: PathBuf,
    head_packed: PathBuf,
    head_scales_f16: PathBuf,
    head_zero_points: Option<PathBuf>,
    #[serde(flatten)]
    pub(super) execution: ProjectionExecution,
}

pub(super) fn load<'a, 'lib>(
    model: Config,
    device: &'a DeviceSession<'lib>,
    config: DecoderConfig,
    group_size: u32,
) -> Result<(TokenHead<DeviceProjection<'a, 'lib>>, [u8; 32]), Box<dyn std::error::Error>> {
    let shape = (config.hidden_size, model.vocab_size);
    let embeddings = std::fs::read(model.embeddings_f16)?;
    let norm = read_f16(&model.final_norm_f16)?;
    let weights = W4WeightInput {
        packed: std::fs::read(model.head_packed)?,
        scales_f16: read_f16(&model.head_scales_f16)?,
        zero_points: model.head_zero_points.map(std::fs::read).transpose()?,
    };
    let mut digest = Sha256::new();
    digest.update(b"plait-decoder-token-model-v1");
    digest.update(weights.digest(shape.0, shape.1, group_size));
    digest.update(&embeddings);
    for value in &norm {
        digest.update(value.to_le_bytes());
    }
    let head = TokenHead::new(
        embeddings,
        &norm,
        model
            .execution
            .prepare(device, &weights, None, shape, group_size)?,
        config.rms_epsilon,
    )?;
    Ok((head, digest.finalize().into()))
}

fn read_f16(path: &Path) -> io::Result<Vec<u16>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() % 2 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "truncated FP16 model artifact",
        ));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|word| u16::from_le_bytes([word[0], word[1]]))
        .collect())
}
