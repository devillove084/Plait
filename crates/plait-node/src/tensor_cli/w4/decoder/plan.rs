use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use plait_core::serving::WorkerGrant;
use plait_core::{DeviceId, NodeId};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{ProjectionExecution, ProjectionInput, ProjectionKernel};

#[derive(Deserialize)]
pub(super) struct Config {
    path: PathBuf,
    device: Option<Binding>,
    #[serde(flatten)]
    mode: Mode,
}

#[derive(Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
enum Mode {
    Load { sha256: Option<[u8; 32]> },
    Save,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
struct Binding {
    device: DeviceId,
    incarnation: u64,
}

#[derive(Clone, PartialEq, Eq, Deserialize, Serialize)]
struct Scope {
    binding: Binding,
    library_sha256: [u8; 32],
    runner_sha256: [u8; 32],
    head: ProjectionExecution,
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    version: u32,
    scope: Scope,
    layers: Vec<Layer>,
}

#[derive(Serialize, Deserialize)]
struct Layer {
    fingerprint: [u8; 32],
    projections: [ProjectionExecution; 7],
}

pub(super) struct Deployment {
    path: PathBuf,
    manifest: Manifest,
    saving: bool,
    sha256: Option<[u8; 32]>,
}

impl Deployment {
    pub(super) fn open(
        config: Config,
        grant: Option<WorkerGrant>,
        ordinal: i32,
        library: &Path,
        head: ProjectionExecution,
        layer_count: u32,
        tuning: bool,
    ) -> io::Result<Self> {
        let saving = matches!(config.mode, Mode::Save);
        if saving != tuning {
            return Err(invalid(
                "saving a decoder plan requires tune_model; loading a plan requires it to be absent",
            ));
        }
        if saving && config.path.try_exists()? {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "decoder plan already exists",
            ));
        }
        let granted = grant.map(|grant| Binding {
            device: grant.device,
            incarnation: grant.incarnation,
        });
        if matches!((granted, config.device), (Some(a), Some(b)) if a != b) {
            return Err(invalid(
                "decoder plan device does not match the worker grant",
            ));
        }
        let binding = granted
            .or(config.device)
            .ok_or_else(|| invalid("standalone decoder plans require device and incarnation"))?;
        if u32::try_from(ordinal).ok() != Some(binding.device.ordinal) {
            return Err(invalid("decoder plan device has a different ordinal"));
        }
        let scope = Scope {
            binding,
            library_sha256: hash_file(library)?,
            runner_sha256: hash_file(&std::env::current_exe()?)?,
            head,
        };
        let path = config.path;
        let (manifest, sha256) = match config.mode {
            Mode::Save => (
                Manifest {
                    version: 1,
                    scope,
                    layers: Vec::with_capacity(layer_count as usize),
                },
                None,
            ),
            Mode::Load { sha256 } => {
                let data = fs::read(&path)?;
                let digest: [u8; 32] = Sha256::digest(&data).into();
                if sha256.is_some_and(|expected| expected != digest) {
                    return Err(invalid("decoder plan checksum does not match"));
                }
                let manifest: Manifest = serde_json::from_slice(&data)?;
                if manifest.version != 1 || manifest.layers.len() != layer_count as usize {
                    return Err(invalid("unsupported decoder plan version or layer count"));
                }
                if manifest.scope.binding != scope.binding {
                    return Err(invalid("decoder plan device or incarnation changed"));
                }
                if manifest.scope.library_sha256 != scope.library_sha256 {
                    return Err(invalid("decoder plan execution library changed"));
                }
                if manifest.scope.runner_sha256 != scope.runner_sha256 {
                    return Err(invalid("decoder plan runner changed"));
                }
                if manifest.scope.head != scope.head {
                    return Err(invalid("decoder plan token head execution changed"));
                }
                (manifest, Some(digest))
            }
        };
        Ok(Self {
            path,
            manifest,
            saving,
            sha256,
        })
    }

    pub(super) fn layer(
        &self,
        index: u32,
        fingerprint: [u8; 32],
        sources: &mut [ProjectionInput; 7],
    ) -> io::Result<()> {
        if self.saving {
            return Ok(());
        }
        let layer = &self.manifest.layers[index as usize];
        if layer.fingerprint != fingerprint {
            return Err(invalid(format!(
                "decoder plan fingerprint changed for layer {index}"
            )));
        }
        for (source, execution) in sources.iter_mut().zip(layer.projections) {
            if !default_execution(source.execution) && source.execution != execution {
                return Err(invalid(format!(
                    "decoder weights specify a different execution plan at layer {index}"
                )));
            }
            source.execution = execution;
        }
        Ok(())
    }

    pub(super) fn record(&mut self, fingerprint: [u8; 32], projections: [ProjectionExecution; 7]) {
        self.manifest.layers.push(Layer {
            fingerprint,
            projections,
        });
    }

    pub(super) fn finish(self) -> io::Result<[u8; 32]> {
        if let Some(sha256) = self.sha256 {
            return Ok(sha256);
        }
        let data = serde_json::to_vec(&self.manifest)?;
        publish(&self.path, &data)?;
        Ok(Sha256::digest(data).into())
    }
}

fn default_execution(settings: ProjectionExecution) -> bool {
    matches!(settings.activation_bits, None | Some(16))
        && settings.quantization.is_none()
        && settings.kernel == ProjectionKernel::Native
}

fn hash_file(file: &Path) -> io::Result<[u8; 32]> {
    let mut digest = Sha256::new();
    io::copy(&mut File::open(file)?, &mut digest)?;
    Ok(digest.finalize().into())
}

fn publish(path: &Path, contents: &[u8]) -> io::Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let temporary = parent.join(format!(".plait-decoder-plan-{}", NodeId::random()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    let result = (|| {
        file.write_all(contents)?;
        file.sync_all()?;
        fs::hard_link(&temporary, path)?;
        Ok::<_, io::Error>(())
    })();
    let removed = fs::remove_file(&temporary);
    result?;
    removed?;
    File::open(parent)?.sync_all()
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_publication_never_replaces_an_existing_file() -> io::Result<()> {
        let dir = std::env::temp_dir().join(format!("plait-plan-test-{}", NodeId::random()));
        fs::create_dir(&dir)?;
        let path = dir.join("decoder.json");
        let result = (|| {
            publish(&path, b"first")?;
            assert_eq!(
                publish(&path, b"second").unwrap_err().kind(),
                io::ErrorKind::AlreadyExists
            );
            assert_eq!(fs::read(&path)?, b"first");
            assert_eq!(fs::read_dir(&dir)?.count(), 1);
            Ok(())
        })();
        fs::remove_dir_all(dir)?;
        result
    }
}
