use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Component, Path};

use serde::de::{DeserializeOwned, Error, MapAccess, Visitor};
use serde::{Deserialize, Deserializer};

const MAX_HEADER_BYTES: u64 = 100_000_000;

#[derive(Deserialize)]
struct Index {
    #[serde(deserialize_with = "unique_map")]
    weight_map: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct Header(#[serde(deserialize_with = "unique_map")] BTreeMap<String, HeaderEntry>);

#[derive(Deserialize)]
#[serde(untagged)]
enum HeaderEntry {
    Tensor(TensorMeta),
    Metadata(BTreeMap<String, String>),
}

#[derive(Deserialize)]
struct TensorMeta {
    dtype: String,
    shape: Vec<u32>,
    data_offsets: [u64; 2],
}

struct TensorLocation {
    file: usize,
    offset: u64,
    length: u64,
    meta: TensorMeta,
}

pub(super) struct Safetensors {
    files: Vec<File>,
    tensors: BTreeMap<String, TensorLocation>,
}

#[derive(Clone, Copy)]
pub(super) enum FloatFormat {
    F16,
    Bf16,
    F32,
}

pub(super) struct Tensor {
    pub(super) bytes: Vec<u8>,
    pub(super) format: FloatFormat,
}

impl FloatFormat {
    fn parse(dtype: &str) -> Option<Self> {
        match dtype {
            "F16" => Some(Self::F16),
            "BF16" => Some(Self::Bf16),
            "F32" => Some(Self::F32),
            _ => None,
        }
    }

    pub(super) fn word_bytes(self) -> usize {
        match self {
            Self::F16 | Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }

    pub(super) fn decode(self, word: &[u8]) -> f32 {
        match self {
            Self::F16 => half::f16::from_le_bytes([word[0], word[1]]).to_f32(),
            Self::Bf16 => half::bf16::from_le_bytes([word[0], word[1]]).to_f32(),
            Self::F32 => f32::from_le_bytes([word[0], word[1], word[2], word[3]]),
        }
    }
}

impl Safetensors {
    pub(super) fn open(root: &Path) -> io::Result<Self> {
        let index = match read_json::<Index>(
            &root.join("model.safetensors.index.json"),
            MAX_HEADER_BYTES,
        ) {
            Ok(index) => Some(index.weight_map),
            Err(error) if error.kind() == io::ErrorKind::NotFound => None,
            Err(error) => return Err(error),
        };
        let names: BTreeSet<&str> = match &index {
            Some(index) if !index.is_empty() => index.values().map(String::as_str).collect(),
            Some(_) => return Err(invalid("safetensors index has no tensors")),
            None => BTreeSet::from(["model.safetensors"]),
        };
        let mut files = Vec::with_capacity(names.len());
        let mut tensors = BTreeMap::new();
        for name in names {
            let mut components = Path::new(name).components();
            if !matches!(components.next(), Some(Component::Normal(_)))
                || components.next().is_some()
                || !name.ends_with(".safetensors")
            {
                return Err(invalid("safetensors shard must be a local filename"));
            }
            let mut file = File::open(root.join(name))?;
            let file_bytes = file.metadata()?.len();
            let (data_offset, header) = read_header(&mut file, file_bytes)?;
            for (tensor, meta) in header {
                if let Some(index) = &index
                    && index.get(&tensor).map(String::as_str) != Some(name)
                {
                    continue;
                }
                let [begin, end] = meta.data_offsets;
                tensors.insert(
                    tensor,
                    TensorLocation {
                        file: files.len(),
                        offset: data_offset + begin,
                        length: end - begin,
                        meta,
                    },
                );
            }
            files.push(file);
        }
        if let Some(index) = &index
            && let Some(missing) = index.keys().find(|tensor| !tensors.contains_key(*tensor))
        {
            return Err(invalid(format!(
                "safetensors index refers to an absent tensor {missing}"
            )));
        }
        Ok(Self { files, tensors })
    }

    pub(super) fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub(super) fn read(&mut self, name: &str, shape: &[u32]) -> io::Result<Tensor> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| invalid(format!("missing checkpoint tensor {name}")))?;
        let format = FloatFormat::parse(&tensor.meta.dtype)
            .ok_or_else(|| invalid(format!("unsupported floating point type for tensor {name}")))?;
        let bytes = shape.iter().try_fold(format.word_bytes(), |product, &dim| {
            product.checked_mul(dim as usize)
        });
        let bytes = bytes
            .filter(|&bytes| tensor.meta.shape == shape && bytes as u64 == tensor.length)
            .ok_or_else(|| {
                invalid(format!(
                    "checkpoint tensor {name} has an invalid shape or byte count"
                ))
            })?;
        let file = &mut self.files[tensor.file];
        file.seek(SeekFrom::Start(tensor.offset))?;
        let mut buffer = Vec::new();
        buffer.try_reserve_exact(bytes).map_err(io::Error::other)?;
        file.take(tensor.length).read_to_end(&mut buffer)?;
        if buffer.len() != bytes {
            return Err(invalid(format!("truncated checkpoint tensor {name}")));
        }
        Ok(Tensor {
            bytes: buffer,
            format,
        })
    }
}

fn read_header(
    file: &mut impl Read,
    file_bytes: u64,
) -> io::Result<(u64, BTreeMap<String, TensorMeta>)> {
    let mut size = [0; 8];
    file.read_exact(&mut size)?;
    let length = u64::from_le_bytes(size);
    if length == 0 || length > MAX_HEADER_BYTES || length > file_bytes.saturating_sub(8) {
        return Err(invalid("invalid safetensors header length"));
    }
    let mut header = vec![0; length as usize];
    file.read_exact(&mut header)?;
    if header[0] != b'{' {
        return Err(invalid("invalid safetensors JSON header"));
    }
    let Header(entries) = serde_json::from_slice(&header).map_err(invalid)?;
    let mut tensors = BTreeMap::new();
    let mut ranges = Vec::with_capacity(entries.len());
    for (name, entry) in entries {
        match entry {
            HeaderEntry::Tensor(meta) if name != "__metadata__" => {
                ranges.push(meta.data_offsets);
                tensors.insert(name, meta);
            }
            HeaderEntry::Metadata(metadata) if name == "__metadata__" => drop(metadata),
            _ => return Err(invalid("invalid safetensors header entry")),
        }
    }
    ranges.sort_unstable();
    let data_offset = length + 8;
    let data_bytes = file_bytes - data_offset;
    let mut offset = 0;
    for [begin, end] in ranges {
        if begin != offset || end < begin || end > data_bytes {
            return Err(invalid(
                "safetensors tensor ranges must be contiguous and in bounds",
            ));
        }
        offset = end;
    }
    if offset != data_bytes {
        return Err(invalid("safetensors header does not cover its data"));
    }
    Ok((data_offset, tensors))
}

pub(super) fn read_json<T: DeserializeOwned>(path: &Path, limit: u64) -> io::Result<T> {
    let file = File::open(path)?;
    let mut data = Vec::new();
    file.take(limit + 1).read_to_end(&mut data)?;
    if data.len() as u64 > limit {
        return Err(invalid("checkpoint JSON exceeds its size limit"));
    }
    serde_json::from_slice(&data).map_err(invalid)
}

fn unique_map<'de, D, V>(deserializer: D) -> Result<BTreeMap<String, V>, D::Error>
where
    D: Deserializer<'de>,
    V: Deserialize<'de>,
{
    struct Unique<V>(std::marker::PhantomData<V>);
    impl<'de, V: Deserialize<'de>> Visitor<'de> for Unique<V> {
        type Value = BTreeMap<String, V>;

        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "a JSON object with unique keys")
        }

        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
            let mut values = BTreeMap::new();
            while let Some((name, value)) = map.next_entry::<String, V>()? {
                if values.insert(name, value).is_some() {
                    return Err(A::Error::custom("duplicate safetensors JSON key"));
                }
            }
            Ok(values)
        }
    }
    deserializer.deserialize_map(Unique(std::marker::PhantomData))
}

pub(super) fn invalid(error: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_duplicate_keys_and_uncovered_data() {
        let tensor = r#"{"dtype":"F16","shape":[1],"data_offsets":[0,2]}"#;
        let read = |json: &str, len: usize| {
            let mut file = (json.len() as u64).to_le_bytes().to_vec();
            file.extend(json.as_bytes());
            file.resize(file.len() + len, 0);
            read_header(&mut file.as_slice(), file.len() as u64)
        };
        assert!(read(&format!(r#"{{"weight":{tensor}}}"#), 2).is_ok());
        assert!(read(&format!(r#"{{"weight":{tensor}}}"#), 4).is_err());
        assert!(read(&format!(r#"{{"weight":{tensor},"weight":{tensor}}}"#), 2).is_err());
        assert!(read(&format!(r#"{{"a":{tensor},"b":{tensor}}}"#), 2).is_err());
        assert!(serde_json::from_str::<Index>(r#"{"weight_map":{"a":"x","a":"y"}}"#).is_err());
    }
}
