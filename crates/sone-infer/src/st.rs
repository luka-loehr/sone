//! Safetensors in and out, without a crate: an 8-byte header length, a JSON
//! header, then raw little-endian data. A loaded network keeps the file's
//! bytes and reads its matrices in place (F16 or F32); each matrix is widened
//! into a scratch buffer right before its GEMM, so no f32 copy of the model
//! is ever made (wasm memory never shrinks, so the peak is what counts).

use std::collections::{BTreeMap, HashMap};
use std::ops::Deref;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail, ensure};

use crate::ops::{f16_to_f32, f32_to_f16};

#[derive(Clone)]
enum Bytes {
    Static(&'static [u8]),
    Shared(Arc<[u8]>),
}

impl Deref for Bytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Self::Static(b) => b,
            Self::Shared(b) => b,
        }
    }
}

#[derive(Debug, Clone)]
struct Entry {
    shape: Vec<usize>,
    f16: bool,
    /// Byte range inside the file.
    start: usize,
    end: usize,
}

/// A parsed weights file. Tensors are looked up by name; every loader takes
/// each tensor once and [`WeightFile::finish`] rejects files with leftovers,
/// so a naming mismatch between exporter and reader cannot pass silently.
pub struct WeightFile {
    bytes: Bytes,
    tensors: HashMap<String, Entry>,
    /// `__metadata__` of the file (the exporter writes `model` and `config`).
    pub metadata: BTreeMap<String, String>,
}

/// A matrix `(rows, cols)` inside the weights file, row-major.
#[derive(Debug, Clone, Copy)]
pub struct Mat {
    start: usize,
    f16: bool,
    pub rows: usize,
    pub cols: usize,
}

impl Mat {
    /// Widen into `out` (cleared first) and return it.
    pub fn load<'a>(&self, file: &WeightFile, out: &'a mut Vec<f32>) -> &'a [f32] {
        let n = self.rows * self.cols;
        out.clear();
        if self.f16 {
            let raw = &file.bytes[self.start..self.start + 2 * n];
            out.extend(raw.as_chunks::<2>().0.iter().map(|&b| f16_to_f32(u16::from_le_bytes(b))));
        } else {
            let raw = &file.bytes[self.start..self.start + 4 * n];
            out.extend(raw.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)));
        }
        out
    }
}

impl WeightFile {
    pub fn from_vec(bytes: Vec<u8>) -> Result<Self> {
        Self::parse(Bytes::Shared(Arc::from(bytes)))
    }

    /// Weights compiled into the binary (`include_bytes!`): read in place.
    pub fn from_static(bytes: &'static [u8]) -> Result<Self> {
        Self::parse(Bytes::Static(bytes))
    }

    fn parse(bytes: Bytes) -> Result<Self> {
        ensure!(bytes.len() >= 8, "weights: truncated");
        let header_len = u64::from_le_bytes(bytes[..8].try_into()?) as usize;
        let data_start = 8 + header_len;
        ensure!(bytes.len() >= data_start, "weights: truncated header");
        let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&bytes[8..data_start])?;
        let mut tensors = HashMap::new();
        let mut metadata = BTreeMap::new();
        for (name, info) in header {
            if name == "__metadata__" {
                if let Some(map) = info.as_object() {
                    for (k, v) in map {
                        metadata.insert(k.clone(), v.as_str().unwrap_or_default().to_string());
                    }
                }
                continue;
            }
            let dtype = info["dtype"].as_str().unwrap_or_default();
            let shape: Vec<usize> = info["shape"]
                .as_array()
                .ok_or_else(|| anyhow!("weights: {name} has no shape"))?
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect();
            let offsets = info["data_offsets"].as_array().ok_or_else(|| anyhow!("weights: {name} has no offsets"))?;
            let (start, end) = (offsets[0].as_u64().unwrap_or(0) as usize, offsets[1].as_u64().unwrap_or(0) as usize);
            let f16 = match dtype {
                "F32" => false,
                "F16" => true,
                other => bail!("weights: {name} has dtype {other}"),
            };
            let size = if f16 { 2 } else { 4 };
            ensure!(start <= end && data_start + end <= bytes.len(), "weights: {name} out of range");
            ensure!(end - start == size * shape.iter().product::<usize>(), "weights: {name} size mismatch");
            tensors.insert(name, Entry { shape, f16, start: data_start + start, end: data_start + end });
        }
        Ok(Self { bytes, tensors, metadata })
    }

    /// The `config` metadata entry, parsed.
    pub fn config<T: serde::de::DeserializeOwned>(&self, model: &str) -> Result<T> {
        self.config_any(&[model])
    }

    /// The `config` metadata entry, accepting any of the given model identifiers.
    pub fn config_any<T: serde::de::DeserializeOwned>(&self, models: &[&str]) -> Result<T> {
        let found = self.metadata.get("model").map(String::as_str).unwrap_or("?");
        ensure!(models.contains(&found), "weights: file holds a {found} network, expected one of {models:?}");
        let text = self.metadata.get("config").context("weights: no config in metadata")?;
        Ok(serde_json::from_str(text)?)
    }

    fn take(&mut self, name: &str, shape: &[usize]) -> Result<Entry> {
        let e = self.tensors.remove(name).with_context(|| format!("weights: {name} missing"))?;
        ensure!(e.shape == shape, "weights: {name} has shape {:?}, expected {:?}", e.shape, shape);
        Ok(e)
    }

    /// A small tensor (norm gains, biases, embeddings) as an owned f32 vector.
    pub fn vec(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let e = self.take(name, shape)?;
        let raw = &self.bytes[e.start..e.end];
        Ok(if e.f16 {
            raw.as_chunks::<2>().0.iter().map(|&b| f16_to_f32(u16::from_le_bytes(b))).collect()
        } else {
            raw.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect()
        })
    }

    /// A `(rows, cols)` matrix, left in the file.
    pub fn mat(&mut self, name: &str, rows: usize, cols: usize) -> Result<Mat> {
        let e = self.take(name, &[rows, cols])?;
        Ok(Mat { start: e.start, f16: e.f16, rows, cols })
    }

    /// Fails when the file holds tensors no loader asked for.
    pub fn finish(&self) -> Result<()> {
        if !self.tensors.is_empty() {
            let mut left: Vec<&String> = self.tensors.keys().collect();
            left.sort();
            bail!("weights: {} unexpected tensors, e.g. {:?}", left.len(), &left[..left.len().min(5)]);
        }
        Ok(())
    }

    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }

    /// Split a unified `sone` WeightFile zero-copy into one WeightFile per organ
    /// (`cochlea.*`, `tympanum.*`, `cortex.*`, `cingulate.*`), rejecting any leftover tensors.
    pub fn split_bundle(self, stages: &[&str]) -> Result<BTreeMap<String, Self>> {
        let found = self.metadata.get("model").map(String::as_str).unwrap_or("?");
        ensure!(
            found == "sone" || found == "bundle",
            "weights: file holds a {found} network, not unified sone bundle"
        );
        let fmt = self
            .metadata
            .get("format")
            .cloned()
            .unwrap_or_else(|| "sone-infer/1".to_string());

        let mut maps: BTreeMap<String, HashMap<String, Entry>> = stages
            .iter()
            .map(|&s| (s.to_string(), HashMap::new()))
            .collect();

        for (full_name, entry) in self.tensors {
            let mut matched = false;
            for &st in stages {
                if let Some(rest) = full_name.strip_prefix(st).and_then(|r| r.strip_prefix('.')) {
                    maps.get_mut(st).unwrap().insert(rest.to_string(), entry.clone());
                    matched = true;
                    break;
                }
            }
            ensure!(matched, "weights: unexpected tensor {full_name} in bundle");
        }

        let mut out = BTreeMap::new();
        for &st in stages {
            let cfg = self
                .metadata
                .get(&format!("config.{st}"))
                .with_context(|| format!("weights: missing config.{st} in bundle metadata"))?
                .clone();
            let mut meta = BTreeMap::new();
            meta.insert("model".to_string(), st.to_string());
            meta.insert("config".to_string(), cfg);
            meta.insert("format".to_string(), fmt.clone());
            out.insert(
                st.to_string(),
                Self {
                    bytes: self.bytes.clone(),
                    tensors: maps.remove(st).unwrap(),
                    metadata: meta,
                },
            );
        }
        Ok(out)
    }
}

/// Losslessly pack multiple single-organ `.safetensors` buffers `(organ, bytes)` into one
/// unified 8-byte-aligned `sone` `.safetensors` bundle without re-encoding raw FP16/FP32 values.
pub fn merge_bundle(parts: &[(&str, &[u8])], extra_meta: &BTreeMap<String, String>) -> Result<Vec<u8>> {
    let mut meta = extra_meta.clone();
    meta.insert("model".to_string(), "sone".to_string());
    meta.insert("format".to_string(), "sone-infer/1".to_string());

    struct RawSlice<'a> {
        name: String,
        dtype: &'static str,
        shape: Vec<usize>,
        raw: &'a [u8],
    }

    let legacy_alias = |stage: &str| -> &'static [&'static str] {
        match stage {
            "cochlea" => &["cochlea", "stems"],
            "tympanum" => &["tympanum", "ear"],
            "cortex" => &["cortex", "plan"],
            "cingulate" => &["cingulate", "critic"],
            _ => &[],
        }
    };

    let mut slices: Vec<RawSlice<'_>> = Vec::new();
    for &(stage, raw_file) in parts {
        let wf = WeightFile::from_vec(raw_file.to_vec())
            .with_context(|| format!("parsing organ {stage}"))?;
        let found = wf.metadata.get("model").map(String::as_str).unwrap_or("?");
        let allowed = legacy_alias(stage);
        ensure!(
            found == stage || allowed.contains(&found),
            "merge_bundle: expected model={stage}, got {found}"
        );
        let mut cfg = wf
            .metadata
            .get("config")
            .with_context(|| format!("merge_bundle: organ {stage} missing config"))?
            .clone();
        if stage == "cortex" {
            cfg = cfg.replace("\"kind\":\"plan\"", "\"kind\":\"cortex\"");
        } else if stage == "cingulate" {
            cfg = cfg.replace("\"kind\":\"critic\"", "\"kind\":\"cingulate\"");
        }
        meta.insert(format!("config.{stage}"), cfg);

        for (tname, entry) in wf.tensors {
            slices.push(RawSlice {
                name: format!("{stage}.{tname}"),
                dtype: if entry.f16 { "F16" } else { "F32" },
                shape: entry.shape,
                raw: &raw_file[entry.start..entry.end],
            });
        }
    }

    slices.sort_by(|a, b| a.name.cmp(&b.name));
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".into(), serde_json::to_value(&meta)?);
    let mut data = Vec::new();
    for s in slices {
        ensure!(!header.contains_key(&s.name), "duplicate tensor {}", s.name);
        let start = data.len();
        data.extend_from_slice(s.raw);
        header.insert(
            s.name,
            serde_json::json!({
                "dtype": s.dtype,
                "shape": s.shape,
                "data_offsets": [start, data.len()],
            }),
        );
    }
    let mut head = serde_json::to_vec(&header)?;
    while (8 + head.len()) % 8 != 0 {
        head.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + head.len() + data.len());
    out.extend_from_slice(&(head.len() as u64).to_le_bytes());
    out.extend_from_slice(&head);
    out.extend_from_slice(&data);
    Ok(out)
}

/// One tensor to write: name, shape, values.
pub type NamedTensor = (String, Vec<usize>, Vec<f32>);

/// Serialise tensors as safetensors, F16 (`half`) or F32, sorted by name.
pub fn write(tensors: &[NamedTensor], half: bool, metadata: &BTreeMap<String, String>) -> Result<Vec<u8>> {
    let mut sorted: Vec<&NamedTensor> = tensors.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    let mut header = serde_json::Map::new();
    if !metadata.is_empty() {
        header.insert("__metadata__".into(), serde_json::to_value(metadata)?);
    }
    let mut data = Vec::new();
    for (name, shape, values) in sorted {
        ensure!(values.len() == shape.iter().product::<usize>(), "{name}: {} values for shape {shape:?}", values.len());
        ensure!(!header.contains_key(name), "{name} twice");
        let start = data.len();
        if half {
            for &v in values {
                data.extend_from_slice(&f32_to_f16(v).to_le_bytes());
            }
        } else {
            for &v in values {
                data.extend_from_slice(&v.to_le_bytes());
            }
        }
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": if half { "F16" } else { "F32" }, "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let mut head = serde_json::to_vec(&header)?;
    // The data section starts 8-byte aligned (the format asks for it).
    while (8 + head.len()) % 8 != 0 {
        head.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + head.len() + data.len());
    out.extend_from_slice(&(head.len() as u64).to_le_bytes());
    out.extend_from_slice(&head);
    out.extend_from_slice(&data);
    Ok(out)
}

/// Metadata every exported network carries.
pub fn metadata(model: &str, config: &impl serde::Serialize) -> Result<BTreeMap<String, String>> {
    let mut m = BTreeMap::new();
    m.insert("model".to_string(), model.to_string());
    m.insert("config".to_string(), serde_json::to_string(config)?);
    m.insert("format".to_string(), "sone-infer/1".to_string());
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_f32_and_f16() {
        let tensors = vec![
            ("b".to_string(), vec![3], vec![1.0, -2.5, 3.25]),
            ("a.w".to_string(), vec![2, 2], vec![0.5, 0.25, -0.125, 1e-3]),
        ];
        let meta = metadata("test", &serde_json::json!({"x": 1})).unwrap();
        for half in [false, true] {
            let bytes = write(&tensors, half, &meta).unwrap();
            let mut f = WeightFile::from_vec(bytes).unwrap();
            assert_eq!(f.metadata["model"], "test");
            let m = f.mat("a.w", 2, 2).unwrap();
            let mut buf = Vec::new();
            let w = m.load(&f, &mut buf).to_vec();
            assert!((w[3] - 1e-3).abs() < if half { 1e-6 } else { 0.0 } + 1e-9);
            assert_eq!(f.vec("b", &[3]).unwrap(), vec![1.0, -2.5, 3.25]);
            f.finish().unwrap();
        }
    }

    #[test]
    fn wrong_shape_and_leftovers_fail() {
        let tensors = vec![("x".to_string(), vec![2], vec![1.0, 2.0]), ("y".to_string(), vec![1], vec![0.0])];
        let bytes = write(&tensors, false, &BTreeMap::new()).unwrap();
        let mut f = WeightFile::from_vec(bytes).unwrap();
        assert!(f.vec("x", &[3]).is_err());
        assert!(f.finish().is_err());
    }
}
