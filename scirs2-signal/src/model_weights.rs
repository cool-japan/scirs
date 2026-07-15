//! Pre-trained model weight loading and saving in oxicode format.
//!
//! [`SignalWeightStore`] is a flat key-value map from tensor names to
//! n-dimensional `f64` arrays.  It mirrors the interface of
//! `scirs2_neural::export::weights::WeightStore` but is self-contained
//! within `scirs2-signal` so the signal crate does not depend on
//! `scirs2-neural`.
//!
//! Two on-disk formats are supported:
//!
//! | [`SignalWeightFormat`] | Description |
//! |------------------------|-------------|
//! | `Oxicode`              | Compact binary via the `oxicode` crate (default) |
//! | `Json`                 | Human-readable JSON (suitable for debugging) |
//!
//! # Example
//!
//! ```rust
//! use scirs2_signal::model_weights::{SignalWeightStore, SignalWeightFormat};
//! use scirs2_core::ndarray::Array2;
//!
//! let mut store = SignalWeightStore::new();
//! store.insert("conv.weight", Array2::<f64>::zeros((8, 16)).into_dyn());
//! store.insert("conv.bias",   Array2::<f64>::zeros((1, 8)).into_dyn());
//!
//! let tmp = std::env::temp_dir().join("my_model.ox");
//! store.save(tmp.to_str().unwrap(), SignalWeightFormat::Oxicode)
//!      .expect("save failed");
//! let loaded = SignalWeightStore::load(tmp.to_str().unwrap(), SignalWeightFormat::Oxicode)
//!              .expect("load failed");
//! assert_eq!(loaded.len(), 2);
//! ```

use crate::error::{SignalError, SignalResult};
use oxicode::{config as oxicode_config, serde as oxicode_serde};
use scirs2_core::ndarray::{Array, IxDyn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// Wire-format types
// ---------------------------------------------------------------------------

/// Serialisable representation of a single tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TensorEntry {
    shape: Vec<usize>,
    data: Vec<f64>,
}

impl TensorEntry {
    fn from_array(arr: &Array<f64, IxDyn>) -> Self {
        Self {
            shape: arr.shape().to_vec(),
            data: arr.iter().copied().collect(),
        }
    }

    fn into_array(self) -> SignalResult<Array<f64, IxDyn>> {
        let expected: usize = self.shape.iter().product();
        if self.data.len() != expected {
            return Err(SignalError::ValueError(format!(
                "TensorEntry shape {:?} expects {} elements but data has {}",
                self.shape,
                expected,
                self.data.len()
            )));
        }
        Array::from_shape_vec(IxDyn(self.shape.as_slice()), self.data)
            .map_err(|e| SignalError::ShapeMismatch(format!("from_shape_vec: {e}")))
    }
}

/// Top-level serialisable payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WeightPayload {
    metadata: HashMap<String, String>,
    tensors: HashMap<String, TensorEntry>,
}

// ---------------------------------------------------------------------------
// SignalWeightFormat
// ---------------------------------------------------------------------------

/// Format used by [`SignalWeightStore::save`] / [`SignalWeightStore::load`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalWeightFormat {
    /// Compact binary via `oxicode`.
    Oxicode,
    /// Human-readable JSON.
    Json,
}

// ---------------------------------------------------------------------------
// SignalWeightStore
// ---------------------------------------------------------------------------

/// A flat key-value store mapping tensor names to dynamic `f64` arrays.
///
/// Weights are serialised in `oxicode` format (default) or JSON.
/// The `oxicode` format uses the same encoder as `scirs2-neural`, so files
/// produced by either crate are interchangeable.
#[derive(Debug, Clone, Default)]
pub struct SignalWeightStore {
    weights: HashMap<String, Array<f64, IxDyn>>,
    metadata: HashMap<String, String>,
}

impl SignalWeightStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    // ------------------------------------------------------------------
    // Core accessors
    // ------------------------------------------------------------------

    /// Insert (or overwrite) a tensor under `name`.
    pub fn insert(&mut self, name: impl Into<String>, tensor: Array<f64, IxDyn>) {
        self.weights.insert(name.into(), tensor);
    }

    /// Retrieve a reference to the tensor named `name`, or `None`.
    pub fn get(&self, name: &str) -> Option<&Array<f64, IxDyn>> {
        self.weights.get(name)
    }

    /// Remove and return the tensor named `name`.
    pub fn remove(&mut self, name: &str) -> Option<Array<f64, IxDyn>> {
        self.weights.remove(name)
    }

    /// Return an alphabetically sorted list of all tensor names.
    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.weights.keys().map(|s| s.as_str()).collect();
        names.sort_unstable();
        names
    }

    /// Number of tensors in the store.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// `true` if no tensors are stored.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    // ------------------------------------------------------------------
    // Metadata
    // ------------------------------------------------------------------

    /// Attach an arbitrary key-value metadata pair.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Retrieve a metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }

    // ------------------------------------------------------------------
    // I/O
    // ------------------------------------------------------------------

    /// Persist the store to disk at `path`.
    pub fn save(&self, path: &str, format: SignalWeightFormat) -> SignalResult<()> {
        let payload = WeightPayload {
            metadata: self.metadata.clone(),
            tensors: self
                .weights
                .iter()
                .map(|(k, v)| (k.clone(), TensorEntry::from_array(v)))
                .collect(),
        };

        match format {
            SignalWeightFormat::Json => {
                let json = serde_json::to_string_pretty(&payload)
                    .map_err(|e| SignalError::ComputationError(format!("JSON serialise: {e}")))?;
                std::fs::write(path, json.as_bytes())
                    .map_err(|e| SignalError::ComputationError(format!("write {path}: {e}")))?;
            }
            SignalWeightFormat::Oxicode => {
                let cfg = oxicode_config::standard();
                let bytes = oxicode_serde::encode_to_vec(&payload, cfg)
                    .map_err(|e| SignalError::ComputationError(format!("oxicode encode: {e}")))?;
                std::fs::write(path, &bytes)
                    .map_err(|e| SignalError::ComputationError(format!("write {path}: {e}")))?;
            }
        }
        Ok(())
    }

    /// Load a store from disk at `path`.
    ///
    /// The `format` must match the one used during [`save`](Self::save).
    pub fn load(path: &str, format: SignalWeightFormat) -> SignalResult<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| SignalError::ComputationError(format!("read {path}: {e}")))?;

        let payload: WeightPayload = match format {
            SignalWeightFormat::Json => serde_json::from_slice(&bytes)
                .map_err(|e| SignalError::ComputationError(format!("JSON deserialise: {e}")))?,
            SignalWeightFormat::Oxicode => {
                let cfg = oxicode_config::standard();
                oxicode_serde::decode_owned_from_slice(&bytes, cfg)
                    .map(|(p, _)| p)
                    .map_err(|e| SignalError::ComputationError(format!("oxicode decode: {e}")))?
            }
        };

        let mut weights = HashMap::new();
        for (name, entry) in payload.tensors {
            let arr = entry.into_array()?;
            weights.insert(name, arr);
        }

        Ok(Self {
            weights,
            metadata: payload.metadata,
        })
    }

    // ------------------------------------------------------------------
    // Path-based convenience wrappers
    // ------------------------------------------------------------------

    /// Save to a [`Path`].  Format is inferred from the extension:
    /// `.json` → JSON; anything else → Oxicode.
    pub fn save_to_path(&self, path: &Path) -> SignalResult<()> {
        let format = infer_format(path);
        let path_str = path_to_str(path)?;
        self.save(path_str, format)
    }

    /// Load from a [`Path`].  Format is inferred from the extension.
    pub fn load_from_path(path: &Path) -> SignalResult<Self> {
        let format = infer_format(path);
        let path_str = path_to_str(path)?;
        Self::load(path_str, format)
    }
}

fn infer_format(path: &Path) -> SignalWeightFormat {
    match path.extension().and_then(|e| e.to_str()) {
        Some("json") => SignalWeightFormat::Json,
        _ => SignalWeightFormat::Oxicode,
    }
}

fn path_to_str(path: &Path) -> SignalResult<&str> {
    path.to_str().ok_or_else(|| {
        SignalError::InvalidArgument(format!("path contains non-UTF-8 characters: {path:?}"))
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array, Array1, Array2, IxDyn};

    fn tmp_path(filename: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(filename)
    }

    fn make_store() -> SignalWeightStore {
        let mut store = SignalWeightStore::new();
        store.insert("fc.weight", Array2::<f64>::zeros((4, 8)).into_dyn());
        store.insert("fc.bias", Array1::<f64>::zeros(4).into_dyn());
        store
    }

    // ------------------------------------------------------------------
    // Basic store operations
    // ------------------------------------------------------------------

    #[test]
    fn test_insert_and_get() {
        let mut store = SignalWeightStore::new();
        let arr: Array<f64, IxDyn> = Array::zeros(IxDyn(&[3, 4]));
        store.insert("layer.weight", arr.clone());
        let got = store.get("layer.weight").expect("tensor not found");
        assert_eq!(got.shape(), &[3, 4]);
    }

    #[test]
    fn test_len_and_is_empty() {
        let store = make_store();
        assert_eq!(store.len(), 2);
        assert!(!store.is_empty());

        let empty = SignalWeightStore::new();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn test_names_sorted() {
        let store = make_store();
        let names = store.names();
        assert_eq!(names.len(), 2);
        assert!(names[0] < names[1], "names should be sorted");
    }

    #[test]
    fn test_remove() {
        let mut store = make_store();
        let removed = store.remove("fc.bias");
        assert!(removed.is_some());
        assert_eq!(store.len(), 1);
        assert!(store.remove("no_such_tensor").is_none());
    }

    #[test]
    fn test_metadata() {
        let mut store = SignalWeightStore::new();
        store.set_metadata("arch", "conv_tasnet");
        store.set_metadata("epoch", "42");
        assert_eq!(store.get_metadata("arch"), Some("conv_tasnet"));
        assert_eq!(store.get_metadata("epoch"), Some("42"));
        assert!(store.get_metadata("missing").is_none());
    }

    // ------------------------------------------------------------------
    // Oxicode round-trip
    // ------------------------------------------------------------------

    #[test]
    fn test_save_load_oxicode_roundtrip() {
        let mut store = SignalWeightStore::new();
        store.set_metadata("model", "deep_filter");
        store.insert("enc.weight", Array2::<f64>::eye(4).into_dyn());
        store.insert(
            "dec.bias",
            Array1::<f64>::from_vec(vec![0.1, 0.2, 0.3, 0.4]).into_dyn(),
        );

        let path = tmp_path("scirs2_signal_test_oxicode.ox");
        let path_str = path.to_str().expect("path is UTF-8");

        store
            .save(path_str, SignalWeightFormat::Oxicode)
            .expect("save failed");
        let loaded =
            SignalWeightStore::load(path_str, SignalWeightFormat::Oxicode).expect("load failed");

        assert_eq!(loaded.len(), 2);
        let w = loaded.get("enc.weight").expect("enc.weight missing");
        assert_eq!(w.shape(), &[4, 4]);
        // diagonal should be 1.0
        assert!((w[[0, 0]] - 1.0).abs() < 1e-12);
        assert!(w[[0, 1]].abs() < 1e-12);
        assert_eq!(loaded.get_metadata("model"), Some("deep_filter"));
    }

    // ------------------------------------------------------------------
    // JSON round-trip
    // ------------------------------------------------------------------

    #[test]
    fn test_save_load_json_roundtrip() {
        let store = make_store();
        let path = tmp_path("scirs2_signal_test_weights.json");
        let path_str = path.to_str().expect("path is UTF-8");

        store
            .save(path_str, SignalWeightFormat::Json)
            .expect("save failed");
        let loaded =
            SignalWeightStore::load(path_str, SignalWeightFormat::Json).expect("load failed");

        assert_eq!(loaded.len(), 2);
        let w = loaded.get("fc.weight").expect("fc.weight missing");
        assert_eq!(w.shape(), &[4, 8]);
    }

    // ------------------------------------------------------------------
    // Path-based API (format inferred from extension)
    // ------------------------------------------------------------------

    #[test]
    fn test_save_load_to_path_oxicode() {
        let store = make_store();
        let path = tmp_path("scirs2_signal_test_path.ox");

        store.save_to_path(&path).expect("save_to_path failed");
        let loaded = SignalWeightStore::load_from_path(&path).expect("load_from_path failed");
        assert_eq!(loaded.len(), 2);
    }

    #[test]
    fn test_save_load_to_path_json() {
        let store = make_store();
        let path = tmp_path("scirs2_signal_test_path.json");

        store
            .save_to_path(&path)
            .expect("save_to_path .json failed");
        let loaded = SignalWeightStore::load_from_path(&path).expect("load_from_path .json failed");
        assert_eq!(loaded.len(), 2);
    }

    // ------------------------------------------------------------------
    // Empty store
    // ------------------------------------------------------------------

    #[test]
    fn test_empty_store_roundtrip() {
        let store = SignalWeightStore::new();
        let path = tmp_path("scirs2_signal_test_empty.ox");
        let path_str = path.to_str().expect("UTF-8");

        store
            .save(path_str, SignalWeightFormat::Oxicode)
            .expect("save");
        let loaded = SignalWeightStore::load(path_str, SignalWeightFormat::Oxicode).expect("load");
        assert!(loaded.is_empty());
    }

    // ------------------------------------------------------------------
    // Bad data
    // ------------------------------------------------------------------

    #[test]
    fn test_load_nonexistent_file_returns_error() {
        let path = tmp_path("no_such_file_xyzzy.ox");
        let path_str = path.to_str().expect("UTF-8");
        let result = SignalWeightStore::load(path_str, SignalWeightFormat::Oxicode);
        assert!(result.is_err(), "should fail on missing file");
    }

    #[test]
    fn test_tensor_entry_shape_mismatch() {
        let entry = TensorEntry {
            shape: vec![2, 3],
            data: vec![1.0, 2.0], // only 2 elements, not 6
        };
        let result = entry.into_array();
        assert!(result.is_err());
    }
}
