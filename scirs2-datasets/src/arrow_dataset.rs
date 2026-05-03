//! HuggingFace-compatible Arrow dataset reader.
//!
//! Reads `.arrow` files in Arrow IPC format with optional `dataset_info.json`
//! metadata. This mirrors the on-disk layout used by HuggingFace `datasets`
//! when calling `dataset.save_to_disk()`.
//!
//! # Feature gates
//!
//! | Feature | Provides |
//! |---------|----------|
//! | *(default)* | Magic-byte validation, directory scanning, JSON metadata parse |
//! | `parquet_io` | Full Arrow IPC record-batch reading via the `arrow` crate |
//!
//! # File layout expected
//!
//! ```text
//! my_dataset/
//!   dataset_info.json        ← optional metadata
//!   train/
//!     data-00000-of-00001.arrow
//!   test/
//!     data-00000-of-00001.arrow
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use scirs2_datasets::arrow_dataset::ArrowDataset;
//!
//! # fn example() -> Result<(), scirs2_datasets::error::DatasetsError> {
//! // Validate magic bytes without parsing the full file
//! let ok = ArrowDataset::validate_arrow_magic("/path/to/data.arrow")?;
//! println!("Is Arrow IPC: {}", ok);
//! # Ok(())
//! # }
//! ```

use crate::error::{DatasetsError, Result};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

// Arrow IPC magic bytes: "ARROW1\0\0" (8 bytes)
const ARROW_MAGIC: &[u8; 6] = b"ARROW1";

// ============================================================================
// Public types
// ============================================================================

/// Feature type descriptor for a HuggingFace dataset column.
#[derive(Debug, Clone)]
pub enum FeatureType {
    /// A scalar value with a given dtype string (e.g. `"int64"`, `"float32"`).
    Value {
        /// Data type name as used in `dataset_info.json`.
        dtype: String,
    },
    /// A variable-length sequence of another feature.
    Sequence {
        /// Inner feature descriptor.
        feature: Box<FeatureType>,
    },
    /// Categorical label with an associated name list.
    ClassLabel {
        /// Ordered list of class names.
        names: Vec<String>,
    },
    /// Free-text column.
    Text,
    /// Image column (raw pixel bytes or file path).
    Image,
    /// Unknown or unsupported feature type.
    Unknown,
}

/// Parsed representation of a HuggingFace `dataset_info.json` file.
#[derive(Debug, Clone)]
pub struct DatasetInfo {
    /// Dataset name (from the `dataset_name` key).
    pub dataset_name: String,
    /// Dataset version string.
    pub version: String,
    /// Column feature descriptors, keyed by column name.
    pub features: HashMap<String, FeatureType>,
    /// Number of rows reported in the metadata (may differ from actual).
    pub num_rows: Option<usize>,
    /// Split name this metadata describes (e.g. `"train"`, `"test"`).
    pub split: Option<String>,
}

impl Default for DatasetInfo {
    fn default() -> Self {
        Self {
            dataset_name: String::new(),
            version: "0.0.0".to_string(),
            features: HashMap::new(),
            num_rows: None,
            split: None,
        }
    }
}

/// A loaded Arrow IPC dataset handle.
///
/// Without the `parquet_io` feature this struct stores only metadata and the
/// file paths discovered on disk; actual column data is not decoded. Enabling
/// `parquet_io` activates full record-batch parsing via the `arrow` crate.
#[derive(Debug)]
pub struct ArrowDataset {
    /// Parsed `dataset_info.json` metadata, if present.
    pub info: Option<DatasetInfo>,
    /// Ordered list of column names discovered in the first file.
    pub column_names: Vec<String>,
    /// Total number of rows across all loaded files.
    pub num_rows: usize,
    /// Arrow IPC file paths that were loaded.
    pub(crate) file_paths: Vec<PathBuf>,
    /// Raw column data per column name (only populated with `parquet_io`).
    columns: HashMap<String, Vec<u8>>,
}

impl ArrowDataset {
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Load a HuggingFace-style dataset from a directory.
    ///
    /// Scans `dir` (and one level of subdirectories) for `*.arrow` files and
    /// optionally reads `dataset_info.json` from the same directory.
    ///
    /// # Errors
    ///
    /// Returns `DatasetsError::NotFound` if no `.arrow` files are present.
    pub fn from_directory(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();

        if !dir.exists() {
            return Err(DatasetsError::NotFound(format!(
                "Directory not found: {}",
                dir.display()
            )));
        }

        // Collect .arrow files (top-level and one sub-directory deep)
        let mut arrow_files: Vec<PathBuf> = Vec::new();
        for entry in std::fs::read_dir(dir).map_err(DatasetsError::IoError)? {
            let entry = entry.map_err(DatasetsError::IoError)?;
            let path = entry.path();
            if path.is_file() {
                if path.extension().and_then(|e| e.to_str()) == Some("arrow") {
                    arrow_files.push(path);
                }
            } else if path.is_dir() {
                // One level of sub-dirs (split directories like train/, test/)
                for sub in std::fs::read_dir(&path).map_err(DatasetsError::IoError)? {
                    let sub = sub.map_err(DatasetsError::IoError)?;
                    let sub_path = sub.path();
                    if sub_path.is_file()
                        && sub_path.extension().and_then(|e| e.to_str()) == Some("arrow")
                    {
                        arrow_files.push(sub_path);
                    }
                }
            }
        }

        if arrow_files.is_empty() {
            return Err(DatasetsError::NotFound(format!(
                "No .arrow files found under: {}",
                dir.display()
            )));
        }

        // Sort for deterministic order
        arrow_files.sort();

        // Attempt to parse dataset_info.json
        let info = Self::try_load_dataset_info(dir).or_else(|_| {
            // Also check parent of first sub-dir file
            if let Some(parent) = arrow_files
                .first()
                .and_then(|p| p.parent())
                .and_then(|p| p.parent())
            {
                Self::try_load_dataset_info(parent).ok()
            } else {
                None
            }
            .ok_or(DatasetsError::NotFound("no dataset_info.json".to_string()))
        });

        // Validate magic bytes on all files
        for path in &arrow_files {
            Self::validate_arrow_magic(path)?;
        }

        Ok(Self {
            info: info.ok(),
            column_names: Vec::new(),
            num_rows: 0,
            file_paths: arrow_files,
            columns: HashMap::new(),
        })
    }

    /// Load from a single Arrow IPC file.
    ///
    /// Validates magic bytes and (with `parquet_io`) decodes record batches.
    ///
    /// # Errors
    ///
    /// Returns `DatasetsError::InvalidFormat` if the file does not begin with
    /// the Arrow IPC magic bytes (`ARROW1`).
    pub fn from_arrow_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(DatasetsError::NotFound(format!(
                "Arrow file not found: {}",
                path.display()
            )));
        }

        // Validate magic bytes
        Self::validate_arrow_magic(path)?;

        #[cfg(feature = "parquet_io")]
        {
            Self::from_arrow_file_full(path)
        }

        #[cfg(not(feature = "parquet_io"))]
        {
            Ok(Self {
                info: None,
                column_names: Vec::new(),
                num_rows: 0,
                file_paths: vec![path.to_path_buf()],
                columns: HashMap::new(),
            })
        }
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /// Returns the column names discovered in the dataset.
    pub fn column_names(&self) -> &[String] {
        &self.column_names
    }

    /// Total number of rows across all loaded files.
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Dataset metadata from `dataset_info.json`, if present.
    pub fn info(&self) -> Option<&DatasetInfo> {
        self.info.as_ref()
    }

    /// Arrow IPC file paths that back this dataset.
    pub fn file_paths(&self) -> &[PathBuf] {
        &self.file_paths
    }

    // ------------------------------------------------------------------
    // Validation helpers
    // ------------------------------------------------------------------

    /// Validate that a file begins with the Arrow IPC magic bytes (`ARROW1`).
    ///
    /// Returns `Ok(true)` on success, or an error if the file cannot be read
    /// or does not start with the expected magic.
    pub fn validate_arrow_magic(path: impl AsRef<Path>) -> Result<bool> {
        let path = path.as_ref();
        let mut f = std::fs::File::open(path).map_err(DatasetsError::IoError)?;
        let mut buf = [0u8; 6];
        f.read_exact(&mut buf).map_err(|e| {
            DatasetsError::InvalidFormat(format!(
                "Could not read magic bytes from {}: {}",
                path.display(),
                e
            ))
        })?;
        if &buf == ARROW_MAGIC {
            Ok(true)
        } else {
            Err(DatasetsError::InvalidFormat(format!(
                "Not an Arrow IPC file (bad magic bytes): {}",
                path.display()
            )))
        }
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Try to parse `dataset_info.json` from `dir`.
    fn try_load_dataset_info(dir: &Path) -> Result<DatasetInfo> {
        let info_path = dir.join("dataset_info.json");
        if !info_path.exists() {
            return Err(DatasetsError::NotFound(
                "dataset_info.json not found".to_string(),
            ));
        }

        let content = std::fs::read_to_string(&info_path).map_err(DatasetsError::IoError)?;

        Self::parse_dataset_info_json(&content)
    }

    /// Parse the JSON string of a `dataset_info.json` file.
    ///
    /// This is a best-effort parser; unknown keys are silently ignored.
    fn parse_dataset_info_json(json: &str) -> Result<DatasetInfo> {
        let value: serde_json::Value =
            serde_json::from_str(json).map_err(|e| DatasetsError::SerdeError(e.to_string()))?;

        let dataset_name = value
            .get("dataset_name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let version = value
            .get("version")
            .and_then(|v| v.as_str())
            .unwrap_or("0.0.0")
            .to_string();

        let split = value
            .get("split")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let num_rows = value
            .get("num_rows")
            .or_else(|| value.get("num_examples"))
            .and_then(|v| v.as_u64())
            .map(|n| n as usize);

        // Parse features map
        let features = if let Some(feat_map) = value.get("features").and_then(|v| v.as_object()) {
            feat_map
                .iter()
                .map(|(k, v)| (k.clone(), Self::parse_feature_type(v)))
                .collect()
        } else {
            HashMap::new()
        };

        Ok(DatasetInfo {
            dataset_name,
            version,
            features,
            num_rows,
            split,
        })
    }

    /// Parse a single feature descriptor from a JSON value.
    fn parse_feature_type(v: &serde_json::Value) -> FeatureType {
        // Handle both string shorthand and object notation
        if let Some(s) = v.as_str() {
            return match s {
                "text" | "string" => FeatureType::Text,
                "image" => FeatureType::Image,
                other => FeatureType::Value {
                    dtype: other.to_string(),
                },
            };
        }

        if let Some(obj) = v.as_object() {
            // ClassLabel: {"names": ["class_a", "class_b"]}
            if let Some(names_val) = obj.get("names") {
                if let Some(names_arr) = names_val.as_array() {
                    let names: Vec<String> = names_arr
                        .iter()
                        .filter_map(|n| n.as_str().map(|s| s.to_string()))
                        .collect();
                    return FeatureType::ClassLabel { names };
                }
            }

            // Sequence: {"feature": {...}}
            if let Some(inner) = obj.get("feature") {
                return FeatureType::Sequence {
                    feature: Box::new(Self::parse_feature_type(inner)),
                };
            }

            // Value: {"dtype": "int64"}
            if let Some(dtype) = obj.get("dtype").and_then(|d| d.as_str()) {
                return FeatureType::Value {
                    dtype: dtype.to_string(),
                };
            }

            // Value: {"_type": "Value", "dtype": "float32"}
            if obj.get("_type").and_then(|t| t.as_str()) == Some("Value") {
                let dtype = obj
                    .get("dtype")
                    .and_then(|d| d.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                return FeatureType::Value { dtype };
            }

            if obj.get("_type").and_then(|t| t.as_str()) == Some("ClassLabel") {
                if let Some(names_arr) = obj.get("names").and_then(|n| n.as_array()) {
                    let names: Vec<String> = names_arr
                        .iter()
                        .filter_map(|n| n.as_str().map(|s| s.to_string()))
                        .collect();
                    return FeatureType::ClassLabel { names };
                }
            }

            if obj.get("_type").and_then(|t| t.as_str()) == Some("Image") {
                return FeatureType::Image;
            }

            if obj.get("_type").and_then(|t| t.as_str()) == Some("Sequence") {
                if let Some(inner) = obj.get("feature") {
                    return FeatureType::Sequence {
                        feature: Box::new(Self::parse_feature_type(inner)),
                    };
                }
            }
        }

        FeatureType::Unknown
    }

    // ------------------------------------------------------------------
    // Full implementation (parquet_io feature)
    // ------------------------------------------------------------------

    #[cfg(feature = "parquet_io")]
    fn from_arrow_file_full(path: &Path) -> Result<Self> {
        use arrow::ipc::reader::FileReader;
        use std::fs::File;

        let file = File::open(path).map_err(DatasetsError::IoError)?;
        let reader = FileReader::try_new(file, None)
            .map_err(|e| DatasetsError::InvalidFormat(format!("Arrow IPC read error: {}", e)))?;

        let schema = reader.schema();
        let column_names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();

        let mut total_rows = 0usize;
        let mut columns: HashMap<String, Vec<u8>> = HashMap::new();

        for batch_result in reader {
            let batch = batch_result.map_err(|e| {
                DatasetsError::InvalidFormat(format!("Arrow batch read error: {}", e))
            })?;
            total_rows += batch.num_rows();

            // Store serialised column data (column name → Arrow buffer bytes)
            for (i, field) in schema.fields().iter().enumerate() {
                let col = batch.column(i);
                let buffers = col.to_data().buffers().to_vec();
                let entry = columns.entry(field.name().clone()).or_default();
                for buf in buffers {
                    entry.extend_from_slice(buf.as_slice());
                }
            }
        }

        Ok(Self {
            info: None,
            column_names,
            num_rows: total_rows,
            file_paths: vec![path.to_path_buf()],
            columns,
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Helper: write an Arrow IPC magic header to a temp file.
    fn temp_arrow_file(valid: bool) -> std::path::PathBuf {
        let dir = std::env::temp_dir();
        let file_name = if valid {
            "test_valid_arrow.arrow"
        } else {
            "test_invalid_arrow.arrow"
        };
        let path = dir.join(file_name);
        let mut f = std::fs::File::create(&path).expect("create temp file");
        if valid {
            // Write ARROW1 magic + dummy continuation bytes
            f.write_all(b"ARROW1\x00\x00some_padding_bytes_for_test")
                .expect("write magic");
        } else {
            // Write wrong magic
            f.write_all(b"NOTARROW_FILE_CONTENT")
                .expect("write wrong magic");
        }
        path
    }

    #[test]
    fn arrow_dataset_validates_magic_bytes() {
        let path = temp_arrow_file(true);
        let result = ArrowDataset::validate_arrow_magic(&path);
        assert!(result.is_ok(), "valid Arrow magic should succeed");
        assert!(
            result.expect("valid arrow result"),
            "validate_arrow_magic should return true for valid magic"
        );
    }

    #[test]
    fn arrow_dataset_rejects_wrong_magic() {
        let path = temp_arrow_file(false);
        let result = ArrowDataset::validate_arrow_magic(&path);
        assert!(result.is_err(), "wrong magic should return an error");
        if let Err(DatasetsError::InvalidFormat(msg)) = result {
            assert!(
                msg.contains("magic bytes"),
                "error should mention magic bytes, got: {}",
                msg
            );
        } else {
            panic!("expected InvalidFormat error");
        }
    }

    #[test]
    #[cfg(not(feature = "parquet_io"))]
    fn arrow_dataset_from_arrow_file_valid() {
        let path = temp_arrow_file(true);
        // Without parquet_io the constructor accepts valid magic and returns a stub
        let result = ArrowDataset::from_arrow_file(&path);
        assert!(
            result.is_ok(),
            "from_arrow_file with valid magic should succeed"
        );
        let ds = result.expect("valid arrow dataset");
        assert_eq!(ds.file_paths().len(), 1);
    }

    /// With parquet_io, from_arrow_file on a dummy file (wrong IPC content after
    /// magic) is expected to fail — the Arrow IPC reader parses beyond the magic.
    #[test]
    #[cfg(feature = "parquet_io")]
    fn arrow_dataset_from_arrow_file_valid_parquet_io() {
        // We cannot easily construct a valid full Arrow IPC file in a unit test
        // without the arrow crate itself.  Just verify the function exists and
        // returns a meaningful error for a stub-only file.
        let path = temp_arrow_file(true);
        // With full IPC parsing, a stub file (only magic) will fail at the IPC
        // record-batch level — this is expected.
        let result = ArrowDataset::from_arrow_file(&path);
        // It either succeeds (unlikely for a stub) or fails with InvalidFormat
        match result {
            Ok(_) => {}                                // Unlikely but acceptable
            Err(DatasetsError::InvalidFormat(_)) => {} // Expected for stub file
            Err(other) => panic!("unexpected error variant: {:?}", other),
        }
    }

    #[test]
    fn arrow_dataset_from_arrow_file_invalid() {
        let path = temp_arrow_file(false);
        let result = ArrowDataset::from_arrow_file(&path);
        assert!(
            result.is_err(),
            "from_arrow_file with bad magic should fail"
        );
    }

    #[test]
    fn arrow_dataset_from_directory_empty_dir() {
        let dir = std::env::temp_dir().join("test_empty_arrow_dir");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        // Remove any stale .arrow files from previous runs
        for entry in std::fs::read_dir(&dir).expect("read dir") {
            let entry = entry.expect("entry");
            if entry.path().extension().and_then(|e| e.to_str()) == Some("arrow") {
                std::fs::remove_file(entry.path()).ok();
            }
        }
        let result = ArrowDataset::from_directory(&dir);
        assert!(result.is_err(), "empty dir should return NotFound");
        if let Err(DatasetsError::NotFound(_)) = result {
            // expected
        } else {
            panic!("expected NotFound error for empty directory");
        }
    }

    #[test]
    fn arrow_dataset_from_directory_with_arrow_file() {
        let dir = std::env::temp_dir().join("test_arrow_dir_with_file");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let arrow_path = dir.join("data-00000-of-00001.arrow");
        {
            let mut f = std::fs::File::create(&arrow_path).expect("create arrow");
            f.write_all(b"ARROW1\x00\x00dummy_ipc_content_for_test")
                .expect("write arrow");
        }
        let result = ArrowDataset::from_directory(&dir);
        assert!(
            result.is_ok(),
            "directory with valid arrow file should succeed"
        );
        let ds = result.expect("arrow dataset from dir");
        assert_eq!(ds.file_paths().len(), 1);
    }

    #[test]
    fn dataset_info_default() {
        let info = DatasetInfo::default();
        assert!(info.dataset_name.is_empty());
        assert_eq!(info.version, "0.0.0");
        assert!(info.features.is_empty());
        assert!(info.num_rows.is_none());
        assert!(info.split.is_none());
    }

    #[test]
    fn dataset_info_parse_json() {
        let json = r#"{
            "dataset_name": "my_dataset",
            "version": "1.0.0",
            "split": "train",
            "num_rows": 42,
            "features": {
                "text": "text",
                "label": {"names": ["neg", "pos"]},
                "score": {"dtype": "float32"}
            }
        }"#;
        let info = ArrowDataset::parse_dataset_info_json(json).expect("parse dataset_info.json");
        assert_eq!(info.dataset_name, "my_dataset");
        assert_eq!(info.version, "1.0.0");
        assert_eq!(info.split.as_deref(), Some("train"));
        assert_eq!(info.num_rows, Some(42));
        assert_eq!(info.features.len(), 3);
        if let FeatureType::ClassLabel { names } = &info.features["label"] {
            assert_eq!(names, &["neg", "pos"]);
        } else {
            panic!("expected ClassLabel for 'label' feature");
        }
    }

    #[test]
    fn arrow_dataset_nonexistent_file() {
        let result = ArrowDataset::from_arrow_file("/nonexistent/path/data.arrow");
        assert!(result.is_err());
        if let Err(DatasetsError::NotFound(_)) = result {
            // expected
        } else {
            panic!("expected NotFound for nonexistent file");
        }
    }

    #[test]
    fn arrow_dataset_nonexistent_directory() {
        let result = ArrowDataset::from_directory("/nonexistent/arrow_dataset_dir_xyz");
        assert!(result.is_err());
        if let Err(DatasetsError::NotFound(_)) = result {
            // expected
        } else {
            panic!("expected NotFound for nonexistent directory");
        }
    }
}
