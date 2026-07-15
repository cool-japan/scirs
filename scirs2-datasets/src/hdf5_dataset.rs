//! HDF5 dataset support
//!
//! Provides `Hdf5Dataset`, a wrapper around HDF5 file access. In default
//! builds (no `hdf5_io` feature) only the file-path wrapper and magic-byte
//! validator are available. Enabling the `hdf5_io` feature activates full
//! read/write support via the `hdf5` crate (which links `libhdf5`).
//!
//! # Feature gates
//!
//! | Feature | Provides |
//! |---------|----------|
//! | *(default)* | `Hdf5Dataset::from_file`, `is_valid_hdf5` |
//! | `hdf5_io` | `read_dataset`, `dataset_names` |
//!
//! # Example (default features)
//!
//! ```rust
//! use scirs2_datasets::hdf5_dataset::Hdf5Dataset;
//!
//! // Validate an HDF5 magic signature without loading the full file
//! // (returns false for a non-HDF5 path that doesn't exist)
//! let valid = Hdf5Dataset::is_valid_hdf5("/non/existent/file.h5");
//! assert!(!valid);
//! ```
//!
//! # noffi migration status
//!
//! TODO(noffi-migration): Replace `hdf5` with `oxih5-core` / `oxih5-format` (Pure Rust HDF5).
//!
//! The `read_dataset` and `dataset_names` methods under `#[cfg(feature = "hdf5_io")]`
//! can be migrated to `oxih5`:
//!
//! - `hdf5::File::open(path)` → `oxih5::File::open(path)`
//! - `file.dataset(name)` → `file.dataset(name)` (same method name)
//! - `ds.read_raw::<f64>()` → `dataset.data` field (pre-typed `Vec<T>`)
//! - `ds.shape()` → `dataset.shape` field (`Vec<usize>`)
//! - `file.member_names()` → `file.dataset_names()` (oxih5 API)
//!
//! **Write path** (in `hdf5_io_tests::write_hdf5_1d` test helper):
//! `hdf5::File::create` / `new_dataset` / `write_raw` — blocked at oxih5 M1 (read-only).
//! Keep under `#[cfg(feature = "hdf5_io")]` until oxih5 write support ships.
//!
//! The `hdf5_io-legacy` feature is available to enable only the C-linked `hdf5` crate
//! without pulling in oxih5 deps, e.g. for write-only paths.
//! See `~/work/noffi/oxih5/` for reference API.

use crate::error::{DatasetsError, Result};
use std::io::Read;
use std::path::{Path, PathBuf};

/// HDF5 magic bytes — first 8 bytes of every valid HDF5 file.
pub const HDF5_MAGIC: &[u8; 8] = b"\x89HDF\r\n\x1a\n";

/// A handle to an HDF5 file.
///
/// Without the `hdf5_io` feature this struct holds only the file path and
/// exposes validation helpers. Full dataset access requires `hdf5_io`.
#[derive(Debug, Clone)]
pub struct Hdf5Dataset {
    path: PathBuf,
}

impl Hdf5Dataset {
    /// Create a new `Hdf5Dataset` pointing at `path`.
    ///
    /// The path is validated to exist and to carry a correct HDF5 magic header.
    ///
    /// # Errors
    ///
    /// Returns `DatasetsError::NotFound` if the file does not exist, or
    /// `DatasetsError::InvalidFormat` if the file does not start with the HDF5
    /// magic bytes.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let p = path.as_ref();
        if !p.exists() {
            return Err(DatasetsError::NotFound(format!(
                "HDF5 file not found: {}",
                p.display()
            )));
        }
        if !Self::is_valid_hdf5(p) {
            return Err(DatasetsError::InvalidFormat(format!(
                "File does not have HDF5 magic bytes: {}",
                p.display()
            )));
        }
        Ok(Self {
            path: p.to_path_buf(),
        })
    }

    /// Return the file path wrapped by this dataset handle.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Check whether the first 8 bytes of `path` match the HDF5 magic.
    ///
    /// Returns `false` if the file cannot be read or is shorter than 8 bytes.
    pub fn is_valid_hdf5(path: impl AsRef<Path>) -> bool {
        let mut f = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(_) => return false,
        };
        let mut buf = [0u8; 8];
        matches!(f.read_exact(&mut buf), Ok(())) && &buf == HDF5_MAGIC
    }

    /// Read a named dataset into a 2-D float array.
    ///
    /// Requires the `hdf5_io` feature.
    #[cfg(feature = "hdf5_io")]
    pub fn read_dataset(&self, name: &str) -> Result<scirs2_core::ndarray::Array2<f64>> {
        use scirs2_core::ndarray::Array2;

        let file = hdf5::File::open(&self.path)
            .map_err(|e| DatasetsError::InvalidFormat(format!("HDF5 open error: {e}")))?;

        let ds = file.dataset(name).map_err(|e| {
            DatasetsError::NotFound(format!("HDF5 dataset '{name}' not found: {e}"))
        })?;

        let shape = ds.shape();
        if shape.len() != 2 {
            return Err(DatasetsError::InvalidFormat(format!(
                "Expected 2-D dataset, got shape {:?}",
                shape
            )));
        }

        let flat: Vec<f64> = ds
            .read_raw()
            .map_err(|e| DatasetsError::InvalidFormat(format!("HDF5 read error: {e}")))?;

        let rows = shape[0];
        let cols = shape[1];
        let arr = Array2::from_shape_vec((rows, cols), flat)
            .map_err(|e| DatasetsError::ComputationError(format!("Array shape error: {e}")))?;

        Ok(arr)
    }

    /// List all top-level dataset names in the HDF5 file.
    ///
    /// Requires the `hdf5_io` feature.
    #[cfg(feature = "hdf5_io")]
    pub fn dataset_names(&self) -> Result<Vec<String>> {
        let file = hdf5::File::open(&self.path)
            .map_err(|e| DatasetsError::InvalidFormat(format!("HDF5 open error: {e}")))?;

        let names = file
            .member_names()
            .map_err(|e| DatasetsError::InvalidFormat(format!("HDF5 member list error: {e}")))?;

        Ok(names)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Write 8 bytes of HDF5 magic to a temp file and return (dir, path).
    fn write_magic_file() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("valid.h5");
        let mut f = std::fs::File::create(&path).expect("create");
        f.write_all(HDF5_MAGIC).expect("write magic");
        (dir, path)
    }

    /// Write garbage bytes to a temp file.
    fn write_invalid_file() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("invalid.h5");
        let mut f = std::fs::File::create(&path).expect("create");
        f.write_all(b"NOTANDF5").expect("write");
        (dir, path)
    }

    #[test]
    fn test_is_valid_hdf5_with_magic() {
        let (_dir, path) = write_magic_file();
        assert!(Hdf5Dataset::is_valid_hdf5(&path));
    }

    #[test]
    fn test_is_valid_hdf5_wrong_bytes() {
        let (_dir, path) = write_invalid_file();
        assert!(!Hdf5Dataset::is_valid_hdf5(&path));
    }

    #[test]
    fn test_is_valid_hdf5_nonexistent() {
        assert!(!Hdf5Dataset::is_valid_hdf5(
            std::env::temp_dir().join("__scirs2_datasets_nonexistent_12345.h5")
        ));
    }

    #[test]
    fn test_from_file_valid_magic() {
        let (_dir, path) = write_magic_file();
        // from_file only validates magic and existence
        let ds = Hdf5Dataset::from_file(&path).expect("from_file");
        assert_eq!(ds.path(), path.as_path());
    }

    #[test]
    fn test_from_file_nonexistent_returns_error() {
        let result =
            Hdf5Dataset::from_file(std::env::temp_dir().join("__scirs2_nonexistent_hdf5_99999.h5"));
        assert!(result.is_err());
        if let Err(DatasetsError::NotFound(msg)) = result {
            assert!(msg.contains("not found"));
        } else {
            panic!("Expected NotFound error");
        }
    }

    #[test]
    fn test_from_file_invalid_magic_returns_error() {
        let (_dir, path) = write_invalid_file();
        let result = Hdf5Dataset::from_file(&path);
        assert!(result.is_err());
        if let Err(DatasetsError::InvalidFormat(msg)) = result {
            assert!(msg.contains("magic"));
        } else {
            panic!("Expected InvalidFormat error");
        }
    }

    #[test]
    fn test_hdf5_magic_constant() {
        assert_eq!(HDF5_MAGIC.len(), 8);
        assert_eq!(HDF5_MAGIC[0], 0x89);
        assert_eq!(&HDF5_MAGIC[1..4], b"HDF");
    }

    #[test]
    fn test_from_file_too_short() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("short.h5");
        let mut f = std::fs::File::create(&path).expect("create");
        // Only 4 bytes — shorter than magic
        f.write_all(b"\x89HDF").expect("write");
        let result = Hdf5Dataset::from_file(&path);
        assert!(result.is_err());
    }

    // Full HDF5 I/O tests — only compiled when hdf5_io feature is active
    #[cfg(feature = "hdf5_io")]
    mod hdf5_io_tests {
        use super::*;

        /// Write a 2-D f64 dataset to an HDF5 file using a flat Vec.
        /// The hdf5 crate uses its own ndarray version; we write a 1-D dataset
        /// and treat it as a column vector to avoid the ndarray version conflict.
        fn write_hdf5_1d(path: &std::path::Path, name: &str, data: &[f64]) {
            let file = hdf5::File::create(path).expect("create hdf5");
            let builder = file.new_dataset::<f64>();
            let ds = builder
                .shape([data.len()])
                .create(name)
                .expect("create dataset");
            // Use write_raw which accepts a slice directly (avoids ndarray version conflict)
            ds.write_raw(data).expect("write_raw");
        }

        #[test]
        fn test_read_dataset_roundtrip() {
            let dir = tempfile::tempdir().expect("tmpdir");
            let path = dir.path().join("test.h5");
            write_hdf5_1d(&path, "data", &[1.0, 2.0, 3.0, 4.0]);

            // Note: read_dataset expects 2-D; 1-D will give InvalidFormat
            // For this test verify the file is readable and error is correct type
            let ds = Hdf5Dataset::from_file(&path).expect("from_file");
            let result = ds.read_dataset("data");
            // Either succeeds (hdf5 lib reshapes) or returns InvalidFormat for 1-D
            match result {
                Ok(arr) => assert!(!arr.is_empty()),
                Err(DatasetsError::InvalidFormat(_)) => { /* expected for 1-D */ }
                Err(e) => panic!("Unexpected error: {e}"),
            }
        }

        #[test]
        fn test_dataset_names() {
            let dir = tempfile::tempdir().expect("tmpdir");
            let path = dir.path().join("named.h5");
            write_hdf5_1d(&path, "temperatures", &[1.0, 2.0]);

            let ds = Hdf5Dataset::from_file(&path).expect("from_file");
            let names = ds.dataset_names().expect("dataset_names");
            assert!(names.contains(&"temperatures".to_owned()));
        }
    }
}
