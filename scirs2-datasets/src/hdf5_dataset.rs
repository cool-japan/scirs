//! HDF5 dataset support
//!
//! Provides `Hdf5Dataset`, a wrapper around HDF5 file access. In default
//! builds (no `hdf5_io` feature) only the file-path wrapper and magic-byte
//! validator are available. Enabling the `hdf5_io` feature activates read
//! support via `oxih5`, a pure-Rust HDF5 implementation (COOLJAPAN Pure Rust
//! Policy — no `libhdf5`, no C toolchain).
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
    /// Any numeric element type is accepted and widened to `f64`, matching the
    /// implicit conversion the previous C-linked backend performed.
    ///
    /// Requires the `hdf5_io` feature.
    #[cfg(feature = "hdf5_io")]
    pub fn read_dataset(&self, name: &str) -> Result<scirs2_core::ndarray::Array2<f64>> {
        use scirs2_core::ndarray::Array2;

        let file = oxih5::File::open(&self.path)
            .map_err(|e| DatasetsError::InvalidFormat(format!("HDF5 open error: {e}")))?;

        let ds = file.dataset(name).map_err(|e| {
            DatasetsError::NotFound(format!("HDF5 dataset '{name}' not found: {e}"))
        })?;

        if ds.shape.len() != 2 {
            return Err(DatasetsError::InvalidFormat(format!(
                "Expected 2-D dataset, got shape {:?}",
                ds.shape
            )));
        }

        let flat = Self::widen_to_f64(&ds)?;

        let rows = ds.shape[0];
        let cols = ds.shape[1];
        let arr = Array2::from_shape_vec((rows, cols), flat)
            .map_err(|e| DatasetsError::ComputationError(format!("Array shape error: {e}")))?;

        Ok(arr)
    }

    /// Decode any fixed-width numeric dataset into `Vec<f64>`.
    ///
    /// oxih5's accessors each match exactly one datatype, so the widening the C
    /// crate did inside `read_raw::<f64>()` has to be spelled out. Without this
    /// the reader would silently accept only datasets already stored as f64.
    #[cfg(feature = "hdf5_io")]
    fn widen_to_f64(ds: &oxih5::Dataset) -> Result<Vec<f64>> {
        use oxih5::Dtype;

        let decode = |what: &str, e: oxih5::OxiH5Error| {
            DatasetsError::InvalidFormat(format!("HDF5 {what} read error: {e}"))
        };
        match &ds.dtype {
            Dtype::Float { size: 8, .. } => ds.as_f64().map_err(|e| decode("f64", e)),
            Dtype::Float { size: 4, .. } => Ok(ds
                .as_f32()
                .map_err(|e| decode("f32", e))?
                .into_iter()
                .map(f64::from)
                .collect()),
            Dtype::Float { size: 2, .. } => Ok(ds
                .as_f16()
                .map_err(|e| decode("f16", e))?
                .into_iter()
                .map(f64::from)
                .collect()),
            Dtype::Int {
                size: 1,
                signed: true,
                ..
            } => Ok(ds
                .as_i8()
                .map_err(|e| decode("i8", e))?
                .into_iter()
                .map(f64::from)
                .collect()),
            Dtype::Int {
                size: 2,
                signed: true,
                ..
            } => Ok(ds
                .as_i16()
                .map_err(|e| decode("i16", e))?
                .into_iter()
                .map(f64::from)
                .collect()),
            Dtype::Int {
                size: 4,
                signed: true,
                ..
            } => Ok(ds
                .as_i32()
                .map_err(|e| decode("i32", e))?
                .into_iter()
                .map(f64::from)
                .collect()),
            Dtype::Int {
                size: 8,
                signed: true,
                ..
            } => Ok(ds
                .as_i64()
                .map_err(|e| decode("i64", e))?
                .into_iter()
                .map(|v| v as f64)
                .collect()),
            Dtype::Int {
                size: 1,
                signed: false,
                ..
            } => Ok(ds
                .as_u8()
                .map_err(|e| decode("u8", e))?
                .into_iter()
                .map(f64::from)
                .collect()),
            Dtype::Int {
                size: 2,
                signed: false,
                ..
            } => Ok(ds
                .as_u16()
                .map_err(|e| decode("u16", e))?
                .into_iter()
                .map(f64::from)
                .collect()),
            Dtype::Int {
                size: 4,
                signed: false,
                ..
            } => Ok(ds
                .as_u32()
                .map_err(|e| decode("u32", e))?
                .into_iter()
                .map(f64::from)
                .collect()),
            Dtype::Int {
                size: 8,
                signed: false,
                ..
            } => Ok(ds
                .as_u64()
                .map_err(|e| decode("u64", e))?
                .into_iter()
                .map(|v| v as f64)
                .collect()),
            other => Err(DatasetsError::InvalidFormat(format!(
                "HDF5 datatype {other} is not numeric and cannot be read as f64"
            ))),
        }
    }

    /// List all top-level object names in the HDF5 file.
    ///
    /// Requires the `hdf5_io` feature.
    #[cfg(feature = "hdf5_io")]
    pub fn dataset_names(&self) -> Result<Vec<String>> {
        let file = oxih5::File::open(&self.path)
            .map_err(|e| DatasetsError::InvalidFormat(format!("HDF5 open error: {e}")))?;

        let root = file
            .root()
            .map_err(|e| DatasetsError::InvalidFormat(format!("HDF5 root group error: {e}")))?;

        // `hdf5::File::member_names` listed datasets *and* groups. `oxih5`
        // separates the two, and `File::dataset_names()` is datasets only — so
        // the group names have to be added back to preserve the old contract.
        let mut names = root
            .datasets()
            .map_err(|e| DatasetsError::InvalidFormat(format!("HDF5 member list error: {e}")))?;
        names
            .extend(root.groups().map_err(|e| {
                DatasetsError::InvalidFormat(format!("HDF5 group list error: {e}"))
            })?);

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

        /// Write a 1-D f64 dataset to an HDF5 file via the pure-Rust writer.
        fn write_hdf5_1d(path: &std::path::Path, name: &str, data: &[f64]) {
            oxih5::FileWriter::new()
                .write_dataset_f64(name, data, &[data.len()])
                .expect("write_dataset_f64")
                .build(path)
                .expect("build hdf5");
        }

        /// Write a 1-D i32 dataset, to exercise the numeric widening path.
        fn write_hdf5_1d_i32(path: &std::path::Path, name: &str, data: &[i32]) {
            oxih5::FileWriter::new()
                .write_dataset_i32(name, data, &[data.len()])
                .expect("write_dataset_i32")
                .build(path)
                .expect("build hdf5");
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

        /// A non-f64 dataset must still be readable: the C backend converted
        /// implicitly, and dropping that would narrow support to f64-only files
        /// without any test noticing.
        #[test]
        fn test_read_dataset_widens_non_f64() {
            let dir = tempfile::tempdir().expect("tmpdir");
            let path = dir.path().join("ints.h5");
            write_hdf5_1d_i32(&path, "counts", &[1, 2, 3, 4]);

            let ds = Hdf5Dataset::from_file(&path).expect("from_file");
            // read_dataset requires rank 2, so go through the widening helper
            // directly with the rank-1 dataset this writer produces.
            let file = oxih5::File::open(&path).expect("open");
            let raw = file.dataset("counts").expect("dataset");
            let widened = Hdf5Dataset::widen_to_f64(&raw).expect("i32 must widen to f64");
            assert_eq!(widened, vec![1.0, 2.0, 3.0, 4.0]);
        }
    }
}
