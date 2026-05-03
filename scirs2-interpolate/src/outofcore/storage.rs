//! Disk-backed flat binary storage for out-of-core coefficient arrays.
//!
//! `DiskStorage` writes/reads f64 values as little-endian bytes in a flat binary
//! file laid out in row-major order: row 0 comes first, then row 1, etc.
//! Each f64 occupies exactly 8 bytes.
//!
//! # Design notes
//! - Files are pre-allocated on [`DiskStorage::create`] to avoid fragmentation.
//! - Partial-row reads and random-access writes are supported via `Seek`.
//! - Drop does **not** delete the backing file; call [`DiskStorage::delete`] explicitly.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::error::InterpolateError;

/// A flat, disk-backed storage for an `(rows × cols)` matrix of `f64` values.
///
/// Values are stored in **row-major** order as little-endian IEEE 754 bytes.
#[derive(Debug, Clone)]
pub struct DiskStorage {
    path: PathBuf,
    rows: usize,
    cols: usize,
}

impl DiskStorage {
    /// Create a new backing file pre-filled with zero bytes.
    ///
    /// Fails if the parent directory does not exist or the file cannot be created.
    pub fn create(
        path: impl AsRef<Path>,
        rows: usize,
        cols: usize,
    ) -> Result<Self, InterpolateError> {
        let path = path.as_ref().to_path_buf();
        let total_bytes = rows
            .checked_mul(cols)
            .and_then(|n| n.checked_mul(8))
            .ok_or_else(|| InterpolateError::InvalidInput {
                message: format!("DiskStorage dimensions overflow: {rows}×{cols}"),
            })?;

        let mut file = File::create(&path).map_err(|e| {
            InterpolateError::IoError(format!("DiskStorage::create '{}': {e}", path.display()))
        })?;
        // Pre-allocate by writing a trailing zero byte
        if total_bytes > 0 {
            let zeros = vec![0u8; (total_bytes).min(65536)];
            let mut written = 0usize;
            while written < total_bytes {
                let chunk = (total_bytes - written).min(zeros.len());
                file.write_all(&zeros[..chunk]).map_err(|e| {
                    InterpolateError::IoError(format!("DiskStorage pre-alloc write: {e}"))
                })?;
                written += chunk;
            }
        }

        Ok(Self { path, rows, cols })
    }

    /// Open an existing backing file.
    pub fn open(
        path: impl AsRef<Path>,
        rows: usize,
        cols: usize,
    ) -> Result<Self, InterpolateError> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            return Err(InterpolateError::IoError(format!(
                "DiskStorage::open: file not found '{}'",
                path.display()
            )));
        }
        Ok(Self { path, rows, cols })
    }

    /// Write `n_rows` rows starting at `row_start`.
    ///
    /// `data.len()` must equal `n_rows * self.cols`.
    pub fn write_rows(&self, row_start: usize, data: &[f64]) -> Result<(), InterpolateError> {
        if data.len() % self.cols != 0 {
            return Err(InterpolateError::ShapeMismatch {
                expected: format!("multiple of {} elements", self.cols),
                actual: format!("{} elements", data.len()),
                object: "DiskStorage::write_rows data".into(),
            });
        }
        let offset = row_start
            .checked_mul(self.cols)
            .and_then(|n| n.checked_mul(8))
            .ok_or_else(|| InterpolateError::ComputationError("write offset overflow".into()))?
            as u64;

        let mut file = OpenOptions::new()
            .write(true)
            .open(&self.path)
            .map_err(|e| InterpolateError::IoError(format!("DiskStorage open for write: {e}")))?;

        file.seek(SeekFrom::Start(offset))
            .map_err(|e| InterpolateError::IoError(format!("DiskStorage seek: {e}")))?;

        // Serialise f64 slice as le-bytes without heap allocation when possible
        let mut buf = [0u8; 8192];
        let mut data_idx = 0usize;
        while data_idx < data.len() {
            let chunk_elems = (data.len() - data_idx).min(buf.len() / 8);
            for k in 0..chunk_elems {
                let bytes = data[data_idx + k].to_le_bytes();
                buf[k * 8..k * 8 + 8].copy_from_slice(&bytes);
            }
            file.write_all(&buf[..chunk_elems * 8])
                .map_err(|e| InterpolateError::IoError(format!("DiskStorage write: {e}")))?;
            data_idx += chunk_elems;
        }
        Ok(())
    }

    /// Read `n_rows` rows starting at `row_start`.
    ///
    /// Returns a `Vec<f64>` of length `n_rows * self.cols` in row-major order.
    pub fn read_rows(&self, row_start: usize, n_rows: usize) -> Result<Vec<f64>, InterpolateError> {
        let n_elems = n_rows
            .checked_mul(self.cols)
            .ok_or_else(|| InterpolateError::ComputationError("read size overflow".into()))?;
        let offset = row_start
            .checked_mul(self.cols)
            .and_then(|n| n.checked_mul(8))
            .ok_or_else(|| InterpolateError::ComputationError("read offset overflow".into()))?
            as u64;

        let mut file = File::open(&self.path)
            .map_err(|e| InterpolateError::IoError(format!("DiskStorage open for read: {e}")))?;

        file.seek(SeekFrom::Start(offset))
            .map_err(|e| InterpolateError::IoError(format!("DiskStorage read seek: {e}")))?;

        let mut bytes = vec![0u8; n_elems * 8];
        file.read_exact(&mut bytes)
            .map_err(|e| InterpolateError::IoError(format!("DiskStorage read_exact: {e}")))?;

        // Deserialise le-bytes -> f64
        let result: Vec<f64> = bytes
            .chunks_exact(8)
            .map(|b| {
                let mut arr = [0u8; 8];
                arr.copy_from_slice(b);
                f64::from_le_bytes(arr)
            })
            .collect();

        Ok(result)
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Path to the backing file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Explicitly delete the backing file.
    ///
    /// [`Drop`] does *not* delete the file automatically; call this when done.
    pub fn delete(self) -> Result<(), InterpolateError> {
        std::fs::remove_file(&self.path)
            .map_err(|e| InterpolateError::IoError(format!("DiskStorage::delete: {e}")))
    }
}
