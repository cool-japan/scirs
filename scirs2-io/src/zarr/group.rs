//! ZarrGroup: in-memory and filesystem Zarr v3 group with high-level array I/O.
//!
//! Provides `MemoryStore`, `FsStore`, and `ZarrGroup` for working with Zarr arrays
//! without needing to manage chunk keys manually.

use std::collections::HashMap;
use std::path::PathBuf;

use serde_json;

use super::types::{ZarrArray, ZarrArrayMeta, ZarrCompressor};
use crate::error::IoError;

// ─────────────────────────────────────────────────────────────────────────────
// ZarrStore trait
// ─────────────────────────────────────────────────────────────────────────────

/// Trait for Zarr v3 key-value stores.
pub trait ZarrKvStore: Send {
    /// Retrieve a value by key. Returns `None` if the key does not exist.
    fn get(&self, key: &str) -> Option<Vec<u8>>;
    /// Store a value under a key.
    fn set(&mut self, key: &str, value: Vec<u8>);
    /// List all keys that start with `prefix`.
    fn list_prefix(&self, prefix: &str) -> Vec<String>;
    /// Delete a key if it exists.
    fn delete(&mut self, key: &str);
}

// ─────────────────────────────────────────────────────────────────────────────
// MemoryStore
// ─────────────────────────────────────────────────────────────────────────────

/// An in-memory Zarr store backed by a `HashMap<String, Vec<u8>>`.
pub struct MemoryStore {
    data: HashMap<String, Vec<u8>>,
}

impl MemoryStore {
    /// Create a new, empty `MemoryStore`.
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ZarrKvStore for MemoryStore {
    fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.data.get(key).cloned()
    }

    fn set(&mut self, key: &str, value: Vec<u8>) {
        self.data.insert(key.to_owned(), value);
    }

    fn list_prefix(&self, prefix: &str) -> Vec<String> {
        let mut keys: Vec<String> = self
            .data
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        keys.sort();
        keys
    }

    fn delete(&mut self, key: &str) {
        self.data.remove(key);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FsStore
// ─────────────────────────────────────────────────────────────────────────────

/// A filesystem-backed Zarr store. Keys are mapped to files under `root`.
pub struct FsStore {
    root: PathBuf,
}

impl FsStore {
    /// Create an `FsStore` rooted at `path`. The directory is created if absent.
    pub fn new(path: &std::path::Path) -> Result<Self, IoError> {
        std::fs::create_dir_all(path)
            .map_err(|e| IoError::FileError(format!("FsStore create_dir_all: {e}")))?;
        Ok(Self {
            root: path.to_path_buf(),
        })
    }

    fn key_to_path(&self, key: &str) -> PathBuf {
        let mut p = self.root.clone();
        for component in key.split('/') {
            if !component.is_empty() {
                p.push(component);
            }
        }
        p
    }
}

impl ZarrKvStore for FsStore {
    fn get(&self, key: &str) -> Option<Vec<u8>> {
        std::fs::read(self.key_to_path(key)).ok()
    }

    fn set(&mut self, key: &str, value: Vec<u8>) {
        let path = self.key_to_path(key);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(path, value);
    }

    fn list_prefix(&self, prefix: &str) -> Vec<String> {
        let prefix_path = self.key_to_path(prefix);
        let mut results = Vec::new();
        collect_fs_keys(&self.root, &prefix_path, &self.root, &mut results);
        results.sort();
        results
    }

    fn delete(&mut self, key: &str) {
        let _ = std::fs::remove_file(self.key_to_path(key));
    }
}

/// Recursively walk `current`, collecting paths that start with `prefix_path`.
fn collect_fs_keys(
    root: &std::path::Path,
    prefix_path: &std::path::Path,
    current: &std::path::Path,
    out: &mut Vec<String>,
) {
    let entries = match std::fs::read_dir(current) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_fs_keys(root, prefix_path, &path, out);
        } else if path.starts_with(prefix_path) {
            if let Ok(rel) = path.strip_prefix(root) {
                let key = rel
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy().into_owned())
                    .collect::<Vec<_>>()
                    .join("/");
                out.push(key);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ZarrGroup
// ─────────────────────────────────────────────────────────────────────────────

/// A Zarr v3 group owning a key-value store and exposing array read/write APIs.
pub struct ZarrGroup {
    store: Box<dyn ZarrKvStore>,
}

impl ZarrGroup {
    /// Create a group backed by an in-memory store.
    pub fn new_memory() -> Self {
        Self {
            store: Box::new(MemoryStore::new()),
        }
    }

    /// Create a group backed by a filesystem store rooted at `path`.
    pub fn new_fs(path: &std::path::Path) -> Result<Self, IoError> {
        Ok(Self {
            store: Box::new(FsStore::new(path)?),
        })
    }

    // ── Key helpers ────────────────────────────────────────────────────────

    /// Metadata key: `{name}/zarr.json`
    pub fn meta_key(name: &str) -> String {
        format!("{name}/zarr.json")
    }

    /// Chunk key: `{name}/c{sep}{c0}{sep}{c1}...`
    pub fn chunk_key(name: &str, coords: &[usize], sep: char) -> String {
        let coord_str: Vec<String> = coords.iter().map(|c| c.to_string()).collect();
        format!("{name}/c{sep}{}", coord_str.join(&sep.to_string()))
    }

    // ── Metadata helpers ───────────────────────────────────────────────────

    fn write_meta(&mut self, name: &str, meta: &ZarrArrayMeta) {
        let json = serde_json::to_vec(meta).unwrap_or_default();
        self.store.set(&Self::meta_key(name), json);
    }

    fn read_meta(&self, name: &str) -> Option<ZarrArrayMeta> {
        let raw = self.store.get(&Self::meta_key(name))?;
        serde_json::from_slice(&raw).ok()
    }

    // ── Public API ─────────────────────────────────────────────────────────

    /// Register a new array by writing its metadata; returns an empty `ZarrArray`.
    pub fn create_array(&mut self, name: &str, meta: ZarrArrayMeta) -> ZarrArray {
        self.write_meta(name, &meta);
        ZarrArray::new(meta, Vec::new())
    }

    /// Write a single chunk. Data is stored as raw little-endian f64 bytes.
    pub fn write_chunk(&mut self, name: &str, chunk_coords: &[usize], data: &[f64]) {
        let sep = self
            .read_meta(name)
            .map(|m| m.dimension_separator)
            .unwrap_or('/');
        let key = Self::chunk_key(name, chunk_coords, sep);
        self.store.set(&key, f64_to_bytes(data));
    }

    /// Read a single chunk. Returns `None` if the chunk was never written.
    pub fn read_chunk(&self, name: &str, chunk_coords: &[usize]) -> Option<Vec<f64>> {
        let sep = self
            .read_meta(name)
            .map(|m| m.dimension_separator)
            .unwrap_or('/');
        let key = Self::chunk_key(name, chunk_coords, sep);
        let raw = self.store.get(&key)?;
        Some(bytes_to_f64(&raw))
    }

    /// Write a full array, automatically splitting into chunks.
    ///
    /// Panics if `meta.shape.len() != meta.chunks.len()`.
    pub fn write_array(&mut self, name: &str, meta: ZarrArrayMeta, data: &[f64]) {
        assert_eq!(
            meta.shape.len(),
            meta.chunks.len(),
            "shape and chunks must have the same number of dimensions"
        );
        self.write_meta(name, &meta);
        let ndim = meta.shape.len();
        if ndim == 0 {
            return;
        }

        let chunks_per_dim: Vec<usize> = (0..ndim)
            .map(|i| div_ceil(meta.shape[i], meta.chunks[i]))
            .collect();
        let sep = meta.dimension_separator;
        let shape = meta.shape.clone();
        let chunk_shape = meta.chunks.clone();

        iterate_coords(&chunks_per_dim, |cc| {
            let chunk_data = extract_chunk(&shape, &chunk_shape, data, cc);
            let key = Self::chunk_key(name, cc, sep);
            self.store.set(&key, f64_to_bytes(&chunk_data));
        });
    }

    /// Read a full array, reassembling it from stored chunks.
    ///
    /// Returns `None` if no metadata exists for the named array.
    pub fn read_array(&self, name: &str) -> Option<(ZarrArrayMeta, Vec<f64>)> {
        let meta = self.read_meta(name)?;
        let ndim = meta.shape.len();
        let total: usize = if ndim == 0 {
            0
        } else {
            meta.shape.iter().product()
        };
        let mut out = vec![meta.fill_value; total];

        if ndim == 0 || total == 0 {
            return Some((meta, out));
        }

        let chunks_per_dim: Vec<usize> = (0..ndim)
            .map(|i| div_ceil(meta.shape[i], meta.chunks[i]))
            .collect();
        let sep = meta.dimension_separator;
        let shape = meta.shape.clone();
        let chunk_shape = meta.chunks.clone();

        iterate_coords(&chunks_per_dim, |cc| {
            let key = Self::chunk_key(name, cc, sep);
            if let Some(raw) = self.store.get(&key) {
                let chunk_data = bytes_to_f64(&raw);
                insert_chunk(&shape, &chunk_shape, cc, &chunk_data, &mut out);
            }
        });

        Some((meta, out))
    }

    /// List names of all arrays stored in this group.
    pub fn list_arrays(&self) -> Vec<String> {
        let all = self.store.list_prefix("");
        let mut names: Vec<String> = all
            .iter()
            .filter_map(|k| k.strip_suffix("/zarr.json").map(str::to_owned))
            .collect();
        names.sort();
        names
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Byte encoding helpers
// ─────────────────────────────────────────────────────────────────────────────

fn f64_to_bytes(data: &[f64]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(data.len() * 8);
    for &v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

fn bytes_to_f64(raw: &[u8]) -> Vec<f64> {
    raw.chunks_exact(8)
        .map(|c| {
            let arr: [u8; 8] = c.try_into().unwrap_or([0u8; 8]);
            f64::from_le_bytes(arr)
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Chunk iteration / extraction helpers
// ─────────────────────────────────────────────────────────────────────────────

fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

/// Iterate over all chunk coordinate tuples in row-major order.
fn iterate_coords<F: FnMut(&[usize])>(counts: &[usize], mut f: F) {
    if counts.is_empty() || counts.contains(&0) {
        return;
    }
    let total: usize = counts.iter().product();
    let ndim = counts.len();
    for flat in 0..total {
        let mut coords = vec![0usize; ndim];
        let mut tmp = flat;
        for i in (0..ndim).rev() {
            coords[i] = tmp % counts[i];
            tmp /= counts[i];
        }
        f(&coords);
    }
}

/// Extract the data for chunk `chunk_coords` from a flat row-major array.
fn extract_chunk(
    shape: &[usize],
    chunk_shape: &[usize],
    data: &[f64],
    chunk_coords: &[usize],
) -> Vec<f64> {
    let ndim = shape.len();
    let starts: Vec<usize> = (0..ndim)
        .map(|i| chunk_coords[i] * chunk_shape[i])
        .collect();
    let ends: Vec<usize> = (0..ndim)
        .map(|i| (starts[i] + chunk_shape[i]).min(shape[i]))
        .collect();
    let sizes: Vec<usize> = (0..ndim).map(|i| ends[i] - starts[i]).collect();

    if sizes.contains(&0) {
        return Vec::new();
    }

    let chunk_total: usize = sizes.iter().product();
    let mut chunk = Vec::with_capacity(chunk_total);

    for flat in 0..chunk_total {
        let mut local_coords = vec![0usize; ndim];
        let mut tmp = flat;
        for i in (0..ndim).rev() {
            local_coords[i] = tmp % sizes[i];
            tmp /= sizes[i];
        }
        let global_flat = row_major_index(shape, &starts, &local_coords);
        chunk.push(if global_flat < data.len() {
            data[global_flat]
        } else {
            0.0
        });
    }
    chunk
}

/// Insert `chunk_data` into the flat output array at the position of `chunk_coords`.
fn insert_chunk(
    shape: &[usize],
    chunk_shape: &[usize],
    chunk_coords: &[usize],
    chunk_data: &[f64],
    out: &mut [f64],
) {
    let ndim = shape.len();
    let starts: Vec<usize> = (0..ndim)
        .map(|i| chunk_coords[i] * chunk_shape[i])
        .collect();
    let ends: Vec<usize> = (0..ndim)
        .map(|i| (starts[i] + chunk_shape[i]).min(shape[i]))
        .collect();
    let sizes: Vec<usize> = (0..ndim).map(|i| ends[i] - starts[i]).collect();

    if sizes.contains(&0) {
        return;
    }

    let chunk_total: usize = sizes.iter().product();

    for (flat, &val) in chunk_data.iter().enumerate().take(chunk_total) {
        let mut local_coords = vec![0usize; ndim];
        let mut tmp = flat;
        for i in (0..ndim).rev() {
            local_coords[i] = tmp % sizes[i];
            tmp /= sizes[i];
        }
        let global_flat = row_major_index(shape, &starts, &local_coords);
        if global_flat < out.len() {
            out[global_flat] = val;
        }
    }
}

/// Compute the flat row-major index for element at `starts + local`.
fn row_major_index(shape: &[usize], starts: &[usize], local: &[usize]) -> usize {
    let ndim = shape.len();
    let mut idx = 0usize;
    let mut stride = 1usize;
    for i in (0..ndim).rev() {
        idx += (starts[i] + local[i]) * stride;
        stride *= shape[i];
    }
    idx
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zarr::types::{ZarrArrayMeta, ZarrCompressor, ZarrDataType};

    fn simple_meta(shape: Vec<usize>, chunks: Vec<usize>) -> ZarrArrayMeta {
        ZarrArrayMeta {
            shape,
            chunks,
            dtype: ZarrDataType::Float64,
            compressor: ZarrCompressor::None,
            fill_value: 0.0,
            zarr_format: 3,
            dimension_separator: '/',
        }
    }

    #[test]
    fn test_memory_store_roundtrip() {
        let mut store = MemoryStore::new();
        store.set("foo/bar", b"hello".to_vec());
        assert_eq!(store.get("foo/bar"), Some(b"hello".to_vec()));
        assert_eq!(store.get("missing"), None);
        store.delete("foo/bar");
        assert_eq!(store.get("foo/bar"), None);
    }

    #[test]
    fn test_memory_store_list_prefix() {
        let mut store = MemoryStore::new();
        store.set("a/zarr.json", b"{}".to_vec());
        store.set("a/c/0", b"data".to_vec());
        store.set("b/zarr.json", b"{}".to_vec());
        let keys = store.list_prefix("a/");
        assert!(keys.contains(&"a/zarr.json".to_owned()));
        assert!(keys.contains(&"a/c/0".to_owned()));
        assert!(!keys.iter().any(|k| k.starts_with("b/")));
    }

    #[test]
    fn test_zarr_group_write_read_array() {
        let mut group = ZarrGroup::new_memory();
        let meta = simple_meta(vec![4], vec![2]);
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        group.write_array("arr", meta, &data);
        let (_, read_back) = group.read_array("arr").expect("array must exist");
        assert_eq!(read_back, data);
    }

    #[test]
    fn test_zarr_group_chunk_key_format() {
        let key = ZarrGroup::chunk_key("arr", &[0, 1], '/');
        assert_eq!(key, "arr/c/0/1");
    }

    #[test]
    fn test_zarr_group_list_arrays() {
        let mut group = ZarrGroup::new_memory();
        group.write_array("x", simple_meta(vec![2], vec![2]), &[1.0, 2.0]);
        group.write_array("y", simple_meta(vec![3], vec![3]), &[0.0, 1.0, 2.0]);
        let names = group.list_arrays();
        assert!(names.contains(&"x".to_owned()));
        assert!(names.contains(&"y".to_owned()));
    }

    #[test]
    fn test_zarr_array_meta_default() {
        let meta = ZarrArrayMeta::default();
        assert_eq!(meta.zarr_format, 3);
        assert_eq!(meta.dimension_separator, '/');
        assert_eq!(meta.fill_value, 0.0);
    }

    #[test]
    fn test_zarr_multi_dimensional_array() {
        let mut group = ZarrGroup::new_memory();
        // 3×4 array chunked 2×2
        let meta = simple_meta(vec![3, 4], vec![2, 2]);
        let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
        group.write_array("mat", meta, &data);
        let (_, read_back) = group.read_array("mat").expect("array must exist");
        assert_eq!(read_back, data);
    }

    #[test]
    fn test_fs_store_roundtrip() {
        let dir = std::env::temp_dir().join("zarr_group_fs_test");
        let _ = std::fs::remove_dir_all(&dir);
        {
            let mut group = ZarrGroup::new_fs(&dir).expect("create fs group");
            let meta = simple_meta(vec![4], vec![4]);
            group.write_array("fsarr", meta, &[10.0, 20.0, 30.0, 40.0]);
            let (_, data) = group.read_array("fsarr").expect("read");
            assert_eq!(data, vec![10.0, 20.0, 30.0, 40.0]);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
