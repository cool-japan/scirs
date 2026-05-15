//! Sharded array storage: partition large arrays across multiple files or chunks.
//!
//! Provides:
//! - `ShardConfig` – shard count, shard size, distribution strategy
//! - `ShardedArray` – partition a `Vec<f64>` across N shards on disk
//! - `RoundRobinSharding`, `HashSharding`, `RangeSharding` strategies
//! - `ShardReader` – parallel shard reading with merge
//! - `ShardWriter` – parallel shard writing with coordination
//! - `VirtualConcatenation` – lazy concatenation of multiple file paths

#![allow(dead_code)]
#![allow(missing_docs)]

use crate::error::{IoError, Result};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;

// ---------------------------------------------------------------------------
// Distribution strategy
// ---------------------------------------------------------------------------

/// Strategy used to decide which shard receives each element.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardingStrategy {
    /// Round-robin: element `i` → shard `i % shard_count`
    RoundRobin,
    /// Hash-based: hash of element index → shard
    Hash,
    /// Range-based: contiguous blocks of equal size per shard
    Range,
}

/// Configuration for a sharded array.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardConfig {
    /// Total number of shards
    pub shard_count: usize,
    /// Maximum elements per shard (0 = unlimited / auto)
    pub shard_size: usize,
    /// Distribution strategy
    pub strategy: ShardingStrategy,
    /// Base directory where shard files are written
    pub base_dir: PathBuf,
    /// File name prefix (e.g. `"data"` → `"data_shard_000.bin"`)
    pub prefix: String,
}

impl ShardConfig {
    /// Create a new `ShardConfig` with sensible defaults.
    pub fn new<P: AsRef<Path>>(base_dir: P, shard_count: usize) -> Self {
        Self {
            shard_count,
            shard_size: 0,
            strategy: ShardingStrategy::Range,
            base_dir: base_dir.as_ref().to_path_buf(),
            prefix: "shard".to_string(),
        }
    }

    /// Set the sharding strategy.
    pub fn with_strategy(mut self, strategy: ShardingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set max elements per shard.
    pub fn with_shard_size(mut self, size: usize) -> Self {
        self.shard_size = size;
        self
    }

    /// Set the file prefix.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Return the file path for shard `index`.
    pub fn shard_path(&self, index: usize) -> PathBuf {
        self.base_dir
            .join(format!("{}_{:04}.bin", self.prefix, index))
    }

    /// Return the metadata file path.
    pub fn meta_path(&self) -> PathBuf {
        self.base_dir.join(format!("{}_meta.json", self.prefix))
    }
}

// ---------------------------------------------------------------------------
// Strategy helpers
// ---------------------------------------------------------------------------

/// Round-robin sharding: assign element at index `i` to shard `i % n`.
pub struct RoundRobinSharding;

impl RoundRobinSharding {
    /// Compute the shard index for element at position `element_index`.
    pub fn shard_for(element_index: usize, shard_count: usize) -> usize {
        if shard_count == 0 {
            return 0;
        }
        element_index % shard_count
    }
}

/// Hash-based sharding: hash the element index to pick a shard.
pub struct HashSharding;

impl HashSharding {
    /// Compute the shard index for element at position `element_index`.
    pub fn shard_for(element_index: usize, shard_count: usize) -> usize {
        if shard_count == 0 {
            return 0;
        }
        let mut h = DefaultHasher::new();
        element_index.hash(&mut h);
        (h.finish() as usize) % shard_count
    }
}

/// Range-based sharding: assign contiguous blocks of elements to shards.
pub struct RangeSharding;

impl RangeSharding {
    /// Compute the shard index for element at position `element_index` when
    /// the total array length is `total`.
    pub fn shard_for(element_index: usize, shard_count: usize, total: usize) -> usize {
        if shard_count == 0 || total == 0 {
            return 0;
        }
        let block = (total + shard_count - 1) / shard_count;
        (element_index / block).min(shard_count - 1)
    }
}

// ---------------------------------------------------------------------------
// ShardedArray metadata
// ---------------------------------------------------------------------------

/// Persistent metadata describing a sharded array on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardedArrayMeta {
    /// Total number of elements
    pub total_elements: usize,
    /// Number of shards
    pub shard_count: usize,
    /// Elements in each shard
    pub shard_sizes: Vec<usize>,
    /// Strategy used
    pub strategy: ShardingStrategy,
    /// Element bit-width (e.g. 64 for f64)
    pub element_bits: u8,
    /// File prefix
    pub prefix: String,
}

// ---------------------------------------------------------------------------
// ShardedArray
// ---------------------------------------------------------------------------

/// A large array of `f64` values partitioned across multiple binary shard files.
///
/// Each shard file stores raw little-endian `f64` values (no header).
/// A companion `<prefix>_meta.json` records the layout.
pub struct ShardedArray {
    config: ShardConfig,
    meta: Option<ShardedArrayMeta>,
}

impl ShardedArray {
    /// Create a `ShardedArray` bound to the given config (does not touch disk yet).
    pub fn new(config: ShardConfig) -> Self {
        Self { config, meta: None }
    }

    /// Write `data` to disk as shards.
    ///
    /// Returns the resulting metadata.
    pub fn write(&mut self, data: &[f64]) -> Result<ShardedArrayMeta> {
        std::fs::create_dir_all(&self.config.base_dir)
            .map_err(|e| IoError::FileError(format!("Cannot create shard dir: {e}")))?;

        let shard_count = self.config.shard_count;
        let total = data.len();

        // Assign each element to a shard
        let mut buckets: Vec<Vec<f64>> = vec![Vec::new(); shard_count];
        for (i, &v) in data.iter().enumerate() {
            let shard_idx = match self.config.strategy {
                ShardingStrategy::RoundRobin => RoundRobinSharding::shard_for(i, shard_count),
                ShardingStrategy::Hash => HashSharding::shard_for(i, shard_count),
                ShardingStrategy::Range => RangeSharding::shard_for(i, shard_count, total),
            };
            buckets[shard_idx].push(v);
        }

        let mut shard_sizes = Vec::with_capacity(shard_count);
        for (idx, bucket) in buckets.iter().enumerate() {
            let path = self.config.shard_path(idx);
            let mut f = std::fs::File::create(&path)
                .map_err(|e| IoError::FileError(format!("Cannot create shard {idx}: {e}")))?;
            for &v in bucket {
                f.write_all(&v.to_le_bytes())
                    .map_err(|e| IoError::FileError(format!("Shard write error: {e}")))?;
            }
            shard_sizes.push(bucket.len());
        }

        let meta = ShardedArrayMeta {
            total_elements: total,
            shard_count,
            shard_sizes,
            strategy: self.config.strategy.clone(),
            element_bits: 64,
            prefix: self.config.prefix.clone(),
        };

        // Persist metadata
        let meta_json = serde_json::to_string_pretty(&meta)
            .map_err(|e| IoError::SerializationError(format!("{e}")))?;
        let mut mf = std::fs::File::create(self.config.meta_path())
            .map_err(|e| IoError::FileError(format!("Cannot write meta: {e}")))?;
        mf.write_all(meta_json.as_bytes())
            .map_err(|e| IoError::FileError(format!("Meta write error: {e}")))?;

        self.meta = Some(meta.clone());
        Ok(meta)
    }

    /// Read back all shards and return the reconstructed `Vec<f64>`.
    ///
    /// For `Range` strategy the order is preserved (contiguous blocks).
    /// For `RoundRobin`/`Hash` strategies, a reconstruction pass restores order.
    pub fn read(&self) -> Result<Vec<f64>> {
        let meta = self.load_meta()?;
        let shard_count = meta.shard_count;
        let mut shards: Vec<Vec<f64>> = Vec::with_capacity(shard_count);
        for idx in 0..shard_count {
            let path = self.config.shard_path(idx);
            let mut f = std::fs::File::open(&path)
                .map_err(|_| IoError::NotFound(format!("Shard {idx} not found")))?;
            let mut buf = Vec::new();
            f.read_to_end(&mut buf)
                .map_err(|e| IoError::FileError(format!("Shard read error: {e}")))?;
            let values: Vec<f64> = buf
                .chunks_exact(8)
                .map(|b| {
                    let arr: [u8; 8] = b.try_into().unwrap_or([0u8; 8]);
                    f64::from_le_bytes(arr)
                })
                .collect();
            shards.push(values);
        }

        let total = meta.total_elements;
        match meta.strategy {
            ShardingStrategy::Range => {
                // Simply concatenate shards in order
                let result: Vec<f64> = shards.into_iter().flatten().collect();
                Ok(result)
            }
            ShardingStrategy::RoundRobin => {
                let mut result = vec![0.0f64; total];
                let mut shard_cursors = vec![0usize; shard_count];
                for i in 0..total {
                    let s = RoundRobinSharding::shard_for(i, shard_count);
                    let cursor = shard_cursors[s];
                    if cursor < shards[s].len() {
                        result[i] = shards[s][cursor];
                        shard_cursors[s] += 1;
                    }
                }
                Ok(result)
            }
            ShardingStrategy::Hash => {
                let mut result = vec![0.0f64; total];
                let mut shard_cursors = vec![0usize; shard_count];
                for i in 0..total {
                    let s = HashSharding::shard_for(i, shard_count);
                    let cursor = shard_cursors[s];
                    if cursor < shards[s].len() {
                        result[i] = shards[s][cursor];
                        shard_cursors[s] += 1;
                    }
                }
                Ok(result)
            }
        }
    }

    fn load_meta(&self) -> Result<ShardedArrayMeta> {
        if let Some(ref m) = self.meta {
            return Ok(m.clone());
        }
        let path = self.config.meta_path();
        let mut f = std::fs::File::open(&path)
            .map_err(|_| IoError::NotFound("Shard metadata not found".to_string()))?;
        let mut buf = String::new();
        f.read_to_string(&mut buf)
            .map_err(|e| IoError::FileError(format!("Cannot read meta: {e}")))?;
        serde_json::from_str(&buf).map_err(|e| IoError::ParseError(format!("Bad meta: {e}")))
    }
}

// ---------------------------------------------------------------------------
// ShardReader / ShardWriter
// ---------------------------------------------------------------------------

/// Parallel shard reader that reads multiple shards concurrently and merges them.
pub struct ShardReader {
    config: ShardConfig,
    num_threads: usize,
}

impl ShardReader {
    /// Create a `ShardReader` for the given config.
    pub fn new(config: ShardConfig) -> Self {
        let cores = num_cpus::get().max(1);
        Self {
            config,
            num_threads: cores,
        }
    }

    /// Set the number of reader threads.
    pub fn with_threads(mut self, n: usize) -> Self {
        self.num_threads = n.max(1);
        self
    }

    /// Read all shards in parallel and merge into a single `Vec<f64>`.
    pub fn read_all(&self) -> Result<Vec<f64>> {
        let shard_count = self.config.shard_count;
        let results: Arc<Mutex<Vec<(usize, Result<Vec<f64>>)>>> =
            Arc::new(Mutex::new(Vec::with_capacity(shard_count)));

        let handles: Vec<_> = (0..shard_count)
            .map(|idx| {
                let path = self.config.shard_path(idx);
                let results = results.clone();
                thread::spawn(move || {
                    let data = read_shard_raw(&path);
                    let mut guard = results.lock().expect("lock poisoned");
                    guard.push((idx, data));
                })
            })
            .collect();

        for h in handles {
            h.join()
                .map_err(|_| IoError::FileError("Shard reader thread panicked".to_string()))?;
        }

        let mut guard = results.lock().expect("lock poisoned");
        guard.sort_by_key(|(idx, _)| *idx);
        let mut merged = Vec::new();
        for (_, result) in guard.drain(..) {
            let data = result?;
            merged.extend(data);
        }
        Ok(merged)
    }

    /// Read a single shard by index.
    pub fn read_shard(&self, index: usize) -> Result<Vec<f64>> {
        read_shard_raw(&self.config.shard_path(index))
    }
}

fn read_shard_raw(path: &Path) -> Result<Vec<f64>> {
    let mut f = std::fs::File::open(path)
        .map_err(|_| IoError::NotFound(format!("Shard not found: {}", path.display())))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)
        .map_err(|e| IoError::FileError(format!("Shard read error: {e}")))?;
    Ok(buf
        .chunks_exact(8)
        .map(|b| {
            let arr: [u8; 8] = b.try_into().unwrap_or([0u8; 8]);
            f64::from_le_bytes(arr)
        })
        .collect())
}

/// Parallel shard writer that distributes data across shards concurrently.
pub struct ShardWriter {
    config: ShardConfig,
}

impl ShardWriter {
    /// Create a `ShardWriter` for the given config.
    pub fn new(config: ShardConfig) -> Self {
        Self { config }
    }

    /// Write `data` to shards in parallel (one thread per shard).
    ///
    /// Returns a list of paths to the written shard files.
    pub fn write(&self, data: &[f64]) -> Result<Vec<PathBuf>> {
        std::fs::create_dir_all(&self.config.base_dir)
            .map_err(|e| IoError::FileError(format!("Cannot create dir: {e}")))?;

        let shard_count = self.config.shard_count;
        let total = data.len();

        // Build per-shard buckets
        let mut buckets: Vec<Vec<f64>> = vec![Vec::new(); shard_count];
        for (i, &v) in data.iter().enumerate() {
            let s = match self.config.strategy {
                ShardingStrategy::RoundRobin => RoundRobinSharding::shard_for(i, shard_count),
                ShardingStrategy::Hash => HashSharding::shard_for(i, shard_count),
                ShardingStrategy::Range => RangeSharding::shard_for(i, shard_count, total),
            };
            buckets[s].push(v);
        }

        let written: Arc<Mutex<Vec<(usize, Result<PathBuf>)>>> =
            Arc::new(Mutex::new(Vec::with_capacity(shard_count)));

        let handles: Vec<_> = buckets
            .into_iter()
            .enumerate()
            .map(|(idx, bucket)| {
                let path = self.config.shard_path(idx);
                let written = written.clone();
                thread::spawn(move || {
                    let result = write_shard_raw(&path, &bucket).map(|()| path.clone());
                    let mut guard = written.lock().expect("lock poisoned");
                    guard.push((idx, result));
                })
            })
            .collect();

        for h in handles {
            h.join()
                .map_err(|_| IoError::FileError("Shard writer thread panicked".to_string()))?;
        }

        let mut guard = written.lock().expect("lock poisoned");
        guard.sort_by_key(|(idx, _)| *idx);
        let mut paths = Vec::with_capacity(shard_count);
        for (_, result) in guard.drain(..) {
            paths.push(result?);
        }
        Ok(paths)
    }
}

fn write_shard_raw(path: &Path, data: &[f64]) -> Result<()> {
    let mut f = std::fs::File::create(path)
        .map_err(|e| IoError::FileError(format!("Cannot create shard: {e}")))?;
    for &v in data {
        f.write_all(&v.to_le_bytes())
            .map_err(|e| IoError::FileError(format!("Shard write error: {e}")))?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// VirtualConcatenation
// ---------------------------------------------------------------------------

/// Lazy concatenation of multiple shard / data files.
///
/// Files are not loaded until `iter()` or `collect()` is called.
pub struct VirtualConcatenation {
    paths: Vec<PathBuf>,
    element_type: ElementType,
}

/// Supported element types for virtual concatenation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementType {
    /// 64-bit float (little-endian)
    F64,
    /// 32-bit float (little-endian)
    F32,
    /// Signed 64-bit integer (little-endian)
    I64,
}

impl VirtualConcatenation {
    /// Create a new virtual concatenation over `paths`.
    pub fn new(paths: Vec<PathBuf>, element_type: ElementType) -> Self {
        Self {
            paths,
            element_type,
        }
    }

    /// Materialise the concatenation into a `Vec<f64>`.
    ///
    /// `F32` and `I64` elements are widened to `f64`.
    pub fn collect_f64(&self) -> Result<Vec<f64>> {
        let mut out = Vec::new();
        for path in &self.paths {
            let mut f = std::fs::File::open(path)
                .map_err(|_| IoError::NotFound(format!("File not found: {}", path.display())))?;
            let mut buf = Vec::new();
            f.read_to_end(&mut buf)
                .map_err(|e| IoError::FileError(format!("Read error: {e}")))?;
            match self.element_type {
                ElementType::F64 => {
                    for b in buf.chunks_exact(8) {
                        let arr: [u8; 8] = b.try_into().unwrap_or([0u8; 8]);
                        out.push(f64::from_le_bytes(arr));
                    }
                }
                ElementType::F32 => {
                    for b in buf.chunks_exact(4) {
                        let arr: [u8; 4] = b.try_into().unwrap_or([0u8; 4]);
                        out.push(f32::from_le_bytes(arr) as f64);
                    }
                }
                ElementType::I64 => {
                    for b in buf.chunks_exact(8) {
                        let arr: [u8; 8] = b.try_into().unwrap_or([0u8; 8]);
                        out.push(i64::from_le_bytes(arr) as f64);
                    }
                }
            }
        }
        Ok(out)
    }

    /// Return total number of elements (requires reading metadata / file sizes).
    pub fn estimated_len(&self) -> usize {
        let bytes_per = match self.element_type {
            ElementType::F64 | ElementType::I64 => 8,
            ElementType::F32 => 4,
        };
        self.paths
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len() as usize / bytes_per)
            .sum()
    }

    /// Number of source files.
    pub fn shard_count(&self) -> usize {
        self.paths.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;
    use uuid::Uuid;

    fn temp_config(strategy: ShardingStrategy) -> ShardConfig {
        let dir = temp_dir().join(format!("scirs2_shard_{}", Uuid::new_v4()));
        ShardConfig::new(&dir, 4).with_strategy(strategy)
    }

    #[test]
    fn test_range_sharding() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let config = temp_config(ShardingStrategy::Range);
        let mut sa = ShardedArray::new(config.clone());
        let meta = sa.write(&data).unwrap();
        assert_eq!(meta.total_elements, 100);
        assert_eq!(meta.shard_sizes.iter().sum::<usize>(), 100);
        let loaded = sa.read().expect("range sharded read should succeed");
        assert_eq!(loaded, data);
        let _ = std::fs::remove_dir_all(&config.base_dir);
    }

    #[test]
    fn test_round_robin_sharding() {
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let config = temp_config(ShardingStrategy::RoundRobin);
        let mut sa = ShardedArray::new(config.clone());
        sa.write(&data).unwrap();
        let loaded = sa.read().expect("round-robin sharded read should succeed");
        assert_eq!(loaded, data);
        let _ = std::fs::remove_dir_all(&config.base_dir);
    }

    #[test]
    fn test_hash_sharding() {
        let data: Vec<f64> = (0..20).map(|i| i as f64 * 2.0).collect();
        let config = temp_config(ShardingStrategy::Hash);
        let mut sa = ShardedArray::new(config.clone());
        sa.write(&data).unwrap();
        let loaded = sa.read().expect("hash sharded read should succeed");
        assert_eq!(loaded, data);
        let _ = std::fs::remove_dir_all(&config.base_dir);
    }

    #[test]
    fn test_shard_writer_and_reader() {
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let config = temp_config(ShardingStrategy::Range);
        let writer = ShardWriter::new(config.clone());
        let paths = writer.write(&data).unwrap();
        assert_eq!(paths.len(), 4);
        let reader = ShardReader::new(config.clone());
        let merged = reader.read_all().unwrap();
        assert_eq!(merged, data);
        let _ = std::fs::remove_dir_all(&config.base_dir);
    }

    #[test]
    fn test_virtual_concatenation() {
        let dir = temp_dir().join(format!("scirs2_vc_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();

        let mut paths = Vec::new();
        for i in 0..3usize {
            let p = dir.join(format!("part_{i}.bin"));
            let mut f = std::fs::File::create(&p).unwrap();
            for v in (i * 10..(i + 1) * 10).map(|x| (x as f64).to_le_bytes()) {
                f.write_all(&v).unwrap();
            }
            paths.push(p);
        }

        let vc = VirtualConcatenation::new(paths, ElementType::F64);
        let data = vc.collect_f64().unwrap();
        assert_eq!(data.len(), 30);
        assert_eq!(data[0], 0.0);
        assert_eq!(data[29], 29.0);
        assert_eq!(vc.estimated_len(), 30);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
