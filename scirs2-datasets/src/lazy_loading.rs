//! Lazy loading and zero-copy streaming for large datasets
//!
//! This module provides advanced lazy loading capabilities with memory-mapped files,
//! adaptive chunking based on memory pressure, and zero-copy views for efficient
//! processing of datasets that exceed available RAM.

use crate::error::{DatasetsError, Result};
use crate::streaming::{DataChunk, StreamConfig};
use crate::utils::Dataset;
use memmap2::{Mmap, MmapOptions};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Configuration for lazy loading operations
#[derive(Debug, Clone)]
pub struct LazyLoadConfig {
    /// Target memory usage in bytes (for adaptive chunking)
    pub target_memory_bytes: usize,
    /// Minimum chunk size in samples
    pub min_chunk_size: usize,
    /// Maximum chunk size in samples
    pub max_chunk_size: usize,
    /// Whether to use memory mapping
    pub use_mmap: bool,
    /// Page size for memory mapping (0 = system default)
    pub page_size: usize,
    /// Whether to prefetch pages
    pub prefetch: bool,
    /// Whether to lock pages in memory
    pub lock_pages: bool,
}

impl Default for LazyLoadConfig {
    fn default() -> Self {
        Self {
            target_memory_bytes: 512 * 1024 * 1024, // 512 MB
            min_chunk_size: 1000,
            max_chunk_size: 100_000,
            use_mmap: true,
            page_size: 0,
            prefetch: true,
            lock_pages: false,
        }
    }
}

impl LazyLoadConfig {
    /// Create a new lazy load configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set target memory usage
    pub fn with_target_memory(mut self, bytes: usize) -> Self {
        self.target_memory_bytes = bytes;
        self
    }

    /// Set chunk size range
    pub fn with_chunk_size_range(mut self, min: usize, max: usize) -> Self {
        self.min_chunk_size = min;
        self.max_chunk_size = max;
        self
    }

    /// Enable or disable memory mapping
    pub fn with_mmap(mut self, use_mmap: bool) -> Self {
        self.use_mmap = use_mmap;
        self
    }

    /// Enable prefetching
    pub fn with_prefetch(mut self, prefetch: bool) -> Self {
        self.prefetch = prefetch;
        self
    }
}

/// Memory-mapped dataset for zero-copy access
pub struct MmapDataset {
    mmap: Arc<Mmap>,
    n_samples: usize,
    n_features: usize,
    data_offset: usize,
    element_size: usize,
    config: LazyLoadConfig,
}

impl MmapDataset {
    /// Create a new memory-mapped dataset from a binary file
    ///
    /// # Arguments
    /// * `path` - Path to the binary file
    /// * `n_samples` - Number of samples in the dataset
    /// * `n_features` - Number of features per sample
    /// * `data_offset` - Byte offset to the start of data
    /// * `config` - Lazy loading configuration
    ///
    /// # Returns
    /// * `Ok(MmapDataset)` - The memory-mapped dataset
    /// * `Err(DatasetsError)` - If mapping fails
    pub fn from_binary<P: AsRef<Path>>(
        path: P,
        n_samples: usize,
        n_features: usize,
        data_offset: usize,
        config: LazyLoadConfig,
    ) -> Result<Self> {
        let file = File::open(path.as_ref()).map_err(DatasetsError::IoError)?;

        let mut mmap_opts = MmapOptions::new();
        if data_offset > 0 {
            mmap_opts.offset(data_offset as u64);
        }

        let mmap = unsafe {
            mmap_opts.map(&file).map_err(|e| {
                DatasetsError::InvalidFormat(format!("Memory mapping failed: {}", e))
            })?
        };

        // Prefetch if enabled (madvise is Unix-only; no-op elsewhere)
        #[cfg(unix)]
        if config.prefetch {
            let _ = mmap.advise(memmap2::Advice::WillNeed);
        }

        Ok(Self {
            mmap: Arc::new(mmap),
            n_samples,
            n_features,
            data_offset,
            element_size: std::mem::size_of::<f64>(),
            config,
        })
    }

    /// Get the number of samples
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Get the number of features
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Get a zero-copy view of a sample range
    ///
    /// # Arguments
    /// * `start` - Starting sample index
    /// * `end` - Ending sample index (exclusive)
    ///
    /// # Returns
    /// * `Ok(Array2<f64>)` - View of the requested samples
    /// * `Err(DatasetsError)` - If range is invalid
    pub fn view_range(&self, start: usize, end: usize) -> Result<Array2<f64>> {
        if start >= self.n_samples || end > self.n_samples || start >= end {
            return Err(DatasetsError::InvalidFormat(format!(
                "Invalid range [{}, {}) for dataset with {} samples",
                start, end, self.n_samples
            )));
        }

        let n_samples_in_range = end - start;
        let start_byte = start * self.n_features * self.element_size;
        let len_bytes = n_samples_in_range * self.n_features * self.element_size;

        // Ensure we don't read past the end
        if start_byte + len_bytes > self.mmap.len() {
            return Err(DatasetsError::InvalidFormat(
                "Range exceeds available data".to_string(),
            ));
        }

        // Convert bytes to f64 slice
        let byte_slice = &self.mmap[start_byte..start_byte + len_bytes];
        let (_, f64_slice, _) = unsafe { byte_slice.align_to::<f64>() };

        // Create array from slice
        let array =
            Array2::from_shape_vec((n_samples_in_range, self.n_features), f64_slice.to_vec())
                .map_err(|e| {
                    DatasetsError::InvalidFormat(format!("Array creation failed: {}", e))
                })?;

        Ok(array)
    }

    /// Calculate optimal chunk size based on memory pressure
    pub fn adaptive_chunk_size(&self) -> usize {
        let bytes_per_sample = self.n_features * self.element_size;
        let ideal_chunk = self.config.target_memory_bytes / bytes_per_sample;

        // Clamp to configured range
        ideal_chunk
            .max(self.config.min_chunk_size)
            .min(self.config.max_chunk_size)
            .min(self.n_samples)
    }

    /// Create an iterator over chunks with adaptive sizing
    pub fn iter_chunks(&self) -> LazyChunkIterator {
        let chunk_size = self.adaptive_chunk_size();
        LazyChunkIterator {
            dataset: self,
            current_pos: 0,
            chunk_size,
        }
    }
}

/// Iterator over lazy-loaded chunks
pub struct LazyChunkIterator<'a> {
    dataset: &'a MmapDataset,
    current_pos: usize,
    chunk_size: usize,
}

impl<'a> Iterator for LazyChunkIterator<'a> {
    type Item = Result<DataChunk>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_pos >= self.dataset.n_samples {
            return None;
        }

        let end = (self.current_pos + self.chunk_size).min(self.dataset.n_samples);
        let chunk_idx = self.current_pos / self.chunk_size;

        let result = self.dataset.view_range(self.current_pos, end).map(|data| {
            let sample_indices: Vec<usize> = (self.current_pos..end).collect();
            let is_last = end >= self.dataset.n_samples;

            DataChunk {
                data,
                target: None,
                chunk_index: chunk_idx,
                sample_indices,
                is_last,
            }
        });

        self.current_pos = end;
        Some(result)
    }
}

/// Lazy dataset wrapper that defers loading until access
pub struct LazyDataset {
    path: PathBuf,
    n_samples: usize,
    n_features: usize,
    data_offset: usize,
    config: LazyLoadConfig,
    mmap_dataset: Option<Arc<MmapDataset>>,
}

impl LazyDataset {
    /// Create a new lazy dataset
    pub fn new<P: AsRef<Path>>(
        path: P,
        n_samples: usize,
        n_features: usize,
        data_offset: usize,
        config: LazyLoadConfig,
    ) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            n_samples,
            n_features,
            data_offset,
            config,
            mmap_dataset: None,
        }
    }

    /// Initialize the memory mapping (called on first access)
    fn ensure_mapped(&mut self) -> Result<()> {
        if self.mmap_dataset.is_none() {
            let mmap = MmapDataset::from_binary(
                &self.path,
                self.n_samples,
                self.n_features,
                self.data_offset,
                self.config.clone(),
            )?;
            self.mmap_dataset = Some(Arc::new(mmap));
        }
        Ok(())
    }

    /// Load a specific range of samples
    pub fn load_range(&mut self, start: usize, end: usize) -> Result<Array2<f64>> {
        self.ensure_mapped()?;
        self.mmap_dataset
            .as_ref()
            .ok_or_else(|| DatasetsError::InvalidFormat("Dataset not mapped".to_string()))?
            .view_range(start, end)
    }

    /// Load all data into memory
    pub fn load_all(&mut self) -> Result<Dataset> {
        let data = self.load_range(0, self.n_samples)?;
        Ok(Dataset {
            data,
            target: None,
            targetnames: None,
            featurenames: None,
            feature_descriptions: None,
            description: None,
            metadata: Default::default(),
        })
    }

    /// Get dataset dimensions
    pub fn shape(&self) -> (usize, usize) {
        (self.n_samples, self.n_features)
    }
}

/// Create a lazy-loaded dataset from a binary file
///
/// # Arguments
/// * `path` - Path to the binary file (f64 in row-major order)
/// * `n_samples` - Number of samples
/// * `n_features` - Number of features
///
/// # Returns
/// * `Ok(LazyDataset)` - The lazy dataset
/// * `Err(DatasetsError)` - If creation fails
pub fn from_binary<P: AsRef<Path>>(
    path: P,
    n_samples: usize,
    n_features: usize,
) -> Result<LazyDataset> {
    Ok(LazyDataset::new(
        path,
        n_samples,
        n_features,
        0,
        LazyLoadConfig::default(),
    ))
}

/// Create a lazy-loaded dataset with custom configuration
pub fn from_binary_with_config<P: AsRef<Path>>(
    path: P,
    n_samples: usize,
    n_features: usize,
    config: LazyLoadConfig,
) -> Result<LazyDataset> {
    Ok(LazyDataset::new(path, n_samples, n_features, 0, config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_lazy_load_config() {
        let config = LazyLoadConfig::new()
            .with_target_memory(256 * 1024 * 1024)
            .with_chunk_size_range(500, 50_000)
            .with_mmap(true)
            .with_prefetch(false);

        assert_eq!(config.target_memory_bytes, 256 * 1024 * 1024);
        assert_eq!(config.min_chunk_size, 500);
        assert_eq!(config.max_chunk_size, 50_000);
        assert!(config.use_mmap);
        assert!(!config.prefetch);
    }

    #[test]
    fn test_mmap_dataset() -> Result<()> {
        // Create a temporary binary file with test data
        let temp_dir = tempfile::tempdir().map_err(|e| {
            DatasetsError::InvalidFormat(format!("Failed to create temp dir: {}", e))
        })?;
        let file_path = temp_dir.path().join("test_data.bin");
        let mut file = File::create(&file_path).map_err(DatasetsError::IoError)?;

        // Write 10 samples with 3 features each
        let data: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<f64>(),
            )
        };
        file.write_all(bytes).map_err(DatasetsError::IoError)?;
        drop(file);

        // Create memory-mapped dataset
        let config = LazyLoadConfig::default();
        let mmap_ds = MmapDataset::from_binary(&file_path, 10, 3, 0, config)?;

        assert_eq!(mmap_ds.n_samples(), 10);
        assert_eq!(mmap_ds.n_features(), 3);

        // Test viewing a range
        let view = mmap_ds.view_range(0, 3)?;
        assert_eq!(view.nrows(), 3);
        assert_eq!(view.ncols(), 3);
        assert_eq!(view[[0, 0]], 0.0);
        assert_eq!(view[[2, 2]], 8.0);

        Ok(())
    }

    #[test]
    fn test_adaptive_chunking() -> Result<()> {
        let temp_dir = tempfile::tempdir().map_err(|e| {
            DatasetsError::InvalidFormat(format!("Failed to create temp dir: {}", e))
        })?;
        let file_path = temp_dir.path().join("test_adaptive.bin");
        let mut file = File::create(&file_path).map_err(DatasetsError::IoError)?;

        // Write 1000 samples with 10 features each
        let data: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<f64>(),
            )
        };
        file.write_all(bytes).map_err(DatasetsError::IoError)?;
        drop(file);

        // Configure for small memory footprint
        let config = LazyLoadConfig::new()
            .with_target_memory(8000) // 8KB = ~100 samples
            .with_chunk_size_range(10, 200);

        let mmap_ds = MmapDataset::from_binary(&file_path, 1000, 10, 0, config)?;
        let chunk_size = mmap_ds.adaptive_chunk_size();

        // Should adapt to fit within memory constraints
        assert!(chunk_size >= 10);
        assert!(chunk_size <= 200);

        Ok(())
    }
}
