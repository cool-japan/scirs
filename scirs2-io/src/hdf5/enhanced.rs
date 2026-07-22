//! Enhanced HDF5 functionality with compression accounting, parallel reads, and
//! extended data type support
//!
//! This module extends the basic HDF5 functionality with:
//! - Compression option modelling (gzip, szip, lzf, shuffle, fletcher32)
//! - Chunk-parallel reads backed by oxih5's hyperslab selection
//! - Extended data type support (all primitive types, compound types)
//! - Proper group hierarchy navigation
//! - Thread-safe operations
//! - Advanced chunking strategies
//!
//! # Backend
//!
//! Everything here runs on `oxih5`, the pure-Rust HDF5 implementation, and is
//! compiled unconditionally — the `hdf5` Cargo feature is a retained no-op alias
//! (see the [parent module](super)).
//!
//! Until this migration the module carried a second, feature-gated "native" path
//! that took priority whenever a `libhdf5` handle existed. That path created the
//! dataset but never wrote its data, so turning the feature on silently discarded
//! everything handed to [`EnhancedHDF5File::create_dataset_with_compression`].
//! It has been deleted, and the path that actually stores data is now the only
//! path.

use crate::error::{IoError, Result};
use crate::hdf5::{CompressionOptions, DatasetOptions, FileMode, HDF5File};
use scirs2_core::ndarray::{ArrayBase, ArrayD, IxDyn};
use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::thread;
use std::time::Instant;

use super::convert::dataset_to_f64;

/// Take a lock, recovering the guard if a previous holder panicked.
///
/// The state behind every lock in this module is accumulated statistics, never
/// an invariant a panic could leave half-updated, so poisoning carries no
/// information worth propagating — and `unwrap()`/`expect()` on a lock is
/// exactly the panic-on-panic pattern the no-unwrap policy exists to remove.
fn lock_mutex<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Shared-read counterpart of [`lock_mutex`].
fn lock_read<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Exclusive-write counterpart of [`lock_mutex`].
fn lock_write<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Extended data type support for HDF5
#[derive(Debug, Clone, PartialEq)]
pub enum ExtendedDataType {
    /// 8-bit signed integer
    Int8,
    /// 8-bit unsigned integer
    UInt8,
    /// 16-bit signed integer
    Int16,
    /// 16-bit unsigned integer
    UInt16,
    /// 32-bit signed integer
    Int32,
    /// 32-bit unsigned integer
    UInt32,
    /// 64-bit signed integer
    Int64,
    /// 64-bit unsigned integer
    UInt64,
    /// 32-bit floating point
    Float32,
    /// 64-bit floating point
    Float64,
    /// Complex 64-bit (32-bit real + 32-bit imaginary)
    Complex64,
    /// Complex 128-bit (64-bit real + 64-bit imaginary)
    Complex128,
    /// Boolean
    Bool,
    /// Variable-length UTF-8 string
    String,
    /// Fixed-length UTF-8 string
    FixedString(usize),
}

/// Parallel I/O configuration
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of parallel workers
    pub num_workers: usize,
    /// Chunk size for parallel processing
    pub chunk_size: usize,
    /// Enable collective I/O (requires MPI)
    pub collective_io: bool,
    /// Buffer size for parallel I/O
    pub buffer_size: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_workers: thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
            chunk_size: 1024 * 1024, // 1MB chunks
            collective_io: false,
            buffer_size: 64 * 1024 * 1024, // 64MB buffer
        }
    }
}

/// Enhanced HDF5 file with compression and parallel I/O support
pub struct EnhancedHDF5File {
    /// Base HDF5 file
    base_file: HDF5File,
    /// Parallel configuration
    parallel_config: Option<ParallelConfig>,
    /// Thread-safe access
    file_lock: Arc<RwLock<()>>,
    /// Compression statistics
    compression_stats: Arc<Mutex<CompressionStats>>,
}

/// Compression statistics
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    /// Original size in bytes
    pub original_size: usize,
    /// Compressed size in bytes
    pub compressed_size: usize,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Compression time in milliseconds
    pub compression_time_ms: f64,
}

impl EnhancedHDF5File {
    /// Create a new enhanced HDF5 file with parallel I/O support
    pub fn create<P: AsRef<Path>>(
        path: P,
        parallel_config: Option<ParallelConfig>,
    ) -> Result<Self> {
        let base_file = HDF5File::create(path)?;

        Ok(Self {
            base_file,
            parallel_config,
            file_lock: Arc::new(RwLock::new(())),
            compression_stats: Arc::new(Mutex::new(CompressionStats::default())),
        })
    }

    /// Open an enhanced HDF5 file with parallel I/O support
    pub fn open<P: AsRef<Path>>(
        path: P,
        mode: FileMode,
        parallel_config: Option<ParallelConfig>,
    ) -> Result<Self> {
        let base_file = HDF5File::open(path, mode)?;

        Ok(Self {
            base_file,
            parallel_config,
            file_lock: Arc::new(RwLock::new(())),
            compression_stats: Arc::new(Mutex::new(CompressionStats::default())),
        })
    }

    /// Create a dataset from `array`, storing every element.
    ///
    /// Elements are widened to `f64` through the `Into<f64>` bound and handed to
    /// [`HDF5File::create_dataset_from_array`]. `options` — chunking,
    /// compression, fletcher32 — is recorded on the dataset; oxih5's writer
    /// currently emits contiguous, uncompressed storage, so those settings do not
    /// yet change the bytes on disk, and
    /// [`EnhancedHDF5File::get_compression_stats`] measures the real ratio rather
    /// than assuming one.
    ///
    /// A chunk shape whose rank disagrees with the array is replaced by
    /// the private `calculate_optimal_chunks` helper instead of being recorded
    /// as given.
    ///
    /// Until the oxih5 migration this method preferred a `libhdf5`-backed branch
    /// whenever the `hdf5` feature was on, and that branch created the dataset
    /// and then wrote nothing. The storing branch — this one — was therefore
    /// skipped in exactly the configuration users enabled in order to get
    /// compression. `test_create_dataset_with_compression_round_trips_values`
    /// pins the repaired behaviour.
    pub fn create_dataset_with_compression<A, D>(
        &mut self,
        path: &str,
        array: &ArrayBase<A, D>,
        _data_type: ExtendedDataType,
        options: DatasetOptions,
    ) -> Result<()>
    where
        A: scirs2_core::ndarray::Data,
        A::Elem: Clone + Into<f64> + std::fmt::Debug,
        D: scirs2_core::ndarray::Dimension,
    {
        let start_time = Instant::now();
        let shape: Vec<usize> = array.shape().to_vec();
        let payload_bytes = array.len() * std::mem::size_of::<f64>();

        let mut options = options;
        if options
            .chunk_size
            .as_ref()
            .is_some_and(|chunks| chunks.len() != shape.len())
        {
            options.chunk_size = Some(self.calculate_optimal_chunks(&shape, array.len()));
        }

        {
            // The handle is cloned so the guard borrows the local `Arc` and
            // leaves `self` free for the `&mut` call underneath it.
            let file_lock = Arc::clone(&self.file_lock);
            let _guard = lock_write(&file_lock);
            self.base_file
                .create_dataset_from_array(path, array, Some(options))?;
        }

        let mut stats = lock_mutex(&self.compression_stats);
        stats.original_size += payload_bytes;
        stats.compression_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }

    /// Calculate optimal chunk sizes based on data shape and size
    fn calculate_optimal_chunks(&self, shape: &[usize], _totalelements: usize) -> Vec<usize> {
        const TARGET_CHUNK_SIZE: usize = 64 * 1024; // 64KB target
        const MIN_CHUNK_SIZE: usize = 1024; // 1KB minimum
        const MAX_CHUNK_SIZE: usize = 1024 * 1024; // 1MB maximum

        let element_size = 8; // Assume f64 for now
        let elements_per_chunk = (TARGET_CHUNK_SIZE / element_size)
            .clamp(MIN_CHUNK_SIZE / element_size, MAX_CHUNK_SIZE / element_size);

        let mut chunks = shape.to_vec();
        let current_chunk_elements: usize = chunks.iter().product();

        if current_chunk_elements > elements_per_chunk {
            // Scale down the chunks proportionally
            let scale_factor = (elements_per_chunk as f64 / current_chunk_elements as f64)
                .powf(1.0 / shape.len() as f64);

            for chunk in &mut chunks {
                *chunk = (*chunk as f64 * scale_factor).max(1.0) as usize;
            }
        }

        chunks
    }

    /// Read a dataset, using chunk-parallel hyperslab reads where they pay off.
    ///
    /// Without a [`ParallelConfig`] this is plain [`HDF5File::read_dataset`].
    ///
    /// # What banding buys, and when
    ///
    /// [`HDF5File::open`] materialises the whole file eagerly, so a read-only
    /// handle already holds every payload in memory. Going back to disk is only
    /// worth it for datasets oxih5 can serve a band of without touching the rest,
    /// which means the *chunked* ones: a hyperslab there decompresses only the
    /// chunks it overlaps, while a **contiguous** dataset is a single flat run of
    /// bytes that oxih5 reads whole and then slices — so banding one across N
    /// threads would do N times the I/O.
    ///
    /// The previous implementation did precisely that, and worse: every thread
    /// called `read_raw` over the *entire* dataset and then `copy_from_slice`d
    /// into a differently-sized window, which panics on any length mismatch. The
    /// layout is now probed up front and banding happens only when it removes
    /// work.
    pub fn read_dataset_parallel(&self, path: &str) -> Result<ArrayD<f64>> {
        let _guard = lock_read(&self.file_lock);

        match self.parallel_config.as_ref() {
            Some(config) => self.read_dataset_parallel_impl(path, config),
            None => self.base_file.read_dataset(path),
        }
    }

    /// Choose between a banded on-disk read and the in-memory copy.
    fn read_dataset_parallel_impl(
        &self,
        path: &str,
        parallel_config: &ParallelConfig,
    ) -> Result<ArrayD<f64>> {
        // Only a read-only handle is guaranteed to agree with the file on disk.
        // Any other mode may hold in-memory edits that were never flushed, and
        // reading the file back would silently discard them.
        if self.base_file.mode != FileMode::ReadOnly {
            return self.base_file.read_dataset(path);
        }

        let file_path = self.base_file.path.clone();
        let dataset_path = path.trim_start_matches('/').to_string();

        // `dataset_data_extent` succeeds for exactly one shape of dataset:
        // contiguous, unfiltered and fixed-size — a flat run of bytes that a
        // single pass already reads optimally. Everything it rejects (chunked,
        // filtered) is where per-band reads and parallel decompression earn
        // their keep.
        if oxih5::dataset_data_extent(&file_path, &dataset_path).is_ok() {
            return self.base_file.read_dataset(path);
        }

        let shape = self.base_file.get_dataset(path)?.shape.clone();
        let bands = Self::split_into_bands(&shape, parallel_config);
        if bands.len() < 2 {
            return self.base_file.read_dataset(path);
        }

        Self::read_bands_parallel(&file_path, &dataset_path, &shape, &bands)
    }

    /// Partition the leading axis into one contiguous row band per worker.
    ///
    /// Bands are first sized so each carries about `chunk_size` elements, then
    /// the count is capped at `num_workers` so no more threads are spawned than
    /// were asked for. A dataset small enough for a single band comes back as one
    /// band, which the caller reads sequentially.
    fn split_into_bands(shape: &[usize], config: &ParallelConfig) -> Vec<Range<usize>> {
        let Some(&rows) = shape.first() else {
            return Vec::new();
        };
        if rows == 0 {
            return Vec::new();
        }
        // Elements in one row of the leading axis (1 for a 1-D dataset).
        let row_len: usize = shape[1..].iter().product::<usize>().max(1);
        let rows_per_band = config.chunk_size.div_ceil(row_len).max(1);
        let band_count = rows
            .div_ceil(rows_per_band)
            .min(config.num_workers.max(1))
            .max(1);
        let rows_per_band = rows.div_ceil(band_count);

        (0..band_count)
            .map(|i| (i * rows_per_band).min(rows)..((i + 1) * rows_per_band).min(rows))
            .filter(|band| !band.is_empty())
            .collect()
    }

    /// Read every band concurrently and stitch the results back together.
    ///
    /// Each worker maps the file itself — a read-only mapping costs page-table
    /// entries rather than I/O — and asks oxih5 for just its own band. Values
    /// arrive through [`super::convert::dataset_to_f64`], so an f32 or integer
    /// dataset widens instead of failing the way oxih5's exact-match `as_f64()`
    /// would.
    fn read_bands_parallel(
        file_path: &str,
        dataset_path: &str,
        shape: &[usize],
        bands: &[Range<usize>],
    ) -> Result<ArrayD<f64>> {
        let row_len: usize = shape[1..].iter().product::<usize>().max(1);
        let total: usize = shape.iter().product();

        let collected: Vec<Result<(usize, Vec<f64>)>> = thread::scope(|scope| {
            let handles: Vec<_> = bands
                .iter()
                .map(|band| {
                    let band = band.clone();
                    scope.spawn(move || -> Result<(usize, Vec<f64>)> {
                        let file = oxih5::File::open_mmap(file_path).map_err(|e| {
                            IoError::FormatError(format!(
                                "Failed to map '{file_path}' for a parallel read: {e}"
                            ))
                        })?;
                        let mut ranges: Vec<Range<usize>> = Vec::with_capacity(shape.len());
                        ranges.push(band.clone());
                        ranges.extend(shape[1..].iter().map(|&len| 0..len));
                        let slice = file.dataset_slice(dataset_path, &ranges).map_err(|e| {
                            IoError::FormatError(format!(
                                "Failed to read rows {}..{} of '{dataset_path}': {e}",
                                band.start, band.end
                            ))
                        })?;
                        Ok((band.start * row_len, dataset_to_f64(&slice)?))
                    })
                })
                .collect();

            handles
                .into_iter()
                .map(|handle| {
                    handle.join().unwrap_or_else(|_| {
                        Err(IoError::Other(
                            "a parallel HDF5 read worker panicked".to_string(),
                        ))
                    })
                })
                .collect()
        });

        let mut full = vec![0.0f64; total];
        let mut written = 0usize;
        for outcome in collected {
            let (offset, values) = outcome?;
            let end = offset
                .checked_add(values.len())
                .filter(|&end| end <= total)
                .ok_or_else(|| {
                    IoError::FormatError(format!(
                        "a band starting at element {offset} returned {} values, past the \
                         {total} the dataset holds",
                        values.len()
                    ))
                })?;
            full[offset..end].copy_from_slice(&values);
            written += values.len();
        }
        if written != total {
            return Err(IoError::FormatError(format!(
                "parallel read of '{dataset_path}' covered {written} of {total} elements"
            )));
        }

        ArrayD::from_shape_vec(IxDyn(shape), full).map_err(|e| IoError::FormatError(e.to_string()))
    }

    /// Measure the stored payload against the file it serialises to.
    ///
    /// `original_size` and `compression_time_ms` are accumulated by
    /// [`EnhancedHDF5File::create_dataset_with_compression`]. `compressed_size`
    /// is obtained by serialising the current tree with oxih5's `FileWriter` and
    /// measuring the result, so it is a real byte count. The previous
    /// implementation never queried it: it left `compressed_size` at zero,
    /// hard-coded the ratio to `1.0`, and derived `original_size` from an
    /// element count it assumed was `f64`-shaped.
    ///
    /// oxih5's writer emits contiguous, uncompressed storage, so a ratio below
    /// `1.0` is the expected and honest answer — the file also carries the
    /// superblock, object headers and group structures that the raw payload does
    /// not.
    ///
    /// # Errors
    ///
    /// Propagates any failure to lay the file out, exactly as
    /// [`HDF5File::write`] would report it.
    pub fn get_compression_stats(&self) -> Result<CompressionStats> {
        let serialized = self.base_file.serialized_len()?;
        let mut stats = lock_mutex(&self.compression_stats);
        stats.compressed_size = serialized;
        stats.compression_ratio = if serialized > 0 {
            stats.original_size as f64 / serialized as f64
        } else {
            0.0
        };
        Ok(stats.clone())
    }

    /// Write several datasets into the file.
    ///
    /// The name is historical: an [`HDF5File`] is a single in-memory tree, so
    /// concurrent writers would serialise on the same lock and gain nothing.
    /// Every entry goes through
    /// [`EnhancedHDF5File::create_dataset_with_compression`], so every entry is
    /// stored. Datasets are written in name order because `HashMap` iteration
    /// order varies between runs, which would otherwise make the bytes of an
    /// identical output file differ run to run.
    pub fn write_datasets_parallel(
        &mut self,
        datasets: HashMap<String, (ArrayD<f64>, ExtendedDataType, DatasetOptions)>,
    ) -> Result<()> {
        let mut ordered: Vec<_> = datasets.into_iter().collect();
        ordered.sort_by(|a, b| a.0.cmp(&b.0));
        for (path, (array, data_type, options)) in ordered {
            self.create_dataset_with_compression(&path, &array, data_type, options)?;
        }
        Ok(())
    }

    /// Close the enhanced file
    pub fn close(self) -> Result<()> {
        self.base_file.close()
    }
}

/// Enhanced write function with compression and parallel I/O
pub fn write_hdf5_enhanced<P: AsRef<Path>>(
    path: P,
    datasets: HashMap<String, (ArrayD<f64>, ExtendedDataType, DatasetOptions)>,
    parallel_config: Option<ParallelConfig>,
) -> Result<()> {
    let mut file = EnhancedHDF5File::create(path, parallel_config)?;
    file.write_datasets_parallel(datasets)?;
    file.close()?;
    Ok(())
}

/// Enhanced read function with parallel I/O
pub fn read_hdf5_enhanced<P: AsRef<Path>>(
    path: P,
    parallel_config: Option<ParallelConfig>,
) -> Result<EnhancedHDF5File> {
    EnhancedHDF5File::open(path, FileMode::ReadOnly, parallel_config)
}

/// Utility function to create optimal compression options
pub fn create_optimal_compression_options(
    data_type: &ExtendedDataType,
    estimated_size: usize,
) -> CompressionOptions {
    let mut options = CompressionOptions::default();

    // Choose compression based on data _type and _size
    match data_type {
        ExtendedDataType::Float32 | ExtendedDataType::Float64 => {
            // Floating point data compresses well with shuffle + gzip
            options.shuffle = true;
            options.gzip = Some(if estimated_size > 1024 * 1024 { 6 } else { 9 });
        }
        ExtendedDataType::Int8 | ExtendedDataType::UInt8 => {
            // Small integers often compress well with LZF for speed
            options.lzf = true;
            options.shuffle = true;
        }
        _ => {
            // Default compression for other types
            options.gzip = Some(6);
            options.shuffle = true;
        }
    }

    options
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array, Array2};

    /// A unique path under the system temp dir, so concurrently running tests
    /// never collide on a filename.
    fn temp_path(tag: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        std::env::temp_dir().join(format!(
            "scirs2_io_enhanced_{tag}_{}_{}.h5",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn test_enhanced_compression_options() {
        let options =
            create_optimal_compression_options(&ExtendedDataType::Float64, 2 * 1024 * 1024);
        assert_eq!(options.gzip, Some(6));
        assert!(options.shuffle);
    }

    #[test]
    fn test_optimal_chunks_calculation() {
        let path = temp_path("chunks");
        let file = EnhancedHDF5File::create(&path, None).expect("create in-memory handle");
        let shape = vec![1000, 1000];
        let total_elements = 1_000_000;

        let chunks = file.calculate_optimal_chunks(&shape, total_elements);
        assert!(chunks.len() == 2);
        assert!(chunks[0] > 0 && chunks[1] > 0);

        let chunk_elements: usize = chunks.iter().product();
        assert!(chunk_elements <= 1024 * 1024 / 8); // Should fit in reasonable memory

        drop(file);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_parallel_config_default() {
        let config = ParallelConfig::default();
        assert!(config.num_workers > 0);
        assert!(config.chunk_size > 0);
        assert!(config.buffer_size > 0);
    }

    /// The regression this whole repair exists for.
    ///
    /// `create_dataset_with_compression` used to hand off to a native branch
    /// that created the dataset and wrote no data, so turning the `hdf5` feature
    /// on silently discarded everything. No test exercised the path with real
    /// bytes, so the loss was invisible.
    #[test]
    fn test_create_dataset_with_compression_round_trips_values() {
        let path = temp_path("round_trip");
        let values = Array2::from_shape_vec((2, 3), vec![1.5, -2.5, 3.0, 4.25, 5.0, -6.75])
            .expect("2x3 literal");

        let mut file = EnhancedHDF5File::create(&path, None).expect("create");
        file.create_dataset_with_compression(
            "measurements",
            &values,
            ExtendedDataType::Float64,
            DatasetOptions::default(),
        )
        .expect("write dataset");
        file.close().expect("flush to disk");

        let reopened = EnhancedHDF5File::open(&path, FileMode::ReadOnly, None).expect("reopen");
        let read_back = reopened
            .read_dataset_parallel("measurements")
            .expect("read dataset");

        assert_eq!(read_back.shape(), &[2, 3]);
        assert_eq!(
            read_back.iter().copied().collect::<Vec<f64>>(),
            values.iter().copied().collect::<Vec<f64>>(),
            "every element must survive the write/read round trip"
        );

        let _ = std::fs::remove_file(&path);
    }

    /// Integer input must reach disk as its real value. The `Into<f64>` bound
    /// replaced an implementation that round-tripped each element through
    /// `format!("{:?}")` and `parse::<f64>()`, silently yielding `0.0` for
    /// anything that did not print as a bare float literal.
    #[test]
    fn test_create_dataset_with_compression_widens_integers() {
        let path = temp_path("widen");
        let values = Array::from_vec(vec![-7i32, 0, 42]).into_dyn();

        let mut file = EnhancedHDF5File::create(&path, None).expect("create");
        file.create_dataset_with_compression(
            "counts",
            &values,
            ExtendedDataType::Int32,
            DatasetOptions::default(),
        )
        .expect("write dataset");
        file.close().expect("flush to disk");

        let reopened = EnhancedHDF5File::open(&path, FileMode::ReadOnly, None).expect("reopen");
        let read_back = reopened.base_file.read_dataset("counts").expect("read");
        assert_eq!(
            read_back.iter().copied().collect::<Vec<f64>>(),
            vec![-7.0, 0.0, 42.0]
        );

        let _ = std::fs::remove_file(&path);
    }

    /// `write_datasets_parallel` funnelled into the same discarding branch, so
    /// `write_hdf5_enhanced` produced files with datasets but no contents.
    #[test]
    fn test_write_datasets_parallel_stores_every_dataset() {
        let path = temp_path("multi");
        let mut datasets = HashMap::new();
        datasets.insert(
            "alpha".to_string(),
            (
                Array::from_vec(vec![1.0, 2.0]).into_dyn(),
                ExtendedDataType::Float64,
                DatasetOptions::default(),
            ),
        );
        datasets.insert(
            "beta".to_string(),
            (
                Array::from_vec(vec![3.0, 4.0, 5.0]).into_dyn(),
                ExtendedDataType::Float64,
                DatasetOptions::default(),
            ),
        );

        write_hdf5_enhanced(&path, datasets, None).expect("write");

        let reopened = EnhancedHDF5File::open(&path, FileMode::ReadOnly, None).expect("reopen");
        let alpha = reopened.base_file.read_dataset("alpha").expect("alpha");
        let beta = reopened.base_file.read_dataset("beta").expect("beta");
        assert_eq!(alpha.iter().copied().collect::<Vec<f64>>(), vec![1.0, 2.0]);
        assert_eq!(
            beta.iter().copied().collect::<Vec<f64>>(),
            vec![3.0, 4.0, 5.0]
        );

        let _ = std::fs::remove_file(&path);
    }

    /// The statistics used to be invented: `compressed_size` stayed at zero and
    /// the ratio was hard-coded to `1.0`.
    #[test]
    fn test_compression_stats_are_measured_not_assumed() {
        let path = temp_path("stats");
        let values = Array::from_vec(vec![0.0f64; 512]).into_dyn();

        let mut file = EnhancedHDF5File::create(&path, None).expect("create");
        file.create_dataset_with_compression(
            "bulk",
            &values,
            ExtendedDataType::Float64,
            DatasetOptions::default(),
        )
        .expect("write dataset");

        let stats = file.get_compression_stats().expect("measure stats");
        assert_eq!(
            stats.original_size,
            512 * 8,
            "the raw payload is counted exactly"
        );
        assert!(
            stats.compressed_size > 0,
            "the serialised size must be queried, not left at zero"
        );
        assert!(
            stats.compressed_size >= stats.original_size,
            "uncompressed storage plus HDF5 metadata cannot be smaller than the payload"
        );
        let expected_ratio = stats.original_size as f64 / stats.compressed_size as f64;
        assert!(
            (stats.compression_ratio - expected_ratio).abs() < f64::EPSILON,
            "the ratio must be derived from the two measured sizes"
        );

        file.close().expect("flush");
        let _ = std::fs::remove_file(&path);
    }

    /// A configured parallel read must return exactly what a sequential read
    /// returns. The old implementation could not: every worker read the whole
    /// dataset and then `copy_from_slice`d a window of a different length, which
    /// panics rather than returning a wrong answer.
    #[test]
    fn test_read_dataset_parallel_matches_sequential() {
        let path = temp_path("parallel");
        let values: Vec<f64> = (0..256).map(|i| f64::from(i) * 0.5).collect();
        let array = Array2::from_shape_vec((32, 8), values.clone()).expect("32x8");

        let mut file = EnhancedHDF5File::create(&path, None).expect("create");
        file.create_dataset_with_compression(
            "grid",
            &array,
            ExtendedDataType::Float64,
            DatasetOptions::default(),
        )
        .expect("write dataset");
        file.close().expect("flush");

        let config = ParallelConfig {
            num_workers: 4,
            chunk_size: 16,
            collective_io: false,
            buffer_size: 1024,
        };
        let parallel = EnhancedHDF5File::open(&path, FileMode::ReadOnly, Some(config))
            .expect("reopen with a parallel config");
        let read_back = parallel
            .read_dataset_parallel("grid")
            .expect("parallel read");

        assert_eq!(read_back.shape(), &[32, 8]);
        assert_eq!(read_back.iter().copied().collect::<Vec<f64>>(), values);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_split_into_bands_tiles_the_leading_axis() {
        let config = ParallelConfig {
            num_workers: 4,
            chunk_size: 8,
            collective_io: false,
            buffer_size: 0,
        };
        let bands = EnhancedHDF5File::split_into_bands(&[10, 4], &config);

        assert!(!bands.is_empty());
        assert_eq!(bands.first().map(|band| band.start), Some(0));
        assert_eq!(bands.last().map(|band| band.end), Some(10));
        assert!(
            bands.len() <= config.num_workers,
            "never more bands than workers were asked for"
        );
        for pair in bands.windows(2) {
            assert_eq!(
                pair[0].end, pair[1].start,
                "bands must tile the axis with no gap and no overlap"
            );
        }
    }

    #[test]
    fn test_split_into_bands_handles_degenerate_shapes() {
        let config = ParallelConfig::default();
        assert!(EnhancedHDF5File::split_into_bands(&[], &config).is_empty());
        assert!(EnhancedHDF5File::split_into_bands(&[0, 5], &config).is_empty());
        // A single row cannot be split, so the caller reads it sequentially.
        assert_eq!(
            EnhancedHDF5File::split_into_bands(&[1, 5], &config).len(),
            1
        );
    }
}

//
// Advanced HDF5 Enhancements
//

use std::collections::BTreeMap;

/// Scientific metadata attribute types
#[derive(Debug, Clone)]
pub enum AttributeValue {
    /// String attribute
    String(String),
    /// Integer attribute
    Integer(i64),
    /// Float attribute
    Float(f64),
    /// Array of floats
    FloatArray(Vec<f64>),
    /// Array of integers
    IntArray(Vec<i64>),
    /// Array of strings
    StringArray(Vec<String>),
    /// Boolean attribute
    Boolean(bool),
}

/// Scientific metadata container
#[derive(Debug, Clone, Default)]
pub struct ScientificMetadata {
    /// Standard attributes
    pub attributes: BTreeMap<String, AttributeValue>,
    /// Units for data
    pub units: Option<String>,
    /// Scale factor for data
    pub scale_factor: Option<f64>,
    /// Add offset for data
    pub add_offset: Option<f64>,
    /// Fill value for missing data
    pub fill_value: Option<f64>,
    /// Valid range for data
    pub valid_range: Option<(f64, f64)>,
    /// Calibration information
    pub calibration: Option<CalibrationInfo>,
    /// Provenance information
    pub provenance: Option<ProvenanceInfo>,
}

/// Calibration information for scientific instruments
#[derive(Debug, Clone)]
pub struct CalibrationInfo {
    /// Calibration date
    pub date: String,
    /// Calibration method
    pub method: String,
    /// Calibration parameters
    pub parameters: BTreeMap<String, f64>,
    /// Accuracy estimate
    pub accuracy: Option<f64>,
    /// Precision estimate
    pub precision: Option<f64>,
}

/// Data provenance information
#[derive(Debug, Clone)]
pub struct ProvenanceInfo {
    /// Data source
    pub source: String,
    /// Processing history
    pub processing_history: Vec<String>,
    /// Creation time
    pub creation_time: String,
    /// Creator information
    pub creator: String,
    /// Software version
    pub software_version: String,
    /// Input files used
    pub input_files: Vec<String>,
}

impl ScientificMetadata {
    /// Create new scientific metadata
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a string attribute
    pub fn add_string_attr<S: Into<String>>(mut self, name: S, value: S) -> Self {
        self.attributes
            .insert(name.into(), AttributeValue::String(value.into()));
        self
    }

    /// Add a numeric attribute
    pub fn add_float_attr<S: Into<String>>(mut self, name: S, value: f64) -> Self {
        self.attributes
            .insert(name.into(), AttributeValue::Float(value));
        self
    }

    /// Add units
    pub fn with_units<S: Into<String>>(mut self, units: S) -> Self {
        self.units = Some(units.into());
        self
    }

    /// Add scale factor and offset
    pub fn with_scaling(mut self, scale_factor: f64, add_offset: f64) -> Self {
        self.scale_factor = Some(scale_factor);
        self.add_offset = Some(add_offset);
        self
    }

    /// Add valid range
    pub fn with_valid_range(mut self, min: f64, max: f64) -> Self {
        self.valid_range = Some((min, max));
        self
    }

    /// Add provenance information
    pub fn with_provenance(mut self, provenance: ProvenanceInfo) -> Self {
        self.provenance = Some(provenance);
        self
    }
}

/// Performance monitoring for HDF5 operations
#[derive(Debug, Clone, Default)]
pub struct HDF5PerformanceMonitor {
    /// Operation timings
    pub timings: BTreeMap<String, Vec<f64>>,
    /// Data transfer statistics
    pub transfer_stats: TransferStats,
    /// Memory usage statistics
    pub memory_stats: MemoryStats,
    /// Compression efficiency
    pub compression_efficiency: Vec<CompressionStats>,
}

/// Bytes moved and how fast they moved, accumulated per direction.
#[derive(Debug, Clone, Default)]
pub struct TransferStats {
    /// Total bytes read
    pub bytes_read: usize,
    /// Total bytes written
    pub bytes_written: usize,
    /// Read operations count
    pub read_operations: usize,
    /// Write operations count
    pub write_operations: usize,
    /// Average read speed (bytes/sec)
    pub avg_read_speed: f64,
    /// Average write speed (bytes/sec)
    pub avg_write_speed: f64,
}

/// Allocation counters and high-water mark for a monitored session.
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Peak memory usage
    pub peak_memory_bytes: usize,
    /// Current memory usage
    pub current_memory_bytes: usize,
    /// Memory allocations count
    pub allocation_count: usize,
    /// Memory deallocations count
    pub deallocation_count: usize,
}

impl HDF5PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an operation timing
    pub fn record_timing(&mut self, operation: &str, durationms: f64) {
        self.timings
            .entry(operation.to_string())
            .or_default()
            .push(durationms);
    }

    /// Record data transfer
    pub fn record_read(&mut self, bytes: usize, durationms: f64) {
        self.transfer_stats.bytes_read += bytes;
        self.transfer_stats.read_operations += 1;

        if durationms > 0.0 {
            let speed = bytes as f64 / (durationms / 1000.0);
            let total_ops = self.transfer_stats.read_operations as f64;
            self.transfer_stats.avg_read_speed =
                (self.transfer_stats.avg_read_speed * (total_ops - 1.0) + speed) / total_ops;
        }
    }

    /// Record data write
    pub fn record_write(&mut self, bytes: usize, durationms: f64) {
        self.transfer_stats.bytes_written += bytes;
        self.transfer_stats.write_operations += 1;

        if durationms > 0.0 {
            let speed = bytes as f64 / (durationms / 1000.0);
            let total_ops = self.transfer_stats.write_operations as f64;
            self.transfer_stats.avg_write_speed =
                (self.transfer_stats.avg_write_speed * (total_ops - 1.0) + speed) / total_ops;
        }
    }

    /// Get average timing for an operation
    pub fn avg_timing(&self, operation: &str) -> Option<f64> {
        self.timings
            .get(operation)
            .map(|times| times.iter().sum::<f64>() / times.len() as f64)
    }

    /// Get performance summary
    pub fn get_summary(&self) -> PerformanceSummary {
        let mut operation_averages = BTreeMap::new();

        for (op, times) in &self.timings {
            let avg = times.iter().sum::<f64>() / times.len() as f64;
            operation_averages.insert(op.clone(), avg);
        }

        PerformanceSummary {
            operation_averages,
            total_bytes_transferred: self.transfer_stats.bytes_read
                + self.transfer_stats.bytes_written,
            avg_read_speed_mbps: self.transfer_stats.avg_read_speed / 1_000_000.0,
            avg_write_speed_mbps: self.transfer_stats.avg_write_speed / 1_000_000.0,
            peak_memory_mb: self.memory_stats.peak_memory_bytes as f64 / 1_000_000.0,
            compression_ratio: self
                .compression_efficiency
                .iter()
                .map(|c| c.compression_ratio)
                .fold(0.0, |acc, x| acc + x)
                / self.compression_efficiency.len().max(1) as f64,
        }
    }
}

/// Performance summary report
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    /// Average timing for each operation type
    pub operation_averages: BTreeMap<String, f64>,
    /// Total bytes transferred
    pub total_bytes_transferred: usize,
    /// Average read speed in MB/s
    pub avg_read_speed_mbps: f64,
    /// Average write speed in MB/s  
    pub avg_write_speed_mbps: f64,
    /// Peak memory usage in MB
    pub peak_memory_mb: f64,
    /// Average compression ratio
    pub compression_ratio: f64,
}

/// Data layout optimization recommendations
#[derive(Debug, Clone)]
pub enum LayoutOptimization {
    /// Row-major layout (C-style)
    RowMajor,
    /// Column-major layout (Fortran-style)
    ColumnMajor,
    /// Chunked layout with specific chunk sizes
    Chunked(Vec<usize>),
    /// Tiled layout for 2D data
    Tiled {
        /// Tile extent along the fastest-varying axis
        tile_width: usize,
        /// Tile extent along the slowest-varying axis
        tile_height: usize,
    },
    /// Strip layout for 1D-like access patterns
    Striped {
        /// Number of elements in one strip
        strip_size: usize,
    },
}

/// Access pattern analysis
#[derive(Debug, Clone)]
pub struct AccessPatternAnalyzer {
    /// Recorded access patterns
    access_patterns: Vec<AccessPattern>,
    /// Current analysis results
    recommendations: Vec<LayoutOptimization>,
}

/// One recorded access, and how often that exact access has been seen.
#[derive(Debug, Clone)]
pub struct AccessPattern {
    /// Operation type (read/write)
    pub operation: String,
    /// Accessed region (start, size for each dimension)
    pub region: Vec<(usize, usize)>,
    /// Frequency of this access pattern
    pub frequency: usize,
    /// Timestamp
    pub timestamp: std::time::Instant,
}

impl AccessPatternAnalyzer {
    /// Create a new access pattern analyzer
    pub fn new() -> Self {
        Self {
            access_patterns: Vec::new(),
            recommendations: Vec::new(),
        }
    }

    /// Record an access pattern
    pub fn record_access(&mut self, operation: String, region: Vec<(usize, usize)>) {
        // Check if this pattern already exists
        for pattern in &mut self.access_patterns {
            if pattern.operation == operation && pattern.region == region {
                pattern.frequency += 1;
                pattern.timestamp = std::time::Instant::now();
                return;
            }
        }

        // Add new pattern
        self.access_patterns.push(AccessPattern {
            operation,
            region,
            frequency: 1,
            timestamp: std::time::Instant::now(),
        });
    }

    /// Analyze patterns and generate recommendations
    pub fn analyze(&mut self) -> &Vec<LayoutOptimization> {
        self.recommendations.clear();

        if self.access_patterns.is_empty() {
            return &self.recommendations;
        }

        // Analyze most frequent patterns
        let mut pattern_analysis = BTreeMap::new();

        for pattern in &self.access_patterns {
            let key = format!("{:?}", pattern.region);
            let entry = pattern_analysis
                .entry(key)
                .or_insert((0, pattern.region.clone()));
            entry.0 += pattern.frequency;
        }

        // Find the most common access pattern
        if let Some((_, (_, most_common_region))) =
            pattern_analysis.iter().max_by_key(|(_, (freq_, _))| *freq_)
        {
            // Generate recommendations based on access patterns
            if most_common_region.len() == 1 {
                // 1D data - recommend striped layout
                let optimal_strip = most_common_region[0].1.max(1024);
                self.recommendations.push(LayoutOptimization::Striped {
                    strip_size: optimal_strip,
                });
            } else if most_common_region.len() == 2 {
                // 2D data - analyze access patterns
                let (_row_access, row_size) = most_common_region[0];
                let (_col_access, col_size) = most_common_region[1];

                if row_size > col_size * 10 {
                    // Row-wise access pattern
                    self.recommendations.push(LayoutOptimization::RowMajor);
                } else if col_size > row_size * 10 {
                    // Column-wise access pattern
                    self.recommendations.push(LayoutOptimization::ColumnMajor);
                } else {
                    // Mixed access - recommend tiled layout
                    let tile_width = col_size.clamp(64, 512);
                    let tile_height = row_size.clamp(64, 512);
                    self.recommendations.push(LayoutOptimization::Tiled {
                        tile_width,
                        tile_height,
                    });
                }
            } else {
                // Multi-dimensional data - recommend chunked layout
                let optimal_chunks: Vec<usize> = most_common_region
                    .iter()
                    .map(|(_, size)| size.clamp(&64, &1024))
                    .cloned()
                    .collect();
                self.recommendations
                    .push(LayoutOptimization::Chunked(optimal_chunks));
            }
        }

        &self.recommendations
    }

    /// Get access pattern statistics
    pub fn get_statistics(&self) -> AccessPatternStats {
        let total_accesses = self.access_patterns.iter().map(|p| p.frequency).sum();
        let unique_patterns = self.access_patterns.len();

        let read_count = self
            .access_patterns
            .iter()
            .filter(|p| p.operation.contains("read"))
            .map(|p| p.frequency)
            .sum();

        let write_count = total_accesses - read_count;

        AccessPatternStats {
            total_accesses,
            unique_patterns,
            read_count,
            write_count,
            most_frequent_pattern: self
                .access_patterns
                .iter()
                .max_by_key(|p| p.frequency)
                .map(|p| p.region.clone()),
        }
    }
}

impl Default for AccessPatternAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Access pattern statistics
#[derive(Debug, Clone)]
pub struct AccessPatternStats {
    /// Total number of accesses recorded
    pub total_accesses: usize,
    /// Number of unique access patterns
    pub unique_patterns: usize,
    /// Number of read operations
    pub read_count: usize,
    /// Number of write operations
    pub write_count: usize,
    /// Most frequently accessed region
    pub most_frequent_pattern: Option<Vec<(usize, usize)>>,
}

/// Enhanced HDF5 file with full monitoring and optimization
pub struct OptimizedHDF5File {
    /// Base enhanced file
    pub base_file: EnhancedHDF5File,
    /// Performance monitor
    pub performance_monitor: Arc<Mutex<HDF5PerformanceMonitor>>,
    /// Access pattern analyzer
    pub access_analyzer: Arc<Mutex<AccessPatternAnalyzer>>,
    /// Metadata cache
    pub metadata_cache: Arc<RwLock<BTreeMap<String, ScientificMetadata>>>,
}

impl OptimizedHDF5File {
    /// Create a new optimized HDF5 file
    pub fn create<P: AsRef<Path>>(
        path: P,
        parallel_config: Option<ParallelConfig>,
    ) -> Result<Self> {
        let base_file = EnhancedHDF5File::create(path, parallel_config)?;

        Ok(Self {
            base_file,
            performance_monitor: Arc::new(Mutex::new(HDF5PerformanceMonitor::new())),
            access_analyzer: Arc::new(Mutex::new(AccessPatternAnalyzer::new())),
            metadata_cache: Arc::new(RwLock::new(BTreeMap::new())),
        })
    }

    /// Open an optimized HDF5 file
    pub fn open<P: AsRef<Path>>(
        path: P,
        mode: FileMode,
        parallel_config: Option<ParallelConfig>,
    ) -> Result<Self> {
        let base_file = EnhancedHDF5File::open(path, mode, parallel_config)?;

        Ok(Self {
            base_file,
            performance_monitor: Arc::new(Mutex::new(HDF5PerformanceMonitor::new())),
            access_analyzer: Arc::new(Mutex::new(AccessPatternAnalyzer::new())),
            metadata_cache: Arc::new(RwLock::new(BTreeMap::new())),
        })
    }

    /// Add scientific metadata to a dataset
    pub fn add_scientific_metadata(
        &mut self,
        dataset_path: &str,
        metadata: ScientificMetadata,
    ) -> Result<()> {
        // Cache the metadata
        {
            let mut cache = lock_write(&self.metadata_cache);
            cache.insert(dataset_path.to_string(), metadata.clone());
        }

        // In a real implementation, this would write the metadata as HDF5 attributes
        // For now, we just cache it for retrieval
        Ok(())
    }

    /// Get scientific metadata for a dataset
    pub fn get_scientific_metadata(&self, datasetpath: &str) -> Option<ScientificMetadata> {
        let cache = lock_read(&self.metadata_cache);
        cache.get(datasetpath).cloned()
    }

    /// Get performance report
    pub fn get_performance_report(&self) -> PerformanceSummary {
        let monitor = lock_mutex(&self.performance_monitor);
        monitor.get_summary()
    }

    /// Get layout optimization recommendations
    pub fn get_layout_recommendations(&self) -> Vec<LayoutOptimization> {
        let mut analyzer = lock_mutex(&self.access_analyzer);
        analyzer.analyze().clone()
    }

    /// Record a data access for optimization analysis
    pub fn record_access(&self, operation: &str, region: Vec<(usize, usize)>) {
        let mut analyzer = lock_mutex(&self.access_analyzer);
        analyzer.record_access(operation.to_string(), region);
    }

    /// Get access pattern statistics
    pub fn get_access_statistics(&self) -> AccessPatternStats {
        let analyzer = lock_mutex(&self.access_analyzer);
        analyzer.get_statistics()
    }

    /// Benchmark a specific operation
    pub fn benchmark_operation<F, R>(&self, operationname: &str, operation: F) -> Result<R>
    where
        F: FnOnce() -> Result<R>,
    {
        let start_time = Instant::now();
        let result = operation()?;
        let duration = start_time.elapsed().as_secs_f64() * 1000.0;

        {
            let mut monitor = lock_mutex(&self.performance_monitor);
            monitor.record_timing(operationname, duration);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod enhanced_tests {
    use super::*;

    #[test]
    fn test_scientific_metadata() {
        let metadata = ScientificMetadata::new()
            .add_string_attr("instrument", "spectrometer")
            .add_float_attr("wavelength", 550.0)
            .with_units("nanometers")
            .with_scaling(1.0, 0.0)
            .with_valid_range(0.0, 1000.0);

        assert_eq!(metadata.units, Some("nanometers".to_string()));
        assert_eq!(metadata.scale_factor, Some(1.0));
        assert_eq!(metadata.valid_range, Some((0.0, 1000.0)));
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = HDF5PerformanceMonitor::new();

        monitor.record_timing("read", 10.0);
        monitor.record_timing("read", 20.0);
        monitor.record_read(1024, 10.0);

        assert_eq!(monitor.avg_timing("read"), Some(15.0));
        assert_eq!(monitor.transfer_stats.bytes_read, 1024);
        assert_eq!(monitor.transfer_stats.read_operations, 1);
    }

    #[test]
    fn test_access_pattern_analyzer() {
        let mut analyzer = AccessPatternAnalyzer::new();

        // Record some access patterns
        analyzer.record_access("read".to_string(), vec![(0, 100), (0, 50)]);
        analyzer.record_access("read".to_string(), vec![(0, 100), (0, 50)]);
        analyzer.record_access("write".to_string(), vec![(100, 100), (50, 50)]);

        let stats = analyzer.get_statistics();
        assert_eq!(stats.total_accesses, 3);
        assert_eq!(stats.unique_patterns, 2);
        assert_eq!(stats.read_count, 2);

        let recommendations = analyzer.analyze();
        assert!(!recommendations.is_empty());
    }
}
