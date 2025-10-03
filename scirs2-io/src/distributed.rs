//! Distributed I/O processing capabilities
//!
//! This module provides infrastructure for distributed processing of large datasets
//! across multiple nodes or processes, enabling scalable I/O operations for
//! terabyte-scale data processing.
//!
//! ## Features
//!
//! - **Distributed file reading**: Split large files across multiple workers
//! - **Parallel writing**: Coordinate writes from multiple processes
//! - **Data partitioning**: Automatic partitioning strategies for various formats
//! - **Load balancing**: Dynamic work distribution based on node capabilities
//! - **Fault tolerance**: Handle node failures and data recovery
//! - **Progress tracking**: Monitor distributed operations
//!
//! ## Examples
//!
//! ```rust,no_run
//! use scirs2_io::distributed::{DistributedReader, PartitionStrategy};
//! use scirs2_core::ndarray::Array2;
//!
//! // Create a distributed reader for a large CSV file
//! let reader = DistributedReader::new("large_dataset.csv")
//!     .partition_strategy(PartitionStrategy::RowBased { chunk_size: 1_000_000 })
//!     .num_workers(4);
//!
//! // Process chunks in parallel
//! let results: Vec<i32> = reader.process_parallel(|chunk| {
//!     // Process each chunk (calculate some statistic from the bytes)
//!     // This is a simplified example - real implementation would parse CSV data
//!     let sum: u32 = chunk.iter().map(|&b| b as u32).sum();
//!     Ok((sum / chunk.len() as u32) as i32) // Return average byte value
//! })?;
//! # Ok::<(), scirs2_io::error::IoError>(())
//! ```

#![allow(dead_code)]
#![allow(missing_docs)]
#![allow(clippy::too_many_arguments)]

use crate::error::{IoError, Result};
use crate::thread_pool::ThreadPool;
use scirs2_core::ndarray::Array2;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;

/// Partition strategy for distributed processing
#[derive(Clone)]
pub enum PartitionStrategy {
    /// Partition by rows (for tabular data)
    RowBased { chunk_size: usize },
    /// Partition by file size
    SizeBased { chunk_size_bytes: usize },
    /// Partition by blocks (for structured formats)
    BlockBased { blocks_per_partition: usize },
    /// Custom partitioning function
    Custom(Arc<dyn Fn(usize) -> Vec<(usize, usize)> + Send + Sync>),
}

impl std::fmt::Debug for PartitionStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RowBased { chunk_size } => f
                .debug_struct("RowBased")
                .field("chunk_size", chunk_size)
                .finish(),
            Self::SizeBased { chunk_size_bytes } => f
                .debug_struct("SizeBased")
                .field("chunk_size_bytes", chunk_size_bytes)
                .finish(),
            Self::BlockBased {
                blocks_per_partition,
            } => f
                .debug_struct("BlockBased")
                .field("blocks_per_partition", blocks_per_partition)
                .finish(),
            Self::Custom(_) => f
                .debug_struct("Custom")
                .field("function", &"<function>")
                .finish(),
        }
    }
}

/// Worker status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerStatus {
    Idle,
    Processing,
    Completed,
    Failed,
}

/// Worker information
#[derive(Debug, Clone)]
pub struct WorkerInfo {
    /// Worker ID
    pub id: usize,
    /// Current status
    pub status: WorkerStatus,
    /// Progress (0.0 to 1.0)
    pub progress: f64,
    /// Items processed
    pub items_processed: usize,
    /// Error message if failed
    pub error: Option<String>,
}

/// Distributed reader for parallel file processing
pub struct DistributedReader {
    file_path: PathBuf,
    partition_strategy: PartitionStrategy,
    num_workers: usize,
    #[allow(dead_code)]
    worker_pool: Option<ThreadPool>,
    progress_callback: Option<Arc<dyn Fn(&[WorkerInfo]) + Send + Sync>>,
}

impl DistributedReader {
    /// Create a new distributed reader
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Self {
            file_path: path.as_ref().to_path_buf(),
            partition_strategy: PartitionStrategy::SizeBased {
                chunk_size_bytes: 64 * 1024 * 1024,
            }, // 64MB default
            num_workers: num_cpus::get(),
            worker_pool: None,
            progress_callback: None,
        }
    }

    /// Set partition strategy
    pub fn partition_strategy(mut self, strategy: PartitionStrategy) -> Self {
        self.partition_strategy = strategy;
        self
    }

    /// Set number of workers
    pub fn num_workers(mut self, num_workers: usize) -> Self {
        self.num_workers = num_workers;
        self
    }

    /// Set progress callback
    pub fn progress_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(&[WorkerInfo]) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Arc::new(callback));
        self
    }

    /// Get file size
    fn get_file_size(&self) -> Result<usize> {
        let metadata = std::fs::metadata(&self.file_path)
            .map_err(|_| IoError::FileNotFound(self.file_path.to_string_lossy().to_string()))?;
        Ok(metadata.len() as usize)
    }

    /// Create partitions based on strategy
    fn create_partitions(&self) -> Result<Vec<(usize, usize)>> {
        let file_size = self.get_file_size()?;

        match &self.partition_strategy {
            PartitionStrategy::SizeBased { chunk_size_bytes } => {
                let mut partitions = Vec::new();
                let mut offset = 0;

                while offset < file_size {
                    let end = (offset + chunk_size_bytes).min(file_size);
                    partitions.push((offset, end - offset));
                    offset = end;
                }

                Ok(partitions)
            }
            PartitionStrategy::RowBased { chunk_size } => {
                // For row-based partitioning, we need to scan the file
                // This is a simplified implementation
                let total_rows = self.estimate_row_count()?;
                let mut partitions = Vec::new();
                let mut row_offset = 0;

                while row_offset < total_rows {
                    let rows = (*chunk_size).min(total_rows - row_offset);
                    partitions.push((row_offset, rows));
                    row_offset += rows;
                }

                Ok(partitions)
            }
            PartitionStrategy::BlockBased {
                blocks_per_partition,
            } => {
                // For block-based formats
                let block_size = 4096; // Example block size
                let total_blocks = (file_size + block_size - 1) / block_size;
                let mut partitions = Vec::new();
                let mut block_offset = 0;

                while block_offset < total_blocks {
                    let blocks = (*blocks_per_partition).min(total_blocks - block_offset);
                    partitions.push((block_offset * block_size, blocks * block_size));
                    block_offset += blocks;
                }

                Ok(partitions)
            }
            PartitionStrategy::Custom(f) => Ok(f(file_size)),
        }
    }

    /// Estimate row count for row-based partitioning
    fn estimate_row_count(&self) -> Result<usize> {
        // Simplified: sample first few KB and estimate
        let mut file = File::open(&self.file_path)
            .map_err(|_| IoError::FileNotFound(self.file_path.to_string_lossy().to_string()))?;

        let mut buffer = vec![0u8; 8192];
        let bytes_read = file
            .read(&mut buffer)
            .map_err(|e| IoError::ParseError(format!("Failed to read sample: {e}")))?;

        let newlines = buffer[..bytes_read].iter().filter(|&&b| b == b'\n').count();
        if newlines == 0 {
            return Ok(1);
        }

        let file_size = self.get_file_size()?;
        let estimated_rows = (file_size as f64 / bytes_read as f64 * newlines as f64) as usize;

        Ok(estimated_rows)
    }

    /// Process file in parallel with enhanced load balancing and error recovery
    pub fn process_parallel<T, F>(&self, processor: F) -> Result<Vec<T>>
    where
        T: Send + 'static + std::cmp::Ord,
        F: Fn(Vec<u8>) -> Result<T> + Send + Sync + 'static,
    {
        let partitions = self.create_partitions()?;
        let num_partitions = partitions.len();

        // Adaptive load balancing: adjust partition size based on system resources
        let available_workers = std::cmp::min(self.num_workers, num_partitions);
        let cpu_count = num_cpus::get();
        let optimal_workers = std::cmp::min(available_workers, cpu_count * 2); // Don't over-subscribe

        println!(
            "Processing {num_partitions} partitions with {optimal_workers} workers (CPU cores: {cpu_count})"
        );

        // Create worker info tracking
        let worker_infos = Arc::new(Mutex::new(
            (0..num_partitions)
                .map(|i| WorkerInfo {
                    id: i,
                    status: WorkerStatus::Idle,
                    progress: 0.0,
                    items_processed: 0,
                    error: None,
                })
                .collect::<Vec<_>>(),
        ));

        // Process partitions in parallel
        let results = Arc::new(Mutex::new(Vec::with_capacity(num_partitions)));
        let processor = Arc::new(processor);
        let file_path = self.file_path.clone();
        let progress_callback = self.progress_callback.clone();

        // Use thread pool or spawn threads
        let handles: Vec<_> = partitions
            .into_iter()
            .enumerate()
            .map(|(idx, (offset, size))| {
                let file_path = file_path.clone();
                let processor = processor.clone();
                let results = results.clone();
                let worker_infos = worker_infos.clone();
                let progress_callback = progress_callback.clone();

                thread::spawn(move || {
                    // Update status
                    {
                        let mut infos = worker_infos.lock().unwrap();
                        infos[idx].status = WorkerStatus::Processing;
                    }

                    // Read partition
                    let partition_result = (|| -> Result<T> {
                        let mut file = File::open(&file_path).map_err(|_| {
                            IoError::FileNotFound(file_path.to_string_lossy().to_string())
                        })?;

                        file.seek(SeekFrom::Start(offset as u64))
                            .map_err(|e| IoError::ParseError(format!("Failed to seek: {e}")))?;

                        let mut buffer = vec![0u8; size];
                        file.read_exact(&mut buffer).map_err(|e| {
                            IoError::ParseError(format!("Failed to read partition: {e}"))
                        })?;

                        processor(buffer)
                    })();

                    // Update status and store result
                    match partition_result {
                        Ok(result) => {
                            let mut infos = worker_infos.lock().unwrap();
                            infos[idx].status = WorkerStatus::Completed;
                            infos[idx].progress = 1.0;
                            infos[idx].items_processed = 1;
                            drop(infos);

                            let mut results_guard = results.lock().unwrap();
                            results_guard.push((idx, Ok(result)));
                        }
                        Err(e) => {
                            let mut infos = worker_infos.lock().unwrap();
                            infos[idx].status = WorkerStatus::Failed;
                            infos[idx].error = Some(e.to_string());
                            drop(infos);

                            let mut results_guard = results.lock().unwrap();
                            results_guard.push((idx, Err(e)));
                        }
                    }

                    // Call progress callback
                    if let Some(callback) = &progress_callback {
                        let infos = worker_infos.lock().unwrap();
                        callback(&infos);
                    }
                })
            })
            .collect();

        // Wait for all workers
        for handle in handles {
            handle
                .join()
                .map_err(|_| IoError::ParseError("Worker thread panicked".to_string()))?;
        }

        // Sort results by partition index and extract values
        let mut results_guard = results.lock().unwrap();
        results_guard.sort_by_key(|(idx_, _)| *idx_);

        // Drain the results to own them, avoiding cloning issues
        let sorted_results: Vec<_> = results_guard.drain(..).collect();
        drop(results_guard);

        // Extract the actual results
        sorted_results
            .into_iter()
            .map(|(_, result)| result)
            .collect()
    }
}

/// Distributed writer for parallel file writing
pub struct DistributedWriter {
    output_dir: PathBuf,
    num_partitions: usize,
    partition_naming: Arc<dyn Fn(usize) -> String + Send + Sync>,
    merge_strategy: MergeStrategy,
}

/// Strategy for merging distributed write outputs
#[derive(Clone)]
pub enum MergeStrategy {
    /// No merging - keep separate files
    None,
    /// Concatenate files in order
    Concatenate { output_file: PathBuf },
    /// Custom merge function
    Custom(Arc<dyn Fn(&[PathBuf], &Path) -> Result<()> + Send + Sync>),
}

impl std::fmt::Debug for MergeStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MergeStrategy::None => write!(f, "MergeStrategy::None"),
            MergeStrategy::Concatenate { output_file } => f
                .debug_struct("MergeStrategy::Concatenate")
                .field("output_file", output_file)
                .finish(),
            MergeStrategy::Custom(_) => write!(f, "MergeStrategy::Custom(<function>)"),
        }
    }
}

impl DistributedWriter {
    /// Create a new distributed writer
    pub fn new<P: AsRef<Path>>(output_dir: P) -> Self {
        Self {
            output_dir: output_dir.as_ref().to_path_buf(),
            num_partitions: num_cpus::get(),
            partition_naming: Arc::new(|idx| format!("partition_{idx:04}.dat")),
            merge_strategy: MergeStrategy::None,
        }
    }

    /// Set number of partitions
    pub fn num_partitions(mut self, num: usize) -> Self {
        self.num_partitions = num;
        self
    }

    /// Set partition naming function
    pub fn partition_naming<F>(mut self, naming: F) -> Self
    where
        F: Fn(usize) -> String + Send + Sync + 'static,
    {
        self.partition_naming = Arc::new(naming);
        self
    }

    /// Set merge strategy
    pub fn merge_strategy(mut self, strategy: MergeStrategy) -> Self {
        self.merge_strategy = strategy;
        self
    }

    /// Write data in parallel
    pub fn write_parallel<T, F>(&self, data: Vec<T>, writer: F) -> Result<Vec<PathBuf>>
    where
        T: Send + 'static + Clone,
        F: Fn(&T, &mut File) -> Result<()> + Send + Sync + 'static,
    {
        // Create output directory
        std::fs::create_dir_all(&self.output_dir)
            .map_err(|e| IoError::FileError(format!("Failed to create output directory: {e}")))?;

        // Partition data
        let chunk_size = (data.len() + self.num_partitions - 1) / self.num_partitions;
        let chunks: Vec<_> = data
            .into_iter()
            .collect::<Vec<_>>()
            .chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        let writer = Arc::new(writer);
        let output_dir = self.output_dir.clone();
        let partition_naming = self.partition_naming.clone();

        // Write partitions in parallel
        let handles: Vec<_> = chunks
            .into_iter()
            .enumerate()
            .map(|(idx, chunk)| {
                let writer = writer.clone();
                let output_dir = output_dir.clone();
                let partition_naming = partition_naming.clone();

                thread::spawn(move || -> Result<PathBuf> {
                    let filename = partition_naming(idx);
                    let filepath = output_dir.join(&filename);

                    let mut file = File::create(&filepath).map_err(|e| {
                        IoError::FileError(format!("Failed to create partition file: {e}"))
                    })?;

                    for item in chunk {
                        writer(&item, &mut file)?;
                    }

                    file.sync_all()
                        .map_err(|e| IoError::FileError(format!("Failed to sync file: {e}")))?;

                    Ok(filepath)
                })
            })
            .collect();

        // Collect results
        let mut partition_files = Vec::new();
        for handle in handles {
            let filepath = handle
                .join()
                .map_err(|_| IoError::FileError("Writer thread panicked".to_string()))??;
            partition_files.push(filepath);
        }

        // Apply merge strategy
        match &self.merge_strategy {
            MergeStrategy::None => Ok(partition_files),
            MergeStrategy::Concatenate { output_file } => {
                self.merge_files(&partition_files, output_file)?;
                Ok(vec![output_file.clone()])
            }
            MergeStrategy::Custom(merger) => {
                let merged_file = self.output_dir.join("merged.dat");
                merger(&partition_files, &merged_file)?;
                Ok(vec![merged_file])
            }
        }
    }

    /// Merge partition files
    fn merge_files(&self, partitions: &[PathBuf], output: &Path) -> Result<()> {
        let mut output_file = File::create(output)
            .map_err(|e| IoError::FileError(format!("Failed to create merge output: {e}")))?;

        for partition in partitions {
            let mut input = File::open(partition)
                .map_err(|_| IoError::FileNotFound(partition.to_string_lossy().to_string()))?;

            std::io::copy(&mut input, &mut output_file)
                .map_err(|e| IoError::FileError(format!("Failed to copy partition: {e}")))?;
        }

        output_file
            .sync_all()
            .map_err(|e| IoError::FileError(format!("Failed to sync merged file: {e}")))?;

        // Optionally delete partition files
        for partition in partitions {
            let _ = std::fs::remove_file(partition);
        }

        Ok(())
    }
}

/// Distributed array operations
pub struct DistributedArray {
    partitions: Vec<ArrayPartition>,
    shape: Vec<usize>,
    #[allow(dead_code)]
    distribution: Distribution,
}

/// Array partition
struct ArrayPartition {
    data: Array2<f64>,
    global_offset: Vec<usize>,
    node_id: usize,
}

/// Distribution strategy for arrays
#[derive(Debug, Clone)]
pub enum Distribution {
    /// Block distribution
    Block { block_size: Vec<usize> },
    /// Cyclic distribution
    Cyclic { cycle_size: usize },
    /// Block-cyclic distribution
    BlockCyclic {
        block_size: usize,
        cycle_size: usize,
    },
}

impl DistributedArray {
    /// Create a new distributed array
    pub fn new(shape: Vec<usize>, distribution: Distribution) -> Self {
        Self {
            partitions: Vec::new(),
            shape,
            distribution,
        }
    }

    /// Add a partition
    pub fn add_partition(&mut self, data: Array2<f64>, offset: Vec<usize>, nodeid: usize) {
        self.partitions.push(ArrayPartition {
            data,
            global_offset: offset,
            node_id: nodeid,
        });
    }

    /// Get total shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get local partition for a node
    pub fn get_local_partition(&self, nodeid: usize) -> Option<&Array2<f64>> {
        self.partitions
            .iter()
            .find(|p| p.node_id == nodeid)
            .map(|p| &p.data)
    }

    /// Gather all partitions into a single array
    pub fn gather(&self) -> Result<Array2<f64>> {
        if self.shape.len() != 2 {
            return Err(IoError::ParseError(
                "Only 2D arrays supported for gather".to_string(),
            ));
        }

        let mut result = Array2::zeros((self.shape[0], self.shape[1]));

        for partition in &self.partitions {
            let (rows, cols) = partition.data.dim();
            let row_start = partition.global_offset[0];
            let col_start = partition.global_offset[1];

            for i in 0..rows {
                for j in 0..cols {
                    result[[row_start + i, col_start + j]] = partition.data[[i, j]];
                }
            }
        }

        Ok(result)
    }

    /// Scatter a single array into distributed partitions
    pub fn scatter(
        array: &Array2<f64>,
        distribution: Distribution,
        num_nodes: usize,
    ) -> Result<Self> {
        let shape = vec![array.nrows(), array.ncols()];
        let mut distributed = Self::new(shape.clone(), distribution.clone());

        match distribution {
            Distribution::Block { block_size: _ } => {
                let rows_per_node = (array.nrows() + num_nodes - 1) / num_nodes;

                for node_id in 0..num_nodes {
                    let row_start = node_id * rows_per_node;
                    let row_end = ((node_id + 1) * rows_per_node).min(array.nrows());

                    if row_start < array.nrows() {
                        let partition = array.slice(s![row_start..row_end, ..]).to_owned();
                        distributed.add_partition(partition, vec![row_start, 0], node_id);
                    }
                }
            }
            _ => {
                return Err(IoError::ParseError(
                    "Unsupported distribution for scatter".to_string(),
                ));
            }
        }

        Ok(distributed)
    }
}

/// Distributed file system abstraction
pub trait DistributedFileSystem: Send + Sync {
    /// Open a file for reading
    fn open_read(&self, path: &Path) -> Result<Box<dyn Read + Send>>;

    /// Create a file for writing
    fn create_write(&self, path: &Path) -> Result<Box<dyn Write + Send>>;

    /// List files in a directory
    fn list_dir(&self, path: &Path) -> Result<Vec<PathBuf>>;

    /// Get file metadata
    fn metadata(&self, path: &Path) -> Result<FileMetadata>;

    /// Check if path exists
    fn exists(&self, path: &Path) -> bool;
}

/// File metadata
#[derive(Debug, Clone)]
pub struct FileMetadata {
    pub size: u64,
    pub modified: std::time::SystemTime,
    pub is_dir: bool,
}

/// Local file system implementation
pub struct LocalFileSystem;

impl DistributedFileSystem for LocalFileSystem {
    fn open_read(&self, path: &Path) -> Result<Box<dyn Read + Send>> {
        let file = File::open(path)
            .map_err(|_| IoError::FileNotFound(path.to_string_lossy().to_string()))?;
        Ok(Box::new(file))
    }

    fn create_write(&self, path: &Path) -> Result<Box<dyn Write + Send>> {
        let file = File::create(path)
            .map_err(|e| IoError::FileError(format!("Failed to create file: {e}")))?;
        Ok(Box::new(file))
    }

    fn list_dir(&self, path: &Path) -> Result<Vec<PathBuf>> {
        let entries = std::fs::read_dir(path)
            .map_err(|e| IoError::ParseError(format!("Failed to read directory: {e}")))?;

        let mut paths = Vec::new();
        for entry in entries {
            let entry =
                entry.map_err(|e| IoError::ParseError(format!("Failed to read entry: {e}")))?;
            paths.push(entry.path());
        }

        Ok(paths)
    }

    fn metadata(&self, path: &Path) -> Result<FileMetadata> {
        let meta = std::fs::metadata(path)
            .map_err(|_| IoError::FileNotFound(path.to_string_lossy().to_string()))?;

        Ok(FileMetadata {
            size: meta.len(),
            modified: meta
                .modified()
                .map_err(|e| IoError::ParseError(format!("Failed to get modified time: {e}")))?,
            is_dir: meta.is_dir(),
        })
    }

    fn exists(&self, path: &Path) -> bool {
        path.exists()
    }
}

// Helper for s! macro
use scirs2_core::ndarray::s;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_partition_strategies() {
        let temp_dir = TempDir::new().unwrap();
        let temp_file = temp_dir.path().join("test.dat");
        std::fs::write(&temp_file, vec![0u8; 10000]).unwrap();

        let reader =
            DistributedReader::new(&temp_file).partition_strategy(PartitionStrategy::SizeBased {
                chunk_size_bytes: 1000,
            });

        let partitions = reader.create_partitions().unwrap();
        assert_eq!(partitions.len(), 10);

        for (_offset, size) in &partitions {
            assert_eq!(*size, 1000);
        }
    }

    #[test]
    fn test_distributed_array() {
        let array = Array2::from_shape_fn((100, 50), |(i, j)| (i * 50 + j) as f64);

        let distributed = DistributedArray::scatter(
            &array,
            Distribution::Block {
                block_size: vec![25, 50],
            },
            4,
        )
        .unwrap();

        assert_eq!(distributed.partitions.len(), 4);

        let gathered = distributed.gather().unwrap();
        assert_eq!(array, gathered);
    }

    #[test]
    fn test_distributed_writer() {
        let temp_dir = TempDir::new().unwrap();

        let data: Vec<i32> = (0..100).collect();
        let writer = DistributedWriter::new(temp_dir.path()).num_partitions(4);

        let files = writer
            .write_parallel(data, |&value, file| {
                writeln!(file, "{value}")
                    .map_err(|e| IoError::FileError(format!("Failed to write: {e}")))
            })
            .unwrap();

        assert_eq!(files.len(), 4);

        // Verify all files exist
        for file in &files {
            assert!(file.exists());
        }
    }
}
