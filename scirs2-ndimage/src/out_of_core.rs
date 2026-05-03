//! Out-of-core image processing pipeline
//!
//! This module provides infrastructure for processing images that are too large
//! to fit in memory by using file-based storage and chunked processing.
//!
//! # Features
//!
//! - **File-based processing**: Work with images stored on disk
//! - **Pipeline composition**: Chain multiple operations efficiently
//! - **Automatic memory management**: Keeps memory usage within bounds
//! - **Checkpointing**: Save intermediate results for resumable processing
//! - **Progress tracking**: Monitor processing progress
//!
//! # Example
//!
//! ```rust,no_run
//! use scirs2_ndimage::out_of_core::{ImagePipeline, PipelineConfig};
//! use std::path::Path;
//!
//! let config = PipelineConfig::default();
//! let pipeline = ImagePipeline::new(config);
//!
//! // Process a large image file
//! let result = pipeline
//!     .input(Path::new("large_image.raw"), (10000, 10000))
//!     .gaussian_filter(2.0)
//!     .threshold(0.5)
//!     .save(Path::new("output.raw"));
//! ```

use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use scirs2_core::ndarray;
use scirs2_core::ndarray::{Array2, ArrayView2, Axis};
use scirs2_core::numeric::{Float, FromPrimitive, NumCast, Zero};
use scirs2_core::parallel_ops::*;

use crate::chunked_processing::{
    ChunkOperation, ChunkRegion, ChunkRegionIterator, ChunkedImageProcessor, ChunkingConfig,
};
use crate::error::{NdimageError, NdimageResult};
use crate::filters::BorderMode;
use crate::zero_copy::MappedImage;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for out-of-core processing
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Maximum memory to use for processing (in bytes)
    pub max_memory_bytes: usize,

    /// Temporary directory for intermediate files
    pub temp_dir: PathBuf,

    /// Whether to enable checkpointing
    pub enable_checkpoints: bool,

    /// Checkpoint interval (number of operations)
    pub checkpoint_interval: usize,

    /// Whether to clean up temporary files after processing
    pub cleanup_temp_files: bool,

    /// Enable parallel processing
    pub enable_parallel: bool,

    /// Number of worker threads (0 = use all available)
    pub num_workers: usize,

    /// Buffer size for file I/O
    pub io_buffer_size: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 512 * 1024 * 1024, // 512 MB
            temp_dir: std::env::temp_dir().join("scirs2_ooc"),
            enable_checkpoints: false,
            checkpoint_interval: 5,
            cleanup_temp_files: true,
            enable_parallel: true,
            num_workers: 0,
            io_buffer_size: 8 * 1024 * 1024, // 8 MB
        }
    }
}

// ============================================================================
// Pipeline Stage
// ============================================================================

/// A stage in the processing pipeline
pub trait PipelineStage<T>: Send + Sync
where
    T: Float + FromPrimitive + Clone + Send + Sync,
{
    /// Get the name of this stage
    fn name(&self) -> &str;

    /// Get the required overlap for this stage
    fn required_overlap(&self) -> usize;

    /// Process a chunk
    fn process_chunk(&self, chunk: &ArrayView2<T>) -> NdimageResult<Array2<T>>;

    /// Whether this stage modifies dimensions
    fn modifies_dimensions(&self) -> bool {
        false
    }

    /// Get the output dimensions given input dimensions
    fn output_dimensions(&self, input_shape: (usize, usize)) -> (usize, usize) {
        input_shape
    }
}

/// A composed pipeline of stages
pub struct ComposedPipeline<T> {
    stages: Vec<Box<dyn PipelineStage<T>>>,
}

impl<T: Float + FromPrimitive + Clone + Send + Sync> ComposedPipeline<T> {
    /// Create a new empty composed pipeline
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Add a stage to the pipeline
    pub fn add_stage(mut self, stage: Box<dyn PipelineStage<T>>) -> Self {
        self.stages.push(stage);
        self
    }

    /// Get the maximum required overlap
    pub fn max_overlap(&self) -> usize {
        self.stages
            .iter()
            .map(|s| s.required_overlap())
            .max()
            .unwrap_or(0)
    }

    /// Get the number of stages
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }

    /// Check if any stage modifies dimensions
    pub fn modifies_dimensions(&self) -> bool {
        self.stages.iter().any(|s| s.modifies_dimensions())
    }
}

impl<T: Float + FromPrimitive + Clone + Send + Sync + Zero + 'static> Default
    for ComposedPipeline<T>
{
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Built-in Pipeline Stages
// ============================================================================

/// Gaussian filter stage
pub struct GaussianStage {
    sigma: f64,
    border_mode: BorderMode,
}

impl GaussianStage {
    pub fn new(sigma: f64, border_mode: BorderMode) -> Self {
        Self { sigma, border_mode }
    }
}

impl<T> PipelineStage<T> for GaussianStage
where
    T: Float
        + FromPrimitive
        + Clone
        + Send
        + Sync
        + Zero
        + 'static
        + std::ops::AddAssign
        + std::ops::DivAssign,
{
    fn name(&self) -> &str {
        "gaussian_filter"
    }

    fn required_overlap(&self) -> usize {
        (self.sigma * 4.0).ceil() as usize
    }

    fn process_chunk(&self, chunk: &ArrayView2<T>) -> NdimageResult<Array2<T>> {
        let chunk_f64 = chunk.mapv(|x| x.to_f64().unwrap_or(0.0));
        let result =
            crate::filters::gaussian_filter(&chunk_f64, self.sigma, Some(self.border_mode), None)?;
        Ok(result.mapv(|x| T::from_f64(x).unwrap_or_else(T::zero)))
    }
}

/// Threshold stage
pub struct ThresholdStage<T> {
    threshold: T,
    above_value: T,
    below_value: T,
}

impl<T: Float> ThresholdStage<T> {
    pub fn new(threshold: T) -> Self {
        Self {
            threshold,
            above_value: T::one(),
            below_value: T::zero(),
        }
    }

    pub fn with_values(threshold: T, above_value: T, below_value: T) -> Self {
        Self {
            threshold,
            above_value,
            below_value,
        }
    }
}

impl<T> PipelineStage<T> for ThresholdStage<T>
where
    T: Float + FromPrimitive + Clone + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        "threshold"
    }

    fn required_overlap(&self) -> usize {
        0
    }

    fn process_chunk(&self, chunk: &ArrayView2<T>) -> NdimageResult<Array2<T>> {
        Ok(chunk.mapv(|x| {
            if x >= self.threshold {
                self.above_value
            } else {
                self.below_value
            }
        }))
    }
}

/// Normalize stage
pub struct NormalizeStage<T> {
    min_val: T,
    max_val: T,
}

impl<T: Float> NormalizeStage<T> {
    pub fn new(min_val: T, max_val: T) -> Self {
        Self { min_val, max_val }
    }
}

impl<T> PipelineStage<T> for NormalizeStage<T>
where
    T: Float + FromPrimitive + Clone + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        "normalize"
    }

    fn required_overlap(&self) -> usize {
        0
    }

    fn process_chunk(&self, chunk: &ArrayView2<T>) -> NdimageResult<Array2<T>> {
        // Find min and max in chunk
        let mut chunk_min = T::infinity();
        let mut chunk_max = T::neg_infinity();

        for &val in chunk.iter() {
            if val < chunk_min {
                chunk_min = val;
            }
            if val > chunk_max {
                chunk_max = val;
            }
        }

        let range = chunk_max - chunk_min;
        let target_range = self.max_val - self.min_val;

        if range <= T::epsilon() {
            // Constant chunk
            return Ok(Array2::from_elem(chunk.raw_dim(), self.min_val));
        }

        Ok(chunk.mapv(|x| self.min_val + (x - chunk_min) / range * target_range))
    }
}

/// Median filter stage
pub struct MedianStage {
    size: usize,
}

impl MedianStage {
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}

impl<T> PipelineStage<T> for MedianStage
where
    T: Float + FromPrimitive + Clone + Send + Sync + Zero + 'static + PartialOrd,
{
    fn name(&self) -> &str {
        "median_filter"
    }

    fn required_overlap(&self) -> usize {
        self.size / 2 + 1
    }

    fn process_chunk(&self, chunk: &ArrayView2<T>) -> NdimageResult<Array2<T>> {
        let chunk_f64 = chunk.mapv(|x| x.to_f64().unwrap_or(0.0));
        let result = crate::filters::median_filter(&chunk_f64, &[self.size, self.size], None)?;
        Ok(result.mapv(|x| T::from_f64(x).unwrap_or_else(T::zero)))
    }
}

/// Custom function stage
pub struct CustomStage<T, F>
where
    F: Fn(&ArrayView2<T>) -> NdimageResult<Array2<T>> + Send + Sync,
{
    name: String,
    overlap: usize,
    func: F,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, F> CustomStage<T, F>
where
    F: Fn(&ArrayView2<T>) -> NdimageResult<Array2<T>> + Send + Sync,
{
    pub fn new(name: &str, overlap: usize, func: F) -> Self {
        Self {
            name: name.to_string(),
            overlap,
            func,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, F> PipelineStage<T> for CustomStage<T, F>
where
    T: Float + FromPrimitive + Clone + Send + Sync + 'static,
    F: Fn(&ArrayView2<T>) -> NdimageResult<Array2<T>> + Send + Sync,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn required_overlap(&self) -> usize {
        self.overlap
    }

    fn process_chunk(&self, chunk: &ArrayView2<T>) -> NdimageResult<Array2<T>> {
        (self.func)(chunk)
    }
}

// ============================================================================
// Progress Tracking
// ============================================================================

/// Progress information for pipeline execution
#[derive(Debug, Clone)]
pub struct ProgressInfo {
    /// Total number of chunks
    pub total_chunks: usize,

    /// Number of chunks processed
    pub chunks_processed: usize,

    /// Current stage name
    pub current_stage: String,

    /// Current stage index
    pub stage_index: usize,

    /// Total number of stages
    pub total_stages: usize,

    /// Elapsed time in seconds
    pub elapsed_seconds: f64,

    /// Estimated remaining time in seconds
    pub eta_seconds: Option<f64>,
}

impl ProgressInfo {
    /// Get the overall progress as a percentage
    pub fn progress_percent(&self) -> f64 {
        if self.total_chunks == 0 {
            0.0
        } else {
            (self.chunks_processed as f64 / self.total_chunks as f64) * 100.0
        }
    }
}

/// Callback type for progress updates
pub type ProgressCallback = Box<dyn Fn(&ProgressInfo) + Send + Sync>;

// ============================================================================
// Image Pipeline
// ============================================================================

/// Out-of-core image processing pipeline
pub struct ImagePipeline<T>
where
    T: Float + FromPrimitive + Clone + Send + Sync + Zero + 'static,
{
    config: PipelineConfig,
    stages: Vec<Box<dyn PipelineStage<T>>>,
    progress_callback: Option<ProgressCallback>,
    cancel_flag: Arc<AtomicBool>,
}

impl<T> ImagePipeline<T>
where
    T: Float + FromPrimitive + Clone + Send + Sync + Zero + 'static,
{
    /// Create a new image pipeline
    pub fn new(config: PipelineConfig) -> Self {
        // Ensure temp directory exists
        if let Err(e) = fs::create_dir_all(&config.temp_dir) {
            eprintln!("Warning: Failed to create temp directory: {}", e);
        }

        Self {
            config,
            stages: Vec::new(),
            progress_callback: None,
            cancel_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(PipelineConfig::default())
    }

    /// Set progress callback
    pub fn on_progress(mut self, callback: ProgressCallback) -> Self {
        self.progress_callback = Some(callback);
        self
    }

    /// Get a handle for cancellation
    pub fn cancel_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel_flag)
    }

    /// Add a gaussian filter stage
    pub fn gaussian_filter(mut self, sigma: f64) -> Self
    where
        T: std::ops::AddAssign + std::ops::DivAssign,
    {
        self.stages
            .push(Box::new(GaussianStage::new(sigma, BorderMode::Reflect)));
        self
    }

    /// Add a threshold stage
    pub fn threshold(mut self, threshold_value: T) -> Self {
        self.stages
            .push(Box::new(ThresholdStage::new(threshold_value)));
        self
    }

    /// Add a normalize stage
    pub fn normalize(mut self, min_val: T, max_val: T) -> Self {
        self.stages
            .push(Box::new(NormalizeStage::new(min_val, max_val)));
        self
    }

    /// Add a median filter stage
    pub fn median_filter(mut self, size: usize) -> Self
    where
        T: PartialOrd,
    {
        self.stages.push(Box::new(MedianStage::new(size)));
        self
    }

    /// Add a custom processing stage
    pub fn custom<F>(mut self, name: &str, overlap: usize, func: F) -> Self
    where
        F: Fn(&ArrayView2<T>) -> NdimageResult<Array2<T>> + Send + Sync + 'static,
    {
        self.stages
            .push(Box::new(CustomStage::new(name, overlap, func)));
        self
    }

    /// Get the maximum required overlap across all stages
    fn max_overlap(&self) -> usize {
        self.stages
            .iter()
            .map(|s| s.required_overlap())
            .max()
            .unwrap_or(0)
    }

    /// Calculate chunk size based on configuration
    fn calculate_chunk_size(&self, element_size: usize) -> usize {
        // Use 1/4 of max memory for each chunk to allow for working memory
        let target_bytes = self.config.max_memory_bytes / 4;
        let target_elements = target_bytes / element_size;
        (target_elements as f64).sqrt() as usize
    }

    /// Process a file-based image
    pub fn process_file(
        &self,
        input_path: &Path,
        output_path: &Path,
        shape: (usize, usize),
    ) -> NdimageResult<()> {
        if self.stages.is_empty() {
            return Err(NdimageError::InvalidInput(
                "No processing stages defined".into(),
            ));
        }

        let element_size = std::mem::size_of::<T>();
        let chunk_size = self.calculate_chunk_size(element_size);
        let overlap = self.max_overlap();

        let chunk_iter = ChunkRegionIterator::new(shape, (chunk_size, chunk_size), overlap);
        let total_chunks = chunk_iter.total_chunks();

        let start_time = Instant::now();
        let chunks_processed = AtomicUsize::new(0);

        // Create output file
        let output_file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(output_path)
            .map_err(NdimageError::IoError)?;
        output_file
            .set_len((shape.0 * shape.1 * element_size) as u64)
            .map_err(NdimageError::IoError)?;

        // Process chunks
        for region in chunk_iter {
            // Check for cancellation
            if self.cancel_flag.load(Ordering::Relaxed) {
                return Err(NdimageError::ComputationError(
                    "Processing cancelled".into(),
                ));
            }

            // Read input chunk
            let chunk = self.read_chunk_from_file(input_path, shape, &region)?;

            // Process through all stages
            let mut result = chunk;
            for (stage_idx, stage) in self.stages.iter().enumerate() {
                result = stage.process_chunk(&result.view())?;

                // Report progress
                if let Some(ref callback) = self.progress_callback {
                    let processed = chunks_processed.load(Ordering::Relaxed);
                    let elapsed = start_time.elapsed().as_secs_f64();
                    let eta = if processed > 0 {
                        Some(elapsed / processed as f64 * (total_chunks - processed) as f64)
                    } else {
                        None
                    };

                    callback(&ProgressInfo {
                        total_chunks,
                        chunks_processed: processed,
                        current_stage: stage.name().to_string(),
                        stage_index: stage_idx,
                        total_stages: self.stages.len(),
                        elapsed_seconds: elapsed,
                        eta_seconds: eta,
                    });
                }
            }

            // Write output chunk
            self.write_chunk_to_file(&result.view(), output_path, shape, &region)?;

            chunks_processed.fetch_add(1, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Process an in-memory image (for smaller images)
    pub fn process_memory(&self, input: &ArrayView2<T>) -> NdimageResult<Array2<T>> {
        if self.stages.is_empty() {
            return Ok(input.to_owned());
        }

        let element_size = std::mem::size_of::<T>();
        let chunk_size = self.calculate_chunk_size(element_size);
        let overlap = self.max_overlap();
        let shape = (input.nrows(), input.ncols());

        // If image is small enough, process directly
        if input.len() * element_size <= self.config.max_memory_bytes / 2 {
            let mut result = input.to_owned();
            for stage in &self.stages {
                // Check for cancellation between stages
                if self.cancel_flag.load(Ordering::Relaxed) {
                    return Err(NdimageError::ComputationError(
                        "Processing cancelled".into(),
                    ));
                }
                result = stage.process_chunk(&result.view())?;
            }
            return Ok(result);
        }

        // Otherwise use chunked processing
        let chunk_iter = ChunkRegionIterator::new(shape, (chunk_size, chunk_size), overlap);
        let mut output = Array2::zeros(shape);

        for region in chunk_iter {
            // Check for cancellation
            if self.cancel_flag.load(Ordering::Relaxed) {
                return Err(NdimageError::ComputationError(
                    "Processing cancelled".into(),
                ));
            }

            // Extract chunk
            let rows = region.padded_start.0..region.padded_end.0;
            let cols = region.padded_start.1..region.padded_end.1;
            let chunk = input.slice(ndarray::s![rows, cols]).to_owned();

            // Process through all stages
            let mut result = chunk;
            for stage in &self.stages {
                result = stage.process_chunk(&result.view())?;
            }

            // Insert into output
            self.insert_chunk_result(&mut output, &result.view(), &region)?;
        }

        Ok(output)
    }

    /// Read a chunk from a file
    fn read_chunk_from_file(
        &self,
        path: &Path,
        image_shape: (usize, usize),
        region: &ChunkRegion,
    ) -> NdimageResult<Array2<T>> {
        let element_size = std::mem::size_of::<T>();
        let mut file = BufReader::with_capacity(
            self.config.io_buffer_size,
            File::open(path).map_err(NdimageError::IoError)?,
        );

        let chunk_rows = region.padded_end.0 - region.padded_start.0;
        let chunk_cols = region.padded_end.1 - region.padded_start.1;
        let mut data = Vec::with_capacity(chunk_rows * chunk_cols);

        for row in region.padded_start.0..region.padded_end.0 {
            let offset = (row * image_shape.1 + region.padded_start.1) * element_size;
            file.seek(SeekFrom::Start(offset as u64))
                .map_err(NdimageError::IoError)?;

            let mut row_buffer = vec![0u8; chunk_cols * element_size];
            file.read_exact(&mut row_buffer)
                .map_err(NdimageError::IoError)?;

            for i in 0..chunk_cols {
                let value =
                    self.bytes_to_value(&row_buffer[i * element_size..(i + 1) * element_size])?;
                data.push(value);
            }
        }

        Array2::from_shape_vec((chunk_rows, chunk_cols), data)
            .map_err(|e| NdimageError::ShapeError(e))
    }

    /// Write a chunk to a file
    fn write_chunk_to_file(
        &self,
        chunk: &ArrayView2<T>,
        path: &Path,
        image_shape: (usize, usize),
        region: &ChunkRegion,
    ) -> NdimageResult<()> {
        let element_size = std::mem::size_of::<T>();
        let file = OpenOptions::new()
            .write(true)
            .open(path)
            .map_err(NdimageError::IoError)?;
        let mut writer = BufWriter::with_capacity(self.config.io_buffer_size, file);

        let overlap = region.overlap();
        let core_start_row = overlap.0 .0;
        let core_start_col = overlap.0 .1;
        let core_end_row = chunk.nrows() - overlap.1 .0;
        let core_end_col = chunk.ncols() - overlap.1 .1;

        for (chunk_row, output_row) in
            (core_start_row..core_end_row).zip(region.start.0..region.end.0)
        {
            let offset = (output_row * image_shape.1 + region.start.1) * element_size;
            writer
                .seek(SeekFrom::Start(offset as u64))
                .map_err(NdimageError::IoError)?;

            for chunk_col in core_start_col..core_end_col {
                let bytes = self.value_to_bytes(chunk[[chunk_row, chunk_col]]);
                writer.write_all(&bytes).map_err(NdimageError::IoError)?;
            }
        }

        writer.flush().map_err(NdimageError::IoError)?;
        Ok(())
    }

    /// Insert a processed chunk into the output array
    fn insert_chunk_result(
        &self,
        output: &mut Array2<T>,
        chunk: &ArrayView2<T>,
        region: &ChunkRegion,
    ) -> NdimageResult<()> {
        let overlap = region.overlap();
        let core_start_row = overlap.0 .0;
        let core_start_col = overlap.0 .1;
        let core_end_row = chunk.nrows() - overlap.1 .0;
        let core_end_col = chunk.ncols() - overlap.1 .1;

        let core_slice = chunk.slice(ndarray::s![
            core_start_row..core_end_row,
            core_start_col..core_end_col
        ]);

        output
            .slice_mut(ndarray::s![
                region.start.0..region.end.0,
                region.start.1..region.end.1
            ])
            .assign(&core_slice);

        Ok(())
    }

    /// Convert bytes to a value of type T
    fn bytes_to_value(&self, bytes: &[u8]) -> NdimageResult<T> {
        let element_size = std::mem::size_of::<T>();

        if element_size == 8 {
            let arr: [u8; 8] = bytes.try_into().map_err(|_| {
                NdimageError::ComputationError("Failed to convert bytes to f64".into())
            })?;
            T::from_f64(f64::from_le_bytes(arr))
                .ok_or_else(|| NdimageError::ComputationError("Failed to convert f64 to T".into()))
        } else if element_size == 4 {
            let arr: [u8; 4] = bytes.try_into().map_err(|_| {
                NdimageError::ComputationError("Failed to convert bytes to f32".into())
            })?;
            T::from_f32(f32::from_le_bytes(arr))
                .ok_or_else(|| NdimageError::ComputationError("Failed to convert f32 to T".into()))
        } else {
            Err(NdimageError::InvalidInput(format!(
                "Unsupported element size: {}",
                element_size
            )))
        }
    }

    /// Convert a value of type T to bytes
    fn value_to_bytes(&self, value: T) -> Vec<u8> {
        let element_size = std::mem::size_of::<T>();

        if element_size == 8 {
            value.to_f64().unwrap_or(0.0).to_le_bytes().to_vec()
        } else {
            value.to_f32().unwrap_or(0.0).to_le_bytes().to_vec()
        }
    }

    /// Clean up temporary files
    pub fn cleanup(&self) -> NdimageResult<()> {
        if self.config.cleanup_temp_files {
            if self.config.temp_dir.exists() {
                fs::remove_dir_all(&self.config.temp_dir).map_err(NdimageError::IoError)?;
            }
        }
        Ok(())
    }
}

// Implement Drop for cleanup
impl<T> Drop for ImagePipeline<T>
where
    T: Float + FromPrimitive + Clone + Send + Sync + Zero + 'static,
{
    fn drop(&mut self) {
        if self.config.cleanup_temp_files {
            let _ = self.cleanup();
        }
    }
}

// ============================================================================
// Checkpoint Support
// ============================================================================

/// Checkpoint data for resumable processing
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// Path to intermediate result
    pub result_path: PathBuf,

    /// Current stage index
    pub stage_index: usize,

    /// Chunks processed
    pub chunks_processed: usize,

    /// Image shape
    pub shape: (usize, usize),

    /// Processing parameters
    pub parameters: HashMap<String, String>,
}

impl Checkpoint {
    /// Save checkpoint to file
    pub fn save(&self, path: &Path) -> NdimageResult<()> {
        let file = File::create(path).map_err(NdimageError::IoError)?;
        let mut writer = BufWriter::new(file);

        // Write checkpoint data as simple key-value pairs
        writeln!(writer, "result_path={}", self.result_path.display())
            .map_err(NdimageError::IoError)?;
        writeln!(writer, "stage_index={}", self.stage_index).map_err(NdimageError::IoError)?;
        writeln!(writer, "chunks_processed={}", self.chunks_processed)
            .map_err(NdimageError::IoError)?;
        writeln!(writer, "shape_rows={}", self.shape.0).map_err(NdimageError::IoError)?;
        writeln!(writer, "shape_cols={}", self.shape.1).map_err(NdimageError::IoError)?;

        for (key, value) in &self.parameters {
            writeln!(writer, "param_{}={}", key, value).map_err(NdimageError::IoError)?;
        }

        writer.flush().map_err(NdimageError::IoError)?;
        Ok(())
    }

    /// Load checkpoint from file
    pub fn load(path: &Path) -> NdimageResult<Self> {
        let file = File::open(path).map_err(NdimageError::IoError)?;
        let reader = BufReader::new(file);

        let mut result_path = PathBuf::new();
        let mut stage_index = 0;
        let mut chunks_processed = 0;
        let mut shape_rows = 0;
        let mut shape_cols = 0;
        let mut parameters = HashMap::new();

        for line in std::io::BufRead::lines(reader) {
            let line = line.map_err(NdimageError::IoError)?;
            if let Some((key, value)) = line.split_once('=') {
                match key {
                    "result_path" => result_path = PathBuf::from(value),
                    "stage_index" => stage_index = value.parse().unwrap_or(0),
                    "chunks_processed" => chunks_processed = value.parse().unwrap_or(0),
                    "shape_rows" => shape_rows = value.parse().unwrap_or(0),
                    "shape_cols" => shape_cols = value.parse().unwrap_or(0),
                    _ if key.starts_with("param_") => {
                        let param_key = key.strip_prefix("param_").unwrap_or(key);
                        parameters.insert(param_key.to_string(), value.to_string());
                    }
                    _ => {}
                }
            }
        }

        Ok(Self {
            result_path,
            stage_index,
            chunks_processed,
            shape: (shape_rows, shape_cols),
            parameters,
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert!(config.max_memory_bytes > 0);
        assert!(config.cleanup_temp_files);
    }

    #[test]
    fn test_pipeline_creation() {
        let pipeline = ImagePipeline::<f64>::with_defaults();
        assert_eq!(pipeline.stages.len(), 0);
    }

    #[test]
    fn test_pipeline_stages() {
        let pipeline = ImagePipeline::<f64>::with_defaults()
            .gaussian_filter(1.0)
            .threshold(0.5)
            .normalize(0.0, 1.0);

        assert_eq!(pipeline.stages.len(), 3);
    }

    #[test]
    fn test_pipeline_memory_processing() {
        let pipeline = ImagePipeline::<f64>::with_defaults().threshold(0.5);

        let input =
            Array2::from_shape_fn((50, 50), |(i, j)| if (i + j) % 2 == 0 { 1.0 } else { 0.0 });

        let result = pipeline.process_memory(&input.view());
        assert!(result.is_ok());

        let output = result.expect("Should succeed");
        assert_eq!(output.shape(), input.shape());

        // Check thresholding worked
        for i in 0..50 {
            for j in 0..50 {
                let expected = if (i + j) % 2 == 0 { 1.0 } else { 0.0 };
                assert_eq!(output[[i, j]], expected);
            }
        }
    }

    #[test]
    fn test_custom_stage() {
        let pipeline = ImagePipeline::<f64>::with_defaults()
            .custom("double", 0, |chunk| Ok(chunk.mapv(|x| x * 2.0)));

        let input = Array2::ones((20, 20));
        let result = pipeline.process_memory(&input.view());

        assert!(result.is_ok());
        let output = result.expect("Should succeed");

        for val in output.iter() {
            assert_eq!(*val, 2.0);
        }
    }

    #[test]
    fn test_progress_info() {
        let progress = ProgressInfo {
            total_chunks: 100,
            chunks_processed: 50,
            current_stage: "test".to_string(),
            stage_index: 0,
            total_stages: 1,
            elapsed_seconds: 10.0,
            eta_seconds: Some(10.0),
        };

        assert_eq!(progress.progress_percent(), 50.0);
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let temp_path = temp_dir().join("test_checkpoint.txt");

        let checkpoint = Checkpoint {
            result_path: std::env::temp_dir().join("result.raw"),
            stage_index: 2,
            chunks_processed: 50,
            shape: (1000, 1000),
            parameters: {
                let mut map = HashMap::new();
                map.insert("sigma".to_string(), "2.0".to_string());
                map
            },
        };

        checkpoint.save(&temp_path).expect("Should save");

        let loaded = Checkpoint::load(&temp_path).expect("Should load");

        assert_eq!(loaded.stage_index, checkpoint.stage_index);
        assert_eq!(loaded.chunks_processed, checkpoint.chunks_processed);
        assert_eq!(loaded.shape, checkpoint.shape);

        std::fs::remove_file(&temp_path).ok();
    }

    #[test]
    fn test_gaussian_stage() {
        let stage = GaussianStage::new(1.0, BorderMode::Reflect);
        assert_eq!(
            <GaussianStage as PipelineStage<f64>>::name(&stage),
            "gaussian_filter"
        );
        assert!(<GaussianStage as PipelineStage<f64>>::required_overlap(&stage) > 0);

        let input = Array2::<f64>::ones((20, 20));
        let result = <GaussianStage as PipelineStage<f64>>::process_chunk(&stage, &input.view());
        assert!(result.is_ok());
    }

    #[test]
    fn test_threshold_stage() {
        let stage = ThresholdStage::new(0.5f64);

        let input = Array2::from_shape_fn((10, 10), |(i, j)| if i < 5 { 0.3 } else { 0.7 });

        let result = stage.process_chunk(&input.view()).expect("Should process");

        for i in 0..10 {
            for j in 0..10 {
                let expected = if i < 5 { 0.0 } else { 1.0 };
                assert_eq!(result[[i, j]], expected);
            }
        }
    }

    #[test]
    fn test_normalize_stage() {
        let stage = NormalizeStage::new(0.0f64, 1.0);

        let input = Array2::from_shape_fn((10, 10), |(i, j)| (i * 10 + j) as f64);

        let result = stage.process_chunk(&input.view()).expect("Should process");

        // Check normalized range
        let min_val = result.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = result.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        assert!((min_val - 0.0).abs() < 1e-10);
        assert!((max_val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pipeline_cancel() {
        let pipeline = ImagePipeline::<f64>::with_defaults().gaussian_filter(1.0);

        let cancel_handle = pipeline.cancel_handle();
        cancel_handle.store(true, Ordering::Relaxed);

        let input = Array2::ones((100, 100));
        let result = pipeline.process_memory(&input.view());

        assert!(result.is_err());
    }
}
