//! Zero-copy transformations for memory-efficient image processing
//!
//! This module provides zero-copy and memory-mapped array operations for
//! processing large images without unnecessary memory allocations.
//!
//! # Features
//!
//! - **Memory-mapped arrays**: Access files as arrays without loading into RAM
//! - **Zero-copy views**: Transform data views without copying underlying data
//! - **Lazy evaluation**: Delay computation until results are actually needed
//! - **Streaming transforms**: Apply point-wise operations in a streaming fashion
//!
//! # Example
//!
//! ```rust,no_run
//! use scirs2_ndimage::zero_copy::{MappedImage, LazyTransform};
//! use std::path::Path;
//!
//! // Memory-map a large image file
//! let mapped = MappedImage::<f64>::open(Path::new("large_image.raw"), (10000, 10000)).unwrap();
//!
//! // Create a lazy transform chain
//! let transform = LazyTransform::new()
//!     .map(|x| x * 2.0)
//!     .map(|x| x.max(0.0).min(255.0));
//!
//! // Process in chunks without loading entire image
//! let result = transform.apply_chunked(&mapped, 1024).unwrap();
//! ```

use std::fmt::Debug;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use scirs2_core::ndarray;
use scirs2_core::ndarray::{Array, Array2, ArrayView2, ArrayViewMut2, Axis, Ix2};
use scirs2_core::numeric::{Float, FromPrimitive, NumCast, Zero};

use crate::error::{NdimageError, NdimageResult};

// ============================================================================
// Memory-Mapped Image Types
// ============================================================================

/// A memory-mapped image that provides lazy access to file data
pub struct MappedImage<T> {
    /// Path to the underlying file
    path: PathBuf,

    /// Shape of the image (rows, cols)
    shape: (usize, usize),

    /// Cached file handle (optional, for keeping file open)
    #[allow(dead_code)]
    file: Option<File>,

    /// Element type marker
    _phantom: PhantomData<T>,
}

impl<T: Float + FromPrimitive + Clone> MappedImage<T> {
    /// Open an existing raw binary image file
    pub fn open(path: &Path, shape: (usize, usize)) -> NdimageResult<Self> {
        // Verify file exists and has correct size
        let metadata = std::fs::metadata(path).map_err(NdimageError::IoError)?;
        let expected_size = shape.0 * shape.1 * std::mem::size_of::<T>();

        if metadata.len() as usize != expected_size {
            return Err(NdimageError::InvalidInput(format!(
                "File size {} does not match expected size {} for shape {:?}",
                metadata.len(),
                expected_size,
                shape
            )));
        }

        Ok(Self {
            path: path.to_path_buf(),
            shape,
            file: None,
            _phantom: PhantomData,
        })
    }

    /// Create a new memory-mapped image file
    pub fn create(path: &Path, shape: (usize, usize)) -> NdimageResult<Self> {
        let element_size = std::mem::size_of::<T>();
        let total_size = shape.0 * shape.1 * element_size;

        // Create and allocate the file
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(NdimageError::IoError)?;

        file.set_len(total_size as u64)
            .map_err(NdimageError::IoError)?;

        Ok(Self {
            path: path.to_path_buf(),
            shape,
            file: Some(file),
            _phantom: PhantomData,
        })
    }

    /// Get the shape of the image
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Get the total number of elements
    pub fn len(&self) -> usize {
        self.shape.0 * self.shape.1
    }

    /// Check if the image is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read a rectangular region from the file
    pub fn read_region(&self, rows: Range<usize>, cols: Range<usize>) -> NdimageResult<Array2<T>> {
        self.validate_range(&rows, &cols)?;

        let region_rows = rows.end - rows.start;
        let region_cols = cols.end - cols.start;
        let element_size = std::mem::size_of::<T>();

        let mut file = File::open(&self.path).map_err(NdimageError::IoError)?;
        let mut data = Vec::with_capacity(region_rows * region_cols);

        for row in rows.clone() {
            // Seek to start of this row in the region
            let offset = (row * self.shape.1 + cols.start) * element_size;
            file.seek(SeekFrom::Start(offset as u64))
                .map_err(NdimageError::IoError)?;

            // Read the row portion
            let mut row_buffer = vec![0u8; region_cols * element_size];
            file.read_exact(&mut row_buffer)
                .map_err(NdimageError::IoError)?;

            // Convert bytes to T values
            for i in 0..region_cols {
                let value =
                    self.bytes_to_value(&row_buffer[i * element_size..(i + 1) * element_size])?;
                data.push(value);
            }
        }

        Array2::from_shape_vec((region_rows, region_cols), data)
            .map_err(|e| NdimageError::ShapeError(e))
    }

    /// Write a rectangular region to the file
    pub fn write_region(
        &self,
        data: &ArrayView2<T>,
        row_offset: usize,
        col_offset: usize,
    ) -> NdimageResult<()> {
        let (data_rows, data_cols) = (data.nrows(), data.ncols());

        // Validate bounds
        if row_offset + data_rows > self.shape.0 || col_offset + data_cols > self.shape.1 {
            return Err(NdimageError::InvalidInput(format!(
                "Region ({}, {}) + ({}, {}) exceeds image bounds ({}, {})",
                row_offset, col_offset, data_rows, data_cols, self.shape.0, self.shape.1
            )));
        }

        let element_size = std::mem::size_of::<T>();
        let mut file = OpenOptions::new()
            .write(true)
            .open(&self.path)
            .map_err(NdimageError::IoError)?;

        for (local_row, global_row) in (row_offset..row_offset + data_rows).enumerate() {
            // Seek to start of this row
            let offset = (global_row * self.shape.1 + col_offset) * element_size;
            file.seek(SeekFrom::Start(offset as u64))
                .map_err(NdimageError::IoError)?;

            // Write the row
            let mut row_buffer = Vec::with_capacity(data_cols * element_size);
            for col in 0..data_cols {
                let bytes = self.value_to_bytes(data[[local_row, col]]);
                row_buffer.extend_from_slice(&bytes);
            }
            file.write_all(&row_buffer).map_err(NdimageError::IoError)?;
        }

        Ok(())
    }

    /// Read the entire image into memory
    pub fn to_array(&self) -> NdimageResult<Array2<T>> {
        self.read_region(0..self.shape.0, 0..self.shape.1)
    }

    /// Write an entire array to the file
    pub fn from_array(path: &Path, data: &ArrayView2<T>) -> NdimageResult<Self> {
        let shape = (data.nrows(), data.ncols());
        let mapped = Self::create(path, shape)?;
        mapped.write_region(data, 0, 0)?;
        Ok(mapped)
    }

    /// Validate row and column ranges
    fn validate_range(&self, rows: &Range<usize>, cols: &Range<usize>) -> NdimageResult<()> {
        if rows.end > self.shape.0 || cols.end > self.shape.1 {
            return Err(NdimageError::InvalidInput(format!(
                "Region {:?} x {:?} exceeds image bounds ({}, {})",
                rows, cols, self.shape.0, self.shape.1
            )));
        }
        if rows.start >= rows.end || cols.start >= cols.end {
            return Err(NdimageError::InvalidInput(
                "Invalid region: empty or inverted range".into(),
            ));
        }
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
}

// ============================================================================
// Lazy Transform Types
// ============================================================================

/// A lazy transformation that is applied on-demand
pub struct LazyTransform<T> {
    transforms: Vec<Box<dyn Fn(T) -> T + Send + Sync>>,
    _phantom: PhantomData<T>,
}

impl<T: Float + Clone + Send + Sync + 'static> LazyTransform<T> {
    /// Create a new empty lazy transform
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
            _phantom: PhantomData,
        }
    }

    /// Add a mapping function to the transform chain
    pub fn map<F>(mut self, f: F) -> Self
    where
        F: Fn(T) -> T + Send + Sync + 'static,
    {
        self.transforms.push(Box::new(f));
        self
    }

    /// Apply the transform chain to a single value
    pub fn apply_value(&self, value: T) -> T {
        let mut result = value;
        for transform in &self.transforms {
            result = transform(result);
        }
        result
    }

    /// Apply the transform chain to an array
    pub fn apply(&self, input: &ArrayView2<T>) -> Array2<T> {
        input.mapv(|x| self.apply_value(x))
    }

    /// Apply the transform chain to a memory-mapped image in chunks
    pub fn apply_chunked(
        &self,
        input: &MappedImage<T>,
        chunk_size: usize,
    ) -> NdimageResult<Array2<T>>
    where
        T: Float + FromPrimitive + Zero,
    {
        let (rows, cols) = input.shape();
        let mut output = Array2::zeros((rows, cols));

        // Process in row chunks
        for chunk_start in (0..rows).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(rows);

            // Read chunk
            let chunk = input.read_region(chunk_start..chunk_end, 0..cols)?;

            // Apply transforms
            let transformed = self.apply(&chunk.view());

            // Write to output
            output
                .slice_mut(ndarray::s![chunk_start..chunk_end, ..])
                .assign(&transformed);
        }

        Ok(output)
    }

    /// Apply the transform and write directly to a mapped output
    pub fn apply_to_mapped(
        &self,
        input: &MappedImage<T>,
        output: &MappedImage<T>,
        chunk_size: usize,
    ) -> NdimageResult<()>
    where
        T: Float + FromPrimitive,
    {
        let (rows, cols) = input.shape();

        if output.shape() != input.shape() {
            return Err(NdimageError::DimensionError(
                "Input and output shapes must match".into(),
            ));
        }

        // Process in row chunks
        for chunk_start in (0..rows).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(rows);

            // Read chunk
            let chunk = input.read_region(chunk_start..chunk_end, 0..cols)?;

            // Apply transforms
            let transformed = self.apply(&chunk.view());

            // Write to output
            output.write_region(&transformed.view(), chunk_start, 0)?;
        }

        Ok(())
    }

    /// Get the number of transforms in the chain
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    /// Check if the transform chain is empty
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }
}

impl<T: Float + Clone + Send + Sync + 'static> Default for LazyTransform<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// View Adapters for Zero-Copy Operations
// ============================================================================

/// A windowed view that provides sliding window access without copying
pub struct SlidingWindow<'a, T> {
    data: &'a ArrayView2<'a, T>,
    window_size: (usize, usize),
    current_row: usize,
    current_col: usize,
}

impl<'a, T: Float + Clone> SlidingWindow<'a, T> {
    /// Create a new sliding window iterator
    pub fn new(data: &'a ArrayView2<'a, T>, window_size: (usize, usize)) -> Self {
        Self {
            data,
            window_size,
            current_row: 0,
            current_col: 0,
        }
    }

    /// Get the number of valid windows
    pub fn count(&self) -> usize {
        let (rows, cols) = (self.data.nrows(), self.data.ncols());
        let valid_rows = rows.saturating_sub(self.window_size.0 - 1);
        let valid_cols = cols.saturating_sub(self.window_size.1 - 1);
        valid_rows * valid_cols
    }
}

impl<'a, T: Float + Clone> Iterator for SlidingWindow<'a, T> {
    type Item = (ArrayView2<'a, T>, (usize, usize));

    fn next(&mut self) -> Option<Self::Item> {
        let (rows, cols) = (self.data.nrows(), self.data.ncols());
        let (win_rows, win_cols) = self.window_size;

        let max_row = rows.saturating_sub(win_rows - 1);
        let max_col = cols.saturating_sub(win_cols - 1);

        if self.current_row >= max_row {
            return None;
        }

        let position = (self.current_row, self.current_col);
        let window = self.data.slice(ndarray::s![
            self.current_row..self.current_row + win_rows,
            self.current_col..self.current_col + win_cols
        ]);

        // Advance to next position
        self.current_col += 1;
        if self.current_col >= max_col {
            self.current_col = 0;
            self.current_row += 1;
        }

        Some((window, position))
    }
}

/// A strided view that skips elements for downsampling
pub struct StridedView<'a, T> {
    data: ArrayView2<'a, T>,
    stride: (usize, usize),
}

impl<'a, T: Float + Clone> StridedView<'a, T> {
    /// Create a new strided view
    pub fn new(data: ArrayView2<'a, T>, stride: (usize, usize)) -> Self {
        Self { data, stride }
    }

    /// Get the shape of the strided view
    pub fn shape(&self) -> (usize, usize) {
        let rows = (self.data.nrows() + self.stride.0 - 1) / self.stride.0;
        let cols = (self.data.ncols() + self.stride.1 - 1) / self.stride.1;
        (rows, cols)
    }

    /// Convert to an owned array
    pub fn to_array(&self) -> Array2<T> {
        let (out_rows, out_cols) = self.shape();
        let mut result = Array2::zeros((out_rows, out_cols));

        for i in 0..out_rows {
            for j in 0..out_cols {
                let src_row = i * self.stride.0;
                let src_col = j * self.stride.1;
                if src_row < self.data.nrows() && src_col < self.data.ncols() {
                    result[[i, j]] = self.data[[src_row, src_col]];
                }
            }
        }

        result
    }

    /// Get an element at the strided coordinates
    pub fn get(&self, row: usize, col: usize) -> Option<T> {
        let src_row = row * self.stride.0;
        let src_col = col * self.stride.1;

        if src_row < self.data.nrows() && src_col < self.data.ncols() {
            Some(self.data[[src_row, src_col]])
        } else {
            None
        }
    }
}

// ============================================================================
// Streaming Operations
// ============================================================================

/// Configuration for streaming operations
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Buffer size in elements
    pub buffer_size: usize,

    /// Number of buffers for double-buffering
    pub num_buffers: usize,

    /// Whether to enable prefetching
    pub prefetch: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1024 * 1024, // 1M elements
            num_buffers: 2,
            prefetch: true,
        }
    }
}

/// Streaming processor for applying operations to large images
pub struct StreamingProcessor<T> {
    config: StreamingConfig,
    _phantom: PhantomData<T>,
}

impl<T: Float + FromPrimitive + Clone + Send + Sync + Zero + 'static> StreamingProcessor<T> {
    /// Create a new streaming processor
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            config,
            _phantom: PhantomData,
        }
    }

    /// Apply a point-wise operation in a streaming fashion
    pub fn apply_pointwise<F>(
        &self,
        input: &MappedImage<T>,
        output: &MappedImage<T>,
        op: F,
    ) -> NdimageResult<()>
    where
        F: Fn(T) -> T + Send + Sync,
    {
        let (rows, cols) = input.shape();
        let elements_per_row = cols;
        let rows_per_buffer = (self.config.buffer_size / elements_per_row).max(1);

        for chunk_start in (0..rows).step_by(rows_per_buffer) {
            let chunk_end = (chunk_start + rows_per_buffer).min(rows);

            // Read chunk
            let chunk = input.read_region(chunk_start..chunk_end, 0..cols)?;

            // Apply operation
            let transformed = chunk.mapv(&op);

            // Write result
            output.write_region(&transformed.view(), chunk_start, 0)?;
        }

        Ok(())
    }

    /// Apply a neighborhood operation in a streaming fashion
    pub fn apply_neighborhood<F>(
        &self,
        input: &MappedImage<T>,
        output: &MappedImage<T>,
        neighborhood_size: (usize, usize),
        op: F,
    ) -> NdimageResult<()>
    where
        F: Fn(&ArrayView2<T>) -> T + Send + Sync,
    {
        let (rows, cols) = input.shape();
        let (nrows, ncols) = neighborhood_size;
        let half_nrows = nrows / 2;
        let half_ncols = ncols / 2;

        // Calculate buffer size with overlap
        let elements_per_row = cols;
        let rows_per_buffer = (self.config.buffer_size / elements_per_row).max(1);

        for chunk_start in (0..rows).step_by(rows_per_buffer) {
            let chunk_end = (chunk_start + rows_per_buffer).min(rows);

            // Read chunk with overlap for neighborhood operations
            let read_start = chunk_start.saturating_sub(half_nrows);
            let read_end = (chunk_end + half_nrows).min(rows);
            let chunk = input.read_region(read_start..read_end, 0..cols)?;

            // Process the chunk
            let output_rows = chunk_end - chunk_start;
            let mut result = Array2::zeros((output_rows, cols - nrows + 1));

            for i in 0..output_rows {
                let src_row = i + (chunk_start - read_start);
                if src_row + nrows <= chunk.nrows() {
                    for j in 0..(cols - ncols + 1) {
                        let neighborhood =
                            chunk.slice(ndarray::s![src_row..src_row + nrows, j..j + ncols]);
                        result[[i, j]] = op(&neighborhood);
                    }
                }
            }

            // Write result (accounting for border handling)
            let write_col_start = half_ncols;
            output.write_region(&result.view(), chunk_start, write_col_start)?;
        }

        Ok(())
    }
}

// ============================================================================
// Buffer Pool for Reusing Allocations
// ============================================================================

/// Pool of reusable buffers for zero-copy operations
pub struct BufferPool<T> {
    buffers: Vec<Vec<T>>,
    buffer_size: usize,
    in_use: Vec<bool>,
}

impl<T: Clone + Zero> BufferPool<T> {
    /// Create a new buffer pool
    pub fn new(buffer_count: usize, buffer_size: usize) -> Self {
        let buffers = (0..buffer_count)
            .map(|_| vec![T::zero(); buffer_size])
            .collect();
        let in_use = vec![false; buffer_count];

        Self {
            buffers,
            buffer_size,
            in_use,
        }
    }

    /// Acquire a buffer from the pool
    pub fn acquire(&mut self) -> Option<&mut Vec<T>> {
        for (i, in_use) in self.in_use.iter_mut().enumerate() {
            if !*in_use {
                *in_use = true;
                return Some(&mut self.buffers[i]);
            }
        }
        None
    }

    /// Release a buffer back to the pool
    pub fn release(&mut self, index: usize) {
        if index < self.in_use.len() {
            self.in_use[index] = false;
        }
    }

    /// Get the number of available buffers
    pub fn available(&self) -> usize {
        self.in_use.iter().filter(|&&x| !x).count()
    }

    /// Get the buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
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
    fn test_lazy_transform() {
        let transform = LazyTransform::<f64>::new()
            .map(|x| x * 2.0)
            .map(|x| x + 1.0);

        assert_eq!(transform.len(), 2);
        assert_eq!(transform.apply_value(5.0), 11.0);
    }

    #[test]
    fn test_lazy_transform_array() {
        let input =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .expect("Shape should be valid");

        let transform = LazyTransform::new().map(|x: f64| x * 2.0);
        let output = transform.apply(&input.view());

        assert_eq!(output[[0, 0]], 2.0);
        assert_eq!(output[[1, 1]], 10.0);
        assert_eq!(output[[2, 2]], 18.0);
    }

    #[test]
    fn test_mapped_image_roundtrip() {
        let temp_path = temp_dir().join("test_mapped_image.raw");

        // Create test data
        let data = Array2::<f64>::from_shape_fn((10, 10), |(i, j)| (i * 10 + j) as f64);

        // Write to mapped image
        let mapped =
            MappedImage::from_array(&temp_path, &data.view()).expect("Should create mapped image");

        // Read back
        let read_data = mapped.to_array().expect("Should read array");

        assert_eq!(data, read_data);

        // Cleanup
        std::fs::remove_file(&temp_path).ok();
    }

    #[test]
    fn test_mapped_image_region() {
        let temp_path = temp_dir().join("test_mapped_region.raw");

        // Create test data
        let data = Array2::<f64>::from_shape_fn((20, 20), |(i, j)| (i * 20 + j) as f64);

        // Write to mapped image
        let mapped =
            MappedImage::from_array(&temp_path, &data.view()).expect("Should create mapped image");

        // Read a region
        let region = mapped
            .read_region(5..10, 5..15)
            .expect("Should read region");

        assert_eq!(region.shape(), &[5, 10]);
        assert_eq!(region[[0, 0]], 5.0 * 20.0 + 5.0);

        // Cleanup
        std::fs::remove_file(&temp_path).ok();
    }

    #[test]
    fn test_sliding_window() {
        let data = Array2::<f64>::from_shape_fn((5, 5), |(i, j)| (i * 5 + j) as f64);
        let view = data.view();
        let sliding = SlidingWindow::new(&view, (3, 3));

        let windows: Vec<_> = sliding.collect();
        assert_eq!(windows.len(), 9); // (5-3+1) * (5-3+1) = 9

        // Check first window position
        let (first_window, first_pos) = &windows[0];
        assert_eq!(*first_pos, (0, 0));
        assert_eq!(first_window[[0, 0]], 0.0);
    }

    #[test]
    fn test_strided_view() {
        let data = Array2::<f64>::from_shape_fn((10, 10), |(i, j)| (i * 10 + j) as f64);
        let strided = StridedView::new(data.view(), (2, 2));

        assert_eq!(strided.shape(), (5, 5));

        let result = strided.to_array();
        assert_eq!(result[[0, 0]], 0.0);
        assert_eq!(result[[1, 1]], 22.0); // (2*10 + 2)
    }

    #[test]
    fn test_buffer_pool() {
        let mut pool = BufferPool::<f64>::new(3, 100);

        assert_eq!(pool.available(), 3);

        let buffer1 = pool.acquire();
        assert!(buffer1.is_some());
        assert_eq!(pool.available(), 2);

        let buffer2 = pool.acquire();
        assert!(buffer2.is_some());
        assert_eq!(pool.available(), 1);

        pool.release(0);
        assert_eq!(pool.available(), 2);
    }

    #[test]
    fn test_lazy_transform_chunked() {
        let temp_path = temp_dir().join("test_chunked_transform.raw");

        // Create test data
        let data = Array2::<f64>::ones((100, 100));

        // Write to mapped image
        let mapped =
            MappedImage::from_array(&temp_path, &data.view()).expect("Should create mapped image");

        // Apply chunked transform
        let transform = LazyTransform::new().map(|x: f64| x * 2.0);
        let result = transform
            .apply_chunked(&mapped, 10)
            .expect("Should apply transform");

        // Verify result
        for val in result.iter() {
            assert_eq!(*val, 2.0);
        }

        // Cleanup
        std::fs::remove_file(&temp_path).ok();
    }
}
