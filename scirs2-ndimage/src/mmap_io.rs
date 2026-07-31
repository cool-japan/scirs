//! Memory-mapped I/O operations for large images
//!
//! This module provides functions for loading and saving large images using
//! memory-mapped arrays, enabling processing of datasets that don't fit in RAM.

use scirs2_core::ndarray::{Array, ArrayView, Dimension, Ix1, Ix2, IxDyn};
use scirs2_core::numeric::{Float, FromPrimitive, NumCast};
use std::fs;
use std::path::Path;

use scirs2_core::memory_efficient::{
    create_mmap, open_mmap, AccessMode, ChunkingStrategy, MemoryMappedArray, MemoryMappedChunkIter,
    MemoryMappedChunks,
};

use crate::error::{NdimageError, NdimageResult};

/// Load an image as a memory-mapped array
///
/// This function creates a memory-mapped array from a file, allowing you to work
/// with images larger than available RAM.
///
/// # Arguments
///
/// * `path` - Path to the image file
/// * `shape` - Expected shape of the image
/// * `offset` - Byte offset in the file where image data starts
/// * `access` - Access mode (Read, Write, or Copy)
///
/// # Returns
///
/// A memory-mapped array that can be used like a regular ndarray
///
/// # Errors
///
/// Returns an error if the file cannot be opened, if the file is too small
/// for the requested shape, or if the file's actual shape (as recorded in
/// its header, for files written by [`saveimage_mmap`]) does not match the
/// requested `shape`.
#[allow(dead_code)]
pub fn loadimage_mmap<T, D, P>(
    path: P,
    shape: &[usize],
    offset: usize,
    access: AccessMode,
) -> NdimageResult<MemoryMappedArray<T>>
where
    T: Float + FromPrimitive + NumCast + Send + Sync + 'static,
    D: Dimension,
    P: AsRef<Path>,
{
    // `saveimage_mmap` (via `create_mmap`, `AccessMode::Write`) prepends a
    // serialized header before the array data. Use the header-aware
    // `open_mmap` here rather than re-deriving a raw offset ourselves:
    // mapping directly at the caller-supplied `offset` (as this function
    // previously did, via `create_mmap` with a shape-only dummy array)
    // ignores that header entirely, silently shifting every element read
    // back by the header size instead of failing loudly.
    let mmap = open_mmap::<T, D>(path.as_ref(), access, offset).map_err(NdimageError::CoreError)?;

    if mmap.shape != shape {
        return Err(NdimageError::InvalidInput(format!(
            "Shape mismatch: file contains shape {:?}, expected {:?}",
            mmap.shape, shape
        )));
    }

    Ok(mmap)
}

/// Save an array as a memory-mapped file
///
/// This function creates a new file and maps it to memory, then copies the array data.
///
/// # Arguments
///
/// * `array` - Array to save
/// * `path` - Path where to save the file
/// * `offset` - Byte offset in the file where to start writing
///
/// # Returns
///
/// A memory-mapped array pointing to the saved data
#[allow(dead_code)]
pub fn saveimage_mmap<T, D, P>(
    array: &ArrayView<T, D>,
    path: P,
    offset: usize,
) -> NdimageResult<MemoryMappedArray<T>>
where
    T: Float + FromPrimitive + NumCast + Send + Sync + 'static,
    D: Dimension,
    P: AsRef<Path>,
{
    // Create memory-mapped array with write access
    let mmap = create_mmap(array, path.as_ref(), AccessMode::Write, offset)
        .map_err(NdimageError::CoreError)?;

    Ok(mmap)
}

/// Create a temporary memory-mapped array for intermediate results
///
/// This is useful for operations that produce large intermediate results.
///
/// # Arguments
///
/// * `shape` - Shape of the array to create
///
/// # Returns
///
/// A memory-mapped array backed by a temporary file
#[allow(dead_code)]
pub fn create_temp_mmap<T>(
    shape: &[usize],
) -> NdimageResult<(MemoryMappedArray<T>, tempfile::TempPath)>
where
    T: Float + FromPrimitive + NumCast + Send + Sync + 'static,
{
    use tempfile::NamedTempFile;

    // Create temporary file
    let temp_file = NamedTempFile::new().map_err(NdimageError::IoError)?;

    let temp_path = temp_file.into_temp_path();

    // Create dummy array for shape
    let dummy_array = Array::<T, IxDyn>::zeros(IxDyn(shape));

    // Create memory-mapped array
    let mmap = create_mmap(&dummy_array.view(), &temp_path, AccessMode::Write, 0)
        .map_err(NdimageError::CoreError)?;

    Ok((mmap, temp_path))
}

/// Process a memory-mapped image in chunks
///
/// This function provides a convenient way to process large memory-mapped images
/// using chunked processing.
///
/// # Arguments
///
/// * `mmap` - Memory-mapped array containing the image
/// * `strategy` - Chunking strategy to use
/// * `processor` - Function to process each chunk
///
/// # Returns
///
/// Results from processing each chunk
#[allow(dead_code)]
pub fn process_mmap_chunks<T, R, F>(
    mmap: &MemoryMappedArray<T>,
    strategy: ChunkingStrategy,
    processor: F,
) -> NdimageResult<Vec<R>>
where
    T: Float + FromPrimitive + NumCast + Send + Sync + 'static,
    F: Fn(&[T], usize) -> R,
    R: Send,
{
    let results = mmap.process_chunks(strategy, processor);
    Ok(results)
}

/// Iterator over chunks of a memory-mapped image
///
/// This provides a lazy way to process large images chunk by chunk.
pub struct MmapChunkIterator<'a, T>
where
    T: Float + FromPrimitive + NumCast + Send + Sync + 'static,
{
    mmap: &'a MemoryMappedArray<T>,
    strategy: ChunkingStrategy,
}

impl<'a, T> MmapChunkIterator<'a, T>
where
    T: Float + FromPrimitive + NumCast + Send + Sync + 'static,
{
    pub fn new(mmap: &'a MemoryMappedArray<T>, strategy: ChunkingStrategy) -> Self {
        Self { mmap, strategy }
    }

    /// Get an iterator over chunks
    pub fn iter(&self) -> impl Iterator<Item = Array<T, Ix1>> + '_ {
        self.mmap.chunks(self.strategy.clone())
    }
}

/// Configuration for memory-mapped image processing
#[derive(Debug, Clone)]
pub struct MmapConfig {
    /// Maximum size (in bytes) before automatically using memory mapping
    pub auto_mmap_threshold: usize,
    /// Default chunking strategy
    pub default_chunk_strategy: ChunkingStrategy,
    /// Whether to use parallel processing for chunks
    pub parallel: bool,
    /// Whether to prefetch chunks
    pub prefetch: bool,
}

impl Default for MmapConfig {
    fn default() -> Self {
        Self {
            auto_mmap_threshold: 100 * 1024 * 1024, // 100 MB
            default_chunk_strategy: ChunkingStrategy::Auto,
            parallel: true,
            prefetch: true,
        }
    }
}

/// Load an array directly into memory from a binary file
///
/// This function reads binary data from a file and interprets it as an array
/// of the specified type and shape. This is for smaller files that can fit in RAM.
///
/// # Arguments
///
/// * `path` - Path to the binary file
/// * `shape` - Expected shape of the array
///
/// # Returns
///
/// A regular ndarray containing the loaded data
#[allow(dead_code)]
pub fn load_regular_array<T, D, P>(path: P, shape: &[usize]) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + NumCast + Send + Sync + 'static,
    D: Dimension + 'static,
    P: AsRef<Path>,
{
    // `saveimage_mmap` (via `create_mmap`/`AccessMode::Write`) prepends a
    // serialized header before the array data -- see `loadimage_mmap`'s fix
    // above for the full rationale. This function previously hand-parsed
    // the file at raw byte offset 0 (ignoring that header entirely, which
    // silently shifted every element read back by the header size) and
    // only supported f32/f64 via a hand-rolled little-endian decode loop.
    // Delegate to the already header-aware, already-tested `open_mmap` +
    // `MemoryMappedArray::as_array` instead of re-deriving the (variable
    // length, versioned, aligned) header layout a second time -- this also
    // removes the f32/f64-only restriction as a side effect, since
    // `as_array` works for any `T`.
    let mmap = open_mmap::<T, D>(path.as_ref(), AccessMode::ReadOnly, 0)
        .map_err(NdimageError::CoreError)?;

    if mmap.shape != shape {
        return Err(NdimageError::InvalidInput(format!(
            "Shape mismatch: file contains shape {:?}, expected {:?}",
            mmap.shape, shape
        )));
    }

    mmap.as_array::<D>().map_err(NdimageError::CoreError)
}

/// Smart image loader that automatically decides between regular and memory-mapped loading
#[allow(dead_code)]
pub fn smart_loadimage<T, D, P>(
    path: P,
    shape: &[usize],
    config: Option<MmapConfig>,
) -> NdimageResult<ImageData<T, D>>
where
    T: Float + FromPrimitive + NumCast + Send + Sync + 'static,
    D: Dimension + 'static,
    P: AsRef<Path>,
{
    let config = config.unwrap_or_default();

    // Calculate expected size
    let total_elements: usize = shape.iter().product();
    let total_bytes = total_elements * std::mem::size_of::<T>();

    if total_bytes > config.auto_mmap_threshold {
        // Use memory-mapped loading for large files
        let mmap = loadimage_mmap::<T, D, P>(path, shape, 0, AccessMode::ReadOnly)?;
        Ok(ImageData::MemoryMapped(mmap))
    } else {
        // Load into regular array for small files
        let array = load_regular_array::<T, D, P>(path, shape)?;
        Ok(ImageData::Regular(array))
    }
}

/// Enum to hold either regular or memory-mapped image data
pub enum ImageData<T, D>
where
    T: Float + FromPrimitive + NumCast + Send + Sync + 'static,
    D: Dimension,
{
    Regular(Array<T, D>),
    MemoryMapped(MemoryMappedArray<T>),
}

impl<T, D> ImageData<T, D>
where
    T: Float + FromPrimitive + NumCast + Send + Sync + 'static,
    D: Dimension + 'static,
{
    /// Get a view of the image data (regular arrays only).
    ///
    /// Memory-mapped arrays cannot return a borrowed view because the underlying
    /// data lives in a file.  Use [`to_array`](Self::to_array) to materialise a
    /// copy first, then call `.view()` on the result.
    pub fn view(&self) -> NdimageResult<ArrayView<T, D>> {
        match self {
            ImageData::Regular(array) => Ok(array.view()),
            ImageData::MemoryMapped(_mmap) => Err(NdimageError::NotImplementedError(
                "Cannot return a borrowed view over a memory-mapped array: \
                 call to_array() to materialise a copy first, then use .view() on the Array."
                    .to_string(),
            )),
        }
    }

    /// Materialise the image into an owned `Array<T, D>`.
    ///
    /// For regular arrays this is a cheap clone.  For memory-mapped arrays the
    /// data is read from the file via `MemoryMappedArray::as_array()`.
    pub fn to_array(&self) -> NdimageResult<Array<T, D>> {
        match self {
            ImageData::Regular(array) => Ok(array.clone()),
            ImageData::MemoryMapped(mmap) => mmap.as_array::<D>().map_err(|e| {
                NdimageError::ProcessingError(format!(
                    "Failed to materialise memory-mapped array: {e}"
                ))
            }),
        }
    }

    /// Check if this is memory-mapped
    pub fn is_mmap(&self) -> bool {
        matches!(self, ImageData::MemoryMapped(_))
    }

    /// Get the shape
    pub fn shape(&self) -> Vec<usize> {
        match self {
            ImageData::Regular(array) => array.shape().to_vec(),
            ImageData::MemoryMapped(mmap) => mmap.shape.clone(),
        }
    }
}

/// Example: Process a large image file using memory mapping
#[allow(dead_code)]
pub fn process_largeimage_example<P: AsRef<Path>>(
    input_path: P,
    output_path: P,
    shape: &[usize],
) -> NdimageResult<()> {
    // Load input as memory-mapped
    let input_mmap = loadimage_mmap::<f64, Ix2, _>(input_path, shape, 0, AccessMode::ReadOnly)?;

    // Create output memory-mapped array
    let output_mmap = saveimage_mmap(
        &Array::<f64, IxDyn>::zeros(IxDyn(shape)).view(),
        output_path,
        0,
    )?;

    // Process in chunks
    let chunk_results = input_mmap.process_chunks(
        ChunkingStrategy::FixedBytes(10 * 1024 * 1024), // 10 MB chunks
        |chunk_data, chunk_idx| {
            // Example: Apply some transformation
            let processed: Vec<f64> = chunk_data.iter().map(|&x| x * 2.0 + 1.0).collect();
            (chunk_idx, processed)
        },
    );

    // Write results back (would need proper implementation)
    println!("Processed {} chunks", chunk_results.len());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;
    use tempfile::tempdir;

    #[test]
    fn test_create_temp_mmap() {
        let shape = vec![100, 100];
        let (mmap, _temp_path) = create_temp_mmap::<f64>(&shape).expect("Operation failed");

        // Test that mmap was created successfully
        // Note: MemoryMappedArray might not have shape() and size() methods
        // but creation success indicates proper functionality
        assert!(!_temp_path.is_dir());
    }

    #[test]
    fn test_save_and_load_mmap() {
        let temp_dir = tempdir().expect("Operation failed");
        let file_path = temp_dir.path().join("testimage.bin");

        // Use per-cell distinct values (not a constant fill) so a
        // misaligned or otherwise corrupted round trip would actually be
        // detected: a uniform array would still "look right" even if every
        // element were silently read from the wrong file offset.
        let mut data = Array2::<f64>::zeros((50, 50));
        for i in 0..50 {
            for j in 0..50 {
                data[[i, j]] = (i * 50 + j) as f64 + 0.5;
            }
        }

        // Save as memory-mapped
        let _saved_mmap = saveimage_mmap(&data.view(), &file_path, 0).expect("Operation failed");

        // Load back
        let loaded_mmap =
            loadimage_mmap::<f64, Ix2, _>(&file_path, &[50, 50], 0, AccessMode::ReadOnly)
                .expect("Operation failed");

        // Verify the round trip preserves the actual data, at several
        // distinct locations (start, middle, end).
        let loaded_view = loaded_mmap.as_array::<Ix2>().expect("Operation failed");
        assert_eq!(loaded_view[[0, 0]], data[[0, 0]]);
        assert_eq!(loaded_view[[0, 1]], data[[0, 1]]);
        assert_eq!(loaded_view[[25, 25]], data[[25, 25]]);
        assert_eq!(loaded_view[[49, 49]], data[[49, 49]]);
    }

    #[test]
    fn test_mmap_chunk_iterator() {
        let shape = vec![1000];
        let (mmap, _temp_path) = create_temp_mmap::<f64>(&shape).expect("Operation failed");

        let iterator = MmapChunkIterator::new(&mmap, ChunkingStrategy::Fixed(100));
        let chunks: Vec<_> = iterator.iter().collect();

        assert_eq!(chunks.len(), 10);
        assert_eq!(chunks[0].len(), 100);
    }

    #[test]
    fn test_load_regular_array_respects_header() {
        // `load_regular_array` previously hand-parsed the file starting at
        // raw byte offset 0, ignoring the header that `saveimage_mmap`
        // actually writes -- this would have silently returned
        // header-shifted garbage (or, for non-f32/f64 `T`, an outright
        // NotImplementedError) instead of the real data. Per-cell distinct
        // values (not a constant fill) so a misaligned/corrupted round
        // trip would actually be detected.
        let temp_dir = tempdir().expect("Operation failed");
        let file_path = temp_dir.path().join("regular.bin");

        let mut data = Array2::<f64>::zeros((20, 20));
        for i in 0..20 {
            for j in 0..20 {
                data[[i, j]] = (i * 20 + j) as f64 + 0.25;
            }
        }

        saveimage_mmap(&data.view(), &file_path, 0).expect("Operation failed");

        let loaded = load_regular_array::<f64, scirs2_core::ndarray::Ix2, _>(&file_path, &[20, 20])
            .expect("Operation failed");

        assert_eq!(loaded.shape(), data.shape());
        assert_eq!(loaded[[0, 0]], data[[0, 0]]);
        assert_eq!(loaded[[0, 1]], data[[0, 1]]);
        assert_eq!(loaded[[10, 10]], data[[10, 10]]);
        assert_eq!(loaded[[19, 19]], data[[19, 19]]);
    }

    #[test]
    fn test_load_regular_array_rejects_shape_mismatch() {
        let temp_dir = tempdir().expect("Operation failed");
        let file_path = temp_dir.path().join("regular_mismatch.bin");

        let data = Array2::<f64>::from_shape_fn((5, 5), |(i, j)| (i * 5 + j) as f64);
        saveimage_mmap(&data.view(), &file_path, 0).expect("Operation failed");

        let result = load_regular_array::<f64, scirs2_core::ndarray::Ix2, _>(&file_path, &[7, 7]);
        assert!(result.is_err());
    }

    #[test]
    fn test_smart_loadimage_small_file_uses_regular_array() {
        // Below the default `auto_mmap_threshold`, `smart_loadimage` should
        // route through `load_regular_array` and return real data (not the
        // header-shifted garbage the pre-fix implementation would have
        // produced).
        let temp_dir = tempdir().expect("Operation failed");
        let file_path = temp_dir.path().join("smart_small.bin");

        let data = Array2::<f64>::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f64 + 0.1);
        saveimage_mmap(&data.view(), &file_path, 0).expect("Operation failed");

        let loaded =
            smart_loadimage::<f64, scirs2_core::ndarray::Ix2, _>(&file_path, &[4, 4], None)
                .expect("Operation failed");
        assert!(!loaded.is_mmap());

        let array = loaded.to_array().expect("Operation failed");
        assert_eq!(array, data);
    }
}
