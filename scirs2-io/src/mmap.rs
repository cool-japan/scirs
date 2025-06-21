//! Memory-mapped file I/O for large arrays
//!
//! This module provides memory-mapped file operations for efficient handling of large arrays
//! without loading them entirely into memory. Memory mapping is particularly useful for:
//!
//! - Processing arrays larger than available RAM
//! - Random access patterns across large datasets
//! - Sharing data between multiple processes
//! - Minimizing memory usage for read-only operations
//! - Fast startup times for large files
//!
//! ## Features
//!
//! - **Memory-Mapped Arrays**: Read arrays from files using memory mapping
//! - **Multi-dimensional Support**: Handle 1D, 2D, and N-dimensional arrays
//! - **Type Safety**: Generic support for different numeric types
//! - **Cross-platform**: Works on Unix and Windows systems
//! - **Performance Optimized**: Minimal memory overhead and fast access
//! - **Error Handling**: Comprehensive error handling for I/O operations
//!
//! ## Examples
//!
//! ```rust,no_run
//! use scirs2_io::mmap::{MmapArray, create_mmap_array};
//! use ndarray::Array2;
//! use std::path::Path;
//!
//! // Create a large array file
//! let data = Array2::from_shape_fn((1000, 1000), |(i, j)| (i + j) as f64);
//! let file_path = Path::new("large_array.bin");
//!
//! // Write array to file
//! create_mmap_array(file_path, &data)?;
//!
//! // Memory-map the array for reading
//! let mmap_array: MmapArray<f64> = MmapArray::open(file_path)?;
//! let shape = mmap_array.shape()?;
//! let array_view = mmap_array.as_array_view(&shape)?;
//!
//! // Access data without loading entire file into memory
//! let slice = mmap_array.as_slice()?;
//! let value = slice[500 * 1000 + 500]; // Access element at (500, 500)
//! println!("Value at (500, 500): {}", value);
//! # Ok::<(), scirs2_io::error::IoError>(())
//! ```

use crate::error::{IoError, Result};
use ndarray::{ArrayBase, ArrayD, ArrayView, ArrayViewMut, Dimension, IxDyn};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::marker::PhantomData;
use std::path::Path;

/// Memory-mapped array that provides efficient access to large datasets
pub struct MmapArray<T> {
    /// Memory-mapped region
    mmap: memmap2::Mmap,
    /// File handle
    _file: File,
    /// Total number of elements
    len: usize,
    /// Phantom data for type safety
    _phantom: PhantomData<T>,
}

/// Mutable memory-mapped array for read-write access
pub struct MmapArrayMut<T> {
    /// Mutable memory-mapped region
    mmap: memmap2::MmapMut,
    /// File handle
    _file: File,
    /// Total number of elements
    len: usize,
    /// Phantom data for type safety
    _phantom: PhantomData<T>,
}

/// Builder for creating memory-mapped arrays
pub struct MmapArrayBuilder<'a> {
    /// Path to the file
    path: &'a Path,
    /// Whether to create the file if it doesn't exist
    create: bool,
    /// Whether to truncate the file if it exists
    truncate: bool,
    /// Buffer size for I/O operations
    buffer_size: usize,
}

/// Configuration for memory-mapped array operations
#[derive(Debug, Clone, Default)]
pub struct MmapConfig {
    /// Enable read-ahead prefetching
    pub prefetch: bool,
    /// Page size for memory mapping (None for system default)
    pub page_size: Option<usize>,
    /// Whether to use sequential access pattern hints
    pub sequential: bool,
    /// Whether to use random access pattern hints
    pub random: bool,
}

impl<'a> MmapArrayBuilder<'a> {
    /// Create a new builder for the specified file path
    pub fn new<P: AsRef<Path>>(path: &'a P) -> Self {
        Self {
            path: path.as_ref(),
            create: true,
            truncate: false,
            buffer_size: 64 * 1024, // 64KB default buffer
        }
    }

    /// Set whether to create the file if it doesn't exist
    pub fn create(mut self, create: bool) -> Self {
        self.create = create;
        self
    }

    /// Set whether to truncate the file if it exists
    pub fn truncate(mut self, truncate: bool) -> Self {
        self.truncate = truncate;
        self
    }

    /// Set the buffer size for I/O operations
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Create a memory-mapped array from an existing ndarray
    pub fn create_from_array<S, D, T>(&self, array: &ArrayBase<S, D>) -> Result<()>
    where
        S: ndarray::Data<Elem = T>,
        D: Dimension,
        T: Clone + bytemuck::Pod,
    {
        let mut file = OpenOptions::new()
            .write(true)
            .create(self.create)
            .truncate(self.truncate)
            .open(self.path)
            .map_err(|e| IoError::FileError(format!("Failed to create file: {}", e)))?;

        // Write array metadata (shape, stride, element count)
        let shape = array.shape();
        let ndim = shape.len() as u64;
        file.write_all(&ndim.to_le_bytes())
            .map_err(|e| IoError::FileError(format!("Failed to write metadata: {}", e)))?;

        for &dim in shape {
            let dim = dim as u64;
            file.write_all(&dim.to_le_bytes())
                .map_err(|e| IoError::FileError(format!("Failed to write shape: {}", e)))?;
        }

        // Write element size
        let element_size = std::mem::size_of::<T>() as u64;
        file.write_all(&element_size.to_le_bytes())
            .map_err(|e| IoError::FileError(format!("Failed to write element size: {}", e)))?;

        // Write array data in chunks to avoid memory pressure
        if array.is_standard_layout() {
            // For contiguous arrays, we can write directly
            let data_slice = bytemuck::cast_slice(array.as_slice().unwrap());
            let mut written = 0;
            while written < data_slice.len() {
                let chunk_size = (data_slice.len() - written).min(self.buffer_size);
                let chunk = &data_slice[written..written + chunk_size];
                file.write_all(chunk)
                    .map_err(|e| IoError::FileError(format!("Failed to write data: {}", e)))?;
                written += chunk_size;
            }
        } else {
            // For non-contiguous arrays, we need to copy to a contiguous buffer
            let owned_array = array.to_owned();
            let data_slice = bytemuck::cast_slice(owned_array.as_slice().unwrap());
            let mut written = 0;
            while written < data_slice.len() {
                let chunk_size = (data_slice.len() - written).min(self.buffer_size);
                let chunk = &data_slice[written..written + chunk_size];
                file.write_all(chunk)
                    .map_err(|e| IoError::FileError(format!("Failed to write data: {}", e)))?;
                written += chunk_size;
            }
        }

        file.sync_all()
            .map_err(|e| IoError::FileError(format!("Failed to sync file: {}", e)))?;

        Ok(())
    }

    /// Create an empty memory-mapped array with the specified shape
    pub fn create_empty<T>(&self, shape: &[usize]) -> Result<()>
    where
        T: bytemuck::Pod,
    {
        let mut file = OpenOptions::new()
            .write(true)
            .create(self.create)
            .truncate(self.truncate)
            .open(self.path)
            .map_err(|e| IoError::FileError(format!("Failed to create file: {}", e)))?;

        // Write metadata
        let ndim = shape.len() as u64;
        file.write_all(&ndim.to_le_bytes())
            .map_err(|e| IoError::FileError(format!("Failed to write metadata: {}", e)))?;

        for &dim in shape {
            let dim = dim as u64;
            file.write_all(&dim.to_le_bytes())
                .map_err(|e| IoError::FileError(format!("Failed to write shape: {}", e)))?;
        }

        let element_size = std::mem::size_of::<T>() as u64;
        file.write_all(&element_size.to_le_bytes())
            .map_err(|e| IoError::FileError(format!("Failed to write element size: {}", e)))?;

        // Write zeros for the data
        let total_elements: usize = shape.iter().product();
        let total_bytes = total_elements * std::mem::size_of::<T>();

        let zero_buffer = vec![0u8; self.buffer_size.min(total_bytes)];
        let mut remaining = total_bytes;

        while remaining > 0 {
            let chunk_size = remaining.min(zero_buffer.len());
            file.write_all(&zero_buffer[..chunk_size])
                .map_err(|e| IoError::FileError(format!("Failed to write zeros: {}", e)))?;
            remaining -= chunk_size;
        }

        file.sync_all()
            .map_err(|e| IoError::FileError(format!("Failed to sync file: {}", e)))?;

        Ok(())
    }
}

impl<T> MmapArray<T>
where
    T: bytemuck::Pod,
{
    /// Open an existing memory-mapped array file for reading
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|e| IoError::FileError(format!("Failed to open file: {}", e)))?;

        let file_size = file
            .metadata()
            .map_err(|e| IoError::FileError(format!("Failed to get file size: {}", e)))?
            .len();

        if file_size < 8 {
            return Err(IoError::FormatError(
                "File too small to contain valid array".to_string(),
            ));
        }

        let mmap = unsafe {
            memmap2::Mmap::map(&file)
                .map_err(|e| IoError::FileError(format!("Failed to create memory map: {}", e)))?
        };

        // Read metadata to determine array size
        let (len, _metadata_size) = Self::read_metadata(&mmap[..])?;

        Ok(Self {
            mmap,
            _file: file,
            len,
            _phantom: PhantomData,
        })
    }

    /// Read metadata from the memory-mapped file
    fn read_metadata(mmap: &[u8]) -> Result<(usize, usize)> {
        if mmap.len() < 8 {
            return Err(IoError::FormatError("Invalid file format".to_string()));
        }

        let mut offset = 0;

        // Read number of dimensions
        let ndim = u64::from_le_bytes(
            mmap[offset..offset + 8]
                .try_into()
                .map_err(|_| IoError::FormatError("Failed to read ndim".to_string()))?,
        ) as usize;
        offset += 8;

        if ndim == 0 || ndim > 32 {
            return Err(IoError::FormatError(
                "Invalid number of dimensions".to_string(),
            ));
        }

        // Read shape
        let mut total_elements = 1;
        for _ in 0..ndim {
            if offset + 8 > mmap.len() {
                return Err(IoError::FormatError("Truncated shape data".to_string()));
            }
            let dim = u64::from_le_bytes(
                mmap[offset..offset + 8]
                    .try_into()
                    .map_err(|_| IoError::FormatError("Failed to read dimension".to_string()))?,
            ) as usize;
            total_elements *= dim;
            offset += 8;
        }

        // Read element size
        if offset + 8 > mmap.len() {
            return Err(IoError::FormatError(
                "Truncated element size data".to_string(),
            ));
        }
        let element_size = u64::from_le_bytes(
            mmap[offset..offset + 8]
                .try_into()
                .map_err(|_| IoError::FormatError("Failed to read element size".to_string()))?,
        ) as usize;
        offset += 8;

        if element_size != std::mem::size_of::<T>() {
            return Err(IoError::FormatError("Element size mismatch".to_string()));
        }

        Ok((total_elements, offset))
    }

    /// Get the shape of the array from the file metadata
    pub fn shape(&self) -> Result<Vec<usize>> {
        let mut offset = 0;

        // Read number of dimensions
        let ndim = u64::from_le_bytes(
            self.mmap[offset..offset + 8]
                .try_into()
                .map_err(|_| IoError::FormatError("Failed to read ndim".to_string()))?,
        ) as usize;
        offset += 8;

        // Read shape
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            let dim = u64::from_le_bytes(
                self.mmap[offset..offset + 8]
                    .try_into()
                    .map_err(|_| IoError::FormatError("Failed to read dimension".to_string()))?,
            ) as usize;
            shape.push(dim);
            offset += 8;
        }

        Ok(shape)
    }

    /// Get the data offset in the file (after metadata)
    fn data_offset(&self) -> Result<usize> {
        let ndim = u64::from_le_bytes(
            self.mmap[0..8]
                .try_into()
                .map_err(|_| IoError::FormatError("Failed to read ndim".to_string()))?,
        ) as usize;

        // 8 bytes for ndim + 8 bytes per dimension + 8 bytes for element size
        Ok(8 + ndim * 8 + 8)
    }

    /// Get a slice view of the raw data
    pub fn as_slice(&self) -> Result<&[T]> {
        let data_offset = self.data_offset()?;
        let data_bytes = &self.mmap[data_offset..];

        if data_bytes.len() < self.len * std::mem::size_of::<T>() {
            return Err(IoError::FormatError(
                "Insufficient data in file".to_string(),
            ));
        }

        Ok(bytemuck::cast_slice(
            &data_bytes[..self.len * std::mem::size_of::<T>()],
        ))
    }

    /// Create an ndarray view of the memory-mapped data
    pub fn as_array_view(&self, shape: &[usize]) -> Result<ArrayView<T, IxDyn>> {
        let data_slice = self.as_slice()?;

        let expected_len: usize = shape.iter().product();
        if expected_len != self.len {
            return Err(IoError::FormatError(format!(
                "Shape mismatch: expected {} elements, got {}",
                expected_len, self.len
            )));
        }

        ArrayView::from_shape(IxDyn(shape), data_slice)
            .map_err(|e| IoError::FormatError(format!("Failed to create array view: {}", e)))
    }

    /// Get the number of elements in the array
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the array is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> MmapArrayMut<T>
where
    T: bytemuck::Pod,
{
    /// Open an existing memory-mapped array file for read-write access
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path.as_ref())
            .map_err(|e| IoError::FileError(format!("Failed to open file: {}", e)))?;

        let file_size = file
            .metadata()
            .map_err(|e| IoError::FileError(format!("Failed to get file size: {}", e)))?
            .len();

        if file_size < 8 {
            return Err(IoError::FormatError(
                "File too small to contain valid array".to_string(),
            ));
        }

        let mmap = unsafe {
            memmap2::MmapMut::map_mut(&file)
                .map_err(|e| IoError::FileError(format!("Failed to create memory map: {}", e)))?
        };

        // Read metadata to determine array size
        let (len, _metadata_size) = Self::read_metadata(&mmap)?;

        Ok(Self {
            mmap,
            _file: file,
            len,
            _phantom: PhantomData,
        })
    }

    /// Read metadata from the memory-mapped file
    fn read_metadata(mmap: &memmap2::MmapMut) -> Result<(usize, usize)> {
        // Similar to read-only version
        MmapArray::<T>::read_metadata(&mmap[..])
    }

    /// Get the shape of the array from the file metadata
    pub fn shape(&self) -> Result<Vec<usize>> {
        let mut offset = 0;

        // Read number of dimensions
        let ndim = u64::from_le_bytes(
            self.mmap[offset..offset + 8]
                .try_into()
                .map_err(|_| IoError::FormatError("Failed to read ndim".to_string()))?,
        ) as usize;
        offset += 8;

        // Read shape
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            let dim = u64::from_le_bytes(
                self.mmap[offset..offset + 8]
                    .try_into()
                    .map_err(|_| IoError::FormatError("Failed to read dimension".to_string()))?,
            ) as usize;
            shape.push(dim);
            offset += 8;
        }

        Ok(shape)
    }

    /// Get the data offset in the file (after metadata)
    fn data_offset(&self) -> Result<usize> {
        let ndim = u64::from_le_bytes(
            self.mmap[0..8]
                .try_into()
                .map_err(|_| IoError::FormatError("Failed to read ndim".to_string()))?,
        ) as usize;

        // 8 bytes for ndim + 8 bytes per dimension + 8 bytes for element size
        Ok(8 + ndim * 8 + 8)
    }

    /// Get a mutable slice view of the raw data
    pub fn as_slice_mut(&mut self) -> Result<&mut [T]> {
        let data_offset = self.data_offset()?;
        let data_len = self.len * std::mem::size_of::<T>();

        if self.mmap.len() < data_offset + data_len {
            return Err(IoError::FormatError(
                "Insufficient data in file".to_string(),
            ));
        }

        let data_bytes = &mut self.mmap[data_offset..data_offset + data_len];
        Ok(bytemuck::cast_slice_mut(data_bytes))
    }

    /// Create a mutable ndarray view of the memory-mapped data
    pub fn as_array_view_mut(&mut self, shape: &[usize]) -> Result<ArrayViewMut<T, IxDyn>> {
        let expected_len: usize = shape.iter().product();
        if expected_len != self.len {
            return Err(IoError::FormatError(format!(
                "Shape mismatch: expected {} elements, got {}",
                expected_len, self.len
            )));
        }

        let data_slice = self.as_slice_mut()?;

        ArrayViewMut::from_shape(IxDyn(shape), data_slice)
            .map_err(|e| IoError::FormatError(format!("Failed to create array view: {}", e)))
    }

    /// Flush changes to disk
    pub fn flush(&self) -> Result<()> {
        self.mmap
            .flush()
            .map_err(|e| IoError::FileError(format!("Failed to flush memory map: {}", e)))
    }

    /// Get the number of elements in the array
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the array is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Convenience function to create a memory-mapped array from an ndarray
pub fn create_mmap_array<P, S, D, T>(path: P, array: &ArrayBase<S, D>) -> Result<()>
where
    P: AsRef<Path>,
    S: ndarray::Data<Elem = T>,
    D: Dimension,
    T: Clone + bytemuck::Pod,
{
    MmapArrayBuilder::new(&path).create_from_array(array)
}

/// Convenience function to read a memory-mapped array as an ndarray
pub fn read_mmap_array<P, T>(path: P) -> Result<ArrayD<T>>
where
    P: AsRef<Path>,
    T: bytemuck::Pod + Clone,
{
    let mmap_array = MmapArray::open(path)?;
    let shape = mmap_array.shape()?;
    let array_view = mmap_array.as_array_view(&shape)?;
    Ok(array_view.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};
    use tempfile::tempdir;

    #[test]
    fn test_mmap_array_1d() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_1d.bin");

        // Create test data
        let data = Array1::from(vec![1.0f64, 2.0, 3.0, 4.0, 5.0]);

        // Write to file
        create_mmap_array(&file_path, &data).unwrap();

        // Read back
        let mmap_array: MmapArray<f64> = MmapArray::open(&file_path).unwrap();
        let shape = mmap_array.shape().unwrap();
        assert_eq!(shape, vec![5]);

        let array_view = mmap_array.as_array_view(&shape).unwrap();
        assert_eq!(array_view.len(), 5);

        for (i, &value) in array_view.iter().enumerate() {
            assert_eq!(value, data[i]);
        }
    }

    #[test]
    fn test_mmap_array_2d() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_2d.bin");

        // Create test data
        let data = array![[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Write to file
        create_mmap_array(&file_path, &data).unwrap();

        // Read back
        let mmap_array: MmapArray<f64> = MmapArray::open(&file_path).unwrap();
        let shape = mmap_array.shape().unwrap();
        assert_eq!(shape, vec![2, 3]);

        let array_view = mmap_array.as_array_view(&shape).unwrap();
        assert_eq!(array_view.shape(), &[2, 3]);

        // Access individual elements using linear indexing
        for i in 0..2 {
            for j in 0..3 {
                let linear_index = i * 3 + j;
                assert_eq!(array_view.as_slice().unwrap()[linear_index], data[[i, j]]);
            }
        }
    }

    #[test]
    fn test_mmap_array_mutable() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_mut.bin");

        // Create test data
        let data: Array2<f64> = Array2::zeros((10, 10));

        // Write to file
        create_mmap_array(&file_path, &data).unwrap();

        // Open for writing
        let mut mmap_array: MmapArrayMut<f64> = MmapArrayMut::open(&file_path).unwrap();
        let shape = mmap_array.shape().unwrap();

        {
            let mut array_view = mmap_array.as_array_view_mut(&shape).unwrap();
            // Modify some values using linear indexing
            let slice = array_view.as_slice_mut().unwrap();
            slice[5 * 10 + 5] = 42.0; // (5, 5) in row-major order
            slice[1 * 10 + 2] = 13.7; // (1, 2) in row-major order
        }

        // Flush changes
        mmap_array.flush().unwrap();

        // Read back and verify
        let read_array: ArrayD<f64> = read_mmap_array(&file_path).unwrap();
        let read_slice = read_array.as_slice().unwrap();
        assert_eq!(read_slice[5 * 10 + 5], 42.0);
        assert_eq!(read_slice[1 * 10 + 2], 13.7);
        assert_eq!(read_slice[0], 0.0);
    }

    #[test]
    fn test_convenience_functions() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_convenience.bin");

        // Create test data
        let original = Array2::from_shape_fn((100, 50), |(i, j)| (i + j) as f64);

        // Write using convenience function
        create_mmap_array(&file_path, &original).unwrap();

        // Read using convenience function
        let read_back: ArrayD<f64> = read_mmap_array(&file_path).unwrap();

        assert_eq!(original.shape(), read_back.shape());
        for (orig, read) in original.iter().zip(read_back.iter()) {
            assert_eq!(orig, read);
        }
    }

    #[test]
    fn test_empty_array_creation() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_empty.bin");

        // Create empty array
        let shape = vec![100, 200];
        MmapArrayBuilder::new(&file_path)
            .create_empty::<f64>(&shape)
            .unwrap();

        // Verify it was created correctly
        let mmap_array = MmapArray::<f64>::open(&file_path).unwrap();
        let read_shape = mmap_array.shape().unwrap();
        assert_eq!(read_shape, shape);
        assert_eq!(mmap_array.len(), 100 * 200);

        let array_view = mmap_array.as_array_view(&shape).unwrap();
        for &value in array_view.iter() {
            assert_eq!(value, 0.0);
        }
    }
}
