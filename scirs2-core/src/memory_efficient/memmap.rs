//! Memory-mapped array implementation for efficient handling of large datasets.
//!
//! This module provides a `MemoryMappedArray` type that uses memory mapping to efficiently
//! access large datasets stored on disk. Memory mapping allows the operating system to
//! page in data as needed, reducing memory usage for very large arrays.
//!
//! Based on `NumPy`'s memmap implementation, this provides similar functionality in Rust.

use super::validation;
use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};
use bincode::{deserialize, serialize};
use memmap2::{Mmap, MmapMut, MmapOptions};
use ndarray::{Array, ArrayBase, Data, Dimension, IxDyn};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::mem;
use std::path::{Path, PathBuf};
use std::slice;
use tempfile::NamedTempFile;

/// Access mode for memory-mapped arrays
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessMode {
    /// Read-only access
    ReadOnly,
    /// Read-write access
    ReadWrite,
    /// Write access (creates a new file or overwrites existing one)
    Write,
    /// Copy-on-write access (changes not saved to disk)
    CopyOnWrite,
}

impl AccessMode {
    /// Convert to string representation
    pub const fn as_str(&self) -> &'static str {
        match self {
            AccessMode::ReadOnly => "r",
            AccessMode::ReadWrite => "r+",
            AccessMode::Write => "w+",
            AccessMode::CopyOnWrite => "c",
        }
    }
}

/// Implement FromStr for AccessMode to allow parsing from string
impl std::str::FromStr for AccessMode {
    type Err = CoreError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "r" => Ok(AccessMode::ReadOnly),
            "r+" => Ok(AccessMode::ReadWrite),
            "w+" => Ok(AccessMode::Write),
            "c" => Ok(AccessMode::CopyOnWrite),
            _ => Err(CoreError::ValidationError(
                ErrorContext::new(format!("Invalid access mode: {s}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )),
        }
    }
}

/// Memory-mapped array that efficiently maps file data directly into memory
#[derive(Debug)]
pub struct MemoryMappedArray<A>
where
    A: Clone + Copy + 'static + Send + Sync + Send + Sync,
{
    /// The shape of the array
    pub shape: Vec<usize>,
    /// The path to the mapped file
    pub file_path: PathBuf,
    /// The access mode
    pub mode: AccessMode,
    /// The offset in the file where the data starts (in bytes)
    pub offset: usize,
    /// The total number of elements
    pub size: usize,
    /// The memory-mapped data (read-only)
    pub(crate) mmap_view: Option<Mmap>,
    /// The memory-mapped data (mutable)
    pub(crate) mmap_view_mut: Option<MmapMut>,
    /// Whether the file is temporary and should be deleted on drop
    pub(crate) is_temp: bool,
    /// Phantom data for type parameters
    pub(crate) _phantom: PhantomData<A>,
}

/// Header information stored at the beginning of the file
#[derive(Serialize, Deserialize, Debug, Clone)]
struct MemoryMappedHeader {
    /// Element type size in bytes
    element_size: usize,
    /// Shape of the array
    shape: Vec<usize>,
    /// Total number of elements
    total_elements: usize,
}

impl<A> Clone for MemoryMappedArray<A>
where
    A: Clone + Copy + 'static + Send + Sync,
{
    fn clone(&self) -> Self {
        // Create a new memory mapping with the same parameters
        // This is safe because we're creating a new mapping to the same file
        Self::new::<ndarray::OwnedRepr<A>, IxDyn>(None, &self.file_path, self.mode, self.offset)
            .expect("Failed to clone memory mapped array")
    }
}

impl<A> MemoryMappedArray<A>
where
    A: Clone + Copy + 'static + Send + Sync + Send + Sync,
{
    /// Create a new reference to the same memory-mapped file
    pub fn clone_ref(&self) -> CoreResult<Self> {
        // Create a new MemoryMappedArray with the same parameters
        // This will properly initialize the mmap views
        let element_size = mem::size_of::<A>();
        let data_size = self.size * element_size;

        // Open file based on access mode
        match self.mode {
            AccessMode::ReadOnly => {
                let file = File::open(&self.file_path)
                    .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;

                let mmap = unsafe {
                    MmapOptions::new()
                        .offset(self.offset as u64)
                        .len(data_size)
                        .map(&file)
                        .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?
                };

                Ok(Self {
                    shape: self.shape.clone(),
                    file_path: self.file_path.clone(),
                    mode: self.mode,
                    offset: self.offset,
                    size: self.size,
                    mmap_view: Some(mmap),
                    mmap_view_mut: None,
                    is_temp: false,
                    _phantom: PhantomData,
                })
            }
            AccessMode::ReadWrite | AccessMode::CopyOnWrite => {
                let file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&self.file_path)
                    .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;

                let mmap = unsafe {
                    MmapOptions::new()
                        .offset(self.offset as u64)
                        .len(data_size)
                        .map_mut(&file)
                        .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?
                };

                Ok(Self {
                    shape: self.shape.clone(),
                    file_path: self.file_path.clone(),
                    mode: self.mode,
                    offset: self.offset,
                    size: self.size,
                    mmap_view: None,
                    mmap_view_mut: Some(mmap),
                    is_temp: false,
                    _phantom: PhantomData,
                })
            }
            AccessMode::Write => {
                // For Write mode, we typically shouldn't clone
                Err(CoreError::InvalidArgument(
                    ErrorContext::new("Cannot clone a write-only memory-mapped array".to_string())
                        .with_location(ErrorLocation::new(file!(), line!())),
                ))
            }
        }
    }
    /// Validate safety preconditions and create a slice from raw parts
    ///
    /// # Safety
    /// This method performs comprehensive validation before creating the slice
    fn validate_slice_creation(&self, ptr: *const A, mmap_len: usize) -> Result<&[A], CoreError> {
        // Validate safety preconditions for from_raw_parts
        if ptr.is_null() {
            return Err(CoreError::MemoryError(
                ErrorContext::new("Memory map pointer is null".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Check alignment
        if (ptr as usize) % std::mem::align_of::<A>() != 0 {
            return Err(CoreError::MemoryError(
                ErrorContext::new(format!(
                    "Memory map pointer is not properly aligned for type {} (alignment: {}, address: 0x{:x})",
                    std::any::type_name::<A>(),
                    std::mem::align_of::<A>(),
                    ptr as usize
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Check size bounds to prevent overflow
        let element_size = std::mem::size_of::<A>();
        if element_size > 0 && self.size > isize::MAX as usize / element_size {
            return Err(CoreError::MemoryError(
                ErrorContext::new(format!(
                    "Array size {} exceeds maximum safe size for slice creation",
                    self.size
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Check that we don't exceed the memory map bounds
        let total_bytes = self.size.checked_mul(element_size).ok_or_else(|| {
            CoreError::MemoryError(
                ErrorContext::new("Array size calculation overflows".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        if total_bytes > mmap_len {
            return Err(CoreError::MemoryError(
                ErrorContext::new(format!(
                    "Requested array size {} bytes exceeds memory map size {} bytes",
                    total_bytes, mmap_len
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Now it's safe to create the slice
        // SAFETY: We have validated:
        // 1. ptr is not null
        // 2. ptr is properly aligned for type A
        // 3. self.size * element_size <= isize::MAX
        // 4. the memory region is valid (within the memory map bounds)
        Ok(unsafe { slice::from_raw_parts(ptr, self.size) })
    }

    /// Get the underlying slice of data
    pub fn as_slice(&self) -> &[A] {
        match (&self.mmap_view, &self.mmap_view_mut) {
            (Some(view), _) => {
                let ptr = view.as_ptr() as *const A;
                // SAFETY: The memory map is valid for the lifetime of self
                unsafe { slice::from_raw_parts(ptr, self.size) }
            }
            (_, Some(view)) => {
                let ptr = view.as_ptr() as *const A;
                // SAFETY: The memory map is valid for the lifetime of self
                unsafe { slice::from_raw_parts(ptr, self.size) }
            }
            _ => &[],
        }
    }

    /// Open an existing memory-mapped array file
    pub fn open(file_path: &Path, shape: &[usize]) -> Result<Self, CoreError> {
        // Calculate total elements
        let size = shape.iter().product();

        // Open the file for reading
        let file = File::open(file_path)
            .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;

        // Get file size
        let file_metadata = file
            .metadata()
            .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;
        let file_size = file_metadata.len() as usize;

        // Calculate expected data size
        let element_size = mem::size_of::<A>();
        let data_size = size * element_size;

        // Check if file has enough data
        if data_size > file_size {
            return Err(CoreError::ValidationError(
                ErrorContext::new(format!(
                    "File too small for specified shape: need {} bytes, but file is only {} bytes",
                    data_size, file_size
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Create memory mapping
        let mmap = unsafe {
            MmapOptions::new()
                .len(data_size)
                .map(&file)
                .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?
        };

        Ok(Self {
            shape: shape.to_vec(),
            file_path: file_path.to_path_buf(),
            mode: AccessMode::ReadOnly,
            offset: 0,
            size,
            mmap_view: Some(mmap),
            mmap_view_mut: None,
            is_temp: false,
            _phantom: PhantomData,
        })
    }

    /// Create a new memory-mapped array from an existing array
    ///
    /// # Arguments
    ///
    /// * `data` - The source array to map to a file
    /// * `file_path` - The path to the file to create or open
    /// * `mode` - The access mode
    /// * `offset` - The offset in the file where the data should start (in bytes)
    ///
    /// # Returns
    ///
    /// A new `MemoryMappedArray` instance
    pub fn new<S, D>(
        data: Option<&ArrayBase<S, D>>,
        file_path: &Path,
        mode: AccessMode,
        offset: usize,
    ) -> Result<Self, CoreError>
    where
        S: Data<Elem = A>,
        D: Dimension,
    {
        let (shape, size) = if let Some(array) = data {
            validation::check_not_empty(array)?;
            (array.shape().to_vec(), array.len())
        } else {
            // If no data is provided, try to read the file header
            let (header, _) = read_header::<A>(file_path)?;
            (header.shape, header.total_elements)
        };

        // Calculate required file size
        let element_size = mem::size_of::<A>();
        let data_size = size * element_size;

        // Create and prepare the file depending on the mode
        match mode {
            AccessMode::ReadOnly => {
                // Open existing file for reading only
                let file = File::open(file_path)
                    .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;

                // Get file size to ensure proper mapping
                let file_metadata = file
                    .metadata()
                    .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;
                let file_size = file_metadata.len() as usize;

                // Ensure the file is large enough
                if offset + data_size > file_size {
                    return Err(CoreError::ValidationError(
                        ErrorContext::new(format!(
                            "File too small: need {needed} bytes, but file is only {file_size} bytes",
                            needed = offset + data_size
                        ))
                        .with_location(ErrorLocation::new(file!(), line!())),
                    ));
                }

                // Create a read-only memory mapping
                let mmap = unsafe {
                    MmapOptions::new()
                        .offset(offset as u64)
                        .len(data_size)
                        .map(&file)
                        .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?
                };

                Ok(Self {
                    shape,
                    file_path: file_path.to_path_buf(),
                    mode,
                    offset,
                    size,
                    mmap_view: Some(mmap),
                    mmap_view_mut: None,
                    is_temp: false,
                    _phantom: PhantomData,
                })
            }
            AccessMode::ReadWrite => {
                // Open existing file for reading and writing
                let file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(file_path)
                    .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;

                // Get file metadata to check size
                let metadata = file
                    .metadata()
                    .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;
                let file_size = metadata.len() as usize;

                // Ensure file has sufficient size before mapping
                if offset + data_size > file_size {
                    file.set_len((offset + data_size) as u64)
                        .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;
                }

                // Create a mutable memory mapping
                let mut mmap = unsafe {
                    MmapOptions::new()
                        .offset(offset as u64)
                        .len(data_size)
                        .map_mut(&file)
                        .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?
                };

                // If data is provided, write it to the mapping
                if let Some(array) = data {
                    // Convert array data to bytes
                    let bytes = unsafe {
                        slice::from_raw_parts(
                            array.as_ptr() as *const u8,
                            array.len() * mem::size_of::<A>(),
                        )
                    };

                    // Copy the data to the memory mapping
                    mmap[..].copy_from_slice(bytes);
                }

                Ok(Self {
                    shape,
                    file_path: file_path.to_path_buf(),
                    mode,
                    offset,
                    size,
                    mmap_view: None,
                    mmap_view_mut: Some(mmap),
                    is_temp: false,
                    _phantom: PhantomData,
                })
            }
            AccessMode::Write => {
                // Create or truncate file for writing
                let mut file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(file_path)
                    .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;

                // Create header
                let header = MemoryMappedHeader {
                    element_size,
                    shape: shape.clone(),
                    total_elements: size,
                };

                // Serialize header to bytes
                let header_bytes = serialize(&header).map_err(|e| {
                    CoreError::ValidationError(
                        ErrorContext::new(format!("Failed to serialize header: {}", e))
                            .with_location(ErrorLocation::new(file!(), line!())),
                    )
                })?;

                // Write header length first (8 bytes)
                let header_len = header_bytes.len() as u64;
                file.write_all(&header_len.to_le_bytes())
                    .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;

                // Write header to file
                file.write_all(&header_bytes)
                    .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;

                // Calculate total file size (header length + header + data)
                let header_size = 8 + header_bytes.len(); // 8 bytes for header length + header bytes
                let total_size = header_size + data_size;

                // Set file length to accommodate header and data
                file.set_len(total_size as u64)
                    .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;

                // Create a mutable memory mapping
                // When we have a header, the actual data starts after the header
                let data_offset = header_size + offset;
                let mut mmap = unsafe {
                    MmapOptions::new()
                        .offset(data_offset as u64)
                        .len(data_size)
                        .map_mut(&file)
                        .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?
                };

                // If data is provided, write it to the mapping
                if let Some(array) = data {
                    // Convert array data to bytes
                    let bytes = unsafe {
                        slice::from_raw_parts(
                            array.as_ptr() as *const u8,
                            array.len() * mem::size_of::<A>(),
                        )
                    };

                    // Copy the data to the memory mapping
                    mmap[..].copy_from_slice(bytes);
                }

                Ok(Self {
                    shape,
                    file_path: file_path.to_path_buf(),
                    mode,
                    offset: data_offset, // Store the actual data offset, not the requested offset
                    size,
                    mmap_view: None,
                    mmap_view_mut: Some(mmap),
                    is_temp: false,
                    _phantom: PhantomData,
                })
            }
            AccessMode::CopyOnWrite => {
                // Open existing file for reading
                let file = File::open(file_path)
                    .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;

                // Create a copy-on-write memory mapping
                let mmap = unsafe {
                    MmapOptions::new()
                        .offset(offset as u64)
                        .len(data_size)
                        .map_copy(&file)
                        .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?
                };

                Ok(Self {
                    shape,
                    file_path: file_path.to_path_buf(),
                    mode,
                    offset,
                    size,
                    mmap_view: None,
                    mmap_view_mut: Some(mmap),
                    is_temp: false,
                    _phantom: PhantomData,
                })
            }
        }
    }

    /// Create a new memory-mapped array with a temporary file
    ///
    /// # Arguments
    ///
    /// * `data` - The source array to map to a temporary file
    /// * `mode` - The access mode
    /// * `offset` - The offset in the file where the data should start (in bytes)
    ///
    /// # Returns
    ///
    /// A new `MemoryMappedArray` instance backed by a temporary file
    pub fn new_temp<S, D>(
        data: &ArrayBase<S, D>,
        mode: AccessMode,
        offset: usize,
    ) -> Result<Self, CoreError>
    where
        S: Data<Elem = A>,
        D: Dimension,
    {
        let temp_file = NamedTempFile::new()
            .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;
        let file_path = temp_file.path().to_path_buf();

        // Manually persist the temp file so it stays around after we return
        let _file = temp_file
            .persist(&file_path)
            .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;

        let mut result = Self::new(Some(data), &file_path, mode, offset)?;
        result.is_temp = true;

        Ok(result)
    }

    /// Get a view of the array data as an ndarray Array with the given dimension
    ///
    /// # Returns
    ///
    /// An ndarray Array view of the memory-mapped data
    pub fn as_array<D>(&self) -> Result<Array<A, D>, CoreError>
    where
        D: Dimension,
    {
        // Get a slice to the memory-mapped data
        let data_slice = match (&self.mmap_view, &self.mmap_view_mut) {
            (Some(view), _) => {
                // Read-only view
                let ptr = view.as_ptr() as *const A;
                self.validate_slice_creation(ptr, view.len())?
            }
            (_, Some(view)) => {
                // Mutable view
                let ptr = view.as_ptr() as *const A;
                self.validate_slice_creation(ptr, view.len())?
            }
            _ => {
                return Err(CoreError::ValidationError(
                    ErrorContext::new("Memory map is not initialized".to_string())
                        .with_location(ErrorLocation::new(file!(), line!())),
                ));
            }
        };

        // No need to create a separate dimension object - use the from_shape_vec method on Array directly
        // This approach works because we're not trying to use the dimension directly
        let shape_vec = self.shape.clone();

        // Create an array from the memory-mapped data
        let array = Array::from_shape_vec(shape_vec, data_slice.to_vec()).map_err(|e| {
            CoreError::ShapeError(
                ErrorContext::new(format!("Cannot reshape data: {}", e))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        // Convert to the requested dimension type
        let array = array.into_dimensionality::<D>().map_err(|e| {
            CoreError::ShapeError(
                ErrorContext::new(format!(
                    "Failed to convert array to requested dimension type: {}",
                    e
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        Ok(array)
    }

    /// Get a mutable view of the array data as an ndarray ArrayViewMut with the given dimension
    ///
    /// # Returns
    ///
    /// A mutable ndarray ArrayViewMut of the memory-mapped data
    ///
    /// # Errors
    ///
    /// Returns an error if the array is in read-only mode
    pub fn as_array_mut<D>(&mut self) -> Result<ndarray::ArrayViewMut<A, D>, CoreError>
    where
        D: Dimension,
    {
        if self.mode == AccessMode::ReadOnly {
            return Err(CoreError::ValidationError(
                ErrorContext::new(
                    "Cannot get mutable view of read-only memory-mapped array".to_string(),
                )
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Get a mutable slice to the memory-mapped data
        let data_slice = if let Some(view) = &mut self.mmap_view_mut {
            let ptr = view.as_mut_ptr() as *mut A;

            // Validate safety preconditions for from_raw_parts_mut
            if ptr.is_null() {
                return Err(CoreError::MemoryError(
                    ErrorContext::new("Memory map pointer is null".to_string())
                        .with_location(ErrorLocation::new(file!(), line!())),
                ));
            }

            // Check alignment
            if (ptr as usize) % std::mem::align_of::<A>() != 0 {
                return Err(CoreError::MemoryError(
                    ErrorContext::new(format!(
                        "Memory map pointer is not properly aligned for type {} (alignment: {}, address: 0x{:x})",
                        std::any::type_name::<A>(),
                        std::mem::align_of::<A>(),
                        ptr as usize
                    ))
                    .with_location(ErrorLocation::new(file!(), line!())),
                ));
            }

            // Check size bounds to prevent overflow
            let element_size = std::mem::size_of::<A>();
            if element_size > 0 && self.size > isize::MAX as usize / element_size {
                return Err(CoreError::MemoryError(
                    ErrorContext::new(format!(
                        "Array size {} exceeds maximum safe size for slice creation",
                        self.size
                    ))
                    .with_location(ErrorLocation::new(file!(), line!())),
                ));
            }

            // Check that we don't exceed the memory map bounds
            let total_bytes = self.size.checked_mul(element_size).ok_or_else(|| {
                CoreError::MemoryError(
                    ErrorContext::new("Array size calculation overflows".to_string())
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;

            if total_bytes > view.len() {
                return Err(CoreError::MemoryError(
                    ErrorContext::new(format!(
                        "Requested array size {} bytes exceeds memory map size {} bytes",
                        total_bytes,
                        view.len()
                    ))
                    .with_location(ErrorLocation::new(file!(), line!())),
                ));
            }

            // Now it's safe to create the slice
            // SAFETY: We have validated:
            // 1. ptr is not null
            // 2. ptr is properly aligned for type A
            // 3. self.size * element_size <= isize::MAX
            // 4. the memory region is valid (within the memory map bounds)
            unsafe { slice::from_raw_parts_mut(ptr, self.size) }
        } else {
            return Err(CoreError::ValidationError(
                ErrorContext::new("Mutable memory map is not initialized".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        };

        // Create a mutable array view from the memory-mapped data
        let array_view = ndarray::ArrayViewMut::from_shape(self.shape.clone(), data_slice)
            .map_err(|e| {
                CoreError::ShapeError(
                    ErrorContext::new(format!("Cannot reshape data: {}", e))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;

        // Convert to the requested dimension type
        let array_view = array_view.into_dimensionality::<D>().map_err(|e| {
            CoreError::ShapeError(
                ErrorContext::new(format!(
                    "Failed to convert array to requested dimension type: {}",
                    e
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        Ok(array_view)
    }

    /// Flush changes to disk if the array is writable
    ///
    /// # Returns
    ///
    /// `Ok(())` if the flush succeeded, or an error
    pub fn flush(&mut self) -> Result<(), CoreError> {
        if let Some(view) = &mut self.mmap_view_mut {
            view.flush()
                .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;
        }

        Ok(())
    }

    /// Reload the memory mapping from disk
    ///
    /// This function is useful when changes have been made to the underlying file
    /// by other processes or by direct file I/O operations.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the reload succeeded, or an error
    pub fn reload(&mut self) -> Result<(), CoreError> {
        // First, flush any pending changes
        let _ = self.flush();

        // Reopen the file with the original mode
        let file_path = self.file_path.clone();
        let mode = self.mode;
        let offset = self.offset;

        // Clear existing memory maps
        self.mmap_view = None;
        self.mmap_view_mut = None;

        // Create the appropriate memory mapping based on the mode
        match mode {
            AccessMode::ReadOnly => {
                // Open existing file for reading only
                let file = File::open(&file_path)
                    .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;

                // Create a read-only memory mapping
                let mmap = unsafe {
                    MmapOptions::new()
                        .offset(offset as u64)
                        .map(&file)
                        .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?
                };

                self.mmap_view = Some(mmap);
            }
            AccessMode::ReadWrite | AccessMode::Write => {
                // Open existing file for reading and writing
                let file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&file_path)
                    .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;

                // Create a mutable memory mapping
                let mmap = unsafe {
                    MmapOptions::new()
                        .offset(offset as u64)
                        .map_mut(&file)
                        .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?
                };

                self.mmap_view_mut = Some(mmap);
            }
            AccessMode::CopyOnWrite => {
                // Open existing file for reading only (copy-on-write doesn't modify the file)
                let file = File::open(&file_path)
                    .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;

                // Create a copy-on-write memory mapping
                let mmap = unsafe {
                    MmapOptions::new()
                        .offset(offset as u64)
                        .map_copy(&file)
                        .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?
                };

                self.mmap_view_mut = Some(mmap);
            }
        }

        Ok(())
    }

    /// Check if the array is temporary
    ///
    /// # Returns
    ///
    /// `true` if the array is backed by a temporary file, `false` otherwise
    pub fn is_temp(&self) -> bool {
        self.is_temp
    }

    /// Get a view of the memory-mapped data as bytes
    ///
    /// # Returns
    ///
    /// A byte slice view of the memory-mapped data
    pub fn as_bytes(&self) -> Result<&[u8], CoreError> {
        match (&self.mmap_view, &self.mmap_view_mut) {
            (Some(view), _) => {
                // Read-only view
                Ok(view)
            }
            (_, Some(view)) => {
                // Mutable view
                Ok(view)
            }
            _ => Err(CoreError::ValidationError(
                ErrorContext::new("Memory map is not initialized".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            )),
        }
    }

    /// Get a mutable view of the memory-mapped data as bytes
    ///
    /// # Returns
    ///
    /// A mutable byte slice view of the memory-mapped data
    pub fn as_bytes_mut(&mut self) -> Result<&mut [u8], CoreError> {
        if self.mode == AccessMode::ReadOnly {
            return Err(CoreError::ValidationError(
                ErrorContext::new(
                    "Cannot get mutable view of read-only memory-mapped array".to_string(),
                )
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        match &mut self.mmap_view_mut {
            Some(view) => {
                // Mutable view
                Ok(view)
            }
            _ => Err(CoreError::ValidationError(
                ErrorContext::new("Mutable memory map is not initialized".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            )),
        }
    }
}

impl<A> Drop for MemoryMappedArray<A>
where
    A: Clone + Copy + 'static + Send + Sync + Send + Sync,
{
    fn drop(&mut self) {
        // Flush any pending changes
        if let Some(view) = &mut self.mmap_view_mut {
            let _ = view.flush();
        }

        // If temporary, remove the file when done
        if self.is_temp {
            let _ = std::fs::remove_file(&self.file_path);
        }
    }
}

/// Helper function to read the header from a file
fn read_header<A: Clone + Copy + 'static + Send + Sync>(
    file_path: &Path,
) -> Result<(MemoryMappedHeader, usize), CoreError> {
    // Open the file
    let mut file =
        File::open(file_path).map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;

    // Try to read the file as a proper memory-mapped file with header
    // First, check if the file is large enough to contain a header
    let file_metadata = file
        .metadata()
        .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;
    let file_size = file_metadata.len() as usize;

    if file_size < 8 {
        // File is too small to have a proper header, treat as raw data
        let element_size = std::mem::size_of::<A>();
        let total_elements = file_size / element_size;

        let header = MemoryMappedHeader {
            element_size,
            shape: vec![total_elements],
            total_elements,
        };

        return Ok((header, 0)); // No header offset for raw files
    }

    // Try to read as a proper memory-mapped file with header
    // Read header length (first 8 bytes)
    let mut header_len_bytes = [0u8; 8];
    if file.read_exact(&mut header_len_bytes).is_err() {
        // Failed to read header length, treat as raw data
        let element_size = std::mem::size_of::<A>();
        let total_elements = file_size / element_size;

        let header = MemoryMappedHeader {
            element_size,
            shape: vec![total_elements],
            total_elements,
        };

        return Ok((header, 0)); // No header offset for raw files
    }

    let header_len = u64::from_ne_bytes(header_len_bytes) as usize;

    // Sanity check: header length should be reasonable
    if header_len > file_size || header_len > 1024 * 1024 {
        // Header length is unreasonable, treat as raw data
        let element_size = std::mem::size_of::<A>();
        let total_elements = file_size / element_size;

        let header = MemoryMappedHeader {
            element_size,
            shape: vec![total_elements],
            total_elements,
        };

        return Ok((header, 0)); // No header offset for raw files
    }

    // Read header data
    let mut header_bytes = vec![0u8; header_len];
    if file.read_exact(&mut header_bytes).is_err() {
        // Failed to read header, treat as raw data
        let element_size = std::mem::size_of::<A>();
        let total_elements = file_size / element_size;

        let header = MemoryMappedHeader {
            element_size,
            shape: vec![total_elements],
            total_elements,
        };

        return Ok((header, 0)); // No header offset for raw files
    }

    // Try to deserialize header
    match deserialize::<MemoryMappedHeader>(&header_bytes) {
        Ok(header) => {
            // Validate header makes sense
            if header.element_size == std::mem::size_of::<A>() {
                Ok((header, 8 + header_len))
            } else {
                // Header element size doesn't match, treat as raw data
                let element_size = std::mem::size_of::<A>();
                let total_elements = file_size / element_size;

                let fallback_header = MemoryMappedHeader {
                    element_size,
                    shape: vec![total_elements],
                    total_elements,
                };

                Ok((fallback_header, 0)) // No header offset for raw files
            }
        }
        Err(_) => {
            // Failed to deserialize header, treat as raw data
            let element_size = std::mem::size_of::<A>();
            let total_elements = file_size / element_size;

            let header = MemoryMappedHeader {
                element_size,
                shape: vec![total_elements],
                total_elements,
            };

            Ok((header, 0)) // No header offset for raw files
        }
    }
}

/// Create a memory-mapped array from an existing file
///
/// # Arguments
///
/// * `file_path` - Path to the file to memory-map
/// * `mode` - Access mode (read-only, read-write, etc.)
/// * `offset` - Offset in bytes from the start of the file
///
/// # Returns
///
/// A new memory-mapped array
pub fn open_mmap<A, D>(
    file_path: &Path,
    mode: AccessMode,
    offset: usize,
) -> Result<MemoryMappedArray<A>, CoreError>
where
    A: Clone + Copy + Send + Sync + 'static,
    D: Dimension,
{
    // Read the header to get shape and element info
    let (header, header_size) = read_header::<A>(file_path)?;

    // Verify element size
    let element_size = std::mem::size_of::<A>();
    if header.element_size != element_size {
        return Err(CoreError::ValidationError(
            ErrorContext::new(format!(
                "Element size mismatch: file has {} bytes, but type requires {} bytes",
                header.element_size, element_size
            ))
            .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }

    // Calculate the effective offset (header size + user offset)
    let effective_offset = header_size + offset;

    // Create the array with the header info and effective offset
    MemoryMappedArray::<A>::new::<ndarray::OwnedRepr<A>, D>(None, file_path, mode, effective_offset)
}

/// Create a new memory-mapped array file
///
/// # Arguments
///
/// * `data` - The array data to write to the file
/// * `file_path` - Path where the memory-mapped file should be created
/// * `mode` - Access mode (should be Write for new files)
/// * `offset` - Offset in bytes from the start of the file
///
/// # Returns
///
/// A new memory-mapped array
pub fn create_mmap<A, S, D>(
    data: &ArrayBase<S, D>,
    file_path: &Path,
    mode: AccessMode,
    offset: usize,
) -> Result<MemoryMappedArray<A>, CoreError>
where
    A: Clone + Copy + 'static + Send + Sync + Send + Sync,
    S: Data<Elem = A>,
    D: Dimension,
{
    MemoryMappedArray::new(Some(data), file_path, mode, offset)
}

/// Create a new temporary memory-mapped array
///
/// # Arguments
///
/// * `data` - The array data to write to the temporary file
/// * `mode` - Access mode
/// * `offset` - Offset in bytes from the start of the file
///
/// # Returns
///
/// A new memory-mapped array backed by a temporary file
pub fn create_temp_mmap<A, S, D>(
    data: &ArrayBase<S, D>,
    mode: AccessMode,
    offset: usize,
) -> Result<MemoryMappedArray<A>, CoreError>
where
    A: Clone + Copy + 'static + Send + Sync + Send + Sync,
    S: Data<Elem = A>,
    D: Dimension,
{
    MemoryMappedArray::new_temp(data, mode, offset)
}
