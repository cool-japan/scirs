//! Zero-copy serialization and deserialization for memory-mapped arrays.
//!
//! This module provides traits and implementations for serializing and deserializing
//! data in memory-mapped arrays with zero-copy operations. It allows efficient
//! loading and saving of data without unnecessary memory allocations.
//!
//! # Overview
//!
//! Zero-copy serialization avoids creating unnecessary copies of data by directly
//! mapping file content to memory and interpreting it in place. This approach is
//! especially beneficial for:
//!
//! - Very large datasets that don't fit comfortably in memory
//! - Applications requiring frequent access to subsets of large arrays
//! - Performance-critical code where minimizing memory copies is important
//! - Systems with limited memory resources
//!
//! # Key Features
//!
//! - Fast serialization and deserialization with minimal memory overhead
//! - Support for metadata reading/updating without loading the entire array
//! - Flexible access modes (ReadOnly, ReadWrite, CopyOnWrite)
//! - Efficient error handling and validation
//! - Seamless integration with ndarray
//!
//! # Usage Examples
//!
//! ## Saving an Array with Zero-Copy Serialization
//!
//! ```no_run
//! use ndarray::Array2;
//! use scirs2_core::memory_efficient::{MemoryMappedArray, ZeroCopySerialization};
//! use serde_json::json;
//! use std::path::Path;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a 2D array
//! let data = Array2::<f64>::from_shape_fn((100, 100), |(i, j)| (i * 100 + j) as f64);
//!
//! // Define metadata
//! let metadata = json!({
//!     "description": "Sample 2D array",
//!     "created": "2023-05-20",
//!     "dimensions": {
//!         "rows": 100,
//!         "cols": 100
//!     }
//! });
//!
//! // Save with zero-copy serialization
//! let file_path = Path::new("array_data.bin");
//! MemoryMappedArray::<f64>::save_array(&data, &file_path, Some(metadata))?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Loading an Array with Zero-Copy Deserialization
//!
//! ```no_run
//! use scirs2_core::memory_efficient::{AccessMode, MemoryMappedArray, ZeroCopySerialization};
//! use std::path::Path;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Load with zero-copy deserialization
//! let file_path = Path::new("array_data.bin");
//! let array = MemoryMappedArray::<f64>::open_zero_copy(&file_path, AccessMode::ReadOnly)?;
//!
//! // Access as a standard ndarray
//! let ndarray = array.readonly_array::<ndarray::Ix2>()?;
//! println!("Value at [10, 20]: {}", ndarray[[10, 20]]);
//! # Ok(())
//! # }
//! ```
//!
//! ## Working with Metadata
//!
//! ```no_run
//! use scirs2_core::memory_efficient::{MemoryMappedArray, ZeroCopySerialization};
//! use serde_json::json;
//! use std::path::Path;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let file_path = Path::new("array_data.bin");
//!
//! // Read metadata without loading the array
//! let metadata = MemoryMappedArray::<f64>::read_metadata(&file_path)?;
//! println!("Description: {}", metadata["description"]);
//!
//! // Update metadata without rewriting the array
//! let updated_metadata = json!({
//!     "description": "Updated sample 2D array",
//!     "created": "2023-05-20",
//!     "updated": "2023-05-21",
//!     "dimensions": {
//!         "rows": 100,
//!         "cols": 100
//!     }
//! });
//! MemoryMappedArray::<f64>::update_metadata(&file_path, updated_metadata)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Modifying Data In-Place
//!
//! ```no_run
//! use scirs2_core::memory_efficient::{AccessMode, MemoryMappedArray, ZeroCopySerialization};
//! use std::path::Path;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let file_path = Path::new("array_data.bin");
//!
//! // Load with read-write access
//! let mut array = MemoryMappedArray::<f64>::open_zero_copy(&file_path, AccessMode::ReadWrite)?;
//!
//! // Modify the array
//! {
//!     let mut ndarray = array.as_array_mut::<ndarray::Ix2>()?;
//!     
//!     // Set all diagonal elements to 1000
//!     for i in 0..100 {
//!         if i < ndarray.shape()[0] && i < ndarray.shape()[1] {
//!             ndarray[[i, i]] = 1000.0;
//!         }
//!     }
//! }
//!
//! // Flush changes to disk
//! array.flush()?;
//! # Ok(())
//! # }
//! ```
//!
//! # File Format
//!
//! Files saved with zero-copy serialization have the following structure:
//!
//! 1. Header Length (8 bytes): u64 indicating the size of the serialized header
//! 2. Header (variable size): Bincode-serialized ZeroCopyHeader struct containing:
//!    - Type name (string)
//!    - Element size in bytes (usize)
//!    - Array shape (Vec<usize>)
//!    - Total number of elements (usize)
//!    - Optional metadata (serde_json::Value or None)
//! 3. Array Data: Raw binary data of the array's elements
//!
//! This format allows efficient operations like:
//! - Reading metadata without loading the full array
//! - Updating metadata without rewriting array data
//! - Direct memory mapping of array data for zero-copy access
//!
//! # Performance Considerations
//!
//! - Zero-copy deserialization is almost instantaneous regardless of array size
//! - First access to memory-mapped data may cause page faults, impacting initial performance
//! - Subsequent accesses benefit from OS caching mechanisms
//! - Memory usage is determined by accessed data, not the entire array size
//! - Prefer sequential access patterns when possible for optimal performance
//!
//! # Custom Type Support
//!
//! This module supports creating custom zero-copy serializable types. To create
//! a custom type that works with zero-copy serialization:
//!
//! 1. Use `#[repr(C)]` or `#[repr(transparent)]` to ensure stable memory layout
//! 2. Implement `Clone + Copy`
//! 3. Only include fields that are themselves zero-copy serializable (primitives)
//! 4. Implement the `ZeroCopySerializable` trait with proper safety checks
//! 5. Optionally override `type_identifier()` for more precise type validation
//!
//! Example of a custom complex number type:
//!
//! ```
//! use std::mem;
//! use std::slice;
//! use scirs2_core::memory_efficient::ZeroCopySerializable;
//! use scirs2_core::error::{CoreResult, CoreError, ErrorContext, ErrorLocation};
//!
//! #[repr(C)]
//! #[derive(Debug, Clone, Copy, PartialEq)]
//! struct Complex64 {
//!     real: f64,
//!     imag: f64,
//! }
//!
//! impl Complex64 {
//!     fn new(real: f64, imag: f64) -> Self {
//!         Self { real, imag }
//!     }
//! }
//!
//! impl ZeroCopySerializable for Complex64 {
//!     unsafe fn from_bytes(bytes: &[u8]) -> CoreResult<Self> {
//!         if !Self::validate_bytes(bytes) {
//!             return Err(CoreError::ValidationError(
//!                 ErrorContext::new(format!(
//!                     "Invalid byte length for Complex64: expected {} got {}",
//!                     mem::size_of::<Self>(),
//!                     bytes.len()
//!                 ))
//!                 .with_location(ErrorLocation::new(file!(), line!())),
//!             ));
//!         }
//!         
//!         let ptr = bytes.as_ptr() as *const Self;
//!         Ok(*ptr)
//!     }
//!     
//!     unsafe fn as_bytes(&self) -> &[u8] {
//!         let ptr = self as *const Self as *const u8;
//!         slice::from_raw_parts(ptr, mem::size_of::<Self>())
//!     }
//!     
//!     // Optional: Override the type identifier
//!     fn type_identifier() -> &'static str {
//!         "Complex64"
//!     }
//! }
//! ```
//!
//! Once implemented, your custom type can be used with all the memory-mapped array
//! functionality, including saving and loading arrays of your type:
//!
//! ```no_run
//! use ndarray::Array2;
//! use scirs2_core::memory_efficient::{AccessMode, MemoryMappedArray, ZeroCopySerialization};
//! use serde_json::json;
//! use std::path::Path;
//!
//! # #[repr(C)]
//! # #[derive(Debug, Clone, Copy, PartialEq)]
//! # struct Complex64 { real: f64, imag: f64 }
//! # impl scirs2_core::memory_efficient::ZeroCopySerializable for Complex64 {
//! #     unsafe fn from_bytes(bytes: &[u8]) -> scirs2_core::error::CoreResult<Self> { unimplemented!() }
//! #     unsafe fn as_bytes(&self) -> &[u8] { unimplemented!() }
//! # }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a 2D array of Complex64 numbers
//! let data = Array2::<Complex64>::from_shape_fn((10, 10), |(i, j)| {
//!     Complex64 { real: i as f64, imag: j as f64 }
//! });
//!
//! // Save with metadata
//! let file_path = Path::new("complex_array.bin");
//! let metadata = json!({
//!     "description": "Complex number array",
//!     "type": "Complex64"
//! });
//!
//! // Save the array with zero-copy serialization
//! MemoryMappedArray::<Complex64>::save_array(&data, &file_path, Some(metadata))?;
//!
//! // Load with zero-copy deserialization
//! let array = MemoryMappedArray::<Complex64>::open_zero_copy(&file_path, AccessMode::ReadOnly)?;
//!
//! // Access as an ndarray
//! let loaded_data = array.readonly_array::<ndarray::Ix2>()?;
//! println!("First element: real={}, imag={}", loaded_data[[0, 0]].real, loaded_data[[0, 0]].imag);
//! # Ok(())
//! # }
//! ```
//!
//! ## Safety Considerations
//!
//! When implementing `ZeroCopySerializable` for custom types, be aware of:
//!
//! - **Platform dependencies**: Memory layout can vary across platforms, so files may not be portable
//! - **Endianness**: Byte order can differ between processors (e.g., little-endian vs. big-endian)
//! - **Padding**: Ensure your type doesn't contain undefined padding bytes
//! - **Pointers**: Avoid references or pointers in custom types as they cannot be serialized safely
//!
//! For maximum compatibility, consider:
//!
//! - Using explicit byte conversions for platform-independent serialization
//! - Converting endianness explicitly (using `to_ne_bytes()` and `from_ne_bytes()`)
//! - Adding a version field to your serialized format for future compatibility

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::mem;
use std::path::{Path, PathBuf};
use std::slice;

use memmap2::{Mmap, MmapMut, MmapOptions};
use ndarray::{Array, Dimension};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use super::memmap::{AccessMode, MemoryMappedArray};
use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};

/// Trait for zero-copy serializable types.
///
/// This trait enables types to be directly mapped between memory and disk without
/// intermediate copies. It's primarily designed for numeric types and other types
/// with stable memory representations.
///
/// Implementations of this trait provide:
/// - Direct memory layout access through raw byte slices
/// - Safe conversion between bytes and typed values
/// - Validation of serialized data for type safety
/// - Size information for memory allocation and validation
///
/// This trait is optimized for performance-critical code where avoiding
/// memory copies is essential, especially with large datasets.
///
/// # Safety Considerations
///
/// Zero-copy serialization relies on the binary representation of types,
/// which depends on:
///
/// - Memory layout (which can differ across platforms)
/// - Endianness (byte order)
/// - Alignment requirements
///
/// For custom types, ensure:
/// - The type has a well-defined memory layout (e.g., #[repr(C)] or #[repr(transparent)])
/// - The type doesn't contain references, pointers, or other indirection
/// - All fields are themselves zero-copy serializable
/// - The type doesn't have any padding bytes with undefined values
pub trait ZeroCopySerializable: Sized + Clone + Copy + 'static {
    /// Convert a byte slice to an instance of this type.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it reads raw bytes and interprets them as a value
    /// of type `Self`. The caller must ensure that the byte slice is valid for the type.
    unsafe fn from_bytes(bytes: &[u8]) -> CoreResult<Self>;

    /// Convert this value to a byte slice.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it returns a raw byte slice. The caller must ensure
    /// that the returned slice is used safely.
    unsafe fn as_bytes(&self) -> &[u8];

    /// Check if the byte slice is valid for this type.
    fn validate_bytes(bytes: &[u8]) -> bool {
        bytes.len() == mem::size_of::<Self>()
    }

    /// Get the size of this type in bytes.
    fn byte_size() -> usize {
        mem::size_of::<Self>()
    }

    /// Get a type identifier for validation during deserialization.
    ///
    /// This method provides a way to identify the type during deserialization.
    /// By default, it returns the type name, but custom implementations may override
    /// this for more specific type checking.
    fn type_identifier() -> &'static str {
        std::any::type_name::<Self>()
    }
}

/// Implement ZeroCopySerializable for common numeric types

/// Macro to implement ZeroCopySerializable for a primitive numeric type
/// that has from_ne_bytes and to_ne_bytes methods
macro_rules! impl_zerocopy_serializable {
    ($type:ty, $bytesize:expr, $name:expr) => {
        impl ZeroCopySerializable for $type {
            unsafe fn from_bytes(bytes: &[u8]) -> CoreResult<Self> {
                if !Self::validate_bytes(bytes) {
                    return Err(CoreError::ValidationError(
                        ErrorContext::new(format!(
                            "Invalid byte length for {}: expected {} got {}",
                            $name,
                            mem::size_of::<Self>(),
                            bytes.len()
                        ))
                        .with_location(ErrorLocation::new(file!(), line!())),
                    ));
                }
                let mut value = [0u8; $bytesize];
                value.copy_from_slice(bytes);
                Ok(<$type>::from_ne_bytes(value))
            }

            unsafe fn as_bytes(&self) -> &[u8] {
                let ptr = self as *const Self as *const u8;
                slice::from_raw_parts(ptr, mem::size_of::<Self>())
            }
        }
    };
}

// Floating-point implementations

// f32 support when requested
#[cfg(feature = "float32")]
impl_zerocopy_serializable!(f32, 4, "f32");

// f64 is always implemented when requested
#[cfg(feature = "float64")]
impl_zerocopy_serializable!(f64, 8, "f64");

// Default f64 implementation when no specific float feature is enabled
#[cfg(all(not(feature = "float32"), not(feature = "float64")))]
impl_zerocopy_serializable!(f64, 8, "f64");

// Integer implementations - default 32-bit support

// Basic i32 support
#[cfg(any(
    feature = "int32",
    all(not(feature = "int64"), not(feature = "all_ints"))
))]
impl_zerocopy_serializable!(i32, 4, "i32");

// Basic u32 support
#[cfg(any(
    feature = "uint32",
    all(not(feature = "uint64"), not(feature = "all_ints"))
))]
impl_zerocopy_serializable!(u32, 4, "u32");

// 64-bit integer types when explicitly requested
#[cfg(feature = "int64")]
impl_zerocopy_serializable!(i64, 8, "i64");

#[cfg(feature = "uint64")]
impl_zerocopy_serializable!(u64, 8, "u64");

// All integer types when the 'all_ints' feature is enabled
#[cfg(feature = "all_ints")]
impl_zerocopy_serializable!(i8, 1, "i8");

#[cfg(feature = "all_ints")]
impl_zerocopy_serializable!(i16, 2, "i16");

#[cfg(feature = "all_ints")]
impl_zerocopy_serializable!(i32, 4, "i32");

#[cfg(feature = "all_ints")]
impl_zerocopy_serializable!(i64, 8, "i64");

#[cfg(feature = "all_ints")]
impl_zerocopy_serializable!(u8, 1, "u8");

#[cfg(feature = "all_ints")]
impl_zerocopy_serializable!(u16, 2, "u16");

#[cfg(feature = "all_ints")]
impl_zerocopy_serializable!(u32, 4, "u32");

#[cfg(feature = "all_ints")]
impl_zerocopy_serializable!(u64, 8, "u64");

// This default implementation block is no longer needed as we have handled
// the default cases directly in the conditional compilation flags above.
// The approach ensures that at minimum, we'll have f64, i32, and u32 types
// available even if no specific feature flags are enabled.

/// Metadata for zero-copy serialized arrays
#[derive(Serialize, Deserialize, Debug, Clone)]
struct ZeroCopyHeader {
    /// Type name of the elements (for validation)
    pub type_name: String,
    /// Type identifier provided by the type for validation
    pub type_identifier: String,
    /// Size of each element in bytes
    pub element_size: usize,
    /// Shape of the array
    pub shape: Vec<usize>,
    /// Total number of elements
    pub total_elements: usize,
    /// Optional extra metadata
    pub metadata: Option<serde_json::Value>,
}

/// An extension trait for MemoryMappedArray to support zero-copy serialization and deserialization.
///
/// This trait provides methods to save and load memory-mapped arrays with zero-copy
/// operations, enabling efficient serialization of large datasets. The zero-copy approach
/// allows data to be memory-mapped directly from files with minimal overhead.
///
/// Key features:
/// - Efficient saving of arrays to disk with optional metadata
/// - Near-instantaneous loading of arrays from disk via memory mapping
/// - Support for different access modes (ReadOnly, ReadWrite, CopyOnWrite)
/// - Direct access to raw byte representation for advanced use cases
/// - Access to the array with specified dimensionality
pub trait ZeroCopySerialization<A: ZeroCopySerializable> {
    /// Save the array to a file with zero-copy serialization.
    ///
    /// This method serializes the memory-mapped array to a file, including:
    /// - A header with array information (type, shape, size)
    /// - Optional metadata as JSON (can be used for array description, creation date, etc.)
    /// - The raw binary data of the array
    ///
    /// # Arguments
    ///
    /// * `path` - Path where the array will be saved
    /// * `metadata` - Optional metadata to include with the array (as JSON)
    ///
    /// # Returns
    ///
    /// `CoreResult<()>` indicating success or an error with context
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use scirs2_core::memory_efficient::{MemoryMappedArray, ZeroCopySerialization};
    /// # use serde_json::json;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let mmap: MemoryMappedArray<f64> = unimplemented!();
    /// let metadata = json!({"description": "Example array", "created": "2023-05-20"});
    /// mmap.save_zero_copy("array.bin", Some(metadata))?;
    /// # Ok(())
    /// # }
    /// ```
    fn save_zero_copy(
        &self,
        path: impl AsRef<Path>,
        metadata: Option<serde_json::Value>,
    ) -> CoreResult<()>;

    /// Load an array from a file with zero-copy deserialization.
    ///
    /// This method memory-maps a file containing a previously serialized array,
    /// allowing near-instantaneous "loading" regardless of array size.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the file containing the serialized array
    /// * `mode` - Access mode (ReadOnly, ReadWrite, or CopyOnWrite)
    ///
    /// # Returns
    ///
    /// A memory-mapped array or an error with context
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use scirs2_core::memory_efficient::{AccessMode, MemoryMappedArray, ZeroCopySerialization};
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let array = MemoryMappedArray::<f64>::load_zero_copy("array.bin", AccessMode::ReadOnly)?;
    /// # Ok(())
    /// # }
    /// ```
    fn load_zero_copy(path: impl AsRef<Path>, mode: AccessMode)
        -> CoreResult<MemoryMappedArray<A>>;

    /// Get the raw byte representation of the array.
    ///
    /// This provides low-level access to the memory-mapped data as a byte slice.
    /// Primarily used for implementing serialization operations.
    ///
    /// # Returns
    ///
    /// A byte slice representing the raw array data or an error
    fn as_bytes_slice(&self) -> CoreResult<&[u8]>;

    /// Get the mutable raw byte representation of the array.
    ///
    /// This provides low-level mutable access to the memory-mapped data.
    /// Primarily used for implementing serialization operations.
    ///
    /// # Returns
    ///
    /// A mutable byte slice or an error if the array is not mutable
    fn as_bytes_slice_mut(&mut self) -> CoreResult<&mut [u8]>;
}

impl<A: ZeroCopySerializable> ZeroCopySerialization<A> for MemoryMappedArray<A> {
    fn save_zero_copy(
        &self,
        path: impl AsRef<Path>,
        metadata: Option<serde_json::Value>,
    ) -> CoreResult<()> {
        let path = path.as_ref();

        // Create header
        let header = ZeroCopyHeader {
            type_name: std::any::type_name::<A>().to_string(),
            type_identifier: A::type_identifier().to_string(),
            element_size: mem::size_of::<A>(),
            shape: self.shape.clone(),
            total_elements: self.size,
            metadata,
        };

        // Serialize header
        let header_bytes = bincode::serialize(&header).map_err(|e| {
            CoreError::SerializationError(
                ErrorContext::new(format!("Failed to serialize header: {}", e))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        // Write header and array data
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .map_err(CoreError::IoError)?;

        // Write header length (for easier reading later)
        let header_len = header_bytes.len() as u64;
        file.write_all(&header_len.to_ne_bytes())
            .map_err(CoreError::IoError)?;

        // Write header
        file.write_all(&header_bytes).map_err(CoreError::IoError)?;

        // Get array bytes
        let array_bytes = self.as_bytes_slice()?;

        // Write array data
        file.write_all(array_bytes).map_err(CoreError::IoError)?;

        Ok(())
    }

    fn load_zero_copy(
        path: impl AsRef<Path>,
        mode: AccessMode,
    ) -> CoreResult<MemoryMappedArray<A>> {
        let path = path.as_ref();

        // Open file
        let mut file = File::open(path).map_err(CoreError::IoError)?;

        // Read header length
        let mut header_len_bytes = [0u8; 8]; // u64
        file.read_exact(&mut header_len_bytes)
            .map_err(CoreError::IoError)?;
        let header_len = u64::from_ne_bytes(header_len_bytes) as usize;

        // Read header
        let mut header_bytes = vec![0u8; header_len];
        file.read_exact(&mut header_bytes)
            .map_err(CoreError::IoError)?;

        // Deserialize header
        let header: ZeroCopyHeader = bincode::deserialize(&header_bytes).map_err(|e| {
            CoreError::SerializationError(
                ErrorContext::new(format!("Failed to deserialize header: {}", e))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        // Validate type
        if header.element_size != mem::size_of::<A>() {
            return Err(CoreError::ValidationError(
                ErrorContext::new(format!(
                    "Element size mismatch: expected {} got {}",
                    mem::size_of::<A>(),
                    header.element_size
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Validate type identifier
        if header.type_identifier != A::type_identifier() {
            return Err(CoreError::ValidationError(
                ErrorContext::new(format!(
                    "Type identifier mismatch: expected '{}' got '{}'",
                    A::type_identifier(),
                    header.type_identifier
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Calculate data offset (8 bytes for header length + header bytes)
        let data_offset = 8 + header_len;

        // Memory map file at the offset of the actual data
        match mode {
            AccessMode::ReadOnly => {
                // Create read-only memory map
                let file = File::open(path).map_err(CoreError::IoError)?;
                let mmap = unsafe {
                    MmapOptions::new()
                        .offset(data_offset as u64)
                        .map(&file)
                        .map_err(CoreError::IoError)?
                };

                // Create MemoryMappedArray
                Ok(MemoryMappedArray {
                    shape: header.shape,
                    file_path: path.to_path_buf(),
                    mode,
                    offset: data_offset,
                    size: header.total_elements,
                    mmap_view: Some(mmap),
                    mmap_view_mut: None,
                    is_temp: false,
                    _phantom: PhantomData,
                })
            }
            AccessMode::ReadWrite => {
                // Create read-write memory map
                let file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(path)
                    .map_err(CoreError::IoError)?;

                let mmap = unsafe {
                    MmapOptions::new()
                        .offset(data_offset as u64)
                        .map_mut(&file)
                        .map_err(CoreError::IoError)?
                };

                // Create MemoryMappedArray
                Ok(MemoryMappedArray {
                    shape: header.shape,
                    file_path: path.to_path_buf(),
                    mode,
                    offset: data_offset,
                    size: header.total_elements,
                    mmap_view: None,
                    mmap_view_mut: Some(mmap),
                    is_temp: false,
                    _phantom: PhantomData,
                })
            }
            AccessMode::CopyOnWrite => {
                // Create copy-on-write memory map
                let file = File::open(path).map_err(CoreError::IoError)?;
                let mmap = unsafe {
                    MmapOptions::new()
                        .offset(data_offset as u64)
                        .map_copy(&file)
                        .map_err(CoreError::IoError)?
                };

                // Create MemoryMappedArray
                Ok(MemoryMappedArray {
                    shape: header.shape,
                    file_path: path.to_path_buf(),
                    mode,
                    offset: data_offset,
                    size: header.total_elements,
                    mmap_view: None,
                    mmap_view_mut: Some(mmap),
                    is_temp: false,
                    _phantom: PhantomData,
                })
            }
            AccessMode::Write => {
                return Err(CoreError::ValidationError(
                    ErrorContext::new("Cannot use Write mode with load_zero_copy".to_string())
                        .with_location(ErrorLocation::new(file!(), line!())),
                ));
            }
        }
    }

    fn as_bytes_slice(&self) -> CoreResult<&[u8]> {
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

    fn as_bytes_slice_mut(&mut self) -> CoreResult<&mut [u8]> {
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

// Extension methods for MemoryMappedArray
impl<A: ZeroCopySerializable> MemoryMappedArray<A> {
    /// Create a new memory-mapped array from an existing array and save with zero-copy serialization.
    ///
    /// This method provides a convenient way to convert a standard ndarray to a memory-mapped
    /// array with zero-copy serialization in a single operation. It's particularly useful for
    /// initializing memory-mapped arrays with data.
    ///
    /// # Arguments
    ///
    /// * `data` - The source ndarray to be converted and saved
    /// * `file_path` - Path where the memory-mapped array will be saved
    /// * `metadata` - Optional metadata to include with the array
    ///
    /// # Returns
    ///
    /// A new memory-mapped array with read-write access to the saved data
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ndarray::Array2;
    /// # use scirs2_core::memory_efficient::MemoryMappedArray;
    /// # use serde_json::json;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// // Create an ndarray
    /// let data = Array2::<f64>::from_shape_fn((100, 100), |(i, j)| (i * 100 + j) as f64);
    ///
    /// // Create metadata
    /// let metadata = json!({"description": "Temperature data", "units": "Celsius"});
    ///
    /// // Convert to memory-mapped array and save
    /// let mmap = MemoryMappedArray::<f64>::save_array(&data, "temperature.bin", Some(metadata))?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn save_array<S, D>(
        data: &ndarray::ArrayBase<S, D>,
        file_path: impl AsRef<Path>,
        metadata: Option<serde_json::Value>,
    ) -> CoreResult<Self>
    where
        S: ndarray::Data<Elem = A>,
        D: Dimension,
    {
        // First create a temporary in-memory memory-mapped array
        let mmap = super::memmap::create_temp_mmap(data, AccessMode::ReadWrite, 0)?;

        // Save to the specified file with zero-copy serialization
        mmap.save_zero_copy(file_path, metadata)?;

        // Open the file we just created with read-write access
        Self::load_zero_copy(file_path, AccessMode::ReadWrite)
    }

    /// Open a zero-copy serialized memory-mapped array from a file.
    ///
    /// This is a convenient wrapper around the `load_zero_copy` method with a more intuitive name.
    /// It memory-maps a file containing a previously serialized array, providing efficient access
    /// to the data.
    ///
    /// # Arguments
    ///
    /// * `file_path` - Path to the file containing the serialized array
    /// * `mode` - Access mode (ReadOnly, ReadWrite, or CopyOnWrite)
    ///
    /// # Returns
    ///
    /// A memory-mapped array or an error with context
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use scirs2_core::memory_efficient::{AccessMode, MemoryMappedArray};
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// // Open a memory-mapped array with read-only access
    /// let array = MemoryMappedArray::<f64>::open_zero_copy("data/temperature.bin", AccessMode::ReadOnly)?;
    ///
    /// // Access the array
    /// let ndarray = array.readonly_array::<ndarray::Ix2>()?;
    /// println!("First value: {}", ndarray[[0, 0]]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn open_zero_copy(file_path: impl AsRef<Path>, mode: AccessMode) -> CoreResult<Self> {
        Self::load_zero_copy(file_path, mode)
    }

    /// Read the metadata from a zero-copy serialized file without loading the entire array.
    ///
    /// This method efficiently extracts just the metadata from a file without memory-mapping
    /// the entire array data. This is useful for checking array properties or file information
    /// before deciding whether to load the full array.
    ///
    /// # Arguments
    ///
    /// * `file_path` - Path to the file containing the serialized array
    ///
    /// # Returns
    ///
    /// The metadata as a JSON value or an empty JSON object if no metadata was stored
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use scirs2_core::memory_efficient::MemoryMappedArray;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// // Read metadata without loading the array
    /// let metadata = MemoryMappedArray::<f64>::read_metadata("data/large_dataset.bin")?;
    ///
    /// // Check properties
    /// if let Some(created) = metadata.get("created") {
    ///     println!("Dataset created on: {}", created);
    /// }
    ///
    /// if let Some(dimensions) = metadata.get("dimensions") {
    ///     println!("Dataset dimensions: {}", dimensions);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_metadata(file_path: impl AsRef<Path>) -> CoreResult<serde_json::Value> {
        let path = file_path.as_ref();

        // Open file
        let mut file = File::open(path).map_err(CoreError::IoError)?;

        // Read header length
        let mut header_len_bytes = [0u8; 8]; // u64
        file.read_exact(&mut header_len_bytes)
            .map_err(CoreError::IoError)?;
        let header_len = u64::from_ne_bytes(header_len_bytes) as usize;

        // Read header
        let mut header_bytes = vec![0u8; header_len];
        file.read_exact(&mut header_bytes)
            .map_err(CoreError::IoError)?;

        // Deserialize header
        let header: ZeroCopyHeader = bincode::deserialize(&header_bytes).map_err(|e| {
            CoreError::SerializationError(
                ErrorContext::new(format!("Failed to deserialize header: {}", e))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        // Return metadata or empty object if none
        Ok(header.metadata.unwrap_or_else(|| serde_json::json!({})))
    }

    /// Get a read-only view of the array as an ndarray Array.
    ///
    /// This method provides a convenient way to access the memory-mapped array as a
    /// standard ndarray Array with the specified dimensionality.
    ///
    /// # Type Parameters
    ///
    /// * `D` - The dimensionality for the returned array (e.g., Ix1, Ix2, IxDyn)
    ///
    /// # Returns
    ///
    /// A read-only ndarray Array view of the memory-mapped data
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ndarray::Ix2;
    /// # use scirs2_core::memory_efficient::{AccessMode, MemoryMappedArray};
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let array = MemoryMappedArray::<f64>::open_zero_copy("matrix.bin", AccessMode::ReadOnly)?;
    ///
    /// // Access as a 2D ndarray
    /// let ndarray = array.readonly_array::<Ix2>()?;
    ///
    /// // Now you can use all the ndarray methods
    /// let sum = ndarray.sum();
    /// let mean = ndarray.mean().unwrap_or(0.0);
    /// println!("Matrix sum: {}, mean: {}", sum, mean);
    /// # Ok(())
    /// # }
    /// ```
    pub fn readonly_array<D>(&self) -> CoreResult<Array<A, D>>
    where
        D: Dimension,
    {
        self.as_array::<D>()
    }

    /// Update metadata in a zero-copy serialized file without rewriting the entire array.
    ///
    /// This method efficiently updates just the metadata portion of a serialized array file
    /// without touching the actual array data. When possible, it performs the update in place
    /// to avoid creating a new file.
    ///
    /// # Arguments
    ///
    /// * `file_path` - Path to the file containing the serialized array
    /// * `metadata` - The new metadata to store (as JSON)
    ///
    /// # Returns
    ///
    /// `CoreResult<()>` indicating success or an error with context
    ///
    /// # Behavior
    ///
    /// - If the new metadata is the same size or smaller than the original, the update is done in-place
    /// - If the new metadata is larger, the entire file is rewritten to maintain proper alignment
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use scirs2_core::memory_efficient::MemoryMappedArray;
    /// # use serde_json::json;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// // Add processing information to the metadata
    /// let updated_metadata = json!({
    ///     "description": "Temperature dataset",
    ///     "processed": true,
    ///     "processing_date": "2023-05-21",
    ///     "normalization_applied": true,
    ///     "outliers_removed": 12
    /// });
    ///
    /// // Update the metadata without affecting the array data
    /// MemoryMappedArray::<f64>::update_metadata("data/temperature.bin", updated_metadata)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn update_metadata(
        file_path: impl AsRef<Path>,
        metadata: serde_json::Value,
    ) -> CoreResult<()> {
        let path = file_path.as_ref();

        // Open file
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(CoreError::IoError)?;

        // Read header length
        let mut header_len_bytes = [0u8; 8]; // u64
        file.read_exact(&mut header_len_bytes)
            .map_err(CoreError::IoError)?;
        let header_len = u64::from_ne_bytes(header_len_bytes) as usize;

        // Read header
        let mut header_bytes = vec![0u8; header_len];
        file.read_exact(&mut header_bytes)
            .map_err(CoreError::IoError)?;

        // Deserialize header
        let mut header: ZeroCopyHeader = bincode::deserialize(&header_bytes).map_err(|e| {
            CoreError::SerializationError(
                ErrorContext::new(format!("Failed to deserialize header: {}", e))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        // Update metadata
        header.metadata = Some(metadata);

        // Serialize updated header
        let new_header_bytes = bincode::serialize(&header).map_err(|e| {
            CoreError::SerializationError(
                ErrorContext::new(format!("Failed to serialize header: {}", e))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        // If new header is same size or smaller, we can update in place
        if new_header_bytes.len() <= header_len {
            // Seek back to header start (after header length)
            file.seek(SeekFrom::Start(8)).map_err(CoreError::IoError)?;

            // Write new header
            file.write_all(&new_header_bytes)
                .map_err(CoreError::IoError)?;

            // If new header is smaller, pad with zeros to maintain original size
            if new_header_bytes.len() < header_len {
                let padding = vec![0u8; header_len - new_header_bytes.len()];
                file.write_all(&padding).map_err(CoreError::IoError)?;
            }

            Ok(())
        } else {
            // If new header is larger, we need to rewrite the entire file
            // First, load the array
            let array = MemoryMappedArray::<A>::load_zero_copy(path, AccessMode::ReadOnly)?;

            // Then save it to a temporary file
            let temp_path = PathBuf::from(format!("{}.temp", path.display()));
            array.save_zero_copy(&temp_path, Some(metadata))?;

            // Replace the original file with the temporary file
            std::fs::rename(&temp_path, path).map_err(CoreError::IoError)?;

            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, Array1, Array2, Array3, ArrayD, Dim, IxDyn};
    use std::fs::File;
    use tempfile::tempdir;

    // Example of a custom complex number type that implements ZeroCopySerializable
    #[repr(C)]
    #[derive(Debug, Copy, Clone, PartialEq)]
    struct Complex64 {
        real: f64,
        imag: f64,
    }

    impl Complex64 {
        fn new(real: f64, imag: f64) -> Self {
            Self { real, imag }
        }

        fn magnitude(&self) -> f64 {
            (self.real * self.real + self.imag * self.imag).sqrt()
        }
    }

    // Implementation of ZeroCopySerializable for our custom Complex64 type
    impl ZeroCopySerializable for Complex64 {
        unsafe fn from_bytes(bytes: &[u8]) -> CoreResult<Self> {
            if !Self::validate_bytes(bytes) {
                return Err(CoreError::ValidationError(
                    ErrorContext::new(format!(
                        "Invalid byte length for Complex64: expected {} got {}",
                        mem::size_of::<Self>(),
                        bytes.len()
                    ))
                    .with_location(ErrorLocation::new(file!(), line!())),
                ));
            }

            // Create a pointer to the bytes and cast it to our type
            let ptr = bytes.as_ptr() as *const Self;
            Ok(*ptr)
        }

        unsafe fn as_bytes(&self) -> &[u8] {
            let ptr = self as *const Self as *const u8;
            slice::from_raw_parts(ptr, mem::size_of::<Self>())
        }

        // Override the type identifier for more specific validation
        fn type_identifier() -> &'static str {
            "Complex64"
        }
    }

    // Test for our custom complex number type
    #[test]
    fn test_custom_complex_type() {
        // Create a complex number
        let complex = Complex64::new(3.14, 2.71);

        // Test zero-copy serialization
        unsafe {
            let bytes = complex.as_bytes();
            assert_eq!(bytes.len(), 16); // 2 * f64 = 16 bytes

            let deserialized = Complex64::from_bytes(bytes).unwrap();
            assert_eq!(complex.real, deserialized.real);
            assert_eq!(complex.imag, deserialized.imag);
        }
    }

    // Test saving and loading an array of our custom type
    #[test]
    fn test_save_and_load_complex_array() {
        // Create a temporary directory
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("complex_array.bin");

        // Create a 2D array of complex numbers
        let data =
            Array2::<Complex64>::from_shape_fn((5, 5), |(i, j)| Complex64::new(i as f64, j as f64));

        // Save with metadata
        let metadata = serde_json::json!({
            "description": "Complex number array",
            "type": "Complex64",
            "shape": [5, 5]
        });

        let array =
            MemoryMappedArray::<Complex64>::save_array(&data, &file_path, Some(metadata.clone()))
                .unwrap();

        // Verify save worked
        assert_eq!(array.shape(), data.shape());
        assert_eq!(array.size, data.len());

        // Load from file
        let loaded =
            MemoryMappedArray::<Complex64>::open_zero_copy(&file_path, AccessMode::ReadOnly)
                .unwrap();

        // Verify load worked
        assert_eq!(loaded.shape(), data.shape());
        assert_eq!(loaded.size, data.len());

        // Convert to ndarray and check values
        let loaded_array = loaded.readonly_array::<ndarray::Ix2>().unwrap();

        for i in 0..5 {
            for j in 0..5 {
                let original = data[[i, j]];
                let loaded = loaded_array[[i, j]];
                assert_eq!(original.real, loaded.real);
                assert_eq!(original.imag, loaded.imag);
            }
        }

        // Read metadata
        let loaded_metadata = MemoryMappedArray::<Complex64>::read_metadata(&file_path).unwrap();
        assert_eq!(loaded_metadata, metadata);
    }

    #[test]
    fn test_zero_copy_serializable_f32() {
        let value: f32 = 3.14159;

        unsafe {
            let bytes = value.as_bytes();
            assert_eq!(bytes.len(), 4);

            let deserialized = f32::from_bytes(bytes).unwrap();
            assert_eq!(value, deserialized);
        }
    }

    #[test]
    fn test_zero_copy_serializable_i32() {
        let value: i32 = -42;

        unsafe {
            let bytes = value.as_bytes();
            assert_eq!(bytes.len(), 4);

            let deserialized = i32::from_bytes(bytes).unwrap();
            assert_eq!(value, deserialized);
        }
    }

    #[test]
    fn test_save_and_load_array_1d() {
        // Create a temporary directory
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_array.bin");

        // Create a 1D array
        let data = Array1::<f64>::linspace(0.0, 9.9, 100);

        // Save with metadata
        let metadata = serde_json::json!({
            "description": "Test 1D array",
            "created": "2023-05-20",
        });
        let array = MemoryMappedArray::<f64>::save_array(&data, &file_path, Some(metadata.clone()))
            .unwrap();

        // Verify save worked
        assert_eq!(array.shape(), data.shape());
        assert_eq!(array.size, data.len());

        // Load from file
        let loaded =
            MemoryMappedArray::<f64>::open_zero_copy(&file_path, AccessMode::ReadOnly).unwrap();

        // Verify load worked
        assert_eq!(loaded.shape(), data.shape());
        assert_eq!(loaded.size, data.len());

        // Convert to ndarray and check values
        let loaded_array = loaded.readonly_array::<ndarray::Ix1>().unwrap();
        assert_eq!(loaded_array.shape(), data.shape());

        for (i, &val) in loaded_array.iter().enumerate() {
            assert_eq!(val, data[i]);
        }

        // Read metadata
        let loaded_metadata = MemoryMappedArray::<f64>::read_metadata(&file_path).unwrap();
        assert_eq!(loaded_metadata, metadata);
    }

    #[test]
    fn test_save_and_load_array_2d() {
        // Create a temporary directory
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_array_2d.bin");

        // Create a 2D array
        let data = Array2::<f32>::from_shape_fn((10, 20), |(i, j)| (i * 20 + j) as f32);

        // Save without metadata
        let array = MemoryMappedArray::<f32>::save_array(&data, &file_path, None).unwrap();

        // Verify save worked
        assert_eq!(array.shape(), data.shape());
        assert_eq!(array.size, data.len());

        // Load from file
        let loaded =
            MemoryMappedArray::<f32>::open_zero_copy(&file_path, AccessMode::ReadOnly).unwrap();

        // Verify load worked
        assert_eq!(loaded.shape(), data.shape());
        assert_eq!(loaded.size, data.len());

        // Convert to ndarray and check values
        let loaded_array = loaded.readonly_array::<ndarray::Ix2>().unwrap();
        assert_eq!(loaded_array.shape(), data.shape());

        for i in 0..10 {
            for j in 0..20 {
                assert_eq!(loaded_array[[i, j]], data[[i, j]]);
            }
        }
    }

    #[test]
    fn test_save_and_load_array_3d() {
        // Create a temporary directory
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_array_3d.bin");

        // Create a 3D array
        let data = Array3::<i32>::from_shape_fn((5, 5, 5), |(i, j, k)| (i * 25 + j * 5 + k) as i32);

        // Save with metadata
        let metadata = serde_json::json!({
            "description": "Test 3D array",
            "dimensions": {
                "x": 5,
                "y": 5,
                "z": 5
            }
        });
        let array =
            MemoryMappedArray::<i32>::save_array(&data, &file_path, Some(metadata)).unwrap();

        // Verify save worked
        assert_eq!(array.shape(), data.shape());
        assert_eq!(array.size, data.len());

        // Load from file
        let loaded =
            MemoryMappedArray::<i32>::open_zero_copy(&file_path, AccessMode::ReadOnly).unwrap();

        // Verify load worked
        assert_eq!(loaded.shape(), data.shape());
        assert_eq!(loaded.size, data.len());

        // Convert to ndarray and check values
        let loaded_array = loaded.readonly_array::<ndarray::Ix3>().unwrap();
        assert_eq!(loaded_array.shape(), data.shape());

        for i in 0..5 {
            for j in 0..5 {
                for k in 0..5 {
                    assert_eq!(loaded_array[[i, j, k]], data[[i, j, k]]);
                }
            }
        }
    }

    #[test]
    fn test_save_and_load_array_dynamic() {
        // Create a temporary directory
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_array_dyn.bin");

        // Create a dynamic-dimension array (4D)
        let shape = IxDyn(&[3, 4, 2, 5]);
        let data = Array::from_shape_fn(shape, |idx| {
            // Convert multidimensional index to a single value for testing
            let mut val = 0;
            let mut factor = 1;
            for &i in idx.slice().iter().rev() {
                val += i * factor;
                factor *= 10;
            }
            val as f64
        });

        // Save with detailed metadata
        let metadata = serde_json::json!({
            "description": "Test dynamic 4D array",
            "dimensions": {
                "dim1": 3,
                "dim2": 4,
                "dim3": 2,
                "dim4": 5
            },
            "created": "2023-05-20",
            "format_version": "1.0"
        });
        let array =
            MemoryMappedArray::<f64>::save_array(&data, &file_path, Some(metadata)).unwrap();

        // Verify save worked
        assert_eq!(array.shape(), data.shape());
        assert_eq!(array.size, data.len());

        // Load from file
        let loaded =
            MemoryMappedArray::<f64>::open_zero_copy(&file_path, AccessMode::ReadOnly).unwrap();

        // Verify load worked
        assert_eq!(loaded.shape(), data.shape());
        assert_eq!(loaded.size, data.len());

        // Convert to ndarray and check values
        let loaded_array = loaded.readonly_array::<IxDyn>().unwrap();
        assert_eq!(loaded_array.shape(), data.shape());

        // Test a few specific indices
        let test_indices = vec![
            IxDyn(&[0, 0, 0, 0]),
            IxDyn(&[1, 2, 1, 3]),
            IxDyn(&[2, 3, 1, 4]),
            IxDyn(&[2, 0, 0, 2]),
        ];

        for idx in test_indices {
            assert_eq!(loaded_array[&idx], data[&idx]);
        }

        // Also test reading as a flattened 1D array
        let loaded_flat = loaded.readonly_array::<ndarray::Ix1>().unwrap();
        let data_flat = data.as_standard_layout().into_shape(data.len()).unwrap();

        for i in 0..data.len() {
            assert_eq!(loaded_flat[i], data_flat[i]);
        }
    }

    #[test]
    fn test_save_and_load_array_mixed_types() {
        // Create a temporary directory
        let dir = tempdir().unwrap();

        // Test different combinations of array types
        let test_cases = vec![
            ("u32_1d.bin", Array1::<u32>::linspace(0, 99, 100)),
            (
                "i64_2d.bin",
                Array2::<i64>::from_shape_fn((5, 10), |(i, j)| (i * 10 + j) as i64),
            ),
            (
                "f32_3d.bin",
                Array3::<f32>::from_shape_fn((3, 4, 5), |(i, j, k)| (i * 20 + j * 5 + k) as f32),
            ),
        ];

        for (filename, data) in test_cases {
            // Save array
            let file_path = dir.path().join(filename);
            let metadata = serde_json::json!({
                "array_type": filename.split('_').next().unwrap(),
                "dimensions": data.ndim(),
                "shape": data.shape().to_vec()
            });

            match filename {
                "u32_1d.bin" => {
                    let typed_data = data.mapv(|v| v as u32);
                    let array = MemoryMappedArray::<u32>::save_array(
                        &typed_data,
                        &file_path,
                        Some(metadata.clone()),
                    )
                    .unwrap();
                    assert_eq!(array.shape(), typed_data.shape());

                    // Load and verify
                    let loaded =
                        MemoryMappedArray::<u32>::open_zero_copy(&file_path, AccessMode::ReadOnly)
                            .unwrap();
                    let loaded_array = loaded.readonly_array::<ndarray::Ix1>().unwrap();

                    for i in 0..typed_data.len() {
                        assert_eq!(loaded_array[i], typed_data[i]);
                    }
                }
                "i64_2d.bin" => {
                    let typed_data = data.mapv(|v| v as i64);
                    let array = MemoryMappedArray::<i64>::save_array(
                        &typed_data,
                        &file_path,
                        Some(metadata.clone()),
                    )
                    .unwrap();
                    assert_eq!(array.shape(), typed_data.shape());

                    // Load and verify
                    let loaded =
                        MemoryMappedArray::<i64>::open_zero_copy(&file_path, AccessMode::ReadOnly)
                            .unwrap();
                    let loaded_array = loaded.readonly_array::<ndarray::Ix2>().unwrap();

                    for i in 0..typed_data.shape()[0] {
                        for j in 0..typed_data.shape()[1] {
                            assert_eq!(loaded_array[[i, j]], typed_data[[i, j]]);
                        }
                    }
                }
                "f32_3d.bin" => {
                    let typed_data = data.mapv(|v| v as f32);
                    let array = MemoryMappedArray::<f32>::save_array(
                        &typed_data,
                        &file_path,
                        Some(metadata.clone()),
                    )
                    .unwrap();
                    assert_eq!(array.shape(), typed_data.shape());

                    // Load and verify
                    let loaded =
                        MemoryMappedArray::<f32>::open_zero_copy(&file_path, AccessMode::ReadOnly)
                            .unwrap();
                    let loaded_array = loaded.readonly_array::<ndarray::Ix3>().unwrap();

                    for i in 0..typed_data.shape()[0] {
                        for j in 0..typed_data.shape()[1] {
                            for k in 0..typed_data.shape()[2] {
                                assert_eq!(loaded_array[[i, j, k]], typed_data[[i, j, k]]);
                            }
                        }
                    }
                }
                _ => unreachable!(),
            }

            // Verify metadata was saved correctly
            let loaded_metadata = match filename {
                "u32_1d.bin" => MemoryMappedArray::<u32>::read_metadata(&file_path).unwrap(),
                "i64_2d.bin" => MemoryMappedArray::<i64>::read_metadata(&file_path).unwrap(),
                "f32_3d.bin" => MemoryMappedArray::<f32>::read_metadata(&file_path).unwrap(),
                _ => unreachable!(),
            };
            assert_eq!(loaded_metadata, metadata);
        }
    }

    #[test]
    fn test_update_metadata() {
        // Create a temporary directory
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_metadata_update.bin");

        // Create a 1D array
        let data = Array1::<f64>::linspace(0.0, 9.9, 100);

        // Save with initial metadata
        let initial_metadata = serde_json::json!({
            "description": "Initial metadata",
            "version": "1.0"
        });
        MemoryMappedArray::<f64>::save_array(&data, &file_path, Some(initial_metadata)).unwrap();

        // Update metadata
        let updated_metadata = serde_json::json!({
            "description": "Updated metadata",
            "version": "2.0",
            "updated": true
        });
        MemoryMappedArray::<f64>::update_metadata(&file_path, updated_metadata.clone()).unwrap();

        // Read metadata
        let loaded_metadata = MemoryMappedArray::<f64>::read_metadata(&file_path).unwrap();
        assert_eq!(loaded_metadata, updated_metadata);

        // Load array and check it's still correct
        let loaded =
            MemoryMappedArray::<f64>::open_zero_copy(&file_path, AccessMode::ReadOnly).unwrap();
        let loaded_array = loaded.readonly_array::<ndarray::Ix1>().unwrap();

        for (i, &val) in loaded_array.iter().enumerate() {
            assert_eq!(val, data[i]);
        }
    }

    #[test]
    fn test_modify_array() {
        // Create a temporary directory
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_modify.bin");

        // Create a 2D array
        let data = Array2::<f32>::from_shape_fn((5, 5), |(i, j)| (i * 5 + j) as f32);

        // Save array
        MemoryMappedArray::<f32>::save_array(&data, &file_path, None).unwrap();

        // Load in read-write mode
        let mut mmap =
            MemoryMappedArray::<f32>::open_zero_copy(&file_path, AccessMode::ReadWrite).unwrap();

        // Modify array through mmap
        {
            let mut array = mmap.as_array_mut::<ndarray::Ix2>().unwrap();
            array[[2, 2]] = 999.0;
        }

        // Flush changes
        mmap.flush().unwrap();

        // Load again to verify changes were saved
        let loaded =
            MemoryMappedArray::<f32>::open_zero_copy(&file_path, AccessMode::ReadOnly).unwrap();
        let loaded_array = loaded.readonly_array::<ndarray::Ix2>().unwrap();

        // Verify only the specified element was changed
        for i in 0..5 {
            for j in 0..5 {
                if i == 2 && j == 2 {
                    assert_eq!(loaded_array[[i, j]], 999.0);
                } else {
                    assert_eq!(loaded_array[[i, j]], data[[i, j]]);
                }
            }
        }
    }

    #[test]
    fn test_copy_on_write_mode() {
        // Create a temporary directory
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_cow.bin");

        // Create a 2D array
        let data = Array2::<f64>::from_shape_fn((10, 10), |(i, j)| (i * 10 + j) as f64);

        // Save array
        MemoryMappedArray::<f64>::save_array(&data, &file_path, None).unwrap();

        // Load in copy-on-write mode
        let mut cow_mmap =
            MemoryMappedArray::<f64>::open_zero_copy(&file_path, AccessMode::CopyOnWrite).unwrap();

        // Modify array through copy-on-write view
        {
            let mut array = cow_mmap.as_array_mut::<ndarray::Ix2>().unwrap();
            // Set diagonal to 100
            for i in 0..10 {
                array[[i, i]] = 100.0;
            }
        }

        // Load the original file to verify it wasn't modified
        let original =
            MemoryMappedArray::<f64>::open_zero_copy(&file_path, AccessMode::ReadOnly).unwrap();
        let original_array = original.readonly_array::<ndarray::Ix2>().unwrap();

        // Check original values weren't changed on disk
        for i in 0..10 {
            for j in 0..10 {
                assert_eq!(original_array[[i, j]], data[[i, j]]);
            }
        }

        // Check our copy-on-write view has the modifications
        let cow_array = cow_mmap.as_array::<ndarray::Ix2>().unwrap();
        for i in 0..10 {
            for j in 0..10 {
                if i == j {
                    assert_eq!(cow_array[[i, j]], 100.0);
                } else {
                    assert_eq!(cow_array[[i, j]], data[[i, j]]);
                }
            }
        }
    }
}
