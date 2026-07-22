//! HDF5 file format support
//!
//! This module provides functionality for reading and writing HDF5 (Hierarchical Data Format version 5) files.
//! HDF5 is a data model, library, and file format for storing and managing data. It supports an unlimited
//! variety of datatypes, and is designed for flexible and efficient I/O and for high volume and complex data.
//!
//! Features:
//! - Reading and writing HDF5 files
//! - Support for groups and datasets
//! - Attributes on groups and datasets
//! - Multiple datatypes (integers, floats, strings, compound types)
//! - Chunking and compression support
//! - Integration with ndarray for efficient array operations
//! - Enhanced functionality with compression and parallel I/O (see `enhanced` module)
//! - Extended data type support including complex numbers and boolean types
//! - Performance optimizations for large datasets
//!
//! # Backend
//!
//! This module is backed by [`oxih5`], a pure-Rust HDF5 implementation — no
//! `libhdf5`, no C toolchain, no `-sys` crate (COOLJAPAN Pure Rust Policy).
//! HDF5 support is therefore unconditional; the `hdf5` Cargo feature is a
//! retained no-op alias and gates nothing.
//!
//! Numeric reads are routed through the widening helpers in the private
//! `convert` submodule, which reproduce the implicit conversion the C crate
//! performed inside `Dataset::read_raw::<T>()`. oxih5's own accessors match one
//! exact datatype each, so calling them directly would restrict SciRS2 to
//! reading only datasets already stored in the requested type.
//!
//! ## Writer limitations
//!
//! Groups nest to any depth, attributes attach to groups and datasets alike,
//! and scalar and array attributes are both supported. Two gaps remain:
//!
//! - **Multi-dimensional string datasets.** [`HDF5File`](crate::hdf5::HDF5File)`::write` returns
//!   [`crate::error::IoError::UnsupportedFormat`] naming the dataset; oxih5 lays
//!   variable-length strings out along a single axis.
//! - **Compression.** [`DatasetOptions`](crate::hdf5::DatasetOptions) is recorded but not applied — oxih5's
//!   writer emits contiguous, uncompressed storage. This predates the migration;
//!   the C-backed writer ignored these options too. See the `enhanced` module
//!   for the compression-aware paths.

// `pub(crate)` rather than private: `crate::universal_reader` reads HDF5 through
// the same widening helpers, and duplicating them there would let the two paths
// drift apart on which datatypes they accept.
pub(crate) mod convert;
pub mod functions;
pub mod types;
pub mod types_2;
pub mod types_3;

/// Enhanced HDF5 functionality with compression and parallel I/O
pub mod enhanced;

// Re-export all types
pub use functions::*;
pub use types::*;
pub use types_2::*;
pub use types_3::*;

// Re-export enhanced functionality for convenience
pub use enhanced::{
    create_optimal_compression_options, read_hdf5_enhanced, write_hdf5_enhanced, CompressionStats,
    EnhancedHDF5File, ExtendedDataType, ParallelConfig,
};

#[cfg(test)]
mod tests;
