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
//! # noffi migration status
//!
//! TODO(noffi-migration): Replace `hdf5` with `oxih5-core` / `oxih5-format` (Pure Rust HDF5).
//!
//! **Read operations** can map to the `oxih5` API:
//! - `hdf5::File::open(path)` → `oxih5::File::open(path)` (returns `oxih5::File`)
//! - `file.dataset(name)` → `file.dataset(name)` (returns `oxih5_core::Dataset`)
//! - `Dataset::read_raw::<T>()` → `Dataset.data` field (pre-typed `Vec<T>`)
//! - Attribute reads → `Dataset.attributes` field (`Vec<oxih5_core::Attribute>`)
//!
//! **Blockers for full migration (remaining after M1):**
//! - `file.datasets()` / `file.groups()` tree enumeration: oxih5 exposes `file.root()?`
//!   then `Group::datasets()` / `Group::groups()`, but the internal `Group` type in oxih5
//!   uses a different traversal model — needs adapter code.
//! - `TypeDescriptor` / `Datatype::from_descriptor`: oxih5 uses `oxih5_core::Dtype` enum
//!   instead. The `convert_hdf5_datatype` and `read_dataset_data` helpers need rewriting
//!   against `Dtype` variants.
//! - `hdf5::types::VarLenUnicode` / `VarLenAscii`: no equivalent in oxih5 (pure byte slices).
//! - **Write operations** (`File::create`, `new_dataset`, `new_attr`, `write_raw`, etc.):
//!   oxih5 is read-only at M1. Keep under `#[cfg(feature = "hdf5")]` until M2+ ships.
//!   See `~/work/noffi/oxih5/` for reference API.

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
