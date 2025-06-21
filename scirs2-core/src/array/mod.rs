//! Array module that provides enhanced array types and utilities
//!
//! This module provides specialized array types for scientific computing:
//! - ``MaskedArray``: Arrays that can mask out values for operations
//! - ``RecordArray``: Arrays with named fields for structured data
//!
//! These types are inspired by and compatible with ``NumPy``'s masked array and record array
//! implementations, providing similar functionality in Rust.

mod masked_array;
mod record_array;

pub use masked_array::{
    is_masked, mask_array, masked_equal, masked_greater, masked_inside, masked_invalid,
    masked_less, masked_outside, masked_where, ArrayError, MaskedArray, NOMASK,
};
pub use record_array::{
    record_array_from_arrays, record_array_from_records, record_array_from_typed_arrays,
    FieldValue, Record, RecordArray,
};

/// Common array types for scientific computing
pub mod prelude {
    pub use super::{
        is_masked, mask_array, masked_equal, masked_greater, masked_inside, masked_invalid,
        masked_less, masked_outside, masked_where, record_array_from_arrays,
        record_array_from_records, record_array_from_typed_arrays, ArrayError, FieldValue,
        MaskedArray, Record, RecordArray, NOMASK,
    };
}
